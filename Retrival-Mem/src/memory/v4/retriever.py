from __future__ import annotations

import calendar
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from memory.clients import EmbeddingClient
from memory.v4.config import V4MemoryConfig
from memory.v4.cross_scope_visibility import is_cross_scope_visible
from memory.results import EpisodicMemoryItem, SemanticMemoryItem
from memory.v4.errors import MemoryNotReadyError
from memory.v4.failure import V4OperationContext, V4RetrievalError, retry_v4_call
from memory.v4.faiss_index import NamespaceFaissIndex
from memory.v4.schemas import (
    EXACT_MENTION_MAX_LENGTH,
    MemoryEdge,
    MemoryNode,
    RetrievalPlan,
    RetrievalProbe,
    stable_id,
)
from memory.v4.storage import V4SQLiteStore


@dataclass
class _Candidate:
    node: MemoryNode
    score: float = 0.0
    dense: float = 0.0
    sparse: float = 0.0
    entity: float = 0.0
    exact: float = 0.0
    probe_ids: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    graph_path: list[dict[str, Any]] = field(default_factory=list)
    hop: int = 0
    rank_fused: bool = False
    probe_scores: dict[str, float] = field(default_factory=dict)


class V4Retriever:
    def __init__(
        self,
        store: V4SQLiteStore,
        index: NamespaceFaissIndex,
        embedding_client: EmbeddingClient,
        config: V4MemoryConfig,
        embedding_identity: tuple[str | None, str | None] | None = None,
    ):
        self.store = store
        self.index = index
        self.embedding_client = embedding_client
        self.config = config
        self.embedding_identity = embedding_identity or (None, config.embedding_model)
        self.retrieval_call_count = 0

    def _retrieves_node(self, node: MemoryNode) -> bool:
        if getattr(self.config, "retrieve_semantic_nodes", True):
            return True
        return node.node_type not in {"chain_head", "semantic_state"}

    def _global_search(
        self, plan: RetrievalPlan, scope_id: str, state: dict[str, Any]
    ) -> dict[str, _Candidate]:
        probes = plan.probes
        embeddings = self._embed_texts(
            [probe.query for probe in probes],
            stage="retrieval_probe_embedding",
            unit_id="|".join(probe.id for probe in probes),
        )
        requested = max((probe.budget for probe in probes), default=1)
        top_k = min(state["node_count"], max(requested * 2, plan.final_top_k))
        dense_rows = self.index.search(
            scope_id, embeddings, max(1, top_k), **_index_state(state)
        )
        merged: dict[str, _Candidate] = {}
        for probe, rows in zip(probes, dense_rows):
            # Normalize each recall channel to [0, 1] per probe (peak = channel max)
            # so dense (cosine), sparse (BM25-style, unbounded) and entity scores
            # become comparable before the weighted sum. Without this, an FTS score
            # spike could dominate or be dominated by dense depending on corpus.
            dense_by_id = _normalize_scores(rows)
            if probe.must_terms or probe.should_terms:
                sparse_rows = self.store.search_fts_structured(
                    scope_id,
                    probe.must_terms,
                    probe.should_terms,
                    top_k,
                )
            else:
                sparse_rows = self.store.search_fts(scope_id, probe.query, top_k)
            entity_rows = self.store.search_entities(
                scope_id, probe.entity_hints, top_k
            )
            all_ids = list(
                dict.fromkeys(
                    [
                        *(node_id for node_id, _ in rows),
                        *(node_id for node_id, _ in sparse_rows),
                        *(node_id for node_id, _ in entity_rows),
                    ]
                )
            )
            sparse_by_id, entity_by_id = (
                _normalize_scores(sparse_rows),
                _normalize_scores(entity_rows),
            )
            candidates = []
            for node in self.store.get_nodes(all_ids, scope_id):
                if not self._retrieves_node(node):
                    continue
                if not self._hard_matches(node, probe):
                    continue
                dense = dense_by_id.get(node.id, 0.0)
                sparse = sparse_by_id.get(node.id, 0.0)
                entity = entity_by_id.get(node.id, 0.0)
                exact = self._exact_mention_score(node, probe)
                soft = self._soft_hint_score(node, probe)
                facet = max(
                    (_facet_relevance(value, node) for value in plan.evidence_facets),
                    default=0.0,
                )
                weights = _score_weights(probe)
                score = (
                    weights["dense"] * dense
                    + weights["sparse"] * sparse
                    + weights["entity"] * entity
                    + weights["exact"] * exact
                    + weights["facet"] * facet
                    + soft
                )
                candidate = _Candidate(
                    node=node,
                    score=score,
                    dense=dense,
                    sparse=sparse,
                    entity=entity,
                    exact=exact,
                    probe_ids={probe.id},
                    sources={
                        name
                        for name, value in (
                            ("dense", dense),
                            ("fts", sparse),
                            ("entity", entity),
                            ("exact", exact),
                        )
                        if value
                    },
                    probe_scores={probe.id: score},
                )
                candidates.append(candidate)
            candidates.sort(key=lambda item: (-item.score, item.node.id))
            candidates = self._ensure_full_aware_first_round_candidates(
                scope_id, probe, plan, candidates
            )
            for candidate in _bounded_probe_candidates(candidates, probe.budget):
                self._merge_candidate(merged, candidate)
        return merged

    def _embed_texts(
        self, texts: list[str], *, stage: str, unit_id: str
    ) -> list[list[float]]:
        def invoke() -> list[list[float]]:
            vectors = self.embedding_client.embed_texts(texts)
            if len(vectors) != len(texts):
                raise ValueError("retrieval embedding count does not match input count")
            dimensions = {len(vector) for vector in vectors}
            if len(dimensions) > 1 or any(not vector for vector in vectors):
                raise ValueError("retrieval embeddings have invalid dimensions")
            if any(
                not math.isfinite(float(value))
                for vector in vectors
                for value in vector
            ):
                raise ValueError("retrieval embeddings contain non-finite values")
            if any(
                sum(float(value) ** 2 for value in vector) <= 0.0 for vector in vectors
            ):
                raise ValueError("retrieval embeddings contain a zero vector")
            return [[float(value) for value in vector] for vector in vectors]

        provider, model = self.embedding_identity
        return retry_v4_call(
            invoke,
            policy=self.config.failure_policy,
            context=V4OperationContext(
                stage=stage,
                unit_id=unit_id,
                provider=provider,
                model=model,
                checkpoint_key=f"retrieval:{stage}:{unit_id}",
            ),
            error_type=V4RetrievalError,
            validation_errors=(ValueError, TypeError),
        )

    def _ensure_full_aware_first_round_candidates(
        self,
        scope_id: str,
        probe: RetrievalProbe,
        plan: RetrievalPlan,
        candidates: list[_Candidate],
    ) -> list[_Candidate]:
        if not self.config.retrieve_semantic_nodes:
            return candidates
        present = {candidate.node.node_type for candidate in candidates}
        # Only inject chain_head seeds: chain_head is a navigation hub needed as
        # an expansion anchor when no semantic hit surfaced one. event /
        # semantic_state are evidence types whose low-score seeds (0.18 lexical)
        # just pollute evidence ranking without competing, so do not inject them.
        missing = [
            node_type for node_type in ("chain_head",) if node_type not in present
        ]
        if not missing:
            return candidates
        extra: list[_Candidate] = []
        for node in self.store.list_nodes(scope_id, missing):
            if not self._hard_matches(node, probe):
                continue
            relevance = _lexical_relevance(probe.query, node)
            facet = max(
                (_facet_relevance(value, node) for value in plan.evidence_facets),
                default=0.0,
            )
            score = 0.18 * relevance + 0.05 * facet + self._soft_hint_score(node, probe)
            if node.node_type == "chain_head":
                score += 0.01
            extra.append(
                _Candidate(
                    node=node,
                    score=score,
                    dense=relevance,
                    probe_ids={probe.id},
                    sources={"type_seed"},
                    probe_scores={probe.id: score},
                )
            )
        extra.sort(key=lambda item: (-item.score, item.node.id))
        return sorted(
            [*candidates, *extra[: max(1, probe.budget)]],
            key=lambda item: (-item.score, item.node.id),
        )

    def _expand_anchor(
        self,
        namespace: str,
        scope_id: str,
        anchor: _Candidate,
        edge_types: list[str],
        direction: str,
        budget: int,
        question: str,
        query_embedding: list[float] | None = None,
        state: dict[str, Any] | None = None,
        expansion_intent: str = "",
        intent_embedding: list[float] | None = None,
    ) -> tuple[list[_Candidate], list[str]]:
        adjacency = self.store.adjacent(
            scope_id, [anchor.node.id], edge_types or None, direction
        )
        nodes = {
            node.id: node
            for node in self.store.get_nodes(
                (neighbor for _, neighbor in adjacency), scope_id
            )
        }
        # Persisted cross-scope EVENT edges: visibility is baked in at build time,
        # so these are first-class graph neighbors reachable for multi-hop traversal.
        cross_adjacency = (
            self.store.cross_scope_adjacent(
                namespace,
                scope_id,
                [anchor.node.id],
                edge_types or None,
                direction,
            )
            if self.config.cross_scope_navigation
            else []
        )
        for node in self.store.get_nodes(
            (neighbor_id for _, neighbor_id, _ in cross_adjacency),
            scope_id=None,
        ):
            nodes[node.id] = node
        intent_aware = bool(expansion_intent.strip())
        intent_scores = (
            self._expansion_intent_scores(
                nodes,
                scope_id,
                state,
                intent_embedding,
            )
            if intent_aware
            else {}
        )
        same_scope_vectors: dict[str, list[float]] = {}
        if not intent_aware and query_embedding is not None and state is not None:
            same_scope_ids = [
                nid for nid, node in nodes.items() if node.scope_id == scope_id
            ]
            if same_scope_ids:
                same_scope_vectors = self.index.vectors(
                    scope_id, same_scope_ids, **_index_state(state)
                )
        ranked_neighbors: list[tuple[_Candidate, str]] = []

        def consider(edge: MemoryEdge, neighbor_id: str, cross_scope: bool) -> None:
            node = nodes.get(neighbor_id)
            if node is None or not self._retrieves_node(node):
                return
            edge_confidence = _clamp01(edge.confidence)
            if (
                intent_aware
                and edge.trust == "derived"
                and edge_confidence < self.config.derived_edge_min_confidence
            ):
                return
            if intent_aware:
                query_relevance = intent_scores[neighbor_id]
                edge_quality = _edge_trust(edge.trust) * edge_confidence
                score = (
                    self.config.expansion_intent_weight * query_relevance
                    + self.config.expansion_edge_quality_weight * edge_quality
                    + self.config.expansion_node_confidence_weight
                    * _clamp01(node.confidence)
                )
            elif cross_scope or query_embedding is None:
                query_relevance = _lexical_relevance(question, node)
                trust_weight = _edge_trust(edge.trust)
                score = (
                    anchor.score
                    * trust_weight
                    * _edge_type_weight(edge.edge_type)
                    * 0.80
                    + 0.20 * query_relevance
                )
            else:
                vector = same_scope_vectors.get(neighbor_id)
                if vector is None:
                    raise MemoryNotReadyError(
                        f"V4 published index is missing vector for node {neighbor_id}"
                    )
                query_relevance = _cosine(query_embedding, vector)
                trust_weight = _edge_trust(edge.trust)
                score = (
                    anchor.score
                    * trust_weight
                    * _edge_type_weight(edge.edge_type)
                    * 0.80
                    + 0.20 * query_relevance
                )
            graph_path = [
                *anchor.graph_path,
                {
                    "edge_type": edge.edge_type,
                    "trust": edge.trust,
                    "confidence": edge.confidence,
                    "explanation": edge.explanation,
                    "cross_scope": cross_scope,
                },
            ]
            ranked_neighbors.append(
                (
                    _Candidate(
                        node=node,
                        score=score,
                        dense=query_relevance,
                        probe_ids=set(anchor.probe_ids),
                        sources={"graph"},
                        hop=anchor.hop + 1,
                        graph_path=graph_path,
                    ),
                    edge.trust,
                )
            )

        for edge, neighbor_id in adjacency:
            consider(edge, neighbor_id, False)
        for edge, neighbor_id, _neighbor_scope in cross_adjacency:
            consider(edge, neighbor_id, True)
        ranked_neighbors.sort(key=lambda item: (-item[0].score, item[0].node.id))
        selected = ranked_neighbors[:budget]
        return [item[0] for item in selected], [item[1] for item in selected]

    def _expansion_intent_scores(
        self,
        nodes: dict[str, MemoryNode],
        scope_id: str,
        state: dict[str, Any] | None,
        intent_embedding: list[float] | None,
    ) -> dict[str, float]:
        vectors: dict[str, list[float]] = {}
        ids_by_scope: dict[str, list[str]] = {}
        for node in nodes.values():
            ids_by_scope.setdefault(node.scope_id, []).append(node.id)
        for neighbor_scope, node_ids in ids_by_scope.items():
            index_state = (
                state
                if neighbor_scope == scope_id
                else self.store.get_scope_index_state(neighbor_scope)
            )
            if index_state is None:
                raise MemoryNotReadyError(
                    f"V4 scope {neighbor_scope!r} has no published index state"
                )
            vectors.update(
                self.index.vectors(
                    neighbor_scope, node_ids, **_index_state(index_state)
                )
            )
        scores: dict[str, float] = {}
        for node_id, node in nodes.items():
            vector = vectors.get(node_id)
            if vector is None:
                raise MemoryNotReadyError(
                    f"V4 published index is missing vector for node {node_id}"
                )
            if intent_embedding is None:
                raise ValueError("intent-aware expansion requires one intent embedding")
            scores[node_id] = _cosine(intent_embedding, vector)
        return scores

    def _virtual_lookup(
        self,
        namespace: str,
        scope_id: str,
        anchor: _Candidate,
        action: str,
        budget: int,
        *,
        constraints: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        intent_embedding: list[float] | None = None,
    ) -> list[_Candidate]:
        """Execute a bounded index lookup without materializing a graph edge."""
        constraints = dict(constraints or {})
        anchor_node = anchor.node
        fetch_limit = _virtual_fetch_limit(action, budget)
        node_types = {
            str(value)
            for value in constraints.get("node_types", ())
            if str(value) in {"event", "semantic_state"}
        }
        node_type = next(iter(node_types)) if len(node_types) == 1 else None
        max_distance_days = constraints.get("max_distance_days")
        if max_distance_days is not None:
            max_distance_days = max(0, int(max_distance_days))
        if action == "SOURCE_EVIDENCE_LOOKUP":
            return self._source_evidence_lookup(anchor, budget, constraints)
        if action == "ENTITY_LOOKUP":
            if not anchor_node.entity_ids:
                return []
            neighbors = self.store.nodes_by_entity_ids(
                namespace,
                anchor_node.entity_ids,
                node_type=node_type,
                match_all=False,
                limit=fetch_limit,
            )
        elif action == "CROSS_ENTITY_LOOKUP":
            minimum = max(2, int(constraints.get("min_entities", 2)))
            entity_ids = _query_entity_ids(
                anchor_node, str(constraints.get("query") or ""), minimum
            )
            if len(entity_ids) < minimum:
                return []
            neighbors = self.store.nodes_by_entity_ids(
                namespace,
                entity_ids,
                node_type=node_type,
                match_all=True,
                limit=fetch_limit,
            )
        elif action == "TIME_OVERLAP_LOOKUP":
            if (
                anchor_node.node_type != "event"
                or not anchor_node.event_time_start
                or not anchor_node.event_time_end
            ):
                return []
            neighbors = self.store.nodes_overlapping_time(
                namespace,
                anchor_node.event_time_start,
                anchor_node.event_time_end,
                node_type="event",
                limit=fetch_limit,
            )
        elif action == "TEMPORAL_BEFORE_LOOKUP":
            if anchor_node.node_type != "event" or not anchor_node.event_time_start:
                return []
            neighbors = self.store.nodes_before_time(
                namespace,
                anchor_node.event_time_start,
                node_type="event",
                max_distance_days=max_distance_days,
                limit=fetch_limit,
            )
        elif action == "TEMPORAL_AFTER_LOOKUP":
            if anchor_node.node_type != "event" or not anchor_node.event_time_start:
                return []
            neighbors = self.store.nodes_after_time(
                namespace,
                anchor_node.event_time_end or anchor_node.event_time_start,
                node_type="event",
                max_distance_days=max_distance_days,
                limit=fetch_limit,
            )
        elif action == "NEAREST_TIME_LOOKUP":
            if anchor_node.node_type != "event" or not anchor_node.event_time_start:
                return []
            neighbors = self.store.nodes_nearest_time(
                namespace,
                anchor_node.event_time_start,
                node_type="event",
                max_distance_days=max_distance_days,
                limit=fetch_limit,
            )
        else:
            return []

        nodes = {
            node.id: node
            for node in neighbors
            if (
                node.id != anchor_node.id
                and node.node_type != "chain_head"
                and self._retrieves_node(node)
            )
        }
        anchor_scope_participants = {
            item.participant_id for item in self.store.scope_participants(scope_id)
        }
        anchor_evidence_participants = self.store.node_evidence_participants(
            anchor_node.id
        )
        visible_nodes: dict[str, MemoryNode] = {}
        for neighbor in nodes.values():
            cross_scope = neighbor.scope_id != scope_id
            if cross_scope:
                if not self.config.cross_scope_navigation:
                    continue
                neighbor_scope_participants = {
                    item.participant_id
                    for item in self.store.scope_participants(neighbor.scope_id)
                }
                neighbor_evidence = self.store.node_evidence_participants(neighbor.id)
                if not is_cross_scope_visible(
                    anchor_scope_participants,
                    anchor_evidence_participants,
                    neighbor_scope_participants,
                    neighbor_evidence,
                ):
                    continue
            visible_nodes[neighbor.id] = neighbor

        semantic_scores: dict[str, float] = {}
        if visible_nodes and intent_embedding is not None and state is not None:
            semantic_scores = self._expansion_intent_scores(
                visible_nodes, scope_id, state, intent_embedding
            )
        query = str(constraints.get("query") or "")
        output: dict[str, _Candidate] = {}
        for neighbor in visible_nodes.values():
            cross_scope = neighbor.scope_id != scope_id
            query_relevance = semantic_scores.get(
                neighbor.id, _lexical_relevance(query, neighbor) if query else 0.0
            )
            output[neighbor.id] = self._make_navigation_candidate(
                anchor,
                neighbor,
                action,
                cross_scope=cross_scope,
                query_relevance=_clamp01(query_relevance),
                entity_coverage=_entity_coverage(anchor_node, neighbor),
                temporal_proximity=_temporal_proximity(anchor_node, neighbor, action),
                constraints=constraints,
            )
        return sorted(output.values(), key=lambda item: (-item.score, item.node.id))[
            :budget
        ]

    def _source_evidence_lookup(
        self,
        anchor: _Candidate,
        budget: int,
        constraints: dict[str, Any] | None = None,
    ) -> list[_Candidate]:
        """Materialize the original evidence records behind a navigation node.

        ``SOURCE_EVIDENCE_LOOKUP`` exists because the Controller may cite a
        ``chain_head`` (or a slimmer semantic state) whose facts are sufficient,
        while the final answer context would otherwise drop the chain head and
        lose the exact source wording. Each evidence record becomes a synthetic
        ``event`` candidate carrying the original text, speaker, participant and
        utterance timestamp, so the final delivery can cite concrete facts.
        """
        constraints = dict(constraints or {})
        anchor_node = anchor.node
        records = self.store.evidence_records_by_node_ids([anchor_node.id]).get(
            anchor_node.id, []
        )
        if not records:
            # Fall back to refs embedded on the node itself (older or compact
            # snapshots may not have materialized v4_evidence_records rows).
            records = [dict(ref) for ref in anchor_node.evidence_refs]
        query = str(constraints.get("query") or "").strip()
        output: list[_Candidate] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            content = str(record.get("content") or "").strip()
            if not content:
                continue
            record_id = str(
                record.get("id") or record.get("evidence_id") or ""
            ).strip()
            observed_at = record.get("observed_at") or anchor_node.observed_at
            participant_id = str(record.get("participant_id") or "")
            role = str(record.get("role") or "unknown")
            turn_id = str(record.get("turn_id") or "")
            node_id = stable_id(
                "source_evidence", anchor_node.id, record_id or content
            )
            source_ref = {
                "evidence_id": record_id or node_id,
                "session_id": record.get("session_id"),
                "turn_id": record.get("turn_id"),
                "role": role,
                "participant_id": participant_id,
                "observed_at": observed_at,
                "content": content,
            }
            title = "Source evidence"
            if turn_id:
                title = f"Source evidence turn {turn_id}"
            node = MemoryNode(
                id=node_id,
                namespace=anchor_node.namespace,
                scope_id=anchor_node.scope_id,
                chain_id=anchor_node.chain_id,
                topic_id=anchor_node.topic_id,
                node_type="event",
                title=title,
                summary=content,
                text=content,
                actors=[participant_id] if participant_id else [],
                event_time_start=observed_at,
                event_time_end=observed_at,
                time_precision=None,
                observed_at=observed_at,
                valid_from=observed_at,
                valid_to=observed_at,
                facts=[],
                evidence_refs=[source_ref],
                confidence=1.0,
                metadata={
                    "source_evidence_lookup": True,
                    "anchor_node_id": anchor_node.id,
                    "record_id": record_id or node_id,
                    "speaker_role": role,
                },
            )
            relevance = _lexical_relevance(query, node) if query else 0.85
            score = _clamp01(0.60 + 0.40 * relevance)
            output.append(
                _Candidate(
                    node=node,
                    score=score,
                    dense=relevance,
                    probe_ids=set(anchor.probe_ids),
                    sources={"source_evidence_lookup"},
                    hop=anchor.hop + 1,
                    graph_path=[
                        *anchor.graph_path,
                        {
                            "virtual_action": "SOURCE_EVIDENCE_LOOKUP",
                            "confidence": 1.0,
                            "cross_scope": False,
                            "query_relevance": relevance,
                            "entity_coverage": 0.0,
                            "temporal_proximity": 0.0,
                            "lookup_constraints": {
                                key: value
                                for key, value in constraints.items()
                                if key != "query"
                            },
                        },
                    ],
                )
            )
        return sorted(output, key=lambda item: (-item.score, item.node.id))[:budget]

    def _make_navigation_candidate(
        self,
        anchor: _Candidate,
        neighbor: MemoryNode,
        action: str,
        *,
        cross_scope: bool,
        query_relevance: float = 0.0,
        entity_coverage: float = 0.0,
        temporal_proximity: float = 0.0,
        constraints: dict[str, Any] | None = None,
    ) -> _Candidate:
        relation_prior = {
            "ENTITY_LOOKUP": 0.65,
            "CROSS_ENTITY_LOOKUP": 0.80,
            "TIME_OVERLAP_LOOKUP": 0.85,
            "TEMPORAL_BEFORE_LOOKUP": 0.80,
            "TEMPORAL_AFTER_LOOKUP": 0.80,
            "NEAREST_TIME_LOOKUP": 0.70,
        }.get(action, 0.50)
        score = (
            0.40 * _clamp01(query_relevance)
            + 0.25 * _clamp01(entity_coverage)
            + 0.20 * _clamp01(temporal_proximity)
            + 0.10 * _clamp01(neighbor.confidence)
            + 0.05 * relation_prior
        )
        public_constraints = {
            key: value
            for key, value in dict(constraints or {}).items()
            if key != "query"
        }
        return _Candidate(
            node=neighbor,
            score=score,
            dense=query_relevance,
            probe_ids=set(anchor.probe_ids),
            sources={"virtual_lookup"},
            hop=anchor.hop + 1,
            graph_path=[
                *anchor.graph_path,
                {
                    "virtual_action": action,
                    "confidence": relation_prior,
                    "cross_scope": cross_scope,
                    "query_relevance": query_relevance,
                    "entity_coverage": entity_coverage,
                    "temporal_proximity": temporal_proximity,
                    "lookup_constraints": public_constraints,
                },
            ],
        )

    @staticmethod
    def _merge_candidate(merged: dict[str, _Candidate], incoming: _Candidate) -> None:
        current = merged.get(incoming.node.id)
        if current is None:
            merged[incoming.node.id] = incoming
            return
        current.probe_ids.update(incoming.probe_ids)
        for probe_id, score in incoming.probe_scores.items():
            current.probe_scores[probe_id] = max(
                score, current.probe_scores.get(probe_id, float("-inf"))
            )
        current.sources.update(incoming.sources)
        current.dense = max(current.dense, incoming.dense)
        current.sparse = max(current.sparse, incoming.sparse)
        current.entity = max(current.entity, incoming.entity)
        if incoming.score > current.score:
            current.score = incoming.score
            current.graph_path = incoming.graph_path
            current.hop = incoming.hop

    def _rank(
        self,
        candidates: dict[str, _Candidate] | list[_Candidate],
        top_k: int,
        include_heads: bool = False,
    ) -> list[_Candidate]:
        values = (
            list(candidates.values())
            if isinstance(candidates, dict)
            else list(candidates)
        )
        values = [item for item in values if self._retrieves_node(item.node)]
        if not include_heads:
            values = [
                item
                for item in values
                if item.node.node_type != "chain_head"
                or (
                    item.node.facts
                    and not item.node.metadata.get("navigation_only", False)
                )
            ]
        # Greedy diversity/redundancy reranking. Relevance remains dominant, while
        # repeated topics, node types, and near-duplicate summaries receive small
        # penalties so multi-facet plans do not collapse onto one chain.
        selected = []
        topic_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        remaining = list(values)
        while remaining and len(selected) < top_k:

            def diversified_score(item: _Candidate) -> float:
                redundancy = max(
                    (
                        _semantic_node_duplicate_penalty(item.node, chosen.node)
                        for chosen in selected
                    ),
                    default=0.0,
                )
                return (
                    self._ranking_score(item)
                    - 0.04 * topic_counts.get(item.node.topic_id, 0)
                    - 0.02 * type_counts.get(item.node.node_type, 0)
                    - redundancy
                )

            item = max(
                remaining,
                key=lambda value: (
                    diversified_score(value),
                    value.node.importance,
                    value.node.id,
                ),
            )
            remaining.remove(item)
            selected.append(item)
            topic_counts[item.node.topic_id] = (
                topic_counts.get(item.node.topic_id, 0) + 1
            )
            type_counts[item.node.node_type] = (
                type_counts.get(item.node.node_type, 0) + 1
            )
        return selected

    @staticmethod
    def _ranking_score(item: _Candidate) -> float:
        score = item.score
        if not item.rank_fused:
            score += min(0.12, 0.03 * (len(item.probe_ids) - 1))
            score -= 0.08 * item.hop
        score += 0.04 * item.node.confidence
        if (
            item.node.node_type in {"chain_head", "semantic_state"}
            and item.node.valid_to is None
        ):
            score += -0.05 if item.node.metadata.get("unresolved_conflict") else 0.03
        return score

    def _memory_items(
        self, namespace: str, scope_id: str, candidates: list[_Candidate]
    ) -> list[EpisodicMemoryItem]:
        output = []
        evidence_by_node = self.store.evidence_records_by_node_ids(
            candidate.node.id for candidate in candidates
        )
        for candidate in candidates:
            node = candidate.node
            source_evidence = evidence_by_node.get(node.id, [])
            if node.metadata.get("semantic_slimming_applied"):
                source_evidence = _selected_source_evidence(
                    source_evidence, node.evidence_refs
                )
            semantic = []
            if node.node_type in {"chain_head", "semantic_state"} and node.facts:
                for fact in node.facts:
                    semantic.append(
                        SemanticMemoryItem(
                            id=f"{node.id}:{fact.get('key')}",
                            namespace=namespace,
                            subject=str(fact.get("subject") or "user"),
                            predicate=str(fact.get("predicate") or fact.get("key")),
                            object=str(fact.get("value")),
                            qualifiers={
                                "valid_from": fact.get("valid_from") or node.valid_from,
                                "valid_to": fact.get("valid_to") or node.valid_to,
                            },
                            confidence=float(fact.get("confidence", node.confidence)),
                            evidence_node_ids=[node.id],
                        )
                    )
            metadata = {
                **node.metadata,
                "scope_id": scope_id,
                "chain_id": node.chain_id,
                "topic_id": node.topic_id,
                "event_time_start": node.event_time_start,
                "event_time_end": node.event_time_end,
                "valid_from": node.valid_from,
                "valid_to": node.valid_to,
                "time_precision": node.time_precision,
                "observed_at": node.observed_at,
                "evidence_refs": node.evidence_refs,
                "retrieval_sources": sorted(candidate.sources),
                "probe_ids": sorted(candidate.probe_ids),
                "graph_path": candidate.graph_path,
                "hop": candidate.hop,
                "absolute_time_start": node.absolute_time_start,
                "absolute_time_end": node.absolute_time_end,
                "source_evidence": source_evidence,
            }
            output.append(
                EpisodicMemoryItem(
                    node_id=node.id,
                    node_type=node.node_type,
                    depth=3,
                    title=node.title,
                    summary=node.summary,
                    text=node.text,
                    timestamp_start=node.absolute_time_start,
                    timestamp_end=node.absolute_time_end,
                    entities=node.entities,
                    source_refs=node.evidence_refs,
                    metadata=metadata,
                    score=self._ranking_score(candidate),
                    semantic_memories=semantic,
                )
            )
        return output

    def _ready_state(self, scope_id: str) -> dict[str, Any]:
        state = self.store.get_scope_index_state(scope_id)
        if state is None:
            raise MemoryNotReadyError(
                f"V4 participant scope {scope_id!r} has no published index generation"
            )
        if len(self.store.list_nodes(scope_id)) != state["node_count"]:
            raise MemoryNotReadyError(
                f"V4 participant scope {scope_id!r} SQLite/index node counts differ"
            )
        if (
            self.config.cross_scope_navigation
            and not self.store.is_scope_stage_complete(
                scope_id, str(state["generation"]), "cross_scope_completion"
            )
        ):
            raise MemoryNotReadyError(
                f"V4 participant scope {scope_id!r} has incomplete cross-scope navigation"
            )
        self.index.validate(scope_id, **_index_state(state))
        return state

    @staticmethod
    def _hard_matches(node: MemoryNode, probe: RetrievalProbe) -> bool:
        constraint = probe.time
        if not constraint.hard or constraint.operator == "any":
            return True
        if node.node_type in {"chain_head", "semantic_state"} and node.facts:
            return any(_fact_matches_time(fact, node, probe) for fact in node.facts)
        node_start, inferred_node_end = _constraint_bounds(
            node.absolute_time_start, node.time_precision
        )
        _, node_end = _constraint_bounds(
            node.absolute_time_end or node.absolute_time_start, node.time_precision
        )
        node_end = node_end or inferred_node_end
        start, at_end = _constraint_bounds(constraint.start, constraint.precision)
        _, end = _constraint_bounds(
            constraint.end or constraint.start, constraint.precision
        )
        if not node_start or not node_end or not start:
            return False
        if constraint.operator == "at":
            return at_end is not None and node_end >= start and node_start <= at_end
        if constraint.operator == "before":
            return node_end < start
        if constraint.operator == "after":
            return at_end is not None and node_start > at_end
        return end is not None and node_end >= start and node_start <= end

    @staticmethod
    def _soft_hint_score(node: MemoryNode, probe: RetrievalProbe) -> float:
        score = 0.0
        if probe.node_types and node.node_type in probe.node_types:
            score += 0.05
        if probe.topic_hints:
            topic = node.topic_id.casefold()
            if any(
                hint.casefold() in topic or topic in hint.casefold()
                for hint in probe.topic_hints
            ):
                score += 0.05
        if probe.entity_hints:
            entities = {entity.casefold() for entity in node.entities}
            if any(hint.casefold() in entities for hint in probe.entity_hints):
                score += 0.05
        if (
            probe.time.operator != "any"
            and not probe.time.hard
            and node.absolute_time_start
        ):
            start, end = _constraint_bounds(probe.time.start, probe.time.precision)
            # Use the node's full precision-bounded interval (matching
            # _hard_matches) instead of a point value, so a month-precision
            # node still earns the bonus when the constraint is a date inside
            # that month.
            n_start, _ = _constraint_bounds(
                node.absolute_time_start, node.time_precision
            )
            _, n_end = _constraint_bounds(
                node.absolute_time_end or node.absolute_time_start, node.time_precision
            )
            n_end = n_end or n_start
            if (
                start
                and n_start
                and n_end
                and (not end or (n_start <= end and n_end >= start))
            ):
                score += 0.06
        # Cap stacked soft hints so they cannot overpower the dense score on
        # weakly-matched nodes (worst case ~0.21 vs dense weight 0.60).
        return min(score, 0.12)

    @staticmethod
    def _exact_mention_score(node: MemoryNode, probe: RetrievalProbe) -> float:
        """Reward verbatim hits against a node's short exact_mentions.

        FTS already includes exact_mentions so they can be recalled; this
        channel gives those verbatim hits a deterministic ranking boost instead
        of letting them be diluted inside the long summary/text FTS content.
        """
        mentions = [
            _normalize_exact(item)
            for item in (node.metadata.get("exact_mentions") or ())
            if str(item).strip() and len(str(item).strip()) <= EXACT_MENTION_MAX_LENGTH
        ]
        if not mentions:
            return 0.0
        terms = _exact_probe_terms(probe)
        if not terms:
            return 0.0
        best = 0.0
        for term in terms:
            for mention in mentions:
                if term == mention:
                    return 1.0
                if len(term) >= 2 and term in mention:
                    best = max(best, 0.8)
                elif len(mention) >= 2 and mention in term:
                    best = max(best, 0.6)
        return best

    def _available_navigation_edges(
        self,
        namespace: str,
        scope_id: str,
        candidate: _Candidate,
        id_to_ref: dict[str, str],
    ) -> list[dict[str, Any]]:
        adjacency = self.store.adjacent(scope_id, [candidate.node.id], None, "both")
        neighbor_ids = [neighbor_id for _, neighbor_id in adjacency]
        nodes = {node.id: node for node in self.store.get_nodes(neighbor_ids, scope_id)}
        # Also surface persisted cross-scope EVENT edges (visibility baked in at
        # build time) so the planner can opt to navigate across scopes.
        cross_adjacency = (
            self.store.cross_scope_adjacent(
                namespace, scope_id, [candidate.node.id], None, "both"
            )
            if self.config.cross_scope_navigation
            else []
        )
        cross_neighbor_ids = [neighbor_id for _, neighbor_id, _ in cross_adjacency]
        for node in self.store.get_nodes(cross_neighbor_ids, scope_id=None):
            nodes[node.id] = node
        by_option: dict[tuple[str, str], dict[str, Any]] = {}
        trust_rank = {"explicit": 0, "inferred": 1, "derived": 2}
        for edge, neighbor_id in adjacency:
            neighbor = nodes.get(neighbor_id)
            if neighbor is None:
                continue
            direction = "out" if edge.source_id == candidate.node.id else "in"
            key = (direction, edge.edge_type)
            option = {
                "edge_type": edge.edge_type,
                "direction": direction,
                "trust": edge.trust,
                "confidence": round(edge.confidence, 6),
                "neighbor_ref": id_to_ref.get(neighbor_id),
                "neighbor_node_type": neighbor.node_type,
                "neighbor_topic": neighbor.topic_id,
                "neighbor_title": neighbor.title,
                "neighbor_count": 1,
            }
            current = by_option.get(key)
            if current is None:
                by_option[key] = option
                continue
            current["neighbor_count"] = int(current.get("neighbor_count", 1)) + 1
            current_rank = (
                trust_rank.get(str(current["trust"]), 9),
                -float(current["confidence"]),
            )
            option_rank = (trust_rank.get(edge.trust, 9), -float(option["confidence"]))
            if option_rank < current_rank:
                option["neighbor_count"] = current["neighbor_count"]
                by_option[key] = option
        for edge, neighbor_id, _neighbor_scope in cross_adjacency:
            neighbor = nodes.get(neighbor_id)
            if neighbor is None:
                continue
            direction = "out" if edge.source_id == candidate.node.id else "in"
            key = (direction, edge.edge_type)
            option = {
                "edge_type": edge.edge_type,
                "direction": direction,
                "trust": edge.trust,
                "confidence": round(edge.confidence, 6),
                "neighbor_ref": id_to_ref.get(neighbor_id),
                "neighbor_node_type": neighbor.node_type,
                "neighbor_topic": neighbor.topic_id,
                "neighbor_title": neighbor.title,
                "neighbor_count": 1,
                "cross_scope": True,
            }
            current = by_option.get(key)
            if current is None:
                by_option[key] = option
                continue
            current["neighbor_count"] = int(current.get("neighbor_count", 1)) + 1
            current_rank = (
                trust_rank.get(str(current["trust"]), 9),
                -float(current["confidence"]),
            )
            option_rank = (trust_rank.get(edge.trust, 9), -float(option["confidence"]))
            if option_rank < current_rank:
                option["neighbor_count"] = current["neighbor_count"]
                by_option[key] = option
        output = list(by_option.values())
        output.sort(
            key=lambda item: (
                trust_rank.get(str(item["trust"]), 9),
                str(item["direction"]),
                str(item["edge_type"]),
                str(item["neighbor_title"]),
            )
        )
        return output


def _index_state(state: dict[str, Any]) -> dict[str, Any]:
    return {key: state[key] for key in ("generation", "node_count")}


def _normalize_scores(rows: list[tuple[str, float]]) -> dict[str, float]:
    """Peak-normalize a recall channel's scores to [0, 1].

    FTS/BM25 scores are unbounded and dense (cosine) lives in ~[0, 1]; summing
    them raw lets the channel with the largest magnitude dominate the weighted
    blend. Per-probe peak normalization makes the three channels comparable.
    """
    if not rows:
        return {}
    clamped = [max(0.0, float(score)) for _, score in rows]
    peak = max(clamped)
    if peak <= 0.0:
        return {node_id: 0.0 for node_id, _ in rows}
    return {node_id: max(0.0, float(score)) / peak for node_id, score in rows}


def _normalize_exact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().strip("\"'“”‘’")).casefold()


def _exact_probe_terms(probe: RetrievalProbe) -> list[str]:
    raw = [
        *(probe.must_terms or ()),
        *(probe.should_terms or ()),
        *(probe.entity_hints or ()),
        *re.findall(r'["“]([^"”]+)["”]', probe.query or ""),
    ]
    terms: list[str] = []
    seen: set[str] = set()
    for item in raw:
        term = _normalize_exact(item)
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _cosine(query_embedding: list[float], vector: list[float]) -> float:
    """Cosine similarity clamped to [0, 1].

    `vector` is the stored L2-normalized node embedding; `query_embedding` is
    normalized here. Used for graph-expansion neighbor relevance.
    """
    if not query_embedding or not vector:
        return 0.0
    query_norm = sum(float(x) * float(x) for x in query_embedding) ** 0.5 or 1.0
    dot = sum(
        (float(x) / query_norm) * float(v) for x, v in zip(query_embedding, vector)
    )
    return max(0.0, min(1.0, dot))


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def _virtual_fetch_limit(action: str, budget: int) -> int:
    """Bound SQL oversampling used before visibility filtering and reranking."""
    factor = 2 if action == "ENTITY_LOOKUP" else 4
    return min(32, max(1, int(budget)) * factor)


def _entity_coverage(anchor: MemoryNode, neighbor: MemoryNode) -> float:
    anchor_ids = set(anchor.entity_ids)
    if not anchor_ids:
        return 0.0
    return len(anchor_ids.intersection(neighbor.entity_ids)) / len(anchor_ids)


def _query_entity_ids(anchor: MemoryNode, query: str, minimum: int) -> list[str]:
    """Select public query-mentioned anchor entities, falling back conservatively."""
    all_ids = list(dict.fromkeys(anchor.entity_ids))
    lowered = query.casefold()
    mentioned = list(
        dict.fromkeys(
            entity_id
            for entity, entity_id in zip(anchor.entities, anchor.entity_ids)
            if entity.strip() and entity.casefold() in lowered
        )
    )
    if len(mentioned) >= minimum:
        return mentioned
    return all_ids[:minimum]


def _temporal_proximity(anchor: MemoryNode, neighbor: MemoryNode, action: str) -> float:
    if action not in {
        "TIME_OVERLAP_LOOKUP",
        "TEMPORAL_BEFORE_LOOKUP",
        "TEMPORAL_AFTER_LOOKUP",
        "NEAREST_TIME_LOOKUP",
    }:
        return 0.0
    anchor_start = _date_value(anchor.event_time_start)
    anchor_end = _date_value(anchor.event_time_end) or anchor_start
    neighbor_start = _date_value(neighbor.event_time_start)
    neighbor_end = _date_value(neighbor.event_time_end) or neighbor_start
    if not anchor_start or not anchor_end or not neighbor_start or not neighbor_end:
        return 0.0
    overlaps = neighbor_start <= anchor_end and neighbor_end >= anchor_start
    if overlaps:
        return 1.0
    if neighbor_end < anchor_start:
        gap_seconds = (anchor_start - neighbor_end).total_seconds()
    else:
        gap_seconds = (neighbor_start - anchor_end).total_seconds()
    gap_days = max(0.0, gap_seconds / 86400.0)
    return 1.0 / (1.0 + gap_days / 30.0)


def _edge_trust(value: str) -> float:
    return {"explicit": 1.0, "inferred": 0.78, "derived": 0.42}.get(value, 0.4)


def _edge_type_weight(value: str) -> float:
    return {
        "CAUSES": 1.0,
        "TRIGGERS_STATE_CHANGE": 1.0,
        "SUPPORTS_STATE": 0.95,
        "CONTRIBUTES_TO": 0.90,
        "CONTEXT_FOR": 0.85,
        "FOLLOW_UP_OF": 0.85,
        "ENABLES": 0.80,
        "PART_OF": 0.80,
        "SAME_EVENT": 0.95,
        "SUPERSEDES": 0.90,
        "CONTRADICTS": 0.85,
        "AFFECTS_TOPIC": 0.70,
        "SEMANTIC_NEXT": 0.75,
        "TEMPORAL_NEXT": 0.65,
    }.get(value, 0.60)


def _bounded_probe_candidates(
    candidates: Sequence[_Candidate], budget: int
) -> list[_Candidate]:
    """Keep a full concrete-evidence budget plus up to three navigation heads."""
    evidence_limit = max(1, int(budget))
    selected: list[_Candidate] = []
    chain_count = 0
    evidence_count = 0
    for candidate in candidates:
        node_type = candidate.node.node_type
        if node_type == "chain_head":
            if chain_count >= 3:
                continue
            chain_count += 1
        else:
            if evidence_count >= evidence_limit:
                continue
            evidence_count += 1
        selected.append(candidate)
        if chain_count == 3 and evidence_count == evidence_limit:
            break
    return selected


def _score_weights(probe: RetrievalProbe) -> dict[str, float]:
    weights = {
        "dense": 0.52,
        "sparse": 0.16,
        "entity": 0.12,
        "exact": 0.10,
        "facet": 0.05,
    }
    if probe.must_terms or probe.should_terms:
        weights["dense"] -= 0.05
        weights["exact"] += 0.05
    if probe.entity_hints:
        weights["dense"] -= 0.08
        weights["entity"] += 0.08
    if probe.time.operator != "any":
        weights["dense"] -= 0.03
        weights["facet"] += 0.03
    if re.search(r"\b(?:who|what|which)\b", probe.query.casefold()):
        weights["dense"] -= 0.06
        weights["sparse"] += 0.06
    weights["dense"] = max(0.45, weights["dense"])
    return weights


def _facet_relevance(facet: str, node: MemoryNode) -> float:
    return _lexical_relevance(facet, node)


def _lexical_relevance(question: str, node: MemoryNode) -> float:
    query = set(re.findall(r"[\w']+", question.casefold()))
    text = set(
        re.findall(
            r"[\w']+",
            f"{node.title} {node.summary} {' '.join(node.entities)}".casefold(),
        )
    )
    return len(query & text) / max(1, len(query))


def _semantic_node_duplicate_penalty(first: MemoryNode, second: MemoryNode) -> float:
    if not first.embedding or not second.embedding:
        return 0.0
    similarity = _cosine(first.embedding, second.embedding)
    if similarity < 0.92 or _node_times_disjoint(first, second):
        return 0.0
    first_sources = {repr(item) for item in first.evidence_refs}
    second_sources = {repr(item) for item in second.evidence_refs}
    source_overlap = bool(first_sources & second_sources)
    entity_overlap = bool(
        {item.casefold() for item in first.entities}
        & {item.casefold() for item in second.entities}
    )
    if not source_overlap and not (
        entity_overlap and _node_times_consistent(first, second)
    ):
        return 0.0
    return min(0.08, max(0.0, similarity - 0.92))


def _node_interval(node: MemoryNode) -> tuple[str | None, str | None]:
    start = node.event_time_start or node.valid_from
    end = node.event_time_end or node.valid_to or start
    return start, end


def _node_times_disjoint(first: MemoryNode, second: MemoryNode) -> bool:
    first_start, first_end = _node_interval(first)
    second_start, second_end = _node_interval(second)
    return bool(
        first_start
        and first_end
        and second_start
        and second_end
        and (first_end < second_start or second_end < first_start)
    )


def _node_times_consistent(first: MemoryNode, second: MemoryNode) -> bool:
    first_start, _ = _node_interval(first)
    second_start, _ = _node_interval(second)
    return bool(
        first_start and second_start and not _node_times_disjoint(first, second)
    )


def _fact_matches_time(
    fact: dict[str, Any], node: MemoryNode, probe: RetrievalProbe
) -> bool:
    constraint = probe.time
    fact_start = fact.get("valid_from") or node.valid_from
    fact_end = fact.get("valid_to") or node.valid_to
    node_start, _ = _constraint_bounds(fact_start, node.time_precision)
    if fact_end is None:
        node_end = datetime.max
    else:
        _, node_end = _constraint_bounds(fact_end, node.time_precision)
    start, at_end = _constraint_bounds(constraint.start, constraint.precision)
    _, end = _constraint_bounds(
        constraint.end or constraint.start, constraint.precision
    )
    if not node_start or not node_end or not start:
        return False
    if constraint.operator == "at":
        return at_end is not None and node_end >= start and node_start <= at_end
    if constraint.operator == "before":
        return node_end < start
    if constraint.operator == "after":
        return at_end is not None and node_start > at_end
    return end is not None and node_end >= start and node_start <= end


def _date_value(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _constraint_bounds(
    value: str | None, precision: str | None
) -> tuple[datetime | None, datetime | None]:
    if not value:
        return None, None
    text = str(value).strip()
    inferred = precision
    if inferred is None:
        if len(text) == 4 and text.isdigit():
            inferred = "year"
        elif len(text) == 7 and text[4] == "-":
            inferred = "month"
        elif "T" in text or re.search(r"\d{1,2}:\d{2}", text):
            inferred = "datetime"
        else:
            inferred = "date"
    try:
        if inferred == "year":
            year = int(text[:4])
            return datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59)
        if inferred == "month":
            year, month = map(int, text[:7].split("-"))
            return datetime(year, month, 1), datetime(
                year, month, calendar.monthrange(year, month)[1], 23, 59, 59
            )
    except (ValueError, TypeError):
        return None, None
    parsed = _date_value(text)
    if parsed is not None and inferred == "date":
        return parsed.replace(
            hour=0, minute=0, second=0, microsecond=0
        ), parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return parsed, parsed


def _selected_source_evidence(
    records: Sequence[dict[str, Any]],
    selected_refs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not selected_refs:
        return []
    selected_ids = {
        str(value)
        for ref in selected_refs
        for value in (ref.get("evidence_id"), ref.get("id"))
        if value not in (None, "")
    }
    selected_keys = {
        tuple(
            str(ref.get(key) or "")
            for key in ("session_id", "turn_id", "participant_id", "content")
        )
        for ref in selected_refs
    }
    output: list[dict[str, Any]] = []
    for record in records:
        record_ids = {
            str(value)
            for value in (record.get("id"), record.get("evidence_id"))
            if value not in (None, "")
        }
        record_key = tuple(
            str(record.get(key) or "")
            for key in ("session_id", "turn_id", "participant_id", "content")
        )
        if selected_ids.intersection(record_ids) or record_key in selected_keys:
            output.append(record)
    return output
