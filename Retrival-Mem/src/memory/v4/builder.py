from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime
from hashlib import sha256
from difflib import SequenceMatcher
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from memory.clients import ChatClient, EmbeddingClient, NoopChatClient, parse_json_object
from memory.v4.builder_request import (
    MEMORY_BUILDER_JSON_SCHEMA,
    MEMORY_BUILDER_PROMPT_VERSION,
    build_memory_builder_messages,
    count_chat_tokens,
    turn_observed_at,
)
from memory.structured_output import (
    JsonSchema,
    array_schema,
    enum_string,
    messages_with_json_schema,
    nullable,
    object_schema,
)
from memory.v4.config import FailurePolicyConfig, V4MemoryConfig
from memory.v4.cross_scope_visibility import is_cross_scope_visible
from memory.v4.failure import V4BuildStageError, V4OperationContext, retry_v4_call
from memory.v4.schemas import (
    EXACT_MENTION_MAX_LENGTH,
    EVENT_EDGE_TYPES,
    MemoryEdge,
    MemoryNode,
    ParticipantScope,
    STRUCTURAL_EDGE_TYPES,
    SemanticFact,
    TopicChain,
    stable_id,
)
from memory.prompt_safety import UNTRUSTED_DATA_INSTRUCTION
from memory.v4.storage import V4SQLiteStore
from memory.v4.semantic import (
    SEMANTIC_REDUCER_PROMPT_VERSION,
    SemanticEpoch,
    SemanticEpochMachine,
    SemanticFactRecord,
    SemanticValidationError,
    SemanticReducer,
    fact_key,
)
from memory.v4.time_normalizer import (
    TimeNormalization,
    TimeNormalizer,
    parse_iso_datetime,
)
from memory.v4.window_planner import MemoryWindow, make_window_planner


_SYMMETRIC_EVENT_EDGE_TYPES = {"SAME_EVENT"}


def _entity_judge_json_schema(pair_ids: Iterable[str]) -> JsonSchema:
    ids = list(dict.fromkeys(str(pair_id) for pair_id in pair_ids))
    decision = object_schema(
        {
            "id": enum_string(ids),
            "decision": {
                "type": "string",
                "enum": ["merge", "separate", "unresolved"],
            },
            "reason_code": {
                "type": "string",
                "enum": [
                    "alias_equivalent",
                    "context_supports",
                    "context_conflicts",
                    "ambiguous",
                    "insufficient_evidence",
                ],
            },
        },
        required=("id", "decision", "reason_code"),
    )
    return object_schema(
        {
            "decisions": array_schema(
                decision, min_items=len(ids), max_items=len(ids)
            )
        },
        required=("decisions",),
    )


def _adjudication_json_schema(
    candidate_pairs: Iterable[dict[str, Any]],
) -> JsonSchema:
    pairs = list(candidate_pairs)
    refs = [
        str(value)
        for pair in pairs
        for value in (pair.get("source_ref"), pair.get("target_ref"))
        if value
    ]
    decision = object_schema(
        {
            "source_ref": enum_string(refs),
            "target_ref": enum_string(refs),
            "edge_type": nullable(
                {"type": "string", "enum": sorted(EVENT_EDGE_TYPES)}
            ),
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "explanation": {"type": "string", "minLength": 1},
        },
        required=(
            "source_ref",
            "target_ref",
            "edge_type",
            "confidence",
            "explanation",
        ),
    )
    return object_schema(
        {
            "decisions": array_schema(
                decision, min_items=len(pairs), max_items=len(pairs)
            )
        },
        required=("decisions",),
    )


def _stable_digest(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _memory_window_payload(window: MemoryWindow) -> dict[str, Any]:
    return {
        **window.to_dict(),
        "turns": window.turns,
    }


def _memory_window_from_payload(value: dict[str, Any]) -> MemoryWindow:
    return MemoryWindow(
        window_id=str(value["window_id"]),
        namespace=str(value["namespace"]),
        scope_id=str(value["scope_id"]),
        session_id=str(value["session_id"]),
        turn_ids=[str(item) for item in value.get("turn_ids") or []],
        turns=[dict(item) for item in value.get("turns") or []],
        order_index=int(value.get("order_index", 0)),
        source=str(value.get("source") or "conversation"),
        planner=str(value.get("planner") or "rule"),
        boundary_metadata=dict(value.get("boundary_metadata") or {}),
    )


def _extracted_window_payload(window: "ExtractedWindow") -> dict[str, Any]:
    return {
        "session_id": window.session_id,
        "turns": window.turns,
        "events": window.events,
        "semantic_updates": window.semantic_updates,
        "relations": window.relations,
        "window_id": window.window_id,
        "order_index": window.order_index,
        "source": window.source,
        "boundary_metadata": window.boundary_metadata,
    }


def _extracted_window_from_payload(value: dict[str, Any]) -> "ExtractedWindow":
    return ExtractedWindow(
        session_id=str(value["session_id"]),
        turns=[dict(item) for item in value.get("turns") or []],
        events=[dict(item) for item in value.get("events") or []],
        semantic_updates=[
            dict(item) for item in value.get("semantic_updates") or []
        ],
        relations=[dict(item) for item in value.get("relations") or []],
        window_id=(
            str(value["window_id"]) if value.get("window_id") is not None else None
        ),
        order_index=int(value.get("order_index", 0)),
        source=str(value.get("source") or "conversation"),
        boundary_metadata=dict(value.get("boundary_metadata") or {}),
    )


def _scope_content_payload(
    chains: Iterable[TopicChain],
    nodes: Iterable[MemoryNode],
    edges: Iterable[MemoryEdge],
    facts: Iterable[SemanticFact],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "chains": [asdict(item) for item in sorted(chains, key=lambda item: item.id)],
        "nodes": [asdict(item) for item in sorted(nodes, key=lambda item: item.id)],
        "edges": [asdict(item) for item in sorted(edges, key=lambda item: item.id)],
        "facts": [asdict(item) for item in sorted(facts, key=lambda item: item.id)],
    }


def _entity_resolution_payload(
    result: "EntityCanonicalizationResult",
) -> dict[str, Any]:
    return {
        "alias_to_id": [
            {"alias": key[0], "entity_type": key[1], "canonical_id": value}
            for key, value in sorted(result.alias_to_id.items())
        ],
        "alias_type_by_key": dict(result.alias_type_by_key),
        "merged_ids": dict(result.merged_ids),
        "enabled": result.enabled,
        "decision_counts": {
            "merge": result.merge_count,
            "separate": result.separate_count,
            "unresolved": result.unresolved_count,
            "failed_batches": result.failed_batch_count,
        },
    }


def _entity_resolution_from_payload(
    value: dict[str, Any],
) -> "EntityCanonicalizationResult":
    decision_counts = dict(value.get("decision_counts") or {})
    return EntityCanonicalizationResult(
        alias_to_id={
            (str(item["alias"]), str(item["entity_type"])): str(
                item["canonical_id"]
            )
            for item in value.get("alias_to_id") or []
        },
        alias_type_by_key={
            str(key): str(item)
            for key, item in dict(value.get("alias_type_by_key") or {}).items()
        },
        merged_ids={
            str(key): str(item)
            for key, item in dict(value.get("merged_ids") or {}).items()
        },
        enabled=bool(value.get("enabled", False)),
        merge_count=int(decision_counts.get("merge", 0)),
        separate_count=int(decision_counts.get("separate", 0)),
        unresolved_count=int(decision_counts.get("unresolved", 0)),
        failed_batch_count=int(decision_counts.get("failed_batches", 0)),
    )


class MemoryBuildError(RuntimeError):
    """The window builder failed; callers must not publish a partial ingest."""


@dataclass
class ScopeBuildArtifact:
    scope: ParticipantScope
    expected_revision: int
    chains: list[TopicChain]
    nodes: list[MemoryNode]
    edges: list[MemoryEdge]
    facts: list[SemanticFact]
    changed: bool = False
    dirty_node_ids: set[str] = field(default_factory=set)
    dirty_chain_ids: set[str] = field(default_factory=set)


@dataclass
class ExtractedWindow:
    session_id: str
    turns: list[dict[str, Any]]
    events: list[dict[str, Any]] = field(default_factory=list)
    semantic_updates: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    window_id: str | None = None
    order_index: int = 0
    source: str = "conversation"
    boundary_metadata: dict[str, Any] = field(default_factory=dict)


_ExtractedWindow = ExtractedWindow


@dataclass
class EntityCanonicalizationResult:
    alias_to_id: dict[tuple[str, str], str] = field(default_factory=dict)
    alias_type_by_key: dict[str, str] = field(default_factory=dict)
    merged_ids: dict[str, str] = field(default_factory=dict)
    enabled: bool = False
    merge_count: int = 0
    separate_count: int = 0
    unresolved_count: int = 0
    failed_batch_count: int = 0


@dataclass(frozen=True)
class EntityMatchDecision:
    pair_id: str
    decision: str
    reason_code: str
    batch_index: int
    failed_batch: bool = False


class ApiRateLimiter:
    def __init__(
        self,
        *,
        enabled: bool = True,
        max_in_flight: int = 16,
        requests_per_minute: int | None = 120,
        tokens_per_minute: int | None = None,
    ):
        self.enabled = enabled
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._in_flight = threading.BoundedSemaphore(max(1, int(max_in_flight)))
        self._lock = threading.Lock()
        self._request_times: deque[float] = deque()

    def __enter__(self) -> "ApiRateLimiter":
        if not self.enabled:
            return self
        self._in_flight.acquire()
        self._throttle_rpm()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled:
            self._in_flight.release()

    def _throttle_rpm(self) -> None:
        if self.requests_per_minute is None:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._request_times and now - self._request_times[0] >= 60.0:
                    self._request_times.popleft()
                if len(self._request_times) < self.requests_per_minute:
                    self._request_times.append(now)
                    return
                sleep_for = max(0.0, 60.0 - (now - self._request_times[0]))
            time.sleep(min(sleep_for, 1.0))


class V4MemoryBuilder:
    _entity_registry_locks_guard = threading.Lock()
    _entity_registry_locks: dict[str, threading.RLock] = {}

    def __init__(
        self,
        store: V4SQLiteStore,
        embedding_client: EmbeddingClient,
        builder_client: ChatClient | None,
        extraction_batch_turns: int = 8,
        extraction_workers: int = 1,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        time_normalizer: TimeNormalizer | None = None,
        config: V4MemoryConfig | None = None,
        adjudication_client: ChatClient | None = None,
        entity_judge_client: ChatClient | None = None,
        window_planner_client: ChatClient | None = None,
        semantic_reducer_client: ChatClient | None = None,
        model_identities: dict[str, tuple[str | None, str | None]] | None = None,
    ):
        self.store = store
        self.embedding_client = embedding_client
        self.builder_client = builder_client
        self.semantic_reducer_client = semantic_reducer_client
        self.adjudication_client = adjudication_client
        self.extraction_workers = max(1, int(extraction_workers))
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.time_normalizer = time_normalizer or TimeNormalizer()
        self.config = config or V4MemoryConfig()
        self.entity_judge_client = entity_judge_client
        self.failure_policy = self.config.failure_policy
        self.model_identities = dict(model_identities or {})
        self.window_planner = make_window_planner(
            self.config.window_planner,
            window_planner_client,
            failure_policy=self.failure_policy,
            model_identity=self.model_identities.get("window_plan"),
        )
        rate_config = self.config.build_parallelism.api_rate_limit
        self.api_rate_limiter = ApiRateLimiter(
            enabled=rate_config.enabled,
            max_in_flight=rate_config.max_in_flight,
            requests_per_minute=rate_config.requests_per_minute,
            tokens_per_minute=rate_config.tokens_per_minute,
        )
        self.window_turns = max(2, int(self.config.conversation_window_turns or extraction_batch_turns))
        self.window_overlap = min(self.window_turns - 1, max(0, int(self.config.conversation_window_overlap)))



    def _checkpoint_descriptor(
        self, namespace: str, scope_id: str, stage: str, unit_id: str,
        input_payload: Any, *, prompt_version: str,
        model_identity: tuple[str | None, str | None] | None = None,
    ) -> dict[str, Any]:
        provider, model = model_identity or (None, None)
        revision_getter = getattr(getattr(self, "store", None), "scope_revision", None)
        return {
            "checkpoint_key": f"{scope_id}:{stage}:{unit_id}",
            "namespace": namespace, "scope_id": scope_id,
            "scope_revision": revision_getter(scope_id) if callable(revision_getter) else 0,
            "stage": stage, "unit_id": unit_id,
            "provider": provider, "model": model, "prompt_version": prompt_version,
        }

    def _load_checkpoint(self, descriptor: dict[str, Any]) -> dict[str, Any] | None:
        loader = getattr(getattr(self, "store", None), "load_build_checkpoint", None)
        if not callable(loader):
            return None
        return loader(descriptor["checkpoint_key"])

    def _record_checkpoint_failure(
        self, descriptor: dict[str, Any], error: Exception, *, attempt: int | None = None
    ) -> None:
        recorder = getattr(
            getattr(self, "store", None), "record_build_checkpoint", None
        )
        if not callable(recorder):
            return
        recorder(
            **descriptor,
            status="failed",
            attempt=attempt or int(getattr(error, "attempts", 1)),
            error={"type": type(error).__name__, "message": str(error)},
        )

    def _record_checkpoint_success(
        self,
        descriptor: dict[str, Any],
        output: Any,
        *,
        embedding: Iterable[float] | None = None,
    ) -> None:
        recorder = getattr(
            getattr(self, "store", None), "record_build_checkpoint", None
        )
        if not callable(recorder):
            return
        recorder(
            **descriptor,
            status="succeeded",
            attempt=1,
            output=output,
            embedding=embedding,
        )

    def build(
        self,
        namespace: str,
        conversation: Any,
        metadata: dict[str, Any] | None = None,
        *,
        participants: list[dict[str, str]] | None = None,
    ) -> ScopeBuildArtifact:
        metadata = dict(metadata or {})
        windows = self.plan_windows(namespace, conversation, metadata, participants=participants)
        extracted = self.extract_windows(windows, metadata)
        return self.build_from_extracted(
            namespace, conversation, extracted, metadata, participants=participants
        )

    def plan_windows(
        self,
        namespace: str,
        conversation: Any,
        metadata: dict[str, Any] | None = None,
        *,
        participants: list[dict[str, str]] | None = None,
    ) -> list[MemoryWindow]:
        metadata = dict(metadata or {})
        turns = self._normalize_turns(conversation)
        inferred = participants if participants is not None else self._derive_participants(turns)
        scope = ParticipantScope.create(namespace, inferred, metadata.get("scope_metadata"))
        descriptor = self._checkpoint_descriptor(
            namespace,
            scope.id,
            "window_plan",
            f"conversation:{len(turns)}",
            {
                "turns": turns,
                "participants": inferred,
                "metadata": metadata,
                "planner": self.window_planner.name,
            },
            prompt_version="window-plan-v4",
            model_identity=self.model_identities.get("window_plan"),
        )
        cached = self._load_checkpoint(descriptor)
        if cached is not None and isinstance(cached.get("output"), list):
            return [_memory_window_from_payload(value) for value in cached["output"]]
        try:
            windows = self.window_planner.plan(namespace, scope.id, turns, metadata)
        except Exception as error:
            self._record_checkpoint_failure(descriptor, error)
            raise
        self._record_checkpoint_success(
            descriptor, [_memory_window_payload(window) for window in windows]
        )
        return windows

    def extract_windows(
        self,
        windows: list[MemoryWindow],
        metadata: dict[str, Any] | None = None,
        *,
        workers: int | None = None,
        rate_limiter: ApiRateLimiter | None = None,
    ) -> list[ExtractedWindow]:
        metadata = dict(metadata or {})
        pending = [
            window for window in sorted(windows, key=lambda item: item.order_index)
            if self._write_gate(window.turns)
        ]
        selected_workers = max(1, int(workers or self.extraction_workers))

        def run(window: MemoryWindow) -> ExtractedWindow:
            limiter = rate_limiter or self.api_rate_limiter
            with limiter:
                return self._extract_memory_window(window, metadata)

        if selected_workers > 1 and len(pending) > 1:
            with ThreadPoolExecutor(max_workers=selected_workers) as executor:
                futures = [
                    executor.submit(run, window)
                    for window in pending
                ]
                # Resolve in input order so downstream IDs and merges remain deterministic.
                extracted = [future.result() for future in futures]
        else:
            extracted = [run(window) for window in pending]
        return sorted(extracted, key=lambda window: window.order_index)

    def build_from_extracted(
        self,
        namespace: str,
        conversation: Any,
        extracted: list[ExtractedWindow],
        metadata: dict[str, Any] | None = None,
        *,
        participants: list[dict[str, str]] | None = None,
    ) -> ScopeBuildArtifact:
        metadata = dict(metadata or {})
        turns = self._normalize_turns(conversation)
        inferred = participants if participants is not None else self._derive_participants(turns)
        scope = ParticipantScope.create(namespace, inferred, metadata.get("scope_metadata"))
        existing_scope = self.store.get_scope(scope.id)
        expected_revision = existing_scope.revision if existing_scope else 0
        if existing_scope:
            scope = existing_scope

        existing_chains = self.store.list_chains(scope.id) if existing_scope else []
        existing_nodes = self.store.list_nodes(scope.id) if existing_scope else []
        existing_edges = self.store.list_edges(scope.id) if existing_scope else []
        existing_facts = self.store.list_facts(scope.id) if existing_scope else []
        existing_payload = _scope_content_payload(
            existing_chains, existing_nodes, existing_edges, existing_facts
        )
        existing_node_ids = {node.id for node in existing_nodes}

        extracted = [
            window for window in extracted
            if window.events or window.semantic_updates
        ]
        if not extracted:
            return ScopeBuildArtifact(
                scope, expected_revision, existing_chains, existing_nodes, existing_edges,
                existing_facts, False,

            )
        entity_resolution = self._canonicalize_window_entities(
            namespace, scope.id, extracted
        )

        chains = {chain.topic_id: chain for chain in existing_chains}
        nodes = {node.id: node for node in existing_nodes}
        if entity_resolution.merged_ids:
            self._apply_entity_id_merges_to_nodes(nodes.values(), entity_resolution.merged_ids)
        explicit_relations: list[tuple[str, str, dict[str, Any]]] = []
        event_refs: dict[str, str] = {}
        semantic_updates: list[tuple[dict[str, Any], list[dict[str, Any]], str]] = []
        new_node_ids: set[str] = set()

        for window in extracted:
            ref_prefix = f"{window.session_id}:{window.turns[0].get('turn_id', 'window')}"
            evidence = self._evidence_refs(window.session_id, window.turns)
            by_turn = {str(ref["turn_id"]): ref for ref in evidence}
            for index, value in enumerate(window.events):
                event = dict(value)
                refs = self._select_evidence(event, evidence, by_turn)
                topic = self._route_topic(event, refs, chains, scope)
                chain = self._ensure_chain(namespace, scope.id, topic, chains)
                node = self._event_node(namespace, scope.id, chain, event, refs, metadata, entity_resolution)
                existing = nodes.get(node.id)
                if existing:
                    existing.evidence_refs = _merge_refs(existing.evidence_refs, node.evidence_refs)
                    existing.confidence = max(existing.confidence, node.confidence)
                    existing.observed_at = max(filter(None, [existing.observed_at, node.observed_at]), default=None)
                    node = existing
                else:
                    nodes[node.id] = node
                    new_node_ids.add(node.id)
                local_ref = str(event.get("ref") or event.get("id") or f"event_{index + 1}")
                event_refs[f"{ref_prefix}:{local_ref}"] = node.id
                for relation in event.get("relations") or []:
                    if isinstance(relation, dict):
                        explicit_relations.append((ref_prefix, local_ref, relation))
            for update in window.semantic_updates:
                if not isinstance(update, dict):
                    continue
                refs = self._select_evidence(update, evidence, by_turn)
                semantic_updates.append((dict(update), refs, ref_prefix))
            for relation in window.relations:
                if isinstance(relation, dict):
                    explicit_relations.append(
                        (ref_prefix, str(relation.get("source") or relation.get("from") or ""), relation)
                    )

        # V4 has no persisted navigation/weak edges. Structural edges are
        # retained for unaffected chains and only re-derived for chains that
        # gained nodes this ingest; state/event edges are always preserved.
        edges = {edge.id: edge for edge in existing_edges}
        self._apply_semantic_updates(
            namespace, scope.id, semantic_updates, chains, nodes, edges, event_refs, metadata
        )
        self._embed_nodes(nodes.values())
        self._rebuild_heads(namespace, scope.id, chains, nodes, metadata)
        self._embed_nodes(node for node in nodes.values() if node.node_type == "chain_head")
        affected_chain_ids = {
            node.chain_id for node in nodes.values() if node.id not in existing_node_ids
        }
        if affected_chain_ids:
            # Drop retained structural edges touching affected chains so they can
            # be re-derived (CURRENT_STATE / FIRST_TIMELINE_NODE pointers move, and
            # TEMPORAL_NEXT / SEMANTIC_NEXT chains may need local re-linking).
            affected_node_ids = {
                node.id for node in nodes.values() if node.chain_id in affected_chain_ids
            }
            edges = {
                edge_id: edge for edge_id, edge in edges.items()
                if not (
                    edge.edge_type in STRUCTURAL_EDGE_TYPES
                    and edge.trust != "explicit"
                    and (edge.source_id in affected_node_ids or edge.target_id in affected_node_ids)
                )
            }
        self._structural_edges(namespace, scope.id, chains, nodes, edges, affected_chain_ids)
        self._support_state_edges(namespace, scope.id, nodes, edges, affected_chain_ids)
        self._explicit_edges(namespace, scope.id, explicit_relations, event_refs, nodes, edges)
        self._cross_window_edges(namespace, scope.id, nodes, edges, new_node_ids, chains, metadata)
        final_chains = sorted(chains.values(), key=lambda item: item.topic_id)
        final_nodes = sorted(
            nodes.values(), key=lambda item: (item.chain_id, item.node_type, item.id)
        )
        final_edges = sorted(edges.values(), key=lambda item: item.id)
        facts = sorted(self._semantic_fact_rows(nodes.values()), key=lambda item: item.id)
        changed = (
            not existing_scope
            or existing_payload
            != _scope_content_payload(final_chains, final_nodes, final_edges, facts)
        )
        if not changed:
            dirty_chain_ids = set()
        elif semantic_updates:
            dirty_chain_ids = {node.chain_id for node in nodes.values()}
        else:
            dirty_chain_ids = set(affected_chain_ids)
        dirty_node_ids = {
            node.id for node in nodes.values() if node.chain_id in dirty_chain_ids
        }
        return ScopeBuildArtifact(
            scope=scope,
            expected_revision=expected_revision,
            chains=final_chains,
            nodes=final_nodes,
            edges=final_edges,
            facts=facts,
            changed=changed,
            dirty_node_ids=dirty_node_ids,
            dirty_chain_ids=dirty_chain_ids,

        )

    def _extract_memory_window(
        self, window: MemoryWindow, metadata: dict[str, Any]
    ) -> ExtractedWindow:
        session_id = window.session_id
        turns = window.turns

        def extracted(
            events: list[dict[str, Any]] | None = None,
            updates: list[dict[str, Any]] | None = None,
            relations: list[dict[str, Any]] | None = None,
        ) -> ExtractedWindow:
            return ExtractedWindow(
                session_id=session_id,
                turns=turns,
                events=list(events or []),
                semantic_updates=list(updates or []),
                relations=list(relations or []),
                window_id=window.window_id,
                order_index=window.order_index,
                source=window.source,
                boundary_metadata=dict(window.boundary_metadata),
            )

        descriptor = self._checkpoint_descriptor(
            window.namespace,
            window.scope_id,
            "window_extraction",
            window.window_id,
            {
                "session_id": session_id,
                "turns": turns,
                "metadata": metadata,
                "boundary_metadata": window.boundary_metadata,
            },
            prompt_version=MEMORY_BUILDER_PROMPT_VERSION,
            model_identity=getattr(self, "model_identities", {}).get(
                "window_extraction"
            ),
        )
        cached = self._load_checkpoint(descriptor)
        if cached is not None and isinstance(cached.get("output"), dict):
            return _extracted_window_from_payload(cached["output"])

        if self.builder_client is None or isinstance(self.builder_client, NoopChatClient):
            deterministic = self._deterministic_extract(session_id, turns)
            deterministic.window_id = window.window_id
            deterministic.order_index = window.order_index
            deterministic.source = window.source
            deterministic.boundary_metadata = dict(window.boundary_metadata)
            self._record_checkpoint_success(
                descriptor, _extracted_window_payload(deterministic)
            )
            return deterministic
        messages = build_memory_builder_messages(session_id, turns, metadata)
        planner_config = self.config.window_planner
        input_tokens = count_chat_tokens(messages, planner_config.token_encoding)
        if input_tokens > planner_config.max_builder_input_tokens:
            raise MemoryBuildError(
                "builder input token budget exceeded before send: "
                f"{input_tokens} > {planner_config.max_builder_input_tokens}"
            )
        def invoke() -> ExtractedWindow:
            assert self.builder_client is not None
            raw = self.builder_client.chat(
                messages_with_json_schema(
                    messages, MEMORY_BUILDER_JSON_SCHEMA
                ),
                json_mode=True,
                json_schema=MEMORY_BUILDER_JSON_SCHEMA,
            )
            data = parse_json_object(raw)
            write_gate = data.get("write")
            if isinstance(write_gate, str):
                write_gate = write_gate.strip().lower()
                if write_gate in ("true", "1", "yes"):
                    data["write"] = True
                elif write_gate in ("false", "0", "no"):
                    data["write"] = False
            if not isinstance(data.get("write"), bool):
                raise ValueError("builder response has no boolean write gate")
            if not data["write"]:
                return extracted()
            events = data.get("events")
            updates = data.get("candidate_claims", data.get("semantic_updates", []))
            relations = data.get("relations", [])
            if not isinstance(events, list) or not isinstance(updates, list) or not isinstance(relations, list):
                raise ValueError("builder response has invalid extraction arrays")
            if not events and not updates:
                raise ValueError("write=true requires at least one event or semantic update")
            if any(
                not isinstance(item, dict)
                for item in [*events, *updates, *relations]
            ):
                raise ValueError(
                    "every extracted event, claim group, and relation must be an object"
                )
            event_count = len(events)
            update_count = len(updates)
            events, updates = self._repair_extracted_evidence_ids(turns, events, updates)
            if len(events) != event_count or len(updates) != update_count:
                raise ValueError(
                    "an extracted item has invalid or unrepairable evidence_turn_ids"
                )
            if not events and not updates:
                return extracted()
            relations = self._normalize_extracted_relations(events, relations)
            self._validate_extracted_items(turns, events, updates, relations)
            self._normalize_extracted_times(turns, events, updates)
            return extracted(events, updates, relations)

        provider, model = getattr(self, "model_identities", {}).get(
            "window_extraction",
            (None, getattr(self.config, "memory_builder_model", None)),
        )
        def record_failure(attempt: int, error: Exception) -> None:
            messages.append({
                "role": "user",
                "content": json.dumps({
                    "validator_feedback": {
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                    "instruction": "Return one corrected extraction JSON object only.",
                }, ensure_ascii=False),
            })
            self._record_checkpoint_failure(descriptor, error, attempt=attempt)

        result = retry_v4_call(
            invoke,
            policy=getattr(self, "failure_policy", FailurePolicyConfig()),
            context=V4OperationContext(
                stage="window_extraction",
                unit_id=window.window_id,
                provider=provider,
                model=model,
                checkpoint_key=descriptor["checkpoint_key"],
            ),
            error_type=V4BuildStageError,
            validation_errors=(ValueError, TypeError),
            on_failure=record_failure,
            validation_retries=1,
        )
        self._record_checkpoint_success(descriptor, _extracted_window_payload(result))
        return result

    @staticmethod
    def _normalize_extracted_relations(
        events: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        event_refs = {
            str(event.get("ref") or "").strip()
            for event in events
            if isinstance(event, dict)
        }
        normalized: list[dict[str, Any]] = []
        pair_indexes: dict[tuple[str, str], int] = {}

        def confidence(relation: dict[str, Any]) -> float:
            value = relation.get("confidence", 1.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return float("-inf")
            return float(value)

        for relation in relations:
            source = str(relation.get("source") or "").strip()
            target = str(relation.get("target") or "").strip()
            if source not in event_refs or target not in event_refs or source == target:
                continue
            pair = (source, target)
            existing_index = pair_indexes.get(pair)
            if existing_index is None:
                pair_indexes[pair] = len(normalized)
                normalized.append(relation)
                continue
            if confidence(relation) > confidence(normalized[existing_index]):
                normalized[existing_index] = relation
        return normalized

    @staticmethod
    def _validate_extracted_items(
        turns: list[dict[str, Any]],
        events: list[dict[str, Any]],
        updates: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> None:
        V4MemoryBuilder._validate_evidence_ids(turns, [*events, *updates])
        event_refs: set[str] = set()
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("every extracted event must be an object")
            event_ref = str(event.get("ref") or "").strip()
            if not event_ref or event_ref in event_refs:
                raise ValueError("every extracted event requires a unique local ref")
            if not str(event.get("title") or event.get("summary") or "").strip():
                raise ValueError("every extracted event requires title or summary evidence")
            event_refs.add(event_ref)
        for update in updates:
            if not isinstance(update, dict):
                raise ValueError("every candidate claim group must be an object")
            claims = update.get("claims")
            if not isinstance(claims, list) or not claims:
                raise ValueError("candidate claim group requires a non-empty claims list")
            for claim in claims:
                if not isinstance(claim, dict):
                    raise ValueError("candidate claim must be an object")
                for field_name in ("subject", "dimension", "aspect"):
                    if not str(claim.get(field_name) or "").strip():
                        raise ValueError(f"candidate claim requires {field_name}")
                if "value" not in claim:
                    raise ValueError("candidate claim requires value")
        seen_relations: set[tuple[str, str]] = set()
        for relation in relations:
            if not isinstance(relation, dict):
                raise ValueError("every extracted relation must be an object")
            source = str(relation.get("source") or "").strip()
            target = str(relation.get("target") or "").strip()
            if source not in event_refs or target not in event_refs or source == target:
                raise ValueError("relation endpoints must be distinct supplied event refs")
            if (source, target) in seen_relations:
                raise ValueError("duplicate directed event relation")
            edge_type = str(relation.get("edge_type") or "").upper()
            if edge_type not in EVENT_EDGE_TYPES:
                raise ValueError("relation has an invalid event edge type")
            confidence = relation.get("confidence", 1.0)
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError("relation confidence must be numeric")
            if not 0.0 <= float(confidence) <= 1.0:
                raise ValueError("relation confidence must be in 0..1")
            relation["edge_type"] = edge_type
            seen_relations.add((source, target))

    def _normalize_extracted_times(
        self,
        turns: list[dict[str, Any]],
        events: list[dict[str, Any]],
        updates: list[dict[str, Any]],
    ) -> None:
        normalizer = getattr(self, "time_normalizer", None) or TimeNormalizer()
        observed_by_turn = {
            str(turn["turn_id"]): turn_observed_at(turn)
            for turn in turns
        }

        def anchor(item: dict[str, Any]) -> str | None:
            return next((
                observed_by_turn.get(str(turn_id))
                for turn_id in item.get("evidence_turn_ids") or ()
                if observed_by_turn.get(str(turn_id))
            ), None)

        def normalize_time(
            absolute_value: Any,
            raw_expression: Any,
            item: dict[str, Any],
        ) -> TimeNormalization | None:
            expressions = dict.fromkeys(
                str(value)
                for value in (absolute_value, raw_expression)
                if value is not None and str(value).strip()
            )
            for expression in expressions:
                normalized = normalizer.normalize(expression, anchor(item))
                if normalized.absolute_time_start is not None:
                    return normalized
            return None

        for event in events:
            explicit_start = event.get("event_time_start")
            raw_expression = event.get("raw_time_expression")
            normalized = normalize_time(
                explicit_start,
                raw_expression,
                event,
            )
            if normalized is None:
                if explicit_start:
                    raise ValueError(
                        "event time could not be resolved to absolute ISO time"
                    )
                continue

            explicit_end = event.get("event_time_end")
            normalized_end = normalize_time(explicit_end, None, event)
            if explicit_end and normalized_end is None:
                raise ValueError(
                    "event end time could not be resolved to absolute ISO time"
                )
            absolute_end = (
                normalized_end.absolute_time_end
                if normalized_end is not None
                else normalized.absolute_time_end
            )
            precision = normalized.time_precision
            if (
                normalized_end is not None
                and absolute_end != normalized.absolute_time_start
            ):
                precision = "range"
            event["event_time_start"] = normalized.absolute_time_start
            event["event_time_end"] = absolute_end
            event["time_precision"] = precision
            if raw_expression is not None:
                event["raw_time_expression"] = str(raw_expression)
        for update in updates:
            explicit_start = update.get("valid_from")
            raw_expression = update.get("raw_time_expression")
            normalized = normalize_time(
                explicit_start,
                raw_expression,
                update,
            )
            if normalized is None:
                if explicit_start:
                    raise ValueError(
                        "candidate claim valid_from could not be resolved to "
                        "absolute ISO time"
                    )
                continue
            update["valid_from"] = normalized.absolute_time_start
            update["time_precision"] = normalized.time_precision
            if raw_expression is not None:
                update["raw_time_expression"] = str(raw_expression)

    @staticmethod
    def _validate_evidence_ids(turns: list[dict[str, Any]], values: Iterable[dict[str, Any]]) -> None:
        valid = {str(turn["turn_id"]) for turn in turns}
        for value in values:
            ids = value.get("evidence_turn_ids")
            if not isinstance(ids, list) or not ids or any(str(turn_id) not in valid for turn_id in ids):
                raise ValueError("every extracted item requires valid evidence_turn_ids")

    @staticmethod
    def _repair_extracted_evidence_ids(
        turns: list[dict[str, Any]],
        events: list[Any],
        updates: list[Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        valid_ids = [str(turn["turn_id"]) for turn in turns]
        repaired_events = V4MemoryBuilder._repair_evidence_list(events, valid_ids)
        repaired_updates = V4MemoryBuilder._repair_evidence_list(updates, valid_ids)
        return repaired_events, repaired_updates

    @staticmethod
    def _repair_evidence_list(values: list[Any], valid_ids: list[str]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for value in values:
            if not isinstance(value, dict):
                continue
            ids = value.get("evidence_turn_ids")
            if not isinstance(ids, list) or not ids:
                continue
            repaired_ids: list[str] = []
            repairs: list[dict[str, str]] = []
            for raw_id in ids:
                repaired, reason = V4MemoryBuilder._repair_evidence_id(raw_id, valid_ids)
                if repaired is None:
                    continue
                repaired_ids.append(repaired)
                if reason:
                    repairs.append({
                        "from": str(raw_id),
                        "to": repaired,
                        "reason": reason,
                    })
            repaired_ids = list(dict.fromkeys(repaired_ids))
            if not repaired_ids:
                continue
            item = dict(value)
            item["evidence_turn_ids"] = repaired_ids
            if repairs:
                item["evidence_id_repairs"] = repairs
            output.append(item)
        return output

    @staticmethod
    def _repair_evidence_id(value: Any, valid_ids: list[str]) -> tuple[str | None, str | None]:
        text = str(value).strip()
        if text in valid_ids:
            return text, None
        normalized = _evidence_id_key(text)
        normalized_matches = [
            turn_id for turn_id in valid_ids
            if _evidence_id_key(turn_id) == normalized
        ]
        if len(normalized_matches) == 1:
            return normalized_matches[0], "unique_normalized_match"
        scored = [
            (SequenceMatcher(None, text, turn_id).ratio(), turn_id)
            for turn_id in valid_ids
        ]
        scored = sorted(scored, reverse=True)
        if scored and scored[0][0] >= 0.92 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.04):
            return scored[0][1], "unique_close_match"
        return None, None

    def _deterministic_extract(self, session_id: str, turns: list[dict[str, Any]]) -> _ExtractedWindow:
        informative = [turn for turn in turns if self._turn_is_informative(turn)]
        if not informative:
            return _ExtractedWindow(session_id, turns)
        user_like = [
            turn for turn in turns
            if str(turn.get("role", "")).lower() not in {"assistant", "agent", "system"}
            and str(turn.get("content") or "").strip()
        ]
        short_answers = {"yes", "no", "yeah", "nope"}
        evidence_turns = (
            [turn for turn in turns if str(turn.get("content") or "").strip()]
            if user_like and all(str(turn.get("content") or "").strip().casefold() in short_answers for turn in user_like)
            else (user_like or informative)
        )
        content = " ".join(str(turn.get("content") or "").strip() for turn in evidence_turns).strip()
        topic_id = _canonical_topic(content)
        observed = next((self._turn_observed_at(turn) for turn in reversed(evidence_turns) if self._turn_observed_at(turn)), None)
        # Capture every time expression in the window, not just the first; a
        # single window can mention several distinct event times and dropping
        # the later ones loses time-targeted retrievability in Noop mode.
        raw_times = self.time_normalizer.extract_expressions(content) or [None]
        entities = _entities(content)
        actors = list(dict.fromkeys(
            str(turn.get("participant_id") or turn.get("speaker") or turn.get("role") or "unknown")
            for turn in evidence_turns
        ))
        evidence_ids = [str(turn["turn_id"]) for turn in evidence_turns]
        events: list[dict[str, Any]] = []
        for index, raw_time in enumerate(raw_times):
            normalized = self.time_normalizer.normalize(raw_time, observed)
            events.append({
                "ref": f"event_{index + 1}", "topic_id": topic_id, "topic_name": topic_id.replace("_", " "),
                "title": _compact(content, 80), "summary": _compact(content, 360),
                "actors": actors,
                "action": _infer_action(content), "objects": entities, "entities": entities,
                "location": _infer_location(content), "event_time_start": normalized.absolute_time_start,
                "event_time_end": normalized.absolute_time_end, "time_precision": normalized.time_precision,
                "raw_time_expression": raw_time, "observed_at": observed,
                "importance": 0.6, "confidence": 0.8,
                "evidence_turn_ids": evidence_ids,
            })
        updates = _deterministic_semantic_updates(content, evidence_ids, observed)
        return _ExtractedWindow(session_id, turns, events, updates, [])

    @staticmethod
    def _write_gate(turns: list[dict[str, Any]]) -> bool:
        return any(V4MemoryBuilder._turn_is_informative(turn) for turn in turns)

    @staticmethod
    def _turn_is_informative(turn: dict[str, Any]) -> bool:
        text = str(turn.get("content") or "").strip()
        if not text:
            return False
        normalized = re.sub(r"[\s!,.?~]+", "", text).casefold()
        non_memory = {
            "hi", "hello", "hey", "thanks", "thankyou", "ok", "okay", "bye", "goodbye",
            "yes", "no", "yeah", "nope",
        }
        if normalized in non_memory or len(normalized) <= 1:
            return False
        lowered = text.casefold().strip()
        operational_prefixes = (
            "please open ", "open settings", "click ", "run command ", "delete this ",
            "show me ",
        )
        return not lowered.startswith(operational_prefixes)

    def _route_topic(
        self, event: dict[str, Any], refs: list[dict[str, Any]], chains: dict[str, TopicChain], scope: ParticipantScope
    ) -> dict[str, Any]:
        raw = str(event.get("topic_id") or event.get("topic") or "").strip()
        if not raw:
            raw = _canonical_topic(" ".join(str(ref.get("content") or "") for ref in refs))
        topic_id = _normalize_topic_id(raw)
        aliases = _strings(event.get("topic_aliases"))
        candidates = {topic_id.casefold(), *(value.casefold() for value in aliases)}
        for chain in chains.values():
            names = {chain.topic_id.casefold(), chain.name.casefold(), *(value.casefold() for value in chain.aliases)}
            if candidates.intersection(names):
                return {"topic_id": chain.topic_id, "name": chain.name, "description": chain.description, "aliases": chain.aliases}
        if chains:
            query_text = " ".join([
                raw, *aliases, str(event.get("topic_name") or ""),
                str(event.get("topic_description") or ""), str(event.get("summary") or ""),
                *(str(ref.get("content") or "") for ref in refs),
            ])
            candidates_chains = list(chains.values())[: self.config.topic_candidate_top_k]
            vectors = self.embedding_client.embed_texts([
                query_text,
                *(" ".join([chain.topic_id, chain.name, chain.description, *chain.aliases, chain.rolling_summary]) for chain in candidates_chains),
            ])
            if len(vectors) == len(candidates_chains) + 1:
                query_vector = np.asarray(vectors[0], dtype=np.float32)
                query_norm = float(np.linalg.norm(query_vector))
                if query_norm:
                    query_vector /= query_norm
                    scored = []
                    for chain, value in zip(candidates_chains, vectors[1:]):
                        vector = np.asarray(value, dtype=np.float32)
                        norm = float(np.linalg.norm(vector))
                        similarity = float(query_vector @ (vector / norm)) if norm else -1.0
                        scored.append((similarity, chain))
                    similarity, chain = max(scored, key=lambda item: item[0])
                    if similarity >= self.config.topic_match_min_similarity:
                        return {"topic_id": chain.topic_id, "name": chain.name, "description": chain.description, "aliases": chain.aliases}
        return {
            "topic_id": topic_id,
            "name": str(event.get("topic_name") or topic_id.replace("_", " ")),
            "description": str(event.get("topic_description") or ""),
            "aliases": aliases,
        }

    @staticmethod
    def _ensure_chain(
        namespace: str, scope_id: str, topic: dict[str, Any], chains: dict[str, TopicChain]
    ) -> TopicChain:
        topic_id = str(topic["topic_id"])
        if topic_id not in chains:
            chains[topic_id] = TopicChain(
                id=stable_id("chain", scope_id, topic_id), namespace=namespace, scope_id=scope_id,
                topic_id=topic_id, name=str(topic.get("name") or topic_id),
                description=str(topic.get("description") or ""), aliases=_strings(topic.get("aliases")),
            )
        return chains[topic_id]

    def _canonicalize_window_entities(
        self, namespace: str, scope_id: str, extracted: list[ExtractedWindow],
    ) -> EntityCanonicalizationResult:
        alias_entries, alias_type_by_key = self._collect_entity_aliases(extracted)
        result = EntityCanonicalizationResult(alias_type_by_key=alias_type_by_key)
        if not alias_entries or not self.config.entity_canonicalization_enabled:
            return result
        required_methods = ("get_entity_vectors", "upsert_entities", "merge_entity_ids")
        if any(not hasattr(self.store, name) for name in required_methods):
            return result
        descriptor = self._checkpoint_descriptor(
            namespace,
            scope_id,
            "entity_resolution",
            "aliases",
            {"aliases": alias_entries, "types": alias_type_by_key},
            prompt_version="entity-resolution-v4",
            model_identity=self.model_identities.get("entity_resolution"),
        )
        cached = self._load_checkpoint(descriptor)
        if cached is not None and isinstance(cached.get("output"), dict):
            return _entity_resolution_from_payload(cached["output"])

        def embed_aliases() -> list[np.ndarray]:
            raw_vectors = self.embedding_client.embed_texts(
                [entry["alias"] for entry in alias_entries]
            )
            if len(raw_vectors) != len(alias_entries):
                raise ValueError(
                    "entity embedding client returned an incorrect number of vectors"
                )
            vectors = [_normalized_vector(vector) for vector in raw_vectors]
            if any(vector is None for vector in vectors):
                raise ValueError("entity embedding client returned an invalid vector")
            return [vector for vector in vectors if vector is not None]

        provider, model = self.model_identities.get(
            "embedding", (None, self.config.embedding_model)
        )
        vectors = retry_v4_call(
            embed_aliases,
            policy=self.failure_policy,
            context=V4OperationContext(
                stage="entity_embedding",
                unit_id=scope_id,
                provider=provider,
                model=model,
                checkpoint_key=descriptor["checkpoint_key"],
            ),
            error_type=V4BuildStageError,
            validation_errors=(ValueError, TypeError),
            on_failure=lambda attempt, error: self._record_checkpoint_failure(
                descriptor, error, attempt=1
            ),
        )
        result.enabled = True
        upserts: dict[str, dict[str, Any]] = {}
        uncertain: list[dict[str, Any]] = []

        with self._entity_registry_lock(namespace):
            registry = self.store.get_entity_vectors(namespace)
            for row in registry:
                row["embedding"] = _normalized_vector(row.get("embedding"))
                row["aliases"] = _stable_unique([str(row.get("representative_alias") or ""), *_strings(row.get("aliases"))])
            exact = self._entity_exact_alias_index(registry)
            registry_ids = {
                str(row.get("canonical_id") or "") for row in registry
            }
            conservative = self.config.entity_auto_merge_policy == "conservative"

            def remember(row: dict[str, Any]) -> None:
                for alias in [str(row.get("representative_alias") or ""), *list(row.get("aliases") or [])]:
                    alias_text = str(alias).strip()
                    if alias_text:
                        exact[(alias_text.casefold(), str(row.get("entity_type") or "entity"))] = row

            def add_alias(row: dict[str, Any], alias: str) -> None:
                row["aliases"] = _stable_unique([*list(row.get("aliases") or []), alias])
                upserts[str(row["canonical_id"])] = row
                remember(row)

            def add_independent(
                alias: str,
                entity_type: str,
                vector: np.ndarray,
                key: tuple[str, str],
            ) -> None:
                row = self._new_entity_row(
                    namespace, alias, entity_type, vector
                )
                canonical_id = str(row["canonical_id"])
                if canonical_id in registry_ids:
                    return
                registry.append(row)
                registry_ids.add(canonical_id)
                remember(row)
                upserts[canonical_id] = row
                result.alias_to_id[key] = canonical_id

            def needs_judgement(
                alias: str,
                candidate_row: dict[str, Any],
                similarity: float,
            ) -> bool:
                return (
                    similarity >= self.config.entity_cosine_high_threshold
                    or (
                        similarity >= self.config.entity_cosine_low_threshold
                        and _entity_char_similarity(alias, candidate_row)
                        >= self.config.entity_char_similarity_threshold
                    )
                )

            for entry, vector in zip(alias_entries, vectors):
                assert vector is not None
                alias = entry["alias"]
                entity_type = entry["entity_type"]
                key = (alias.casefold(), entity_type)
                exact_row = exact.get(key)
                if exact_row is not None:
                    canonical_id = str(exact_row["canonical_id"])
                    if not conservative or entity_type in {"gpe", "object"}:
                        result.alias_to_id[key] = canonical_id
                    else:
                        uncertain.append({
                            "kind": "exact",
                            "alias": alias,
                            "entity_type": entity_type,
                            "mention_context": entry.get("mention_context") or {},
                            "candidate": exact_row,
                            "vector": vector,
                            "key": key,
                        })
                    candidate = self._best_entity_candidate(registry, entry, vector, exclude_id=canonical_id)
                    if candidate is None:
                        continue
                    candidate_row, similarity = candidate
                    if (
                        not conservative
                        and similarity >= self.config.entity_cosine_high_threshold
                    ):
                        self._record_entity_merge(result, exact_row, candidate_row)
                    elif needs_judgement(alias, candidate_row, similarity):
                        uncertain.append({
                            "kind": "merge",
                            "alias": alias,
                            "entity_type": entity_type,
                            "mention_context": entry.get("mention_context") or {},
                            "source": exact_row,
                            "candidate": candidate_row,
                        })
                    continue

                candidate = self._best_entity_candidate(registry, entry, vector)
                if candidate is not None:
                    candidate_row, similarity = candidate
                    if (
                        not conservative
                        and similarity >= self.config.entity_cosine_high_threshold
                    ):
                        add_alias(candidate_row, alias)
                        result.alias_to_id[key] = str(candidate_row["canonical_id"])
                        continue
                    if needs_judgement(alias, candidate_row, similarity):
                        uncertain.append({
                            "kind": "alias",
                            "alias": alias,
                            "entity_type": entity_type,
                            "mention_context": entry.get("mention_context") or {},
                            "candidate": candidate_row,
                            "vector": vector,
                            "key": key,
                        })
                        continue

                add_independent(alias, entity_type, vector, key)

            decisions = self._judge_entity_matches(
                uncertain, checkpoint_descriptor=descriptor
            )
            failed_batches: set[int] = set()
            for item, decision in zip(uncertain, decisions):
                if decision.decision == "merge":
                    result.merge_count += 1
                elif decision.decision == "separate":
                    result.separate_count += 1
                else:
                    result.unresolved_count += 1
                if decision.failed_batch:
                    failed_batches.add(decision.batch_index)
                if item["kind"] == "merge":
                    if decision.decision == "merge":
                        self._record_entity_merge(result, item["source"], item["candidate"])
                    continue
                alias = str(item["alias"])
                key = item["key"]
                if item["kind"] == "exact":
                    if decision.decision == "merge":
                        result.alias_to_id[key] = str(
                            item["candidate"]["canonical_id"]
                        )
                    continue
                if decision.decision == "merge":
                    candidate_row = item["candidate"]
                    add_alias(candidate_row, alias)
                    result.alias_to_id[key] = str(candidate_row["canonical_id"])
                else:
                    add_independent(
                        alias,
                        str(item["entity_type"]),
                        item["vector"],
                        key,
                    )
            result.failed_batch_count = len(failed_batches)

            if upserts:
                self.store.upsert_entities(namespace, upserts.values())
            if result.merged_ids:
                collapsed = {
                    source: self._resolved_entity_id(source, result.merged_ids)
                    for source in result.merged_ids
                }
                for source, target in sorted(collapsed.items()):
                    if source != target:
                        self.store.merge_entity_ids(namespace, source, target)
                result.alias_to_id = {
                    key: self._resolved_entity_id(value, collapsed)
                    for key, value in result.alias_to_id.items()
                }
                result.merged_ids = collapsed
        self._record_checkpoint_success(
            descriptor, _entity_resolution_payload(result)
        )
        return result

    @staticmethod
    def _collect_entity_aliases(
        extracted: list[ExtractedWindow],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        precedence = {"object": 0, "entity": 1, "gpe": 2}
        by_key: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        def consider(
            alias: str,
            entity_type: str,
            *,
            event: dict[str, Any],
            window: ExtractedWindow,
        ) -> None:
            text = str(alias).strip()
            if not text:
                return
            key = text.casefold()
            turn_index = {
                str(turn.get("turn_id")): index
                for index, turn in enumerate(window.turns)
            }
            evidence_ids = [
                str(item) for item in event.get("evidence_turn_ids") or ()
            ]
            index = next(
                (turn_index[item] for item in evidence_ids if item in turn_index),
                None,
            )
            turn = window.turns[index] if index is not None else {}
            context = {
                "mention": text,
                "speaker": str(
                    turn.get("speaker")
                    or turn.get("participant_id")
                    or turn.get("role")
                    or ""
                ),
                "context": _compact(str(turn.get("content") or ""), 300),
                "before": _compact(
                    str(window.turns[index - 1].get("content") or ""), 300
                ) if index is not None and index > 0 else "",
                "after": _compact(
                    str(window.turns[index + 1].get("content") or ""), 300
                ) if index is not None and index + 1 < len(window.turns) else "",
                "source_event_summary": _compact(
                    str(event.get("summary") or event.get("title") or ""), 300
                ),
            }
            candidate = {
                "alias": text,
                "entity_type": entity_type,
                "mention_context": context,
            }
            if key not in by_key:
                order.append(key)
                by_key[key] = candidate
                return
            current_type = by_key[key]["entity_type"]
            if precedence[entity_type] > precedence[current_type]:
                by_key[key] = candidate

        for window in extracted:
            for event in window.events:
                if not isinstance(event, dict):
                    continue
                if event.get("location"):
                    consider(
                        str(event["location"]), "gpe", event=event, window=window
                    )
                for alias in _strings(event.get("entities")):
                    consider(alias, "entity", event=event, window=window)
                for alias in _strings(event.get("objects")):
                    consider(alias, "object", event=event, window=window)
        alias_type_by_key = {key: by_key[key]["entity_type"] for key in order}
        return [by_key[key] for key in order], alias_type_by_key

    @staticmethod
    def _entity_exact_alias_index(registry: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        exact: dict[tuple[str, str], dict[str, Any]] = {}
        for row in registry:
            entity_type = str(row.get("entity_type") or "entity")
            aliases = _stable_unique([
                str(row.get("representative_alias") or ""),
                *[str(alias) for alias in row.get("aliases") or []],
            ])
            for alias in aliases:
                exact[(alias.casefold(), entity_type)] = row
        return exact

    def _best_entity_candidate(
        self,
        registry: list[dict[str, Any]],
        entry: dict[str, str],
        vector: np.ndarray,
        *,
        exclude_id: str | None = None,
    ) -> tuple[dict[str, Any], float] | None:
        scored: list[tuple[float, str, dict[str, Any]]] = []
        entity_type = entry["entity_type"]
        for row in registry:
            if str(row.get("entity_type") or "entity") != entity_type:
                continue
            canonical_id = str(row.get("canonical_id") or "")
            if exclude_id and canonical_id == exclude_id:
                continue
            embedding = _normalized_vector(row.get("embedding"))
            if embedding is None or embedding.shape != vector.shape:
                continue
            scored.append((float(vector @ embedding), canonical_id, row))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        capped = scored[: min(self.config.entity_embed_top_k, self.config.entity_registry_max_candidates)]
        similarity, _, row = capped[0]
        return row, similarity

    @staticmethod
    def _new_entity_row(
        namespace: str, alias: str, entity_type: str, vector: np.ndarray,
    ) -> dict[str, Any]:
        canonical_id = stable_id("entity", namespace, entity_type, _normalize_text(alias))
        return {
            "namespace": namespace,
            "canonical_id": canonical_id,
            "entity_type": entity_type,
            "representative_alias": alias,
            "aliases": [alias],
            "embedding": vector.tolist(),
            "created_at": "",
            "updated_at": "",
        }

    def _judge_entity_matches(
        self,
        pairs: list[dict[str, Any]],
        *,
        checkpoint_descriptor: dict[str, Any] | None = None,
    ) -> list[EntityMatchDecision]:
        if not pairs:
            return []
        indexed_pairs = [
            (f"p{index + 1}", item) for index, item in enumerate(pairs)
        ]
        batch_size = self.config.entity_judge_batch_size
        batches = [
            indexed_pairs[index:index + batch_size]
            for index in range(0, len(indexed_pairs), batch_size)
        ]

        def unresolved_batch(
            batch_index: int,
            batch: list[tuple[str, dict[str, Any]]],
            *,
            failed: bool,
        ) -> list[EntityMatchDecision]:
            return [
                EntityMatchDecision(
                    pair_id=pair_id,
                    decision="unresolved",
                    reason_code="insufficient_evidence",
                    batch_index=batch_index,
                    failed_batch=failed,
                )
                for pair_id, _ in batch
            ]

        if self.entity_judge_client is None or isinstance(
            self.entity_judge_client, NoopChatClient
        ):
            return [
                decision
                for batch_index, batch in enumerate(batches)
                for decision in unresolved_batch(batch_index, batch, failed=False)
            ]

        valid_decisions = {"merge", "separate", "unresolved"}
        valid_reason_codes = {
            "alias_equivalent",
            "context_supports",
            "context_conflicts",
            "ambiguous",
            "insufficient_evidence",
        }
        provider, model = self.model_identities.get("entity_judge", (None, None))

        def judge_batch(
            batch_index: int,
            batch: list[tuple[str, dict[str, Any]]],
        ) -> list[EntityMatchDecision]:
            pair_refs = dict(batch)
            payload = {
                "task": "entity_canonicalization_judge",
                "pairs": [
                    {
                        "id": pair_id,
                        "alias": str(item["alias"]),
                        "entity_type": str(item["entity_type"]),
                        "mention_context": {
                            key: _compact(str(value or ""), 300)
                            for key, value in dict(
                                item.get("mention_context") or {}
                            ).items()
                        },
                        "candidate": {
                            "canonical_id": str(
                                item["candidate"].get("canonical_id") or ""
                            ),
                            "representative_alias": str(
                                item["candidate"].get("representative_alias") or ""
                            ),
                            "aliases": list(
                                item["candidate"].get("aliases") or []
                            ),
                        },
                    }
                    for pair_id, item in batch
                ],
                "instructions": [
                    "Judge whether alias and candidate refer to the same real-world third-party entity.",
                    "Use unresolved when evidence is ambiguous, generic, or insufficient.",
                    "Return exactly one compact decision for every supplied pair id.",
                    UNTRUSTED_DATA_INSTRUCTION,
                ],
            }

            def invoke() -> list[EntityMatchDecision]:
                assert self.entity_judge_client is not None
                schema = _entity_judge_json_schema(pair_refs)
                raw = self.entity_judge_client.chat(
                    messages_with_json_schema(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "Return only valid JSON. "
                                    + UNTRUSTED_DATA_INSTRUCTION
                                ),
                            },
                            {
                                "role": "user",
                                "content": json.dumps(payload, ensure_ascii=False),
                            },
                        ],
                        schema,
                    ),
                    json_mode=True,
                    json_schema=schema,
                )
                data = parse_json_object(raw)
                raw_decisions = data.get("decisions")
                if not isinstance(raw_decisions, list):
                    raise ValueError("entity judge response has no decisions list")
                by_id: dict[str, EntityMatchDecision] = {}
                for value in raw_decisions:
                    if not isinstance(value, dict):
                        raise ValueError(
                            "entity judge response contains a non-object decision"
                        )
                    pair_id = str(value.get("id") or "")
                    if pair_id not in pair_refs:
                        raise ValueError(
                            f"entity judge response contains invalid id {pair_id!r}"
                        )
                    if pair_id in by_id:
                        raise ValueError(
                            f"entity judge response repeats id {pair_id}"
                        )
                    decision = str(value.get("decision") or "")
                    reason_code = str(value.get("reason_code") or "")
                    if decision not in valid_decisions:
                        raise ValueError(
                            f"entity judge response has invalid decision {decision!r}"
                        )
                    if reason_code not in valid_reason_codes:
                        raise ValueError(
                            "entity judge response has invalid reason_code "
                            f"{reason_code!r}"
                        )
                    by_id[pair_id] = EntityMatchDecision(
                        pair_id=pair_id,
                        decision=decision,
                        reason_code=reason_code,
                        batch_index=batch_index,
                    )
                missing = [pair_id for pair_id, _ in batch if pair_id not in by_id]
                if missing:
                    raise ValueError(
                        "entity judge response omitted decisions for "
                        + ", ".join(missing)
                    )
                return [by_id[pair_id] for pair_id, _ in batch]

            scope_id = str(
                checkpoint_descriptor.get("scope_id")
                if checkpoint_descriptor
                else "entity"
            )
            checkpoint_key = (
                f"{checkpoint_descriptor['checkpoint_key']}:batch-{batch_index + 1}"
                if checkpoint_descriptor
                else None
            )
            return retry_v4_call(
                invoke,
                policy=self.failure_policy,
                context=V4OperationContext(
                    stage="entity_judge",
                    unit_id=f"{scope_id}:batch-{batch_index + 1}",
                    provider=provider,
                    model=model,
                    checkpoint_key=checkpoint_key,
                ),
                error_type=V4BuildStageError,
                validation_errors=(ValueError, TypeError),
            )

        max_workers = min(self.config.entity_judge_workers, len(batches))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(judge_batch, batch_index, batch)
                for batch_index, batch in enumerate(batches)
            ]
            ordered: list[EntityMatchDecision] = []
            for batch_index, (batch, future) in enumerate(zip(batches, futures)):
                try:
                    ordered.extend(future.result())
                except Exception as error:
                    if checkpoint_descriptor is not None:
                        self._record_checkpoint_failure(
                            checkpoint_descriptor,
                            error,
                            attempt=int(getattr(error, "attempts", 1)),
                        )
                    ordered.extend(
                        unresolved_batch(batch_index, batch, failed=True)
                    )
        return ordered

    def _record_entity_merge(
        self,
        result: EntityCanonicalizationResult,
        first: dict[str, Any],
        second: dict[str, Any],
    ) -> None:
        target, source = _preferred_entity_merge_target(first, second)
        target_id = self._resolved_entity_id(str(target["canonical_id"]), result.merged_ids)
        source_id = self._resolved_entity_id(str(source["canonical_id"]), result.merged_ids)
        if source_id != target_id:
            result.merged_ids[source_id] = target_id

    @classmethod
    def _entity_registry_lock(cls, namespace: str) -> threading.RLock:
        with cls._entity_registry_locks_guard:
            lock = cls._entity_registry_locks.get(namespace)
            if lock is None:
                lock = threading.RLock()
                cls._entity_registry_locks[namespace] = lock
            return lock

    @staticmethod
    def _resolved_entity_id(canonical_id: str, merged_ids: dict[str, str]) -> str:
        current = canonical_id
        seen: set[str] = set()
        while current in merged_ids and current not in seen:
            seen.add(current)
            current = merged_ids[current]
        return current

    @staticmethod
    def _apply_entity_id_merges_to_nodes(nodes: Iterable[MemoryNode], merged_ids: dict[str, str]) -> None:
        for node in nodes:
            node.entity_ids = [
                V4MemoryBuilder._resolved_entity_id(entity_id, merged_ids)
                for entity_id in (node.entity_ids or node.entities)
            ]

    @staticmethod
    def _event_entity_ids(
        entities: list[str],
        actors: list[str],
        entity_resolution: EntityCanonicalizationResult | None,
    ) -> list[str]:
        if entity_resolution is None or not entity_resolution.enabled:
            return list(entities)
        actor_keys = {actor.casefold() for actor in actors}
        output: list[str] = []
        for entity in entities:
            key = entity.casefold()
            if key in actor_keys:
                output.append(entity)
                continue
            entity_type = entity_resolution.alias_type_by_key.get(key)
            canonical_id = (
                entity_resolution.alias_to_id.get((key, entity_type))
                if entity_type else None
            )
            output.append(canonical_id or entity)
        return output

    def _event_node(
        self, namespace: str, scope_id: str, chain: TopicChain, event: dict[str, Any],
        refs: list[dict[str, Any]], metadata: dict[str, Any],
        entity_resolution: EntityCanonicalizationResult | None = None,
    ) -> MemoryNode:
        summary = str(event.get("summary") or event.get("title") or "").strip()
        if not summary:
            summary = _compact(" ".join(str(ref.get("content") or "") for ref in refs), 360)
        observed = _optional_string(event.get("observed_at")) or max(
            (str(ref["observed_at"]) for ref in refs if ref.get("observed_at")), default=None
        )
        start = _optional_string(event.get("event_time_start"))
        end = _optional_string(event.get("event_time_end"))
        precision = _optional_string(event.get("time_precision"))
        raw_expression = _optional_string(event.get("raw_time_expression"))
        # Trust but verify: the builder LLM is instructed to copy only explicit
        # absolute dates into event_time_start, but it sometimes returns a
        # relative phrase ("last week") instead. A non-ISO value would be stored
        # as absolute and poison every downstream time index, so demote it to
        # the deterministic normalizer (which resolves against the observed_at
        # anchor and leaves vague expressions unresolved).
        anchor_mismatch = False
        if start and parse_iso_datetime(start) is None:
            start = None
            anchor_mismatch = True
        if start and not precision:
            precision = _infer_time_precision(start)
        normalization_status = "absolute" if start else "unresolved"
        normalization_method = "builder_explicit" if start else None
        if not start:
            normalized = self.time_normalizer.normalize(raw_expression, observed)
            start, end, precision = normalized.absolute_time_start, normalized.absolute_time_end, normalized.time_precision
            normalization_status = normalized.normalization_status
            normalization_method = normalized.normalization_method
        # Precision de-inflation: a datetime event_time whose raw expression
        # never mentioned a time-of-day almost always got the anchor's HMS
        # (e.g. LoCoMo session time) copied in by the builder LLM. Drop the
        # spurious HMS so it cannot leak into retrieval bounds or the answer
        # prompt. Genuine "3pm" events keep HMS because raw_expression carries
        # the time token.
        start, end, precision, datetime_demoted = _demote_spurious_datetime(
            start, end, precision, raw_expression
        )
        actors = _strings(event.get("actors"))
        objects = _strings(event.get("objects"))
        entities = _stable_unique([*_strings(event.get("entities")), *actors, *objects, *([str(event["location"])] if event.get("location") else [])])
        entity_ids = self._event_entity_ids(entities, actors, entity_resolution)
        action = str(event.get("action") or "experienced").strip()
        exact_mentions = [
            mention
            for mention in _strings(event.get("exact_mentions"))
            if len(mention) <= EXACT_MENTION_MAX_LENGTH
        ]
        identity = str(event.get("event_key") or "") or "|".join(
            [_normalize_text(summary), _normalize_text(action), str(start or "")]
        )
        node_metadata = {
            **metadata, "event_time_source": "extracted" if normalization_method == "builder_explicit" else "normalized",
            "raw_time_expression": raw_expression, "time_anchor": observed,
            "normalization_status": normalization_status,
            "normalization_method": normalization_method,
            "time_anchor_mismatch": anchor_mismatch,
            "datetime_demoted": datetime_demoted,
        }
        if exact_mentions:
            node_metadata["exact_mentions"] = exact_mentions
        return MemoryNode(
            id=stable_id("event", scope_id, chain.topic_id, identity), namespace=namespace,
            scope_id=scope_id, chain_id=chain.id, topic_id=chain.topic_id, node_type="event",
            title=str(event.get("title") or _compact(summary, 80)), summary=summary,
            text="\n".join(str(ref.get("content") or "") for ref in refs), actors=actors,
            action=action, objects=objects, location=_optional_string(event.get("location")),
            event_time_start=start, event_time_end=end, time_precision=precision,
            observed_at=observed, entities=entities, entity_ids=entity_ids,
            importance=_float01(event.get("importance"), 0.6), confidence=_float01(event.get("confidence"), 0.9),
            evidence_refs=refs, metadata=node_metadata,
        )

    def _apply_semantic_updates(
        self, namespace: str, scope_id: str,
        updates: list[tuple[dict[str, Any], list[dict[str, Any]], str]],
        chains: dict[str, TopicChain], nodes: dict[str, MemoryNode], edges: dict[str, MemoryEdge],
        event_refs: dict[str, str], metadata: dict[str, Any],
    ) -> None:
        for update, evidence, ref_prefix in updates:
            topic = self._route_topic(update, evidence, chains, ParticipantScope(scope_id, namespace, []))
            chain = self._ensure_chain(namespace, scope_id, topic, chains)
            update_fingerprint = _stable_digest({
                "ref_prefix": ref_prefix,
                "update": update,
                "evidence": evidence,
            })
            applied_updates = list(
                chain.metadata.get("semantic_update_fingerprints") or []
            )
            if update_fingerprint in applied_updates:
                continue

            def mark_applied() -> None:
                chain.metadata["semantic_update_fingerprints"] = sorted({
                    *applied_updates, update_fingerprint,
                })

            if "decision" in update:
                if self._apply_reducer_update(
                    namespace, scope_id, chain, update, evidence, ref_prefix,
                    nodes, edges, event_refs, metadata,
                ):
                    mark_applied()
                continue
            if self.semantic_reducer_client is None or isinstance(
                self.semantic_reducer_client, NoopChatClient
            ):
                raise ValueError(
                    "V4 candidate semantic claims require a configured semantic reducer"
                )
            valid_from = self._semantic_boundary_time(update, evidence)
            current_node = self._current_semantic_node(
                nodes, chain.id, valid_from
            )
            payload = {
                "chain": {"topic": chain.topic_id, "name": chain.name},
                "active_state": current_node.summary if current_node else None,
                "fact_snapshot": current_node.facts if current_node else [],
                "event": {
                    "valid_from": valid_from,
                    "trigger_event_ref": update.get("trigger_event_ref"),
                    "evidence": [ref.get("content") for ref in evidence],
                },
                "candidate_claims": update.get("claims") or update.get("facts") or [update],
                "known_fact_keys": [
                    fact.get("key")
                    for fact in (current_node.facts if current_node else [])
                ],
                "local_event_refs": sorted(
                    key.removeprefix(ref_prefix + ":")
                    for key in event_refs
                    if key.startswith(ref_prefix + ":")
                ),
            }
            unit_id = _stable_digest({
                "ref_prefix": ref_prefix,
                "update": update,
                "evidence": evidence,
            })[:24]
            descriptor = self._checkpoint_descriptor(
                namespace,
                scope_id,
                "semantic_update",
                unit_id,
                payload,
                prompt_version=SEMANTIC_REDUCER_PROMPT_VERSION,
                model_identity=self.model_identities.get("semantic_update"),
            )
            cached = self._load_checkpoint(descriptor)
            if cached is not None and isinstance(cached.get("output"), dict):
                reduced = dict(cached["output"])
                if not self._apply_reducer_update(
                    namespace, scope_id, chain, reduced, evidence, ref_prefix,
                    nodes, edges, event_refs, metadata,
                    raise_on_validation=True,
                ):
                    raise ValueError("validated semantic checkpoint no longer applies")
                mark_applied()
                continue
            def validate_and_apply(output: dict[str, Any]) -> dict[str, Any]:
                reduced = {**update, **output}
                if not self._apply_reducer_update(
                    namespace, scope_id, chain, reduced, evidence, ref_prefix,
                    nodes, edges, event_refs, metadata,
                    raise_on_validation=True,
                ):
                    raise ValueError("semantic reducer output failed validation")
                return reduced

            provider, model = self.model_identities.get(
                "semantic_update",
                (None, self.config.resolved_semantic_state_model),
            )
            reducer = SemanticReducer(
                self.semantic_reducer_client,
                failure_policy=self.failure_policy,
                provider=provider,
                model=model,
            )
            reduced = reducer.reduce(
                payload=payload,
                validate=validate_and_apply,
                unit_id=unit_id,
                checkpoint_key=descriptor["checkpoint_key"],
                on_failure=lambda attempt, error: self._record_checkpoint_failure(
                    descriptor, error, attempt=attempt
                ),
            )
            mark_applied()
            self._record_checkpoint_success(descriptor, reduced)

    def _semantic_boundary_time(
        self,
        update: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> str | None:
        explicit = _optional_string(update.get("valid_from"))
        candidates = [explicit] if explicit else [
            _optional_string(ref.get("observed_at"))
            for ref in evidence
            if _optional_string(ref.get("observed_at"))
        ]
        normalized: list[str] = []
        for value in candidates:
            result = self.time_normalizer.normalize(value, None)
            if result.absolute_time_start is None:
                raise SemanticValidationError(
                    "semantic update boundary could not be resolved to absolute ISO time"
                )
            normalized.append(result.absolute_time_start)
        return max(normalized, default=None)

    def _current_semantic_node(
        self,
        nodes: dict[str, MemoryNode],
        chain_id: str,
        boundary_time: str | None,
    ) -> MemoryNode | None:
        boundary = parse_iso_datetime(boundary_time)
        if boundary_time is not None and boundary is None:
            raise SemanticValidationError(
                "semantic update boundary must be an absolute ISO time"
            )

        eligible: list[tuple[MemoryNode, datetime | None]] = []
        for node in nodes.values():
            if (
                node.chain_id != chain_id
                or node.node_type not in {"chain_head", "semantic_state"}
                or not node.facts
                or node.metadata.get("unresolved_conflict")
            ):
                continue
            node_time_value = self._semantic_boundary_time(
                {"valid_from": node.valid_from}, []
            )
            node_time = parse_iso_datetime(node_time_value)
            if boundary is not None and node_time is not None and node_time > boundary:
                continue
            eligible.append((node, node_time))

        selected = max(
            eligible,
            key=lambda item: (
                int(item[0].metadata.get("state_epoch", 0)),
                item[1] or datetime.min,
                item[0].id,
            ),
            default=None,
        )
        return selected[0] if selected is not None else None

    def _apply_reducer_update(
        self, namespace: str, scope_id: str, chain: TopicChain,
        update: dict[str, Any], evidence: list[dict[str, Any]], ref_prefix: str,
        nodes: dict[str, MemoryNode], edges: dict[str, MemoryEdge],
        event_refs: dict[str, str], metadata: dict[str, Any],
        *, raise_on_validation: bool = False,
    ) -> bool:
        head_id = stable_id("head", scope_id, chain.topic_id)
        valid_from = self._semantic_boundary_time(update, evidence)
        current_node = self._current_semantic_node(
            nodes, chain.id, valid_from
        )
        current = self._semantic_epoch(current_node) if current_node else None
        reducer_update = dict(update)
        reducer_operations = [dict(value) for value in (update.get("operations") or [])]
        replay_only = bool(reducer_operations and current is not None)
        for operation in reducer_operations:
            if str(operation.get("operation") or "").lower() != "add":
                replay_only = False
                continue
            key = fact_key(
                str(operation.get("subject") or ""),
                str(operation.get("dimension") or ""),
                str(operation.get("aspect") or ""),
            )
            existing = current.facts.get(key) if current else None
            if existing is None or existing.value != operation.get("value"):
                replay_only = False
                continue
            operation["operation"] = "reinforce"
            operation["fact_key"] = key
        if replay_only:
            reducer_update["decision"] = "reinforce"
            reducer_update["operations"] = reducer_operations
        local_refs = _stable_unique(
            ref for operation in (update.get("operations") or [])
            if isinstance(operation, dict)
            for ref in _strings(operation.get("evidence_event_refs"))
        )
        unknown_local_refs = [
            ref for ref in local_refs
            if f"{ref_prefix}:{ref}" not in event_refs
        ]
        if unknown_local_refs:
            error = SemanticValidationError(
                "semantic operation references evidence outside supplied local event refs"
            )
            if raise_on_validation:
                raise error
            return False
        trigger_ids = _stable_unique(
            event_refs.get(f"{ref_prefix}:{ref}") for ref in local_refs
            if event_refs.get(f"{ref_prefix}:{ref}")
        )
        trigger_ref = _optional_string(update.get("trigger_event_ref"))
        if trigger_ref:
            trigger = event_refs.get(f"{ref_prefix}:{trigger_ref}")
            if trigger:
                trigger_ids = _stable_unique([*trigger_ids, trigger])
        try:
            transition = SemanticEpochMachine().apply(
                participant_scope=scope_id,
                chain_id=chain.id,
                current=current,
                reducer_output=reducer_update,
                boundary_time=valid_from,
                trigger_event_refs=local_refs or trigger_ids,
            )
        except SemanticValidationError as exc:
            for trigger_id in trigger_ids:
                event = nodes.get(trigger_id)
                if event:
                    event.metadata["semantic_reconciliation_status"] = "failed"
                    event.metadata["semantic_reconciliation_error"] = {
                        "code": "reducer_validation_failed", "message": str(exc),
                    }
            if raise_on_validation:
                raise
            return False
        for trigger_id in trigger_ids:
            event = nodes.get(trigger_id)
            if event:
                event.metadata["semantic_reconciliation_status"] = "succeeded"
                event.metadata.pop("semantic_reconciliation_error", None)
        if transition.pending is not None:
            for trigger_id in trigger_ids:
                event = nodes.get(trigger_id)
                if event:
                    event.metadata.setdefault("pending_semantic_conflicts", []).append({
                        "reason": transition.pending.reason,
                        "boundary_time": transition.pending.boundary_time,
                        "operations": list(transition.pending.candidate_operations),
                    })
            return True
        if transition.state is None:
            return True
        if transition.created_state and current_node is not None and transition.state.id == current_node.id:
            merged = _merge_refs(
                _merge_refs(current_node.evidence_refs, evidence),
                self._trigger_evidence_refs(nodes, trigger_ids),
            )
            if len(merged) != len(current_node.evidence_refs):
                current_node.evidence_refs = merged
            if current_node.node_type in {"chain_head", "semantic_state"}:
                current_node.text = ""
                current_node.embedding = None
            return True

        target_id = head_id if current_node is None else transition.state.id
        target_type = "chain_head" if current_node is None else (
            "semantic_state" if transition.created_state else current_node.node_type
        )
        target = current_node if current_node is not None and not transition.created_state else None
        if target is None:
            target = MemoryNode(
                id=target_id, namespace=namespace, scope_id=scope_id, chain_id=chain.id,
                topic_id=chain.topic_id, node_type=target_type, title=chain.name,
                summary=transition.state.summary, valid_from=transition.state.valid_from,
                evidence_refs=evidence, metadata={**metadata},
            )
            nodes[target.id] = target
        target.summary = transition.state.summary
        target.facts = [self._semantic_fact_dict(fact) for fact in transition.state.facts.values()]
        target.changed_keys = list(transition.changed_keys)
        target.valid_from = transition.state.valid_from
        target.valid_to = transition.state.valid_to
        target.trigger_event_ids = trigger_ids
        target.evidence_refs = _merge_refs(
            _merge_refs(target.evidence_refs, evidence),
            self._trigger_evidence_refs(nodes, trigger_ids),
        )
        target.entities = _stable_unique(fact.subject for fact in transition.state.facts.values())
        if target.node_type in {"chain_head", "semantic_state"}:
            # Aggregation nodes are represented by their reducer-generated
            # summary + facts + entities. Raw turn wording stays in
            # evidence_refs / v4_evidence_records and must not leak into the
            # node text (and therefore embedding/FTS) again.
            target.text = ""
        target.confidence = min(
            (fact.confidence for fact in transition.state.facts.values()), default=1.0
        )
        target.embedding = None
        target.metadata.update({
            "semantic_role": "active",
            "state_epoch": transition.state.epoch,
            "state_summary": transition.state.summary,
            "summary_status": "reducer_generated",
            "current_state_id": target.id,
        })
        if target.node_type == "chain_head":
            target.metadata["navigation_summary"] = transition.state.summary
        if transition.created_state and current_node is not None:
            current_node.valid_to = valid_from
            current_node.metadata["semantic_role"] = "historical"
            current_node.metadata["current_state_id"] = target.id
            target.previous_state_id = current_node.id
            self._add_edge(
                edges, namespace, scope_id, current_node.id, target.id, "SEMANTIC_NEXT",
                confidence=target.confidence, trust="explicit", evidence=evidence,
                explanation="validated semantic revision",
            )
            self._add_edge(
                edges, namespace, scope_id, target.id, current_node.id, "SUPERSEDES",
                confidence=target.confidence, trust="explicit", evidence=evidence,
                explanation="validated semantic revision supersedes prior epoch",
            )
            for key in transition.changed_keys:
                self._add_edge(
                    edges, namespace, scope_id, target.id, current_node.id, "CONTRADICTS",
                    confidence=target.confidence, trust="explicit", evidence=evidence,
                    explanation=f"explicit replace/retract of {key}",
                    edge_metadata={"fact_keys": list(transition.changed_keys)},
                )
            if target.id != head_id:
                self._add_edge(
                    edges, namespace, scope_id, head_id, target.id, "CURRENT_STATE",
                    trust="explicit", explanation="chain current semantic state",
                )
        for trigger_id in trigger_ids:
            edge_type = "TRIGGERS_STATE_CHANGE" if transition.created_state and current_node is not None else "SUPPORTS_STATE"
            self._add_edge(
                edges, namespace, scope_id, trigger_id, target.id, edge_type,
                confidence=target.confidence, trust="explicit", evidence=evidence,
                explanation="semantic reducer operation evidence",
                edge_metadata={
                    "fact_keys": _stable_unique(
                        [*transition.changed_keys, *(
                            operation.get("fact_key") for operation in (update.get("operations") or [])
                            if isinstance(operation, dict) and operation.get("fact_key")
                        )]
                    ),
                    "confidence_source": "semantic_reducer",
                },
            )
        return True

    @staticmethod
    def _semantic_epoch(node: MemoryNode) -> SemanticEpoch:
        facts = {}
        for value in node.facts:
            key = str(value.get("key") or "")
            facts[key] = SemanticFactRecord(
                key=key, subject=str(value.get("subject") or "user"),
                dimension=str(value.get("dimension") or value.get("predicate") or "state"),
                aspect=str(value.get("aspect") or "value"), value=value.get("value"),
                valid_from=value.get("valid_from") or node.valid_from,
                valid_to=value.get("valid_to") or node.valid_to,
                confidence=_float01(value.get("confidence"), node.confidence),
                evidence_event_refs=_strings(value.get("evidence_event_refs")),
                created_at=value.get("created_at"), updated_at=value.get("updated_at"),
                history_origin=value.get("history_origin"),
            )
        return SemanticEpoch(
            id=node.id, chain_id=node.chain_id,
            epoch=int(node.metadata.get("state_epoch", 0)),
            summary=node.summary, facts=facts, valid_from=node.valid_from,
            valid_to=node.valid_to, is_chain_head=node.node_type == "chain_head",
        )

    @staticmethod
    def _semantic_fact_dict(fact: SemanticFactRecord) -> dict[str, Any]:
        return {
            "key": fact.key, "subject": fact.subject,
            "dimension": fact.dimension, "aspect": fact.aspect,
            "predicate": f"{fact.dimension}.{fact.aspect}", "value": fact.value,
            "valid_from": fact.valid_from, "valid_to": fact.valid_to,
            "confidence": fact.confidence,
            "evidence_event_refs": list(fact.evidence_event_refs),
            "created_at": fact.created_at, "updated_at": fact.updated_at,
            "history_origin": fact.history_origin,
        }

    @staticmethod
    def _trigger_evidence_refs(
        nodes: dict[str, MemoryNode], trigger_ids: Iterable[str]
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for trigger_id in trigger_ids:
            event = nodes.get(trigger_id)
            if event is not None and event.evidence_refs:
                output.extend(event.evidence_refs)
        return output

    def _support_state_edges(
        self, namespace: str, scope_id: str, nodes: dict[str, MemoryNode],
        edges: dict[str, MemoryEdge], affected_chain_ids: set[str] | None,
    ) -> None:
        # V4 support edges are created only from validated semantic operations.
        # Time overlap alone is not semantic support and must not manufacture an
        # edge during rebuild.
        return

    def _rebuild_heads(
        self, namespace: str, scope_id: str, chains: dict[str, TopicChain],
        nodes: dict[str, MemoryNode], metadata: dict[str, Any],
    ) -> None:
        for chain in chains.values():
            existing_head = nodes.get(stable_id("head", scope_id, chain.topic_id))
            members = [node for node in nodes.values() if node.chain_id == chain.id and node.node_type != "chain_head"]
            if not members:
                continue
            # Sort key is the ISO string in event_time_start (or observed_at).
            # Mixed precision sorts correctly because lexicographic ISO order
            # matches chronological order for equal-precision values; a bare
            # date "2024-04-01" sorts before a datetime "2024-04-01T10:00:00"
            # at the same day, which is the desired within-day ordering.
            events = sorted(
                [node for node in members if node.node_type == "event"],
                key=lambda item: (item.event_time_start or item.observed_at or "9999", item.id),
            )
            states = sorted(
                [node for node in members if node.node_type == "semantic_state"],
                key=lambda item: (item.valid_from or "", item.id),
            )
            ordered_states = [state for state in states if not state.metadata.get("unresolved_conflict")]
            entities = _stable_unique(entity for node in members for entity in node.entities)[:16]
            entity_ids = _stable_unique(entity_id for node in members for entity_id in node.entity_ids)[:16]
            chain.representative_entities = entities
            if existing_head is not None and existing_head.facts:
                confirmed = [existing_head, *ordered_states]
                current = max(
                    confirmed,
                    key=lambda item: (int(item.metadata.get("state_epoch", 0)), item.valid_from or "", item.id),
                )
                navigation = current.summary
                if current.id != existing_head.id:
                    navigation = f"Current: {current.summary} Historical: {existing_head.summary}"
                existing_head.entities = entities
                existing_head.entity_ids = entity_ids
                existing_head.text = ""
                existing_head.embedding = None
                existing_head.metadata.update({
                    **metadata,
                    "node_count": len(members),
                    "state_summary": existing_head.summary,
                    "navigation_summary": navigation,
                    "current_state_id": current.id,
                    "semantic_role": "active" if current.id == existing_head.id else "historical",
                    "summary_status": "reducer_generated",
                    "navigation_only": False,
                })
                chain.rolling_summary = navigation[:1000]
                continue
            chain.rolling_summary = " ".join(node.summary for node in members[-6:])[:1000]
            head_id = stable_id("head", scope_id, chain.topic_id)
            head = MemoryNode(
                id=head_id, namespace=namespace, scope_id=scope_id, chain_id=chain.id,
                topic_id=chain.topic_id, node_type="chain_head", title=chain.name,
                summary=chain.rolling_summary, entities=entities,
                entity_ids=entity_ids,
                event_time_start=events[0].event_time_start if events else None,
                event_time_end=(events[-1].event_time_end or events[-1].event_time_start) if events else None,
                importance=0.5, confidence=1.0, evidence_refs=[], metadata={
                    **metadata, "node_count": len(members),
                    "first_timeline_node_id": events[0].id if events else None,
                    "initial_state_id": ordered_states[0].id if ordered_states else None,
                    "current_state_id": ordered_states[-1].id if ordered_states else None,
                    "recent_important_node_ids": [node.id for node in sorted(members, key=lambda item: item.importance, reverse=True)[:5]],
                    "navigation_only": True,
                },
            )
            nodes[head_id] = head

    def _structural_edges(
        self, namespace: str, scope_id: str, chains: dict[str, TopicChain],
        nodes: dict[str, MemoryNode], edges: dict[str, MemoryEdge],
        affected_chain_ids: set[str] | None = None,
    ) -> None:
        # When affected_chain_ids is provided (incremental rebuild), only re-derive
        # structural edges for chains that gained nodes this ingest; unaffected
        # chains keep their retained structural edges. None means re-derive all.
        for chain in chains.values():
            if affected_chain_ids is not None and chain.id not in affected_chain_ids:
                continue
            head = nodes.get(stable_id("head", scope_id, chain.topic_id))
            if head is None:
                continue
            events = sorted(
                [node for node in nodes.values() if node.chain_id == chain.id and node.node_type == "event"],
                key=lambda item: (item.event_time_start or item.observed_at or "9999", item.id),
            )
            states = sorted(
                [node for node in nodes.values() if node.chain_id == chain.id and node.node_type == "semantic_state"],
                key=lambda item: (item.valid_from or "", item.id),
            )
            if events:
                self._add_edge(edges, namespace, scope_id, head.id, events[0].id, "FIRST_TIMELINE_NODE", trust="derived")
                for first, second in zip(events, events[1:]):
                    self._add_edge(edges, namespace, scope_id, first.id, second.id, "TEMPORAL_NEXT", trust="derived")
            ordered = [state for state in states if not state.metadata.get("unresolved_conflict")]
            if ordered:
                self._add_edge(edges, namespace, scope_id, head.id, ordered[0].id, "INITIAL_STATE", trust="derived")
                current = ordered[-1]
                self._add_edge(edges, namespace, scope_id, head.id, current.id, "CURRENT_STATE", trust="derived")
                for first, second in zip(ordered, ordered[1:]):
                    self._add_edge(edges, namespace, scope_id, first.id, second.id, "SEMANTIC_NEXT", trust="derived")
                    self._add_edge(
                        edges, namespace, scope_id, second.id, first.id, "SUPERSEDES",
                        confidence=second.confidence, trust="inferred", evidence=second.evidence_refs,
                        explanation="new state supersedes previous snapshot",
                    )
            for node in [*events, *states]:
                self._add_edge(edges, namespace, scope_id, node.id, head.id, "SUPPORTED_BY", trust="derived")

    def _explicit_edges(
        self, namespace: str, scope_id: str, relations: list[tuple[str, str, dict[str, Any]]],
        event_refs: dict[str, str], nodes: dict[str, MemoryNode], edges: dict[str, MemoryEdge],
    ) -> None:
        for ref_prefix, source_ref, relation in relations:
            target_ref = str(relation.get("target") or relation.get("to") or "")
            source = event_refs.get(f"{ref_prefix}:{source_ref}")
            target = event_refs.get(f"{ref_prefix}:{target_ref}")
            edge_type = str(relation.get("edge_type") or relation.get("type") or "").upper()
            if not source or not target or source not in nodes or target not in nodes:
                continue
            try:
                self._add_edge(
                    edges, namespace, scope_id, source, target, edge_type,
                    confidence=_float01(relation.get("confidence"), 0.9), trust="explicit",
                    explanation=str(relation.get("explanation") or "explicitly extracted relation"),
                )
            except ValueError:
                continue

    def _cross_window_edges(
        self, namespace: str, scope_id: str, nodes: dict[str, MemoryNode],
        edges: dict[str, MemoryEdge], new_node_ids: set[str],
        chains: dict[str, TopicChain], metadata: dict[str, Any],
    ) -> None:
        """Cross-window EVENT edges (#1, #3) + AFFECTS_TOPIC (#2).

        Runs on global (scope-wide) nodes so relations can cross extraction
        windows. Deterministic candidate signals are adjudicated by the
        adjudication LLM (explicit) or, in Noop mode, emitted as derived.
        Incremental: only pairs touching a newly-ingested node are considered.
        """
        if not new_node_ids:
            return
        events = [node for node in nodes.values() if node.node_type == "event"]
        by_id = {node.id: node for node in events}
        candidates = self._cross_window_candidates(events, new_node_ids)
        if not candidates:
            return
        decided = self._adjudicate_cross_window(candidates, by_id)
        chain_id_to_topic = {chain.id: chain.topic_id for chain in chains.values()}
        node_chain = {node.id: node.chain_id for node in events}
        for source_id, target_id, edge_type, confidence, trust, explanation in decided:
            source_node = by_id.get(source_id)
            target_node = by_id.get(target_id)
            if source_node is None or target_node is None:
                continue
            source_id, target_id = _oriented_event_endpoints(source_node, target_node, edge_type)
            try:
                self._add_edge(
                    edges, namespace, scope_id, source_id, target_id, edge_type,
                    confidence=confidence, trust=trust, explanation=explanation,
                )
            except ValueError:
                continue
            # AFFECTS_TOPIC: a confirmed cross-chain EVENT edge also links the
            # source event to the target chain's head as a navigation shortcut.
            if (
                edge_type in EVENT_EDGE_TYPES
                and node_chain.get(source_id)
                and node_chain.get(source_id) != node_chain.get(target_id)
            ):
                target_chain_id = node_chain.get(target_id)
                topic_id = chain_id_to_topic.get(target_chain_id)
                if topic_id:
                    head_id = stable_id("head", scope_id, topic_id)
                    if head_id in nodes:
                        try:
                            self._add_edge(
                                edges, namespace, scope_id, source_id, head_id, "AFFECTS_TOPIC",
                                confidence=confidence * 0.8, trust="inferred",
                                explanation=f"event affects topic via {edge_type} relation",
                            )
                        except ValueError:
                            pass

    def _cross_window_candidates(
        self, events: list[MemoryNode], new_node_ids: set[str],
    ) -> list[tuple[str, str, str, str]]:
        """Deterministic candidate pairs: (source_id, target_id, proposed_type, signal).

        Signals (strongest first; a pair keeps the first match):
          - SAME_EVENT (with time): normalized action equal + precision-aligned
            time equal + entity Jaccard >= cross_window_same_event_entity_jaccard.
          - SAME_EVENT (no time): normalized action equal + entity Jaccard >=
            cross_window_same_event_entity_jaccard_no_time, when aligned time is
            unavailable on either side.
          - SAME_EVENT (title): different action but title/summary token overlap
            + entity Jaccard both clear the no-time bar — same event, different
            wording across windows.
          - FOLLOW_UP_OF: same topic, disjoint session, time gap <= config gap.
          - CONTEXT_FOR: shared significant entity, different chain.
        At least one endpoint must be in new_node_ids. Capped by config.
        """
        threshold = self.config.cross_window_same_event_entity_jaccard
        no_time_threshold = self.config.cross_window_same_event_entity_jaccard_no_time
        title_overlap_threshold = self.config.cross_window_same_event_title_token_overlap
        gap_seconds = self.config.cross_window_followup_gap_seconds
        max_pairs = self.config.cross_window_candidate_max
        seen: set[tuple[str, str, str]] = set()
        output: list[tuple[str, str, str, str]] = []
        new_events = [node for node in events if node.id in new_node_ids]
        for first in new_events:
            for second in events:
                if second.id == first.id:
                    continue
                proposed = self._propose_cross_window_relation(
                    first, second, threshold, no_time_threshold,
                    title_overlap_threshold, gap_seconds,
                )
                if proposed:
                    source_id, target_id = _oriented_event_endpoints(first, second, proposed[0])
                    pair = _event_candidate_key(source_id, target_id, proposed[0])
                    if pair in seen:
                        continue
                    seen.add(pair)
                    output.append((source_id, target_id, proposed[0], proposed[1]))
                if len(output) >= max_pairs:
                    return output
        return output

    @staticmethod
    def _propose_cross_window_relation(
        first: MemoryNode, second: MemoryNode, jaccard_threshold: float,
        no_time_threshold: float, title_overlap_threshold: float, gap_seconds: int,
    ) -> tuple[str, str] | None:
        first_action = _normalize_text(first.action)
        second_action = _normalize_text(second.action)
        first_time = first.event_time_start
        second_time = second.event_time_start
        jaccard = _entity_jaccard(first.entity_ids, second.entity_ids)
        if first_action and first_action == second_action:
            if first_time and second_time and _time_precision_aligned(first_time, second_time):
                if jaccard >= jaccard_threshold:
                    return ("SAME_EVENT", "same action + aligned time + entity overlap")
            elif jaccard >= no_time_threshold:
                # Same action with strong entity overlap but no aligned time —
                # likely the same event surfaced in two overlapping windows
                # where one extraction dropped or never had the time token.
                return ("SAME_EVENT", "same action + strong entity overlap (no aligned time)")
        elif (
            jaccard >= no_time_threshold
            and _title_token_overlap(first.title or first.summary, second.title or second.summary) >= title_overlap_threshold
        ):
            # Different action verb but near-identical title/summary plus strong
            # entity overlap — the same event described with different wording.
            return ("SAME_EVENT", "strong title/summary + entity overlap")
        if first.topic_id and first.topic_id == second.topic_id:
            first_sessions = _node_sessions(first)
            second_sessions = _node_sessions(second)
            if first_sessions and second_sessions and not first_sessions.intersection(second_sessions):
                gap = _time_gap_seconds(first, second)
                if 0 <= gap <= gap_seconds:
                    return ("FOLLOW_UP_OF", "same topic, cross-session, small time gap")
        if first.chain_id != second.chain_id:
            shared = _shared_significant_entity(first.entity_ids, second.entity_ids)
            if shared:
                return ("CONTEXT_FOR", f"shared significant entity: {shared}")
        return None

    def _adjudicate_cross_window(
        self, candidates: list[tuple[str, str, str, str]], by_id: dict[str, MemoryNode],
    ) -> list[tuple[str, str, str, float, str, str]]:
        """Return (source, target, edge_type, confidence, trust, explanation).

        With an adjudication LLM: batch-confirm pairs (LLM may return null to
        reject, or override edge_type). Without one (Noop/None): emit all
        candidates as derived with their proposed type.
        """
        if self.adjudication_client is None or isinstance(self.adjudication_client, NoopChatClient):
            return [
                (source, target, proposed, 0.7, "derived", f"cross-window {signal}")
                for source, target, proposed, signal in candidates
            ]
        if len(candidates) > 16:
            output: list[tuple[str, str, str, float, str, str]] = []
            for start in range(0, len(candidates), 16):
                output.extend(
                    self._adjudicate_cross_window(candidates[start:start + 16], by_id)
                )
            return output
        node_ids = sorted({sid for pair in candidates for sid in (pair[0], pair[1])})
        ref_of = {node_id: f"r{index + 1}" for index, node_id in enumerate(node_ids)}
        nodes_payload = [
            {
                "ref": ref_of[node_id],
                "title": by_id[node_id].title,
                "summary": _compact(by_id[node_id].summary, 512),
                "evidence": _compact(
                    " ".join(
                        str(ref.get("content") or "")
                        for ref in by_id[node_id].evidence_refs
                    ) or str(getattr(by_id[node_id], "text", "")),
                    512,
                ),
                "topic": by_id[node_id].topic_id,
                "time": {
                    "start": by_id[node_id].event_time_start,
                    "end": by_id[node_id].event_time_end,
                },
                "entities": by_id[node_id].entities,
                "action": by_id[node_id].action,
                "actors": list(getattr(by_id[node_id], "actors", []) or []),
                "objects": list(getattr(by_id[node_id], "objects", []) or []),
            }
            for node_id in node_ids
        ]
        pairs_payload = [
            {
                "source_ref": ref_of[source],
                "target_ref": ref_of[target],
                "proposed_edge_type": proposed,
                "signal": signal,
            }
            for source, target, proposed, signal in candidates
        ]
        payload = {
            "task": "cross_window_relation_adjudication",
            "nodes": nodes_payload,
            "candidate_pairs": pairs_payload,
            "instructions": [
                "For each candidate pair, judge whether the two events truly relate.",
                "proposed_edge_type and signal are non-binding candidate-generation hints; accept, override, or reject them from the supplied evidence.",
                "Return one of EVENT_EDGE_TYPES (CAUSES, CONTRIBUTES_TO, CONTEXT_FOR, ENABLES, "
                "PART_OF, FOLLOW_UP_OF, SAME_EVENT) or null to reject.",
                "Be conservative: reject if the relation direction is not clearly supported by the summaries and evidence.",
                "Keep each explanation concise and no longer than 20 English words.",
                UNTRUSTED_DATA_INSTRUCTION,
            ],
        }
        first_node = by_id[node_ids[0]]
        unit_id = _stable_digest(candidates)[:24]
        descriptor = self._checkpoint_descriptor(
            first_node.namespace,
            first_node.scope_id,
            "adjudication",
            unit_id,
            payload,
            prompt_version="cross-window-adjudication-v4",
            model_identity=self.model_identities.get("adjudication"),
        )
        cached = self._load_checkpoint(descriptor)
        if cached is not None and isinstance(cached.get("output"), list):
            return [
                (
                    str(item[0]), str(item[1]), str(item[2]), float(item[3]),
                    str(item[4]), str(item[5]),
                )
                for item in cached["output"]
            ]

        def invoke() -> list[tuple[str, str, str, float, str, str]]:
            assert self.adjudication_client is not None
            schema = _adjudication_json_schema(pairs_payload)
            raw = self.adjudication_client.chat(
                messages_with_json_schema(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Return only valid JSON grounded in the supplied evidence. "
                                + UNTRUSTED_DATA_INSTRUCTION
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                    schema,
                ),
                json_mode=True,
                json_schema=schema,
            )
            data = parse_json_object(raw)
            decisions = data.get("decisions")
            if not isinstance(decisions, list):
                raise ValueError("adjudication response has no decisions list")
            ref_to_id = {ref: node_id for node_id, ref in ref_of.items()}
            output: list[tuple[str, str, str, float, str, str]] = []
            seen_pairs: set[tuple[str, str]] = set()
            for decision in decisions:
                if not isinstance(decision, dict):
                    raise ValueError("adjudication decision must be an object")
                source_ref = str(decision.get("source_ref") or "")
                target_ref = str(decision.get("target_ref") or "")
                source = ref_to_id.get(source_ref)
                target = ref_to_id.get(target_ref)
                if source is None or target is None:
                    raise ValueError("adjudication decision references an unknown node")
                if (source_ref, target_ref) in seen_pairs:
                    raise ValueError("adjudication response duplicated a candidate pair")
                seen_pairs.add((source_ref, target_ref))
                confidence = decision.get("confidence")
                if (
                    isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or not 0.0 <= float(confidence) <= 1.0
                ):
                    raise ValueError("adjudication confidence must be numeric in 0..1")
                explanation = str(decision.get("explanation") or "").strip()
                if not explanation:
                    raise ValueError("adjudication decision requires an explanation")
                edge_type = str(decision.get("edge_type") or "").upper()
                if not edge_type or edge_type == "NULL":
                    continue
                if edge_type not in EVENT_EDGE_TYPES:
                    raise ValueError("adjudication decision has an invalid edge type")
                output.append((
                    source,
                    target,
                    edge_type,
                    float(confidence),
                    "explicit",
                    explanation,
                ))
            expected_pairs = {
                (item["source_ref"], item["target_ref"])
                for item in pairs_payload
            }
            if seen_pairs != expected_pairs:
                raise ValueError("adjudication response omitted candidate pairs")
            return output

        provider, model = self.model_identities.get("adjudication", (None, None))
        result = retry_v4_call(
            invoke,
            policy=self.failure_policy,
            context=V4OperationContext(
                stage="adjudication",
                unit_id=unit_id,
                provider=provider,
                model=model,
                checkpoint_key=descriptor["checkpoint_key"],
            ),
            error_type=V4BuildStageError,
            validation_errors=(ValueError, TypeError),
            on_failure=lambda attempt, error: self._record_checkpoint_failure(
                descriptor, error, attempt=1
            ),
        )
        self._record_checkpoint_success(descriptor, result)
        return result

    def build_cross_scope_edges(self, namespace: str, touched_scope_id: str) -> None:
        """Persisted cross-scope EVENT edges (#7).

        For the just-ingested scope, find other scopes sharing a participant,
        generate cross-scope candidate pairs, apply the bidirectional visibility
        filter (each node's evidence participants must be a subset of the other
        scope's participants), adjudicate, and persist to v4_cross_scope_edges.
        Cross-scope edges touching this scope are rewritten; others are untouched.
        """
        if not self.config.cross_scope_navigation:
            return
        store = self.store
        touched = store.get_scope(touched_scope_id)
        if touched is None:
            return
        touched_events = [n for n in store.list_nodes(touched_scope_id) if n.node_type == "event"]
        if not touched_events:
            store.replace_cross_scope_edges_for_scope(namespace, touched_scope_id, [])
            return
        touched_participants = {p.participant_id for p in store.scope_participants(touched_scope_id)}
        # Map participant_id -> other scopes containing that participant.
        partner_scopes: dict[str, ParticipantScope] = {}
        for participant_id in touched_participants:
            for scope in store.scopes_by_participant(namespace, participant_id):
                if scope.id != touched_scope_id:
                    partner_scopes[scope.id] = scope
        if not partner_scopes:
            store.replace_cross_scope_edges_for_scope(namespace, touched_scope_id, [])
            return
        max_pairs = self.config.cross_scope_candidate_max
        threshold = self.config.cross_window_same_event_entity_jaccard
        no_time_threshold = self.config.cross_window_same_event_entity_jaccard_no_time
        title_overlap_threshold = self.config.cross_window_same_event_title_token_overlap
        gap_seconds = self.config.cross_window_followup_gap_seconds
        all_candidates: list[tuple[str, str, str, str]] = []
        by_id: dict[str, MemoryNode] = {node.id: node for node in touched_events}
        evidence_cache: dict[str, set[str]] = {}

        def evidence_participants(node_id: str) -> set[str]:
            if node_id not in evidence_cache:
                evidence_cache[node_id] = store.node_evidence_participants(node_id)
            return evidence_cache[node_id]

        for partner in partner_scopes.values():
            partner_events = [n for n in store.list_nodes(partner.id) if n.node_type == "event"]
            partner_participants = {p.participant_id for p in store.scope_participants(partner.id)}
            for partner_node in partner_events:
                by_id[partner_node.id] = partner_node
            # Bidirectional visibility filter: a pair (N in touched, M in partner)
            # is allowed iff partner_participants ⊇ ev(N) AND touched_participants ⊇ ev(M).
            for n in touched_events:
                ev_n = evidence_participants(n.id)
                for m in partner_events:
                    ev_m = evidence_participants(m.id)
                    if not is_cross_scope_visible(
                        touched_participants,
                        ev_n,
                        partner_participants,
                        ev_m,
                    ):
                        continue
                    proposed = self._propose_cross_window_relation(
                        n, m, threshold, no_time_threshold, title_overlap_threshold, gap_seconds,
                    )
                    if proposed:
                        source_id, target_id = _oriented_event_endpoints(n, m, proposed[0])
                        all_candidates.append((source_id, target_id, proposed[0], proposed[1]))
                    if len(all_candidates) >= max_pairs:
                        break
                if len(all_candidates) >= max_pairs:
                    break
            if len(all_candidates) >= max_pairs:
                break
        if not all_candidates:
            store.replace_cross_scope_edges_for_scope(namespace, touched_scope_id, [])
            return
        # Dedup by pair (keep first / strongest signal).
        seen: set[tuple[str, str]] = set()
        unique: list[tuple[str, str, str, str]] = []
        for source_id, target_id, proposed, signal in all_candidates:
            pair = (source_id, target_id)
            if pair in seen:
                continue
            seen.add(pair)
            unique.append((source_id, target_id, proposed, signal))
        decided = self._adjudicate_cross_window(unique, by_id)
        replacement_edges: list[tuple[MemoryEdge, str]] = []
        for source_id, target_id, edge_type, confidence, trust, explanation in decided:
            if edge_type not in EVENT_EDGE_TYPES:
                continue
            source_node = by_id.get(source_id)
            target_node = by_id.get(target_id)
            if source_node is None or target_node is None:
                continue
            source_id, target_id = _oriented_event_endpoints(source_node, target_node, edge_type)
            source_node = by_id.get(source_id)
            target_node = by_id.get(target_id)
            if source_node is None or target_node is None:
                continue
            source_scope_id, target_scope_id = source_node.scope_id, target_node.scope_id
            if source_scope_id == target_scope_id:
                continue
            edge = MemoryEdge(
                id=stable_id("xedge", namespace, source_scope_id, target_scope_id,
                             source_id, target_id, edge_type),
                namespace=namespace, scope_id=source_scope_id, source_id=source_id,
                target_id=target_id, edge_type=edge_type, confidence=confidence,
                trust=trust, evidence_refs=[],
                created_method="cross_scope_builder", explanation=explanation,
            )
            replacement_edges.append((edge, target_scope_id))
        store.replace_cross_scope_edges_for_scope(
            namespace, touched_scope_id, replacement_edges
        )

    @staticmethod
    def _node_embedding_text(node: MemoryNode) -> str:
        parts = [node.topic_id, node.title, node.summary]
        if node.node_type not in {"chain_head", "semantic_state"}:
            parts.append(node.text)
        parts.append(" ".join(node.entities))
        if node.node_type in {"chain_head", "semantic_state"}:
            parts.append(_dump_facts(node.facts))
        return "\n".join(part for part in parts if part)

    def _embed_nodes(self, nodes: Iterable[MemoryNode]) -> None:
        node_list = list(nodes)
        descriptors: dict[str, dict[str, Any]] = {}
        for node in node_list:
            if node.embedding is not None:
                continue
            text = self._node_embedding_text(node)
            descriptor = self._checkpoint_descriptor(
                node.namespace,
                node.scope_id,
                "embedding_batch",
                node.id,
                {"node_id": node.id, "text": text},
                prompt_version="node-embedding-v1",
                model_identity=self.model_identities.get("embedding"),
            )
            descriptors[node.id] = descriptor
            cached = self._load_checkpoint(descriptor)
            if cached is not None and isinstance(cached.get("embedding"), list):
                node.embedding = [float(value) for value in cached["embedding"]]
        pending = [node for node in node_list if node.embedding is None]
        if not pending:
            return
        known_dimensions = {
            len(node.embedding)
            for node in node_list
            if node.embedding is not None
        }
        if len(known_dimensions) > 1:
            raise MemoryBuildError(
                f"existing node embeddings have mixed dimensions: {sorted(known_dimensions)}"
            )
        expected_dimension = next(iter(known_dimensions), None)
        texts = [self._node_embedding_text(node) for node in pending]

        def invoke() -> list[list[float]]:
            embeddings = self.embedding_client.embed_texts(texts)
            if len(embeddings) != len(pending):
                raise ValueError(
                    "embedding client returned an incorrect number of vectors"
                )
            normalized: list[list[float]] = []
            for node, embedding in zip(pending, embeddings):
                vector = np.asarray(embedding, dtype=np.float32)
                if vector.ndim != 1:
                    raise ValueError(f"node {node.id!r} embedding is not a vector")
                if expected_dimension is not None and int(vector.shape[0]) != int(expected_dimension):
                    raise ValueError(
                        f"node {node.id!r} embedding dimension {int(vector.shape[0])} "
                        f"does not match existing dimension {int(expected_dimension)}"
                    )
                norm = float(np.linalg.norm(vector))
                if not norm:
                    raise ValueError(f"node {node.id!r} has a zero embedding")
                normalized.append((vector / norm).tolist())
            return normalized

        provider, model = self.model_identities.get(
            "embedding", (None, self.config.embedding_model)
        )

        def record_embedding_failure(_attempt: int, error: Exception) -> None:
            for node in pending:
                self._record_checkpoint_failure(
                    descriptors[node.id], error, attempt=1
                )

        normalized = retry_v4_call(
            invoke,
            policy=self.failure_policy,
            context=V4OperationContext(
                stage="embedding_batch",
                unit_id=_stable_digest([node.id for node in pending])[:24],
                provider=provider,
                model=model,
                checkpoint_key=(
                    "embedding_batch:"
                    + _stable_digest(
                        [
                            descriptor["checkpoint_key"]
                            for descriptor in descriptors.values()
                        ]
                    )[:24]
                ),
            ),
            error_type=V4BuildStageError,
            validation_errors=(ValueError, TypeError),
            on_failure=record_embedding_failure,
        )
        for node, embedding in zip(pending, normalized):
            node.embedding = embedding
            self._record_checkpoint_success(
                descriptors[node.id],
                {"node_id": node.id, "dimension": len(embedding)},
                embedding=embedding,
            )

    def _semantic_fact_rows(self, nodes: Iterable[MemoryNode]) -> list[SemanticFact]:
        output = []
        for node in nodes:
            if node.node_type not in {"chain_head", "semantic_state"}:
                continue
            for value in node.facts:
                key = str(value.get("key") or "")
                output.append(SemanticFact(
                    id=stable_id("fact", node.id, key), namespace=node.namespace, scope_id=node.scope_id,
                    chain_id=node.chain_id, state_node_id=node.id, key=key,
                    subject=str(value.get("subject") or "user"), predicate=str(value.get("predicate") or key),
                    dimension=str(value.get("dimension") or value.get("predicate") or "state"),
                    aspect=str(value.get("aspect") or "value"),
                    value=value.get("value"), confidence=_float01(value.get("confidence"), node.confidence),
                    evidence_refs=node.evidence_refs,
                    valid_from=value.get("valid_from") or node.valid_from,
                    valid_to=value.get("valid_to") or node.valid_to,
                    created_at=value.get("created_at"), updated_at=value.get("updated_at"),
                    history_origin=value.get("history_origin"),
                ))
        return output

    @staticmethod
    def _add_edge(
        edges: dict[str, MemoryEdge], namespace: str, scope_id: str, source: str, target: str,
        edge_type: str, *, confidence: float = 1.0, trust: str = "explicit",
        evidence: list[dict[str, Any]] | None = None, explanation: str = "",
        edge_metadata: dict[str, Any] | None = None,
    ) -> None:
        if source == target:
            return
        edge_type = edge_type.upper()
        edge_id = stable_id("edge", scope_id, source, target, edge_type)
        current = edges.get(edge_id)
        if current is not None and current.trust == "explicit" and trust != "explicit":
            return
        edge = MemoryEdge(
            id=edge_id, namespace=namespace,
            scope_id=scope_id, source_id=source, target_id=target, edge_type=edge_type,
            confidence=confidence, trust=trust, evidence_refs=list(evidence or []),
            created_method="memory_builder", explanation=explanation,
            metadata=dict(edge_metadata or {}),
        )
        edges[edge.id] = edge

    @staticmethod
    def _select_evidence(
        item: dict[str, Any], all_refs: list[dict[str, Any]], by_turn: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        ids = item.get("evidence_turn_ids")
        if isinstance(ids, list) and ids:
            return [by_turn[str(value)] for value in ids if str(value) in by_turn]
        return list(all_refs)

    @staticmethod
    def _evidence_refs(session_id: str, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for turn in turns:
            turn_id = str(turn["turn_id"])
            observed = V4MemoryBuilder._turn_observed_at(turn)
            output.append({
                "evidence_id": stable_id("evidence", session_id, turn_id),
                "session_id": session_id, "turn_id": turn_id,
                "dia_id": _optional_string(turn.get("dia_id")),
                "role": str(turn.get("role") or "unknown"),
                "participant_id": _optional_string(
                    turn.get("participant_id") or turn.get("speaker_id") or turn.get("user_id")
                ),
                "observed_at": observed,
                "content": str(turn.get("content") or ""),
            })
        return output

    @staticmethod
    def _turn_observed_at(turn: dict[str, Any]) -> str | None:
        return turn_observed_at(turn)

    @staticmethod
    def _derive_participants(turns: list[dict[str, Any]]) -> list[dict[str, str]]:
        output: dict[tuple[str, str], dict[str, str]] = {}
        for turn in turns:
            raw_role = str(turn.get("role") or turn.get("speaker") or "participant").strip()
            role_key = raw_role.casefold()
            role = "agent" if role_key in {"assistant", "agent"} else (role_key if role_key in {"user", "system"} else "participant")
            participant_id = str(
                turn.get("participant_id") or turn.get("speaker_id") or turn.get("user_id")
                or turn.get("speaker") or raw_role
            ).strip()
            output[(role, participant_id.casefold())] = {"participant_id": participant_id, "role": role}
        if not output:
            raise ValueError("cannot derive participants from an empty conversation")
        return [output[key] for key in sorted(output)]

    @staticmethod
    def _normalize_turns(conversation: Any) -> list[dict[str, Any]]:
        if isinstance(conversation, dict):
            for key in ("turns", "conversation", "messages"):
                if key in conversation:
                    conversation = conversation[key]
                    break
            else:
                conversation = [conversation]
        output: list[dict[str, Any]] = []
        for outer_index, item in enumerate(conversation or []):
            if isinstance(item, dict) and isinstance(item.get("turns"), list):
                session_id = str(item.get("session_id") or item.get("id") or outer_index)
                session_timestamp = item.get("timestamp") or item.get("session_timestamp")
                for inner_index, raw in enumerate(item["turns"]):
                    turn = dict(raw) if isinstance(raw, dict) else {"content": str(raw)}
                    turn.setdefault("session_id", session_id)
                    turn.setdefault("session_timestamp", session_timestamp)
                    turn.setdefault("turn_id", inner_index + 1)
                    turn.setdefault("role", turn.get("speaker", "unknown"))
                    turn.setdefault("content", turn.get("text", turn.get("message", "")))
                    output.append(turn)
                continue
            turn = dict(item) if isinstance(item, dict) else {"content": str(item)}
            turn.setdefault("session_id", turn.get("session", "default"))
            turn.setdefault("turn_id", outer_index + 1)
            turn.setdefault("role", turn.get("speaker", "unknown"))
            turn.setdefault("content", turn.get("text", turn.get("message", "")))
            output.append(turn)
        return output


def _deterministic_semantic_updates(text: str, evidence: list[str], observed_at: str | None) -> list[dict[str, Any]]:
    patterns = [
        (r"\bI\s+(?:like|love)\s*([^,.!?]+)", "preference", "likes", True),
        (r"\bI\s+(?:dislike|hate)\s*([^,.!?]+)", "preference", "likes", False),
        (r"\b(?:can't eat|cannot eat|avoid)\s*([^,.!?]+)", "food_restriction", "can_eat", False),
        (r"\ballergic to\s*([^,.!?]+)", "allergy", "has_allergy", True),
    ]
    output = []
    for pattern, key, predicate, value in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            obj = match.group(1).strip()
            output.append({
                "topic_id": "food_preference" if "eat" in predicate or key == "preference" else "health",
                "facts": [{"key": f"{key}.{_normalize_topic_id(obj)}", "subject": "user", "predicate": predicate, "value": value}],
                "valid_from": observed_at, "trigger_event_ref": "event_1",
                "evidence_turn_ids": evidence,
            })
    return output


def _canonical_topic(text: str) -> str:
    tokens = set(re.findall(r"[a-z0-9']+", text.casefold()))
    rules = {
        "travel": (("travel", "trip", "journey", "beijing", "shanghai"), ()),
        "health": (("health", "doctor", "hospital", "pain"), ("allerg",)),
        "food_preference": (("food", "eat", "spicy", "restaurant"), ()),
        "work": (("work", "job", "company", "project"), ()),
        "relationship": (("friend", "family", "marry", "relationship"), ()),
        "education": (("school", "university", "study", "degree"), ()),
        "long_term_plan": (("plan", "goal", "will", "intend"), ()),
    }
    for topic, (exact, prefixes) in rules.items():
        if tokens.intersection(exact) or any(
            any(token.startswith(prefix) for token in tokens) for prefix in prefixes
        ):
            return topic
    return "general"


def _normalize_topic_id(value: str) -> str:
    aliases = {
        "tourism": "travel", "trip": "travel", "trips": "travel",
        "vacation": "travel", "journey": "travel",
        "health_history": "health", "medical": "health", "allergy": "health",
        "food": "food_preference",
        "diet": "food_preference", "food_preferences": "food_preference",
    }
    if value.strip().casefold() in aliases:
        return aliases[value.strip().casefold()]
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.strip().casefold()).strip("_")
    return normalized or "general"


def _infer_action(text: str) -> str:
    topic = _canonical_topic(text)
    return {"travel": "travel", "health": "health_event", "food_preference": "food_event", "work": "work_event"}.get(topic, "experienced")


def _infer_location(text: str) -> str | None:
    for value in ("Beijing", "Shanghai"):
        if value.casefold() in text.casefold():
            return value
    match = re.search(r"\b(?:in|at|to)\s+([A-Z][\w-]+)", text)
    return match.group(1) if match else None


def _entities(text: str) -> list[str]:
    latin = re.findall(r"\b[A-Z][A-Za-z0-9_-]{1,}\b", text)
    return _stable_unique(latin)[:24]


def _normalized_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if vector.ndim != 1 or vector.size == 0:
        return None
    norm = float(np.linalg.norm(vector))
    if not norm:
        return None
    return vector / norm


def _entity_char_similarity(alias: str, row: dict[str, Any]) -> float:
    left = _normalize_text(alias)
    candidates = [
        str(row.get("representative_alias") or ""),
        *[str(value) for value in row.get("aliases") or []],
    ]
    return max(
        (SequenceMatcher(None, left, _normalize_text(candidate)).ratio() for candidate in candidates if candidate),
        default=0.0,
    )


def _preferred_entity_merge_target(
    first: dict[str, Any], second: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    first_created = str(first.get("created_at") or "9999")
    second_created = str(second.get("created_at") or "9999")
    first_id = str(first.get("canonical_id") or "")
    second_id = str(second.get("canonical_id") or "")
    if (second_created, second_id) < (first_created, first_id):
        return second, first
    return first, second


def _infer_time_precision(value: str) -> str | None:
    text = value.strip()
    if re.fullmatch(r"(?:19|20)\d{2}", text):
        return "year"
    if re.fullmatch(r"(?:19|20)\d{2}-\d{1,2}", text):
        return "month"
    if re.fullmatch(r"(?:19|20)\d{2}-\d{1,2}-\d{1,2}", text):
        return "date"
    if re.fullmatch(
        r"(?:19|20)\d{2}-\d{1,2}-\d{1,2}[T ]\d{1,2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?",
        text,
    ):
        return "datetime"
    return None


_TIME_OF_DAY_RE = re.compile(
    r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*[ap]\.?m\b",
    re.IGNORECASE,
)


def _demote_spurious_datetime(
    start: str | None,
    end: str | None,
    precision: str | None,
    raw_expression: str | None,
) -> tuple[str | None, str | None, str | None, bool]:
    """Collapse a datetime precision to date when the HMS is spurious.

    A datetime event_time is spurious when raw_time_expression is a non-empty
    phrase that contains no time-of-day token (no ``HH:MM`` and no ``am/pm``).
    That almost always means the builder LLM copied the anchor's HMS into the
    event time. Returns ``(start, end, precision, demoted)``.
    """
    if precision != "datetime" or not start:
        return start, end, precision, False
    if not raw_expression:
        # No source phrase to check — leave genuine datetimes alone.
        return start, end, precision, False
    if _TIME_OF_DAY_RE.search(raw_expression):
        # The source itself states a time-of-day; keep HMS.
        return start, end, precision, False
    parsed = parse_iso_datetime(start)
    if parsed is None:
        return start, end, precision, False
    date_text = parsed.date().isoformat()
    new_end = (end or "")[:10] or date_text
    return date_text, new_end, "date", True


def _event_candidate_key(source_id: str, target_id: str, edge_type: str) -> tuple[str, str, str]:
    edge_type = str(edge_type).upper()
    if edge_type in _SYMMETRIC_EVENT_EDGE_TYPES:
        left, right = sorted((source_id, target_id))
        return (edge_type, left, right)
    return (edge_type, source_id, target_id)


def _oriented_event_endpoints(first: MemoryNode, second: MemoryNode, edge_type: str) -> tuple[str, str]:
    edge_type = str(edge_type).upper()
    if edge_type in _SYMMETRIC_EVENT_EDGE_TYPES:
        return (first.id, second.id) if first.id < second.id else (second.id, first.id)
    if edge_type == "FOLLOW_UP_OF":
        first_start = parse_iso_datetime(first.absolute_time_start)
        second_start = parse_iso_datetime(second.absolute_time_start)
        if first_start is not None and second_start is not None:
            if second_start > first_start:
                return second.id, first.id
            if first_start > second_start:
                return first.id, second.id
        first_end = parse_iso_datetime(first.absolute_time_end or first.absolute_time_start)
        second_end = parse_iso_datetime(second.absolute_time_end or second.absolute_time_start)
        if first_end is not None and second_end is not None and second_end > first_end:
            return second.id, first.id
    return first.id, second.id


def _time_gap_seconds(first: MemoryNode, second: MemoryNode) -> float:
    """Minimal temporal distance between two nodes' time intervals.

    Returns 0.0 if they overlap; otherwise the gap between the earlier node's end
    and the later node's start. Falls back to a large constant when times cannot
    be parsed so unparseable nodes sort last rather than first.
    """
    first_start = first.absolute_time_start
    second_start = second.absolute_time_start
    if not first_start or not second_start:
        return 1e12
    first_end = first.absolute_time_end or first_start
    second_end = second.absolute_time_end or second_start
    a_start = parse_iso_datetime(first_start)
    a_end = parse_iso_datetime(first_end)
    b_start = parse_iso_datetime(second_start)
    b_end = parse_iso_datetime(second_end)
    if a_start is None or b_start is None:
        return 1e12
    a_end = a_end or a_start
    b_end = b_end or b_start
    if a_end < b_start:
        return (b_start - a_end).total_seconds()
    if b_end < a_start:
        return (a_start - b_end).total_seconds()
    return 0.0


def _time_precision_aligned(first: str, second: str) -> bool:
    """True when two event times plausibly refer to the same instant.

    Parses both and requires they fall within one calendar day. Coarser
    precisions (year/month) parse to the first day of their period, so two
    "2024-04" or "2024" values align; a date and a same-day datetime also align.
    """
    a = parse_iso_datetime(first)
    b = parse_iso_datetime(second)
    if a is None or b is None:
        return False
    return abs((a - b).total_seconds()) <= 86400.0


def _entity_jaccard(first: Iterable[str], second: Iterable[str]) -> float:
    a = {str(item).casefold() for item in first if str(item).strip()}
    b = {str(item).casefold() for item in second if str(item).strip()}
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _title_token_overlap(first: str, second: str) -> float:
    """Token overlap ratio (intersection / longer token set) for titles/summaries.

    Tokens are alphanumeric, casefolded, length >= 2. Returns 0.0 when either
    side has no usable tokens. Using the longer set as denominator is lenient
    toward a short summary partially echoed in a longer one — appropriate for
    spotting the same event surfaced in two overlapping extraction windows.
    """
    a = {tok for tok in re.split(r"\W+", _normalize_text(first), flags=re.UNICODE) if len(tok) >= 2}
    b = {tok for tok in re.split(r"\W+", _normalize_text(second), flags=re.UNICODE) if len(tok) >= 2}
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


def _node_sessions(node: MemoryNode) -> set[str]:
    return {
        str(ref.get("session_id"))
        for ref in node.evidence_refs
        if ref.get("session_id") and str(ref.get("session_id")).strip()
    }


_DEFAULT_ENTITY_ACTORS = {"user", "agent", "system", "assistant", "participant"}


def _shared_significant_entity(first: Iterable[str], second: Iterable[str]) -> str | None:
    """First shared significant entity (excluding default-role pseudo-entities)."""
    b = {str(item).casefold() for item in second if str(item).strip()}
    for item in first:
        key = str(item).casefold()
        if key in b and key not in _DEFAULT_ENTITY_ACTORS and len(key) >= 2:
            return str(item)
    return None


def _merge_refs(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = {str(ref.get("evidence_id") or _normalize_text(_canonical_ref(ref))): ref for ref in first}
    for ref in second:
        output[str(ref.get("evidence_id") or _normalize_text(_canonical_ref(ref)))] = ref
    return list(output.values())


def _canonical_ref(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _stable_unique(str(item).strip() for item in value if str(item).strip())


def _stable_unique(values: Iterable[str]) -> list[str]:
    output, seen = [], set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _optional_string(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def _float01(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _evidence_id_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().casefold())


def _compact(value: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _dump_facts(values: list[dict[str, Any]]) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True)
