from __future__ import annotations

import threading
import hashlib
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from api_history import ApiHistoryLogger, LoggedChatClient
from memory.base import RetrievalResult
from memory.clients import (
    ChatClient,
    EmbeddingClient,
    NoopChatClient,
    make_chat_client,
    make_embedding_client,
    with_output_token_limit,
)
from memory.config import AppConfig, ModelConfig, load_config
from memory.v4.config import V4MemoryConfig
from memory.v4.failure import V4BuildStageError, V4OperationContext, retry_v4_call
from memory.v4.builder import V4MemoryBuilder
from memory.v4.faiss_index import NamespaceFaissIndex
from memory.v4.controller import V4MultiRoundController
from memory.v4.intent import VIRTUAL_ACTIONS
from memory.v4.retriever import V4Retriever
from memory.v4.storage import RevisionConflictError, V4SQLiteStore
from memory.retrieve.base import RetrievalExecutionContext
from memory.retrieve.types import (
    BackendCapabilities,
    Expansion,
    Probe,
    RetrievalTrajectory,
    RuntimeCandidate,
)
from memory.v4.schemas import (
    EDGE_TYPES,
    TimeConstraint,
    RetrievalPlan as V4RetrievalPlan,
    RetrievalProbe as V4RetrievalProbe,
)


LOGGER = logging.getLogger(__name__)


def _model_identity(config: ModelConfig | None) -> tuple[str | None, str | None]:
    if config is None:
        return (None, None)
    return (config.provider, config.model)


def _combined_model_identity(
    *configs: ModelConfig | None,
) -> tuple[str | None, str | None]:
    identities = [_model_identity(config) for config in configs if config is not None]
    if not identities:
        return (None, None)
    return (
        "+".join(str(provider or "unknown") for provider, _ in identities),
        "|".join(str(model or "unknown") for _, model in identities),
    )


def _named_model_config(config: AppConfig, name: str | None) -> ModelConfig | None:
    if not name:
        return None
    raw_models = dict((config.raw or {}).get("models") or {})
    value = raw_models.get(name)
    return ModelConfig.from_dict(dict(value)) if isinstance(value, dict) else None


class MemorySystem:
    backend_id = "v4"
    _scope_locks_guard = threading.Lock()
    _scope_locks: dict[str, threading.RLock] = {}
    capabilities = BackendCapabilities(
        backend_id="v4",
        backend_version="4",
        node_types=frozenset({"chain_head", "event", "semantic_state"}),
        edge_directions={edge: frozenset({"in", "out"}) for edge in EDGE_TYPES},
        max_hops=1,
        max_total_budget=32,
        plan_schema_version=6,
        virtual_actions=VIRTUAL_ACTIONS,
        time_search=True,
        graph_expansion=True,
        static_plans=False,
        absolute_time_correction=True,
        intent_aware_multi_round=True,
    )

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        store: V4SQLiteStore | None = None,
        embedding_client: EmbeddingClient | None = None,
        controller_client: ChatClient | None = None,
        semantic_slimming_client: ChatClient | None = None,
        decomposition_gate_client: ChatClient | None = None,
        rerank_client: ChatClient | None = None,
        memory_builder_client: ChatClient | None = None,
        window_planner_client: ChatClient | None = None,
        api_history_logger: ApiHistoryLogger | None = None,
        store_read_only: bool = False,
        faiss_index: NamespaceFaissIndex | None = None,
        backend_config: V4MemoryConfig | None = None,
        semantic_state_client: ChatClient | None = None,
    ):
        self.config = config or load_config()
        self.v4_config = backend_config or V4MemoryConfig.from_app_config(self.config)
        self.capabilities = replace(
            type(self).capabilities,
            max_total_budget=self.v4_config.total_candidate_budget,
        )
        self.api_history_logger = api_history_logger
        self.store = store or V4SQLiteStore(
            self.v4_config.database_path, read_only=store_read_only
        )
        self.embedding_client = embedding_client or make_embedding_client(
            self.config.embedding, resilient=False
        )
        builder_client = memory_builder_client
        if builder_client is None:
            builder_client = make_chat_client(
                with_output_token_limit(
                    self.config.memory_builder,
                    self.v4_config.prompt_output_tokens.memory_builder,
                )
            )
        semantic_model_config = (
            _named_model_config(self.config, self.v4_config.semantic_state_model)
            or self.config.memory_builder
        )
        reducer_client = semantic_state_client
        if reducer_client is None:
            reducer_client = make_chat_client(
                with_output_token_limit(
                    semantic_model_config,
                    self.v4_config.prompt_output_tokens.semantic_reducer,
                )
            )
        if api_history_logger is not None and not isinstance(
            builder_client, NoopChatClient
        ):
            builder_client = LoggedChatClient(
                builder_client,
                api_history_logger,
                "v4_memory_builder",
                self.config.memory_builder.provider,
                self.config.memory_builder.model,
            )
        if (
            api_history_logger is not None
            and reducer_client is not None
            and not isinstance(reducer_client, NoopChatClient)
        ):
            reducer_client = LoggedChatClient(
                reducer_client,
                api_history_logger,
                "v4_semantic_reducer",
                semantic_model_config.provider,
                semantic_model_config.model,
            )
        planner_client = self._build_window_planner_client(
            window_planner_client, api_history_logger
        )
        faiss_dir = self.v4_config.faiss_path
        self.faiss_index = faiss_index or NamespaceFaissIndex(
            faiss_dir, read_only=store_read_only
        )
        adjudication_client = self._build_adjudication_client(api_history_logger)
        entity_judge_client = self._build_entity_judge_client(api_history_logger)
        self.builder = V4MemoryBuilder(
            self.store,
            self.embedding_client,
            builder_client,
            self.config.memory.memory_extraction_batch_turns,
            self.config.memory.memory_extraction_workers,
            self.config.memory.llm_max_retries,
            self.config.memory.llm_retry_backoff_seconds,
            config=self.v4_config,
            adjudication_client=adjudication_client,
            entity_judge_client=entity_judge_client,
            window_planner_client=planner_client,
            semantic_reducer_client=reducer_client,
            model_identities={
                "window_plan": _model_identity(
                    self.config.window_planner or self.config.slm
                ),
                "window_extraction": _model_identity(self.config.memory_builder),
                "semantic_update": (
                    semantic_model_config.provider,
                    semantic_model_config.model,
                ),
                "entity_resolution": _combined_model_identity(
                    self.config.embedding, self.config.entity_judge
                ),
                "embedding": _model_identity(self.config.embedding),
                "entity_judge": _model_identity(self.config.entity_judge),
                "adjudication": _model_identity(self.config.adjudication_model),
            },
        )
        self.retriever = V4Retriever(
            self.store,
            self.faiss_index,
            self.embedding_client,
            self.v4_config,
            embedding_identity=_model_identity(self.config.embedding),
        )
        # V4 has one strong Controller model; Router adapters are intentionally
        # unsupported until V4 trajectories are regenerated in a future cycle.
        self._controller_client = controller_client
        self._semantic_slimming_client = semantic_slimming_client
        self._decomposition_gate_client = decomposition_gate_client
        self._rerank_client = (
            rerank_client if rerank_client is not None else controller_client
        )
        self._intent_embedding_cache: dict[str, list[float]] = {}

    @classmethod
    def from_config_file(
        cls, config_path: str | Path = "configs/default.yaml"
    ) -> "MemorySystem":
        return cls(load_config(config_path))

    def _build_adjudication_client(
        self, api_history_logger: ApiHistoryLogger | None
    ) -> ChatClient | None:
        """Construct the explicitly configured adjudication client."""
        model_config = self.config.adjudication_model
        if model_config is None:
            return None
        client = make_chat_client(
            with_output_token_limit(
                model_config,
                self.v4_config.prompt_output_tokens.cross_window_adjudication,
            )
        )
        if api_history_logger is not None:
            return LoggedChatClient(
                client,
                api_history_logger,
                "v4_adjudication",
                model_config.provider,
                model_config.model,
            )
        return client

    def _build_entity_judge_client(
        self, api_history_logger: ApiHistoryLogger | None
    ) -> ChatClient | None:
        model_config = self.config.entity_judge
        if model_config is None or model_config.provider.lower() == "noop":
            return None
        client = make_chat_client(
            with_output_token_limit(
                model_config,
                self.v4_config.prompt_output_tokens.entity_judge,
            )
        )
        if api_history_logger is not None:
            return LoggedChatClient(
                client,
                api_history_logger,
                "v4_entity_judge",
                model_config.provider,
                model_config.model,
            )
        return client

    def _build_window_planner_client(
        self,
        injected: ChatClient | None,
        api_history_logger: ApiHistoryLogger | None,
    ) -> ChatClient | None:
        if str(self.v4_config.window_planner.mode).lower() == "rule":
            return injected
        model_config = self.config.window_planner or self.config.slm
        client = injected
        if client is None and model_config.provider.lower() == "noop":
            return None
        if client is None:
            client = make_chat_client(
                with_output_token_limit(
                    model_config,
                    self.v4_config.prompt_output_tokens.window_boundary,
                )
            )
        if api_history_logger is not None and not isinstance(client, NoopChatClient):
            return LoggedChatClient(
                client,
                api_history_logger,
                "v4_window_planner",
                model_config.provider,
                model_config.model,
            )
        return client

    def ingest_conversation(
        self,
        namespace: str,
        conversation: Any,
        metadata: dict[str, Any] | None = None,
        *,
        participants: list[dict[str, str]] | None = None,
    ) -> None:
        metadata = dict(metadata or {})
        # Compute scope identity without a write so concurrent ingests serialize
        # only for publication; extraction can proceed outside this lock.
        turns = self.builder._normalize_turns(conversation)
        inferred = (
            participants
            if participants is not None
            else self.builder._derive_participants(turns)
        )
        from memory.v4.schemas import participant_scope_id

        scope_id = participant_scope_id(namespace, inferred)
        self._resume_cross_scope_completion(namespace, scope_id)
        windows = self.builder.plan_windows(
            namespace, turns, metadata, participants=participants
        )
        extracted = self.builder.extract_windows(
            windows,
            metadata,
            workers=self.v4_config.build_parallelism.window_api_workers_per_namespace,
        )
        lock = self._scope_lock(scope_id)
        with lock:
            for attempt in range(3):
                artifact = self.builder.build_from_extracted(
                    namespace, turns, extracted, metadata, participants=participants
                )
                if not artifact.changed and self._scope_is_ready(scope_id):
                    return
                try:
                    self.publish_artifact(artifact)
                except RevisionConflictError:
                    if attempt == 2:
                        raise
                    continue
                return

    def ingest(
        self, namespace: str, conversation: Any, metadata=None, *, participants=None
    ) -> None:
        self.ingest_conversation(
            namespace, conversation, metadata, participants=participants
        )

    def publish_artifact(self, artifact) -> None:
        generation = uuid4().hex
        staged = self.faiss_index.stage(
            artifact.scope.id,
            generation,
            [node.id for node in artifact.nodes],
            [node.embedding or [] for node in artifact.nodes],
            [
                {
                    "node_type": node.node_type,
                    "scope_id": node.scope_id,
                    "chain_id": node.chain_id,
                }
                for node in artifact.nodes
            ],
        )
        try:
            self.store.publish_scope(
                artifact.scope,
                artifact.chains,
                artifact.nodes,
                artifact.edges,
                artifact.facts,
                generation=staged.generation,
                node_count=staged.node_count,
                expected_revision=artifact.expected_revision,
            )
        except Exception:
            self.faiss_index.discard(artifact.scope.id, generation)
            raise
        self.store.record_scope_stage(
            artifact.scope.id,
            generation,
            "scope_publish",
            "succeeded",
        )
        publish_checkpoint = self.builder._checkpoint_descriptor(
            artifact.scope.namespace,
            artifact.scope.id,
            "scope_publish",
            artifact.scope.id,
            {
                "generation": generation,
                "node_count": staged.node_count,
            },
            prompt_version="scope-publish-v1",
        )
        self.builder._record_checkpoint_success(
            publish_checkpoint,
            {
                "generation": generation,
                "node_count": staged.node_count,
            },
        )
        try:
            self.faiss_index.cleanup(artifact.scope.id, generation)
        except Exception as error:
            LOGGER.warning(
                "V4 FAISS cleanup deferred for scope %s generation %s: %s",
                artifact.scope.id,
                generation,
                error,
            )
        if self.v4_config.cross_scope_navigation:
            self._complete_cross_scope(
                artifact.scope.namespace,
                artifact.scope.id,
                generation,
            )

    def _complete_cross_scope(
        self,
        namespace: str,
        scope_id: str,
        generation: str,
    ) -> None:
        checkpoint = self.builder._checkpoint_descriptor(
            namespace,
            scope_id,
            "cross_scope_completion",
            generation,
            {"generation": generation},
            prompt_version="cross-scope-v1",
            model_identity=self.builder.model_identities.get("adjudication"),
        )
        context = V4OperationContext(
            stage="cross_scope_completion",
            unit_id=scope_id,
            checkpoint_key=f"{scope_id}:cross_scope_completion:{generation}",
        )
        try:
            retry_v4_call(
                lambda: self.builder.build_cross_scope_edges(namespace, scope_id),
                policy=self.v4_config.failure_policy,
                context=context,
                error_type=V4BuildStageError,
            )
        except Exception as error:
            self.builder._record_checkpoint_failure(checkpoint, error)
            self.store.record_scope_stage(
                scope_id,
                generation,
                "cross_scope_completion",
                "failed",
                error={"type": type(error).__name__, "message": str(error)},
            )
            raise
        self.builder._record_checkpoint_success(
            checkpoint, {"generation": generation, "completed": True}
        )
        self.store.record_scope_stage(
            scope_id,
            generation,
            "cross_scope_completion",
            "succeeded",
        )

    def _resume_cross_scope_completion(
        self, namespace: str, scope_id: str
    ) -> bool:
        if not self.v4_config.cross_scope_navigation:
            return False
        state = self.store.get_scope_index_state(scope_id)
        if state is None:
            return False
        generation = str(state["generation"])
        stage = self.store.get_scope_stage(
            scope_id, generation, "cross_scope_completion"
        )
        if (
            stage is None
            or stage.get("status") != "failed"
        ):
            return False
        self._complete_cross_scope(namespace, scope_id, generation)
        return True


    def _scope_is_ready(self, scope_id: str) -> bool:
        scope = self.store.get_scope(scope_id)
        state = self.store.get_scope_index_state(scope_id)
        if scope is None or state is None or state["revision"] != scope.revision:
            return False
        if len(self.store.list_nodes(scope_id)) != state["node_count"]:
            return False
        if (
            self.v4_config.cross_scope_navigation
            and not self.store.is_scope_stage_complete(
                scope_id, str(state["generation"]), "cross_scope_completion"
            )
        ):
            return False
        try:
            self.faiss_index.validate(
                scope_id,
                generation=state["generation"],
                node_count=state["node_count"],
            )
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    def global_search(
        self, execution: RetrievalExecutionContext, probes: Sequence[Probe]
    ) -> list[RuntimeCandidate]:
        """Expose V4 native global recall to the backend-neutral runtimes."""
        scope = self.store.resolve_scope(
            execution.namespace, list(execution.participants) or None
        )
        state = self.retriever._ready_state(scope.id)
        probes_by_id = {probe.id: probe for probe in probes}
        native_probes = [
            V4RetrievalProbe(
                id=probe.id,
                query=probe.query,
                node_types=list(probe.node_types),
                topic_hints=list(probe.attributes.get("topic_hints") or ()),
                entity_hints=list(probe.entities),
                must_terms=list(probe.must_terms),
                should_terms=list(probe.should_terms),
                facets=list(probe.facets),
                time=TimeConstraint.from_dict(probe.time_constraint),
                budget=probe.budget,
            )
            for probe in probes
        ]
        plan = V4RetrievalPlan(
            probes=native_probes, final_top_k=sum(p.budget for p in probes)
        )
        rows = self.retriever._global_search(plan, scope.id, state)
        ranked = self.retriever._rank(rows, plan.final_top_k, include_heads=True)
        head_evidence_getter = getattr(self.store, "evidence_records_by_node_ids", None)
        head_evidence = (
            head_evidence_getter(
                item.node.id for item in ranked
            )
            if callable(head_evidence_getter)
            else {}
        )
        return [
            RuntimeCandidate(
                key=item.node.id,
                node_type=item.node.node_type,
                score=item.score,
                attributes={
                    "topic": item.node.topic_id,
                    "summary": item.node.summary or item.node.text,
                    "salient_facts": _complete_salient_facts(item.node),
                    "_semantic_evidence": (
                        head_evidence.get(item.node.id, list(item.node.evidence_refs))
                        if item.node.node_type in {"chain_head", "semantic_state"} else []
                    ),
                    "event_time_start": item.node.event_time_start,
                    "event_time_end": item.node.event_time_end,
                    "valid_from": item.node.valid_from,
                    "valid_to": item.node.valid_to,
                    "location": item.node.location,
                    **dict(item.node.metadata),
                },
                payload=item,
                sources=set(item.probe_ids),
                observation=self._observation(
                    item, execution.namespace, scope.id,
                    source_records=head_evidence.get(item.node.id) or item.node.evidence_refs,
                ),
                action_scores=dict(item.probe_scores),
                subproblem_ids={
                    probes_by_id[probe_id].subproblem_id
                    for probe_id in item.probe_ids
                    if (
                        probe_id in probes_by_id
                        and probes_by_id[probe_id].subproblem_id
                    )
                },
            )
            for item in ranked
        ]

    def expand(
        self,
        execution: RetrievalExecutionContext,
        anchors: Sequence[RuntimeCandidate],
        expansion: Expansion,
    ) -> list[RuntimeCandidate]:
        """Expand only supplied runtime anchors through declared V4 native edges."""
        scope = self.store.resolve_scope(
            execution.namespace, list(execution.participants) or None
        )
        state = self.retriever._ready_state(scope.id)
        output: list[RuntimeCandidate] = []
        expansion_intent = expansion.evidence_facet.strip()
        lookup_constraints = dict(expansion.selector.attributes)
        ranking_intent = str(
            lookup_constraints.get("query") or expansion_intent
        ).strip()
        intent_embedding: list[float] | None = None
        if ranking_intent and expansion.edge_type != "SOURCE_EVIDENCE_LOOKUP":
            round_key = expansion.id.split("_", 1)[0]
            cache_key = hashlib.sha256(
                f"{round_key}:{expansion.edge_type}:{ranking_intent}".encode("utf-8")
            ).hexdigest()
            intent_embedding = self._intent_embedding_cache.get(cache_key)
            if intent_embedding is None:
                intent_embedding = self.retriever._embed_texts(
                    [ranking_intent],
                    stage="retrieval_expansion_embedding",
                    unit_id=round_key,
                )[0]
                self._intent_embedding_cache[cache_key] = intent_embedding
        for anchor in anchors:
            if expansion.edge_type in self.capabilities.virtual_actions:
                additions = self.retriever._virtual_lookup(
                    execution.namespace,
                    scope.id,
                    anchor.payload,
                    expansion.edge_type,
                    expansion.budget,
                    constraints=lookup_constraints,
                    state=state,
                    intent_embedding=intent_embedding,
                )
            else:
                additions, _ = self.retriever._expand_anchor(
                    execution.namespace,
                    scope.id,
                    anchor.payload,
                    [expansion.edge_type],
                    expansion.direction,
                    expansion.budget,
                    "",
                    state=state,
                    expansion_intent=expansion_intent,
                    intent_embedding=intent_embedding,
                )
            head_evidence_getter = getattr(self.store, "evidence_records_by_node_ids", None)
            head_evidence = (
                head_evidence_getter(
                    item.node.id for item in additions
                )
                if callable(head_evidence_getter)
                else {}
            )
            output.extend(
                RuntimeCandidate(
                    key=item.node.id,
                    node_type=item.node.node_type,
                    score=item.score,
                    attributes={
                        "topic": item.node.topic_id,
                        "summary": item.node.summary or item.node.text,
                        "salient_facts": _complete_salient_facts(item.node),
                        "_semantic_evidence": (
                            head_evidence.get(item.node.id, list(item.node.evidence_refs))
                            if item.node.node_type in {"chain_head", "semantic_state"} else []
                        ),
                        "event_time_start": item.node.event_time_start,
                        "event_time_end": item.node.event_time_end,
                        "valid_from": item.node.valid_from,
                        "valid_to": item.node.valid_to,
                        "location": item.node.location,
                        **dict(item.node.metadata),
                    },
                    payload=item,
                    observation=self._observation(
                        item, execution.namespace, scope.id,
                        source_records=head_evidence.get(item.node.id) or item.node.evidence_refs,
                    ),
                )
                for item in additions
            )
        return output[: expansion.budget]

    def finalize(
        self,
        execution: RetrievalExecutionContext,
        candidates: Sequence[RuntimeCandidate],
        question: str,
        final_top_k: int,
        trace: RetrievalTrajectory,
    ) -> RetrievalResult:
        scope = self.store.resolve_scope(
            execution.namespace, list(execution.participants) or None
        )
        reranked = bool(candidates) and all(
            "rerank_score" in candidate.attributes for candidate in candidates
        )
        native_candidates = []
        for candidate in candidates:
            item = candidate.payload
            if reranked or candidate.rank_contributions:
                item = replace(item, score=candidate.score, rank_fused=True)
            native_candidates.append(item)
        ranked = (
            native_candidates[:final_top_k]
            if reranked
            else self.retriever._rank(native_candidates, final_top_k)
        )
        memories = self.retriever._memory_items(execution.namespace, scope.id, ranked)
        from memory.results import RetrievalRoundTrace

        return RetrievalResult(
            question,
            execution.namespace,
            memories,
            [
                RetrievalRoundTrace(
                    round_index=len(trace.rounds) or 1,
                    budget=final_top_k,
                    action="retrieval",
                    query_plan=trace.to_dict(),
                    returned_node_ids=[item.node_id for item in memories],
                    sufficient=bool(memories),
                    reason="backend-neutral retrieval runtime",
                )
            ],
            trajectory=trace,
        )

    def _observation(
        self, candidate: Any, namespace: str, scope_id: str,
        *, source_records: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        node = candidate.node
        available_edges = [
            {
                "edge_type": option["edge_type"],
                "direction": option["direction"],
            }
            for option in self.retriever._available_navigation_edges(
                namespace, scope_id, candidate, {}
            )
        ]
        return {
            "node_type": node.node_type,
            # Preserve source text independently of later semantic slimming.
            # No IDs, added speaker/time labels, or length limits in this field.
            "raw_turns": [
                record["content"] for record in source_records
                if isinstance(record.get("content"), str) and record["content"]
            ],
            "has_source_evidence": bool(node.evidence_refs),
            "summary": node.summary or node.text,
            "salient_facts": _complete_salient_facts(node),
            "event_time_start": node.event_time_start,
            "event_time_end": node.event_time_end,
            "valid_from": node.valid_from,
            "valid_to": node.valid_to,
            "location": node.location,
            "confidence": candidate.score,
            "available_edges": available_edges,
        }

    def get_controller_client(self, profile: str = "teacher") -> ChatClient:
        if profile != "teacher":
            raise ValueError("V4 supports only the strong teacher Controller profile")
        if self._controller_client is not None:
            return self._controller_client
        model_config = self._controller_model_config()
        client = make_chat_client(
            with_output_token_limit(
                model_config,
                self.v4_config.prompt_output_tokens.multi_round_controller,
            )
        )
        if self.api_history_logger is not None:
            client = LoggedChatClient(
                client,
                self.api_history_logger,
                "v4_controller",
                model_config.provider,
                model_config.model,
            )
        self._controller_client = client
        return client

    def get_rerank_client(self, profile: str = "teacher") -> ChatClient:
        if profile != "teacher":
            raise ValueError("V4 supports only the strong teacher Controller profile")
        if self._rerank_client is not None:
            return self._rerank_client
        model_config = self._controller_model_config()
        client = make_chat_client(
            with_output_token_limit(
                model_config,
                self.v4_config.prompt_output_tokens.rerank,
            )
        )
        if self.api_history_logger is not None:
            client = LoggedChatClient(
                client,
                self.api_history_logger,
                "v4_rerank",
                model_config.provider,
                model_config.model,
            )
        self._rerank_client = client
        return client

    def get_semantic_slimming_client(self, profile: str = "teacher") -> ChatClient:
        if profile != "teacher":
            raise ValueError("V4 supports only the strong teacher Controller profile")
        existing = getattr(self, "_semantic_slimming_client", None)
        if existing is not None:
            return existing
        model_config = self._semantic_slimming_model_config()
        prompt_tokens = getattr(
            getattr(self, "v4_config", None),
            "prompt_output_tokens",
            None,
        )
        output_tokens = getattr(prompt_tokens, "semantic_slimming", 4096)
        client = make_chat_client(
            with_output_token_limit(
                model_config,
                output_tokens,
            )
        )
        api_history_logger = getattr(self, "api_history_logger", None)
        if api_history_logger is not None:
            client = LoggedChatClient(
                client,
                api_history_logger,
                "v4_semantic_slimming",
                model_config.provider,
                model_config.model,
            )
        self._semantic_slimming_client = client
        return client

    def get_decomposition_gate_client(self, profile: str = "teacher") -> ChatClient:
        if profile != "teacher":
            raise ValueError("V4 supports only the strong teacher Controller profile")
        if self._decomposition_gate_client is not None:
            return self._decomposition_gate_client
        model_config = self._decomposition_gate_model_config()
        client = make_chat_client(
            with_output_token_limit(
                model_config,
                self.v4_config.prompt_output_tokens.decomposition_gate,
            )
        )
        if self.api_history_logger is not None:
            client = LoggedChatClient(
                client,
                self.api_history_logger,
                "v4_decomposition_gate",
                model_config.provider,
                model_config.model,
            )
        self._decomposition_gate_client = client
        return client

    def get_multi_round_controller(
        self, profile: str = "teacher"
    ) -> V4MultiRoundController | None:
        if profile != "teacher":
            raise ValueError("V4 supports only the strong teacher Controller profile")
        if self._controller_model_config().provider.lower() == "noop":
            return None
        backend_config = getattr(self, "v4_config", None)
        return V4MultiRoundController(
            self.get_controller_client(profile),
            rerank_client=self.get_rerank_client(profile),
            semantic_slimming_client=self.get_semantic_slimming_client(profile),
            decomposition_gate_client=self.get_decomposition_gate_client(profile),
            max_subproblems=getattr(backend_config, "max_subproblems", 4),
            max_actions_per_round=getattr(backend_config, "max_actions_per_round", 3),
        )

    def _controller_model_config(self) -> ModelConfig:
        return (
            self.config.controller
            or _named_model_config(self.config, "controller")
            or self.config.slm
        )

    def _semantic_slimming_model_config(self) -> ModelConfig:
        backend_config = getattr(self, "v4_config", None)
        model_name = getattr(backend_config, "semantic_slimming_model", "controller")
        return (
            _named_model_config(self.config, model_name)
            or self._controller_model_config()
        )

    def _decomposition_gate_model_config(self) -> ModelConfig:
        return (
            getattr(self.config, "decomposition_gate", None)
            or _named_model_config(self.config, "decomposition_gate")
            or self._controller_model_config()
        )

    def is_namespace_ready(
        self, namespace: str, *, participants: list[dict[str, str]] | None = None
    ) -> bool:
        return not self.namespace_readiness_issues(namespace, participants=participants)

    def namespace_readiness_issues(
        self, namespace: str, *, participants: list[dict[str, str]] | None = None
    ) -> list[str]:
        """Explain every condition that prevents a V4 namespace from being ready."""
        try:
            scopes = (
                [self.store.resolve_scope(namespace, participants)]
                if participants is not None
                else self.store.list_scopes(namespace)
            )
        except (KeyError, ValueError) as exc:
            return [f"scope_resolution: {type(exc).__name__}: {exc}"]
        if not scopes:
            return ["no participant scopes"]
        issues: list[str] = []
        for scope in scopes:
            state = self.store.get_scope_index_state(scope.id)
            if state is None:
                issues.append(f"{scope.id}: missing scope index state")
                continue
            actual_node_count = len(self.store.list_nodes(scope.id))
            if actual_node_count != state["node_count"]:
                issues.append(
                    f"{scope.id}: node_count mismatch: "
                    f"database={actual_node_count}, index_state={state['node_count']}"
                )
            if (
                self.v4_config.cross_scope_navigation
                and not self.store.is_scope_stage_complete(
                    scope.id, str(state["generation"]), "cross_scope_completion"
                )
            ):
                issues.append(
                    f"{scope.id}: missing cross_scope_completion for "
                    f"generation={state['generation']}"
                )
            try:
                self.faiss_index.validate(
                    scope.id,
                    generation=state["generation"],
                    node_count=state["node_count"],
                )
            except (OSError, RuntimeError, ValueError) as exc:
                issues.append(
                    f"{scope.id}: faiss_validation: {type(exc).__name__}: {exc}"
                )
        return issues

    def list_memory_nodes(
        self,
        namespace: str,
        *,
        participants: list[dict[str, str]] | None = None,
    ) -> list[Any]:
        scope = self.store.resolve_scope(namespace, participants)
        return list(self.store.list_nodes(scope.id))

    @property
    def window_api_workers_per_namespace(self) -> int:
        return self.v4_config.build_parallelism.window_api_workers_per_namespace

    def is_ready(
        self, namespace: str, *, participants: list[dict[str, str]] | None = None
    ) -> bool:
        return self.is_namespace_ready(namespace, participants=participants)


    def close(self) -> None:
        self.store.close()

    @classmethod
    def _scope_lock(cls, scope_id: str) -> threading.RLock:
        with cls._scope_locks_guard:
            return cls._scope_locks.setdefault(scope_id, threading.RLock())


def _complete_salient_facts(node: Any) -> list[Any]:
    """Return every structured fact; exact duplicates are removed by Controller."""
    return [
        *(node.metadata.get("salient_facts") or ()),
        *(node.facts or ()),
    ]
