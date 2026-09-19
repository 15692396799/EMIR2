from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from memory.config import BuildParallelismConfig, WindowPlannerConfig


def _default_v4_window_planner() -> WindowPlannerConfig:
    return WindowPlannerConfig.from_dict(
        {
            "mode": "model",
            "token_encoding": "cl100k_base",
            "min_builder_input_tokens": 2048,
            "target_builder_input_tokens": 8192,
            "max_builder_input_tokens": 30000,
            "max_window_turns": 0,
            "max_window_atoms": 64,
        }
    )


@dataclass(frozen=True)
class FailurePolicyConfig:
    max_attempts: int = 3
    retry_initial_delay_seconds: float = 5.0
    retry_backoff_multiplier: float = 2.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "FailurePolicyConfig":
        value = dict(data or {})
        max_attempts = int(value.get("max_attempts", 3))
        initial_delay = float(value.get("retry_initial_delay_seconds", 5.0))
        multiplier = float(value.get("retry_backoff_multiplier", 2.0))
        if max_attempts < 1:
            raise ValueError("failure_policy.max_attempts must be positive")
        if initial_delay < 0:
            raise ValueError(
                "failure_policy.retry_initial_delay_seconds must be non-negative"
            )
        if multiplier < 1:
            raise ValueError(
                "failure_policy.retry_backoff_multiplier must be at least 1"
            )
        return cls(max_attempts, initial_delay, multiplier)


@dataclass(frozen=True)
class PromptOutputTokenConfig:
    decomposition_gate: int = 32
    window_boundary: int = 256
    multi_round_controller: int = 2048
    semantic_slimming: int = 4096
    rerank: int = 4096
    entity_judge: int = 1024
    semantic_reducer: int = 2048
    cross_window_adjudication: int = 2048
    memory_builder: int = 8192

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PromptOutputTokenConfig":
        raw = dict(data or {})
        values = {
            field_name: int(raw.get(field_name, default))
            for field_name, default in (
                ("decomposition_gate", 32),
                ("window_boundary", 256),
                ("multi_round_controller", 2048),
                ("semantic_slimming", 4096),
                ("rerank", 4096),
                ("entity_judge", 1024),
                ("semantic_reducer", 2048),
                ("cross_window_adjudication", 2048),
                ("memory_builder", 8192),
            )
        }
        invalid = sorted(name for name, value in values.items() if value < 1)
        if invalid:
            raise ValueError(
                "memory.backends.v4.prompt_output_tokens values must be positive: "
                + ", ".join(invalid)
            )
        return cls(**values)


@dataclass
class V4MemoryConfig:
    """Configuration owned by V4 and parsed only from ``memory.backends.v4``."""

    database_path: str = "data/memory_v4.sqlite3"
    faiss_path: str = "data/faiss_v4"
    memory_builder_model: str | None = None
    semantic_state_model: str | None = None
    semantic_slimming_model: str = "controller"
    embedding_model: str | None = None
    failure_policy: FailurePolicyConfig = field(default_factory=FailurePolicyConfig)
    prompt_output_tokens: PromptOutputTokenConfig = field(
        default_factory=PromptOutputTokenConfig
    )
    simple_probe_budget: int = 8
    complex_probe_budget: int = 12
    followup_probe_budget: int = 6
    expansion_budget_per_anchor: int = 3
    final_top_k: int = 16
    semantic_dedup_threshold: float = 0.95
    retrieve_semantic_nodes: bool = True
    semantic_slimming_enabled: bool = True
    max_rounds: int = 8
    zero_growth_stop_rounds: int = 2
    graph_hops_per_round: int = 1
    total_candidate_budget: int = 32
    bootstrap_top_k: int = 5
    rerank_batch_size: int = 16
    rerank_llm_weight: float = 0.7
    rerank_retrieval_weight: float = 0.3
    model_evidence_filter_enabled: bool = False
    max_subproblems: int = 4
    max_actions_per_round: int = 3
    max_actions_per_source: int = 3
    expansion_intent_weight: float = 0.75
    expansion_edge_quality_weight: float = 0.15
    expansion_node_confidence_weight: float = 0.10
    derived_edge_min_confidence: float = 0.50
    rank_fusion_constant: int = 60
    topic_candidate_top_k: int = 8
    topic_match_min_similarity: float = 0.55
    cross_window_candidate_max: int = 100
    cross_window_same_event_entity_jaccard: float = 0.6
    cross_window_same_event_entity_jaccard_no_time: float = 0.75
    cross_window_same_event_title_token_overlap: float = 0.6
    cross_window_followup_gap_seconds: int = 604800
    cross_scope_navigation: bool = True
    cross_scope_max_neighbors: int = 8
    cross_scope_candidate_max: int = 100
    entity_canonicalization_enabled: bool = True
    entity_embed_top_k: int = 10
    entity_cosine_high_threshold: float = 0.85
    entity_cosine_low_threshold: float = 0.75
    entity_char_similarity_threshold: float = 0.85
    entity_registry_max_candidates: int = 50
    entity_judge_batch_size: int = 8
    entity_judge_workers: int = 2
    entity_auto_merge_policy: str = "conservative"
    rebuild_full_scope_reorder_ratio: float = 0.2
    conversation_window_turns: int = 24
    conversation_window_overlap: int = 6
    window_planner: WindowPlannerConfig = field(
        default_factory=_default_v4_window_planner
    )
    build_parallelism: BuildParallelismConfig = field(
        default_factory=BuildParallelismConfig
    )
    faiss_index: str = "flat_ip"

    @property
    def faiss_dir(self) -> str:
        return self.faiss_path

    @property
    def resolved_semantic_state_model(self) -> str | None:
        return self.semantic_state_model or self.memory_builder_model

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "V4MemoryConfig":
        if data is None:
            raise ValueError("memory.backends.v4 is required for the v4 backend")
        raw = dict(data)
        removed = sorted(
            {
                "semantic_neighbor_top_k",
                "semantic_neighbor_min_similarity",
                "shared_entity_max_edges",
                "temporal_overlap_max_edges",
                "max_initial_probes",
                "max_expansions_per_round",
                "max_global_probes_per_followup_round",
            }.intersection(raw)
        )
        if removed:
            raise ValueError(
                "V4 does not support removed retrieval/weak-edge settings: "
                + ", ".join(removed)
            )
        faiss_index = str(raw.get("faiss_index", "flat_ip")).lower()
        if faiss_index != "flat_ip":
            raise ValueError("V4 currently supports only faiss_index=flat_ip")
        turns = _validated_int(raw, "conversation_window_turns", 24, 2, 128)
        overlap = _validated_int(raw, "conversation_window_overlap", 6, 0, 16)
        planner = dict(raw.get("window_planner") or {})
        planner.setdefault("token_encoding", "cl100k_base")
        planner.setdefault("min_builder_input_tokens", 2048)
        planner.setdefault("target_builder_input_tokens", 8192)
        planner.setdefault("max_builder_input_tokens", 30000)
        planner.setdefault("max_window_turns", 0)
        planner.setdefault("max_window_atoms", 64)
        planner.setdefault("mode", "model")
        planner_mode = str(planner.get("mode") or "model").lower()
        if planner_mode not in {"model", "rule"}:
            raise ValueError(
                "memory.backends.v4.window_planner.mode must be model or rule; "
                "hybrid is not supported by V4"
            )
        planner["mode"] = planner_mode
        entity_auto_merge_policy = (
            str(raw.get("entity_auto_merge_policy") or "conservative").strip().lower()
        )
        if entity_auto_merge_policy not in {"conservative", "legacy"}:
            raise ValueError(
                "memory.backends.v4.entity_auto_merge_policy must be "
                "conservative or legacy"
            )
        rerank_llm_weight = _ratio(raw, "rerank_llm_weight", 0.7)
        rerank_retrieval_weight = _ratio(raw, "rerank_retrieval_weight", 0.3)
        if not math.isclose(
            rerank_llm_weight + rerank_retrieval_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "rerank_llm_weight and rerank_retrieval_weight must sum to 1"
            )
        config = cls(
            database_path=str(raw.get("database_path") or cls.database_path),
            faiss_path=str(raw.get("faiss_path") or cls.faiss_path),
            memory_builder_model=_optional(raw.get("memory_builder_model")),
            semantic_state_model=_optional(raw.get("semantic_state_model")),
            semantic_slimming_model=(
                _optional(raw.get("semantic_slimming_model")) or "controller"
            ),
            embedding_model=_optional(raw.get("embedding_model")),
            failure_policy=FailurePolicyConfig.from_dict(raw.get("failure_policy")),
            prompt_output_tokens=PromptOutputTokenConfig.from_dict(
                raw.get("prompt_output_tokens")
            ),
            simple_probe_budget=_validated_int(raw, "simple_probe_budget", 8, 1, 128),
            complex_probe_budget=_validated_int(
                raw, "complex_probe_budget", 12, 1, 128
            ),
            followup_probe_budget=_validated_int(
                raw, "followup_probe_budget", 6, 1, 128
            ),
            expansion_budget_per_anchor=_validated_int(
                raw, "expansion_budget_per_anchor", 3, 1, 64
            ),
            final_top_k=_validated_int(raw, "final_top_k", 16, 1, 512),
            semantic_dedup_threshold=_ratio(raw, "semantic_dedup_threshold", 0.95),
            retrieve_semantic_nodes=bool(raw.get("retrieve_semantic_nodes", True)),
            semantic_slimming_enabled=bool(raw.get("semantic_slimming_enabled", True)),
            max_rounds=_validated_int(raw, "max_rounds", 8, 1, 64),
            zero_growth_stop_rounds=_validated_int(
                raw, "zero_growth_stop_rounds", 2, 0, 16
            ),
            graph_hops_per_round=1,
            total_candidate_budget=_validated_int(
                raw, "total_candidate_budget", 32, 1, 512
            ),
            bootstrap_top_k=_validated_int(raw, "bootstrap_top_k", 5, 1, 512),
            rerank_batch_size=_validated_int(raw, "rerank_batch_size", 16, 1, 64),
            rerank_llm_weight=rerank_llm_weight,
            rerank_retrieval_weight=rerank_retrieval_weight,
            model_evidence_filter_enabled=bool(
                raw.get("model_evidence_filter_enabled", False)
            ),
            max_subproblems=_validated_int(raw, "max_subproblems", 4, 2, 32),
            max_actions_per_round=_validated_int(
                raw, "max_actions_per_round", 3, 1, 64
            ),
            max_actions_per_source=_validated_int(
                raw, "max_actions_per_source", 3, 1, 64
            ),
            expansion_intent_weight=_ratio(raw, "expansion_intent_weight", 0.75),
            expansion_edge_quality_weight=_ratio(
                raw, "expansion_edge_quality_weight", 0.15
            ),
            expansion_node_confidence_weight=_ratio(
                raw, "expansion_node_confidence_weight", 0.10
            ),
            derived_edge_min_confidence=_ratio(
                raw, "derived_edge_min_confidence", 0.50
            ),
            rank_fusion_constant=_validated_int(
                raw, "rank_fusion_constant", 60, 1, 10000
            ),
            topic_candidate_top_k=_validated_int(
                raw, "topic_candidate_top_k", 8, 1, 64
            ),
            topic_match_min_similarity=_ratio(raw, "topic_match_min_similarity", 0.55),
            cross_window_candidate_max=_validated_int(
                raw, "cross_window_candidate_max", 100, 0, 1000
            ),
            cross_window_same_event_entity_jaccard=_ratio(
                raw, "cross_window_same_event_entity_jaccard", 0.6
            ),
            cross_window_same_event_entity_jaccard_no_time=_ratio(
                raw, "cross_window_same_event_entity_jaccard_no_time", 0.75
            ),
            cross_window_same_event_title_token_overlap=_ratio(
                raw, "cross_window_same_event_title_token_overlap", 0.6
            ),
            cross_window_followup_gap_seconds=_validated_int(
                raw, "cross_window_followup_gap_seconds", 604800, 0, 315360000
            ),
            cross_scope_navigation=bool(raw.get("cross_scope_navigation", True)),
            cross_scope_max_neighbors=_validated_int(
                raw, "cross_scope_max_neighbors", 8, 0, 64
            ),
            cross_scope_candidate_max=_validated_int(
                raw, "cross_scope_candidate_max", 100, 0, 1000
            ),
            entity_canonicalization_enabled=bool(
                raw.get("entity_canonicalization_enabled", True)
            ),
            entity_embed_top_k=_validated_int(raw, "entity_embed_top_k", 10, 1, 100),
            entity_cosine_high_threshold=_ratio(
                raw, "entity_cosine_high_threshold", 0.85
            ),
            entity_cosine_low_threshold=_ratio(
                raw, "entity_cosine_low_threshold", 0.75
            ),
            entity_char_similarity_threshold=_ratio(
                raw, "entity_char_similarity_threshold", 0.85
            ),
            entity_registry_max_candidates=_validated_int(
                raw, "entity_registry_max_candidates", 50, 1, 1000
            ),
            entity_judge_batch_size=_validated_int(
                raw, "entity_judge_batch_size", 8, 1, 32
            ),
            entity_judge_workers=_validated_int(raw, "entity_judge_workers", 2, 1, 8),
            entity_auto_merge_policy=entity_auto_merge_policy,
            rebuild_full_scope_reorder_ratio=_ratio(
                raw, "rebuild_full_scope_reorder_ratio", 0.2
            ),
            conversation_window_turns=turns,
            conversation_window_overlap=overlap,
            window_planner=WindowPlannerConfig.from_dict(planner),
            build_parallelism=BuildParallelismConfig.from_dict(
                raw.get("build_parallelism")
            ),
            faiss_index=faiss_index,
        )
        if config.final_top_k > config.total_candidate_budget:
            raise ValueError(
                "memory.backends.v4.final_top_k must not exceed total_candidate_budget"
            )
        if config.bootstrap_top_k > config.total_candidate_budget:
            raise ValueError(
                "memory.backends.v4.bootstrap_top_k must not exceed total_candidate_budget"
            )
        if config.conversation_window_overlap >= config.conversation_window_turns:
            raise ValueError(
                "memory.backends.v4.conversation_window_overlap must be smaller than conversation_window_turns"
            )
        return config

    @classmethod
    def from_app_config(cls, config: Any) -> "V4MemoryConfig":
        memory = (getattr(config, "raw", {}) or {}).get("memory") or {}
        backends = memory.get("backends") or {}
        return cls.from_dict(backends.get("v4"))


def _validated_int(
    data: Mapping[str, Any], key: str, default: int, low: int, high: int
) -> int:
    value = int(data.get(key, default))
    if not low <= value <= high:
        raise ValueError(f"memory.backends.v4.{key} must be between {low} and {high}")
    return value


def _ratio(data: Mapping[str, Any], key: str, default: float) -> float:
    value = float(data.get(key, default))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"memory.backends.v4.{key} must be between 0 and 1")
    return value


def _optional(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None
