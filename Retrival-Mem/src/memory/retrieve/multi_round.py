from __future__ import annotations

from typing import Any, Mapping

from memory.retrieve.base import RetrievalRequest
from memory.retrieve.runtime import MultiRoundRetriever


class MultiRoundStrategy:
    strategy_id = "multi_round"

    def __init__(self, controller: Any = None):
        self.controller = controller

    def retrieve(
        self, backend: Any, request: RetrievalRequest, config: Mapping[str, Any]
    ):
        controller = self.controller
        profile = request.controller_profile or str(
            config.get("controller_profile") or "teacher"
        )
        provider_getter = getattr(backend, "get_multi_round_controller", None)
        if controller is None and callable(provider_getter):
            controller = provider_getter(profile)
        backend_config = getattr(backend, "v4_config", None)
        if controller is None:
            explicit_question_only = (
                str(config.get("controller_mode") or "").lower() == "question_only"
            )
            legacy_common_default = not bool(
                backend.capabilities.intent_aware_multi_round
            )
            if not explicit_question_only and not legacy_common_default:
                raise ValueError(
                    "Multi-round retrieval requires a configured Controller; "
                    "set controller_mode=question_only for the explicit baseline"
                )
            controller = _QuestionController(backend)
        if request.metadata.get("retrieval_prompt_profile") == "locomo_open_ended_v1":
            bind_prompts = getattr(controller, "with_open_ended_prompts", None)
            if callable(bind_prompts):
                controller = bind_prompts()
        default_rank_fusion_constant = getattr(
            backend_config, "rank_fusion_constant", 60
        )
        return MultiRoundRetriever(
            backend,
            controller,
            max_rounds=int(
                config.get("max_rounds", getattr(backend_config, "max_rounds", 8))
            ),
            zero_growth_stop_rounds=int(
                config.get(
                    "zero_growth_stop_rounds",
                    getattr(backend_config, "zero_growth_stop_rounds", 2),
                )
            ),
            simple_probe_budget=int(
                config.get(
                    "simple_probe_budget",
                    getattr(backend_config, "simple_probe_budget", 8),
                )
            ),
            complex_probe_budget=int(
                config.get(
                    "complex_probe_budget",
                    getattr(backend_config, "complex_probe_budget", 12),
                )
            ),
            followup_probe_budget=int(
                config.get(
                    "followup_probe_budget",
                    getattr(backend_config, "followup_probe_budget", 6),
                )
            ),
            expansion_budget_per_anchor=int(
                config.get(
                    "expansion_budget_per_anchor",
                    getattr(backend_config, "expansion_budget_per_anchor", 3),
                )
            ),
            final_top_k=int(
                config.get("final_top_k", getattr(backend_config, "final_top_k", 16))
            ),
            semantic_dedup_threshold=float(
                config.get(
                    "semantic_dedup_threshold",
                    getattr(backend_config, "semantic_dedup_threshold", 0.95),
                )
            ),
            total_candidate_budget=int(
                config.get(
                    "total_candidate_budget",
                    getattr(backend_config, "total_candidate_budget", 32),
                )
            ),
            controller_max_retries=int(config.get("controller_max_retries", 2)),
            rank_fusion_constant=int(
                config.get("rank_fusion_constant", default_rank_fusion_constant)
            ),
            bootstrap_top_k=int(
                config.get(
                    "bootstrap_top_k",
                    getattr(backend_config, "bootstrap_top_k", 5),
                )
            ),
            rerank_batch_size=int(
                config.get(
                    "rerank_batch_size",
                    getattr(backend_config, "rerank_batch_size", 16),
                )
            ),
            rerank_llm_weight=float(
                config.get(
                    "rerank_llm_weight",
                    getattr(backend_config, "rerank_llm_weight", 0.7),
                )
            ),
            rerank_retrieval_weight=float(
                config.get(
                    "rerank_retrieval_weight",
                    getattr(backend_config, "rerank_retrieval_weight", 0.3),
                )
            ),
            model_evidence_filter_enabled=bool(
                config.get(
                    "model_evidence_filter_enabled",
                    getattr(
                        backend_config,
                        "model_evidence_filter_enabled",
                        False,
                    ),
                )
            ),
            semantic_slimming_enabled=bool(
                config.get(
                    "semantic_slimming_enabled",
                    getattr(
                        backend_config,
                        "semantic_slimming_enabled",
                        True,
                    ),
                )
            ),
            max_subproblems=int(
                config.get(
                    "max_subproblems",
                    getattr(backend_config, "max_subproblems", 4),
                )
            ),
            max_actions_per_round=int(
                config.get(
                    "max_actions_per_round",
                    getattr(backend_config, "max_actions_per_round", 3),
                )
            ),
            max_actions_per_source=int(
                config.get(
                    "max_actions_per_source",
                    getattr(backend_config, "max_actions_per_source", 3),
                )
            ),
            failure_policy=getattr(backend_config, "failure_policy", None),
        ).retrieve(request.question, request.execution_context())


class _QuestionController:
    def __init__(self, backend: Any):
        self.backend = backend

    def set_bootstrap_evidence(self, evidence: Any) -> None:
        self.bootstrap_evidence = tuple(evidence)

    def initial_plan(self, question: str, context: Any = None) -> Mapping[str, Any]:
        return {"queries": [question]}

    def assess(
        self, question: str, results: Any, round_index: int, context: Any = None
    ):
        return {
            "sufficient": True,
            "missing_facets": [],
            "global_query": None,
            "action_ids": [],
        }
