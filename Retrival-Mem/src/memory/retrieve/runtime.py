from __future__ import annotations

import hashlib
import json
import math
import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from memory.retrieve.base import RetrievalExecutionContext
from memory.retrieve.checkpoints import RerankCheckpointStore
from memory.retrieve.structured import (
    derive_probe_hints,
    normalize_subproblem,
    normalize_subproblems,
)
from memory.retrieve.types import (
    Expansion,
    MultiRoundController,
    Probe,
    RetrievalBackend,
    RetrievalPlan,
    RetrievalRound,
    RetrievalTrajectory,
    RuntimeCandidate,
    Selector,
    validate_plan,
    validate_runtime_expansion,
)
from memory.v4.failure import V4OperationContext, V4RetrievalError, retry_v4_call
from memory.v4.intent import FACETS, QuestionIntentAnalyzer, RetrievalAction


_EVIDENCE_FILTER_BATCH_SIZE = 16
_MAX_EXPANSION_SOURCES = 5
_DUPLICATE_QUERY_REFORMULATIONS = 1


@dataclass
class _TargetState:
    model_probe_growth: int = 0
    model_action_growth: int = 0
    fallback_growth: int = 0
    total_growth: int = 0
    fallback_used: bool = False
    zero_growth_rounds: int = 0
    attempted_probe_fingerprints: set[tuple[Any, ...]] = field(default_factory=set)
    attempted_query_texts: set[str] = field(default_factory=set)
    duplicate_query_events: int = 0
    attempted_expansion_paths: set[tuple[str, ...]] = field(default_factory=set)
    reformulation_stage: int = 0
    exhausted: bool = False
    reformulation_exhausted: bool = False
    exhausted_action_snapshot: frozenset[tuple[str, ...]] = frozenset()
    last_probe: Probe | None = None
    failed_queries: tuple[str, ...] = ()
    delivery_missing: bool = False

    def public_state(self) -> dict[str, Any]:
        if self.delivery_missing:
            return {
                "status": "missing_answer_evidence",
                "instruction": "Recover original evidence for cited navigation facts before stopping.",
            }
        if self.exhausted:
            return {"status": "exhausted"}
        if not self.zero_growth_rounds:
            return {}
        feedback: dict[str, Any] = {"status": "zero_growth"}
        if self.last_probe is not None:
            feedback["failed_query"] = self.last_probe.query
        if self.failed_queries:
            feedback["failed_queries"] = list(self.failed_queries)
        return feedback


@dataclass(frozen=True)
class _ExpansionRecord:
    expansion_id: str
    anchor_key: str
    result_keys: frozenset[str]


@dataclass(frozen=True)
class _TargetGrowth:
    model_probe_growth: int
    model_action_growth: int
    fallback_growth: int
    total_growth: int
    fallback_used: bool

    @property
    def model_growth(self) -> int:
        return self.model_probe_growth + self.model_action_growth


@dataclass(frozen=True)
class _SemanticSlimmingView:
    summary: str
    facts: tuple[dict[str, Any], ...]
    evidence: tuple[dict[str, Any], ...]
    kept_items: int
    dropped_items: int


def _retained_target_growth(
    model_probe_keys: set[str],
    model_action_keys: set[str],
    fallback_keys: set[str],
    surviving_new: set[str],
    *,
    fallback_used: bool,
) -> _TargetGrowth:
    retained_probes = model_probe_keys & surviving_new
    retained_actions = model_action_keys & surviving_new
    retained_fallback = fallback_keys & surviving_new
    return _TargetGrowth(
        model_probe_growth=len(retained_probes),
        model_action_growth=len(retained_actions),
        fallback_growth=len(retained_fallback),
        total_growth=len((retained_probes | retained_actions | retained_fallback)),
        fallback_used=fallback_used,
    )


class MultiRoundRetriever:
    """V4 evidence-complete Controller loop with deterministic action rules."""

    def __init__(
        self,
        backend: RetrievalBackend,
        controller: MultiRoundController,
        *,
        max_rounds: int = 8,
        zero_growth_stop_rounds: int = 2,
        simple_probe_budget: int = 8,
        complex_probe_budget: int = 12,
        followup_probe_budget: int = 6,
        expansion_budget_per_anchor: int = 3,
        final_top_k: int = 16,
        total_candidate_budget: int | None = 32,
        controller_max_retries: int = 2,
        rank_fusion_constant: int = 60,
        bootstrap_top_k: int = 5,
        rerank_batch_size: int = 16,
        rerank_llm_weight: float = 0.7,
        rerank_retrieval_weight: float = 0.3,
        failure_policy: Any = None,
        analyzer: QuestionIntentAnalyzer | None = None,
        max_subproblems: int = 4,
        max_actions_per_round: int = 3,
        max_actions_per_source: int = 3,
        semantic_dedup_threshold: float = 0.95,
        model_evidence_filter_enabled: bool = False,
        semantic_slimming_enabled: bool = True,
    ):
        positive_values = {
            "max_rounds": max_rounds,
            "simple_probe_budget": simple_probe_budget,
            "complex_probe_budget": complex_probe_budget,
            "followup_probe_budget": followup_probe_budget,
            "expansion_budget_per_anchor": expansion_budget_per_anchor,
            "final_top_k": final_top_k,
            "total_candidate_budget": total_candidate_budget,
            "max_subproblems": max_subproblems,
            "max_actions_per_round": max_actions_per_round,
            "max_actions_per_source": max_actions_per_source,
        }
        invalid = [
            name
            for name, value in positive_values.items()
            if value is None or int(value) < 1
        ]
        if invalid:
            raise ValueError(
                "V4 retrieval values must be positive: " + ", ".join(invalid)
            )
        self.backend = backend
        self.controller = controller
        self.max_rounds = int(max_rounds)
        self.zero_growth_stop_rounds = int(zero_growth_stop_rounds)
        if self.zero_growth_stop_rounds < 0:
            raise ValueError("zero_growth_stop_rounds must be non-negative")
        self.simple_probe_budget = int(simple_probe_budget)
        self.complex_probe_budget = int(complex_probe_budget)
        self.followup_probe_budget = int(followup_probe_budget)
        self.expansion_budget_per_anchor = int(expansion_budget_per_anchor)
        self.max_subproblems = int(max_subproblems)
        self.max_actions_per_round = int(max_actions_per_round)
        self.max_actions_per_source = int(max_actions_per_source)
        self.final_top_k = int(final_top_k)
        self.total_candidate_budget = int(total_candidate_budget)
        if self.final_top_k > self.total_candidate_budget:
            raise ValueError("final_top_k must not exceed total_candidate_budget")
        if (
            not math.isfinite(semantic_dedup_threshold)
            or not 0.0 <= float(semantic_dedup_threshold) <= 1.0
        ):
            raise ValueError("semantic_dedup_threshold must be within 0..1")
        self.semantic_dedup_threshold = float(semantic_dedup_threshold)
        self.model_evidence_filter_enabled = bool(model_evidence_filter_enabled)
        self.semantic_slimming_enabled = bool(semantic_slimming_enabled)
        self.controller_max_retries = int(controller_max_retries)
        if self.controller_max_retries < 0:
            raise ValueError("controller_max_retries must be non-negative")
        self.rank_fusion_constant = int(rank_fusion_constant)
        if self.rank_fusion_constant < 1:
            raise ValueError("rank_fusion_constant must be positive")
        self.bootstrap_top_k = int(bootstrap_top_k)
        self.rerank_batch_size = int(rerank_batch_size)
        self.rerank_llm_weight = float(rerank_llm_weight)
        self.rerank_retrieval_weight = float(rerank_retrieval_weight)
        if not 1 <= self.bootstrap_top_k <= self.total_candidate_budget:
            raise ValueError(
                "bootstrap_top_k must be between 1 and total_candidate_budget"
            )
        if not 1 <= self.rerank_batch_size <= 64:
            raise ValueError("rerank_batch_size must be between 1 and 64")
        if (
            not math.isfinite(self.rerank_llm_weight)
            or not math.isfinite(self.rerank_retrieval_weight)
            or self.rerank_llm_weight < 0.0
            or self.rerank_retrieval_weight < 0.0
            or not math.isclose(
                self.rerank_llm_weight + self.rerank_retrieval_weight,
                1.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "rerank weights must be finite, non-negative, and sum to 1"
            )
        self.rank_fusion = bool(backend.capabilities.intent_aware_multi_round)
        self.failure_policy = failure_policy
        self.strict_v4 = (
            backend.capabilities.backend_id == "v4" and failure_policy is not None
        )
        if int(backend.capabilities.max_total_budget) < self.total_candidate_budget:
            raise ValueError(
                f"V4 backend max_total_budget must allow {self.total_candidate_budget} candidates"
            )
        controller_analyzer = getattr(controller, "analyzer", None)
        self.analyzer = analyzer or controller_analyzer or QuestionIntentAnalyzer()

    def retrieve(self, question: str, execution: RetrievalExecutionContext) -> Any:
        trajectory = RetrievalTrajectory(
            question=question,
            backend_id=self.backend.capabilities.backend_id,
        )
        semantic_slimming_by_key: dict[str, _SemanticSlimmingView | None] = {}
        bootstrap_probe = Probe(
            id="baseline",
            query=question,
            budget=self.complex_probe_budget,
            lane="baseline",
        )
        bootstrap_rows, bootstrap_by_source = _execute_probe_batch(
            self.backend, execution, (bootstrap_probe,)
        )
        trajectory.retrieval_calls = 1
        bootstrap_candidates = _deduplicate(
            bootstrap_by_source.get(bootstrap_probe.id, bootstrap_rows)
        )
        bootstrap_candidates = self._slim_semantic_candidates(
            question,
            execution,
            bootstrap_candidates,
            semantic_slimming_by_key,
            trajectory,
        )
        set_bootstrap = getattr(self.controller, "set_bootstrap_evidence", None)
        if not callable(set_bootstrap):
            raise TypeError(
                "Multi-round Controller must implement set_bootstrap_evidence"
            )
        set_bootstrap(
            _public_candidate_batch(
                bootstrap_candidates[: self.bootstrap_top_k], ref_prefix="b"
            )
        )
        try:
            plan = self._initial_plan(question, execution, trajectory)
        except Exception as error:
            if self.strict_v4:
                raise
            trajectory.controller_errors.append(str(error))
            trajectory.termination_reason = "controller_error"
            return _finalize(
                self.backend, execution, [], question, self.final_top_k, trajectory
            )

        model_probes = tuple(probe for probe in plan.probes if probe.lane == "model")
        subproblems = tuple(probe.public_subproblem() for probe in model_probes)
        required_target_ids = {
            str(item["subproblem_id"])
            for item in subproblems
            if item.get("required", True)
        }
        baseline_probe = next(
            probe for probe in plan.probes if probe.lane == "baseline"
        )
        model_rows: list[RuntimeCandidate] = []
        model_by_source: dict[str, list[RuntimeCandidate]] = {}
        if model_probes:
            model_rows, model_by_source = _execute_probe_batch(
                self.backend, execution, model_probes
            )
            trajectory.retrieval_calls += 1
        baseline_rows = bootstrap_candidates[: baseline_probe.budget]
        by_source = {baseline_probe.id: baseline_rows, **model_by_source}
        if self.rank_fusion:
            _record_action_ranks(
                bootstrap_candidates,
                baseline_probe.id,
                self.rank_fusion_constant,
            )
            for probe in model_probes:
                _record_action_ranks(
                    by_source.get(probe.id, ()),
                    probe.id,
                    self.rank_fusion_constant,
                )
        all_candidates = _deduplicate(
            [*bootstrap_candidates, *model_rows],
            rank_fusion=self.rank_fusion,
        )
        all_candidates = self._slim_semantic_candidates(
            question,
            execution,
            all_candidates,
            semantic_slimming_by_key,
            trajectory,
        )
        protected_keys = {
            candidate.key for candidate in baseline_rows[: baseline_probe.budget]
        }
        target_states = _initial_target_states(plan.probes)
        attempted: set[str] = set()
        previous_support_keys: set[str] = set()
        pending_expansions: tuple[_ExpansionRecord, ...] = ()
        rewarded_expansions: set[str] = set()
        usefulness_by_key: dict[str, bool] = {}
        useful_candidates = self._filter_useful_candidates(
            question,
            execution,
            all_candidates,
            usefulness_by_key,
            trajectory,
        )
        candidates = _select_controller_evidence(
            useful_candidates,
            subproblems,
            previous_support_keys,
            protected_keys=protected_keys,
        )
        observations, refs = _observations(candidates)
        trajectory.rounds.append(
            RetrievalRound(
                round_index=1,
                initial_probes=plan.probes,
                observations=tuple(observations),
                growth=len(candidates),
                total_growth=len(candidates),
                controller_calls=1,
                retrieval_calls=trajectory.retrieval_calls,
                reason="initial_global_search",
            )
        )
        for round_index in range(2, self.max_rounds + 1):
            refs_at_round_start = dict(refs)
            retrieval_state: dict[str, dict[str, Any]] = {}
            for target_id, state in target_states.items():
                feedback = state.public_state()
                if feedback:
                    retrieval_state[target_id] = feedback
            metadata = {
                "retrieval_state": retrieval_state,
            }
            prepare = getattr(self.controller, "set_round_metadata", None)
            if callable(prepare):
                prepare(metadata)
            try:
                expansion_source_refs = self._select_expansion_sources(
                    question,
                    observations,
                    round_index,
                    execution,
                    trajectory,
                )
                selected_ref_set = set(expansion_source_refs)
                expansion_sources = [
                    item
                    for item in observations
                    if str(item["result_ref"]) in selected_ref_set
                ]
                catalog = self._action_catalog(question, expansion_sources)
                prepared_catalogs = _prepare_action_catalogs_by_target(
                    catalog, refs_at_round_start, target_states
                )
                set_available_actions = getattr(
                    self.controller,
                    "set_available_actions_by_target",
                    None,
                )
                if callable(set_available_actions):
                    set_available_actions(prepared_catalogs)
                    decision_catalogs = prepared_catalogs
                else:
                    decision_catalogs = _filter_action_catalogs_by_target(
                        catalog, refs_at_round_start, target_states
                    )
                metadata["expansion_source_refs"] = list(expansion_source_refs)
                set_expansion_sources = getattr(
                    self.controller, "set_expansion_source_refs", None
                )
                if callable(set_expansion_sources):
                    set_expansion_sources(expansion_source_refs)
                _refresh_exhausted_targets(target_states, catalog, refs_at_round_start)
                decision = self._followup_decision(
                    question,
                    subproblems,
                    observations,
                    round_index,
                    execution,
                    decision_catalogs,
                    trajectory,
                )
            except Exception as error:
                if self.strict_v4:
                    raise
                trajectory.controller_errors.append(str(error))
                trajectory.termination_reason = "controller_error"
                break

            root_assessment = dict(decision["root_assessment"])
            assessments = {
                str(item["subproblem_id"]): dict(item)
                for item in decision["assessments"]
            }
            cited_keys = _cited_candidate_keys(decision, refs_at_round_start)
            previous_support_keys = cited_keys
            # A navigation summary is not sufficient unless its concrete facts
            # have a question-specific, renderable view. Check before stopping,
            # while the loop can still recover the missing source evidence.
            delivery_gaps = self._check_support_delivery(
                root_assessment, assessments, refs_at_round_start, trajectory
            )
            if delivery_gaps:
                target_states["root"].delivery_missing = True
                decision["target_ids"] = ["root"]
            else:
                target_states["root"].delivery_missing = False
            _reinforce_cited_expansion_anchors(
                candidates,
                pending_expansions,
                cited_keys,
                rewarded_expansions,
            )
            if (
                root_assessment["sufficient"]
                and not target_states["root"].reformulation_exhausted
            ):
                target_states["root"].exhausted = False
                target_states["root"].exhausted_action_snapshot = frozenset()
            for target_id, assessment in assessments.items():
                if (
                    assessment["sufficient"]
                    and not target_states[target_id].reformulation_exhausted
                ):
                    target_states[target_id].exhausted = False
                    target_states[target_id].exhausted_action_snapshot = frozenset()
            exhausted = {
                target_id
                for target_id, state in target_states.items()
                if state.exhausted
            }
            statuses = _subproblem_statuses(
                subproblems, assessments, attempted, exhausted
            )
            root_status = _root_status(root_assessment, attempted, exhausted)
            if _root_and_required_sufficient(root_assessment, subproblems, assessments):
                trajectory.rounds.append(
                    RetrievalRound(
                        round_index=round_index,
                        observations=tuple(observations),
                        sufficient=True,
                        root_status=root_status,
                        subproblem_statuses=statuses,
                        controller_calls=2,
                        reason="root_and_required_sufficient",
                    )
                )
                trajectory.termination_reason = "root_and_required_sufficient"
                break

            actionable = _actionable_targets(
                root_assessment, subproblems, assessments, exhausted
            )
            if not actionable:
                trajectory.rounds.append(
                    RetrievalRound(
                        round_index=round_index,
                        observations=tuple(observations),
                        root_status=root_status,
                        subproblem_statuses=statuses,
                        controller_calls=2,
                        reason="all_missing_exhausted",
                    )
                )
                trajectory.termination_reason = "all_missing_exhausted"
                break
            target_ids = [
                target_id
                for target_id in decision["target_ids"]
                if target_id in actionable
            ]
            if not target_ids:
                trajectory.controller_errors.append(
                    f"Replaced non-actionable target_ids in round {round_index}"
                )
                target_ids = actionable[:1]

            fresh_by_target: dict[str, Probe] = {}
            expansions_by_target: dict[str, list[Expansion]] = {}
            missing_by_target: dict[str, tuple[str, ...]] = {}
            for target_id in target_ids:
                target = (
                    root_assessment if target_id == "root" else assessments[target_id]
                )
                missing = tuple(target["missing_facets"])
                missing_by_target[target_id] = missing
                state = target_states[target_id]
                fresh_probe = self._fresh_probe_for_target(
                    target.get("followup_probe"),
                    round_index,
                    target_id,
                    state,
                    trajectory,
                    missing_facets=missing,
                    allow_reformulation=target_id in required_target_ids,
                )
                if fresh_probe is not None:
                    fresh_by_target[target_id] = fresh_probe
                if state.reformulation_exhausted:
                    continue
                expansions = self._parse_actions(
                    target.get("selected_actions", ()),
                    round_index,
                    refs_at_round_start,
                    missing,
                    target_id,
                    state.attempted_expansion_paths,
                )
                if len(expansions) > self.max_actions_per_source:
                    trajectory.controller_errors.append(
                        f"Dropped {len(expansions) - self.max_actions_per_source} "
                        f"actions for {target_id} in round {round_index}"
                    )
                if expansions:
                    expansions_by_target[target_id] = expansions[
                        : self.max_actions_per_source
                    ]

            rule_fallback_targets: set[str] = set()
            if not fresh_by_target and not expansions_by_target:
                for target_id in target_ids:
                    state = target_states[target_id]
                    if state.reformulation_exhausted:
                        continue
                    available_snapshot = _available_expansion_paths(
                        target_id, catalog, refs_at_round_start
                    )
                    available = available_snapshot - state.attempted_expansion_paths
                    fallback = self._rule_expansion_for_target(
                        target_id,
                        state,
                        catalog,
                        refs_at_round_start,
                        missing_by_target[target_id],
                        round_index,
                    )
                    if fallback is not None:
                        expansions_by_target[target_id] = [fallback]
                        rule_fallback_targets.add(target_id)
                        continue
                    if not available and not _has_reformulation_opportunity(
                        state,
                        allow_reformulation=target_id in required_target_ids,
                    ):
                        state.exhausted = True
                        state.exhausted_action_snapshot = frozenset(available_snapshot)
                if not expansions_by_target:
                    exhausted = {
                        target_id
                        for target_id, state in target_states.items()
                        if state.exhausted
                    }
                    statuses = _subproblem_statuses(
                        subproblems, assessments, attempted, exhausted
                    )
                    trajectory.rounds.append(
                        RetrievalRound(
                            round_index=round_index,
                            observations=tuple(observations),
                            missing_facets=tuple(
                                dict.fromkeys(
                                    facet
                                    for target_id in target_ids
                                    for facet in missing_by_target[target_id]
                                )
                            ),
                            root_status=_root_status(
                                root_assessment, attempted, exhausted
                            ),
                            subproblem_statuses=statuses,
                            controller_calls=2,
                            reason="targets_exhausted"
                            if any(target_states[item].exhausted for item in target_ids)
                            else "no_actions_selected",
                        )
                    )
                    if not _actionable_targets(
                        root_assessment, subproblems, assessments, exhausted
                    ):
                        trajectory.termination_reason = "all_missing_exhausted"
                        break
                    continue

            round_start_keys = {candidate.key for candidate in candidates}
            additions: list[RuntimeCandidate] = []
            model_probe_additions: dict[str, set[str]] = {
                target_id: set() for target_id in target_ids
            }
            model_action_additions: dict[str, set[str]] = {
                target_id: set() for target_id in target_ids
            }
            fallback_additions: dict[str, set[str]] = {
                target_id: set() for target_id in target_ids
            }
            fallback_used_by_target = {
                target_id: target_id in rule_fallback_targets
                for target_id in target_ids
            }
            retrieval_calls = 0
            fresh_probes = tuple(fresh_by_target.values())
            if fresh_probes:
                fresh_rows, fresh_by_source = _execute_probe_batch(
                    self.backend, execution, fresh_probes
                )
                if self.rank_fusion:
                    for probe in fresh_probes:
                        _record_action_ranks(
                            fresh_by_source.get(probe.id, ()),
                            probe.id,
                            self.rank_fusion_constant,
                        )
                for target_id, probe in fresh_by_target.items():
                    model_probe_additions[target_id].update(
                        row.key for row in fresh_by_source.get(probe.id, ())
                    )
                additions.extend(fresh_rows)
                retrieval_calls += 1
            expansion_records: list[_ExpansionRecord] = []
            for target_id, expansions in expansions_by_target.items():
                state = target_states[target_id]
                for expansion in expansions:
                    stable_path = _expansion_path_key(
                        target_id,
                        refs_at_round_start[expansion.source].key,
                        expansion.edge_type,
                        expansion.direction,
                        expansion.selector.attributes,
                    )
                    state.attempted_expansion_paths.add(stable_path)
                    anchor = refs_at_round_start[expansion.source]
                    rows = list(self.backend.expand(execution, (anchor,), expansion))[
                        : expansion.budget
                    ]
                    if expansion.edge_type == "SOURCE_EVIDENCE_LOOKUP":
                        trajectory.evidence_backtrack_calls += 1
                    for row in rows:
                        row.sources.add(expansion.id)
                        if target_id != "root":
                            row.subproblem_ids.add(target_id)
                    if self.rank_fusion:
                        _record_action_ranks(
                            rows, expansion.id, self.rank_fusion_constant
                        )
                    additions.extend(rows)
                    expansion_keys = {row.key for row in rows}
                    if target_id in rule_fallback_targets:
                        fallback_additions[target_id].update(expansion_keys)
                    else:
                        model_action_additions[target_id].update(expansion_keys)
                    expansion_records.append(
                        _ExpansionRecord(
                            expansion_id=expansion.id,
                            anchor_key=anchor.key,
                            result_keys=frozenset(row.key for row in rows),
                        )
                    )
                    retrieval_calls += 1

            all_candidates = _deduplicate(
                [*all_candidates, *additions],
                rank_fusion=self.rank_fusion,
            )
            all_candidates = self._slim_semantic_candidates(
                question,
                execution,
                all_candidates,
                semantic_slimming_by_key,
                trajectory,
            )
            useful_candidates = self._filter_useful_candidates(
                question,
                execution,
                all_candidates,
                usefulness_by_key,
                trajectory,
            )
            candidates = _select_controller_evidence(
                useful_candidates,
                subproblems,
                cited_keys,
                protected_keys=protected_keys,
            )
            surviving_new = {
                candidate.key for candidate in candidates
            } - round_start_keys
            preliminary_model_growth = {
                target_id: len(
                    (
                        model_probe_additions[target_id]
                        | model_action_additions[target_id]
                    )
                    & surviving_new
                )
                for target_id in target_ids
            }

            fallback_executed = False
            fallback_target_ids = [
                target_id
                for target_id in target_ids
                if preliminary_model_growth.get(target_id, 0) == 0
                and target_id not in rule_fallback_targets
            ]
            for target_id in fallback_target_ids:
                if len(expansion_records) >= self.max_actions_per_round:
                    break
                state = target_states[target_id]
                fallback = self._rule_expansion_for_target(
                    target_id,
                    state,
                    catalog,
                    refs_at_round_start,
                    missing_by_target[target_id],
                    round_index,
                )
                if fallback is None:
                    continue
                stable_path = _expansion_path_key(
                    target_id,
                    refs_at_round_start[fallback.source].key,
                    fallback.edge_type,
                    fallback.direction,
                    fallback.selector.attributes,
                )
                state.attempted_expansion_paths.add(stable_path)
                anchor = refs_at_round_start[fallback.source]
                rows = list(self.backend.expand(execution, (anchor,), fallback))[
                    : fallback.budget
                ]
                if fallback.edge_type == "SOURCE_EVIDENCE_LOOKUP":
                    trajectory.evidence_backtrack_calls += 1
                for row in rows:
                    row.sources.add(fallback.id)
                    if target_id != "root":
                        row.subproblem_ids.add(target_id)
                if self.rank_fusion:
                    _record_action_ranks(rows, fallback.id, self.rank_fusion_constant)
                additions.extend(rows)
                fallback_additions[target_id].update(row.key for row in rows)
                fallback_used_by_target[target_id] = True
                expansion_records.append(
                    _ExpansionRecord(
                        expansion_id=fallback.id,
                        anchor_key=anchor.key,
                        result_keys=frozenset(row.key for row in rows),
                    )
                )
                retrieval_calls += 1
                fallback_executed = True

            if fallback_executed:
                all_candidates = _deduplicate(
                    [*all_candidates, *additions],
                    rank_fusion=self.rank_fusion,
                )
                all_candidates = self._slim_semantic_candidates(
                    question,
                    execution,
                    all_candidates,
                    semantic_slimming_by_key,
                    trajectory,
                )
                useful_candidates = self._filter_useful_candidates(
                    question,
                    execution,
                    all_candidates,
                    usefulness_by_key,
                    trajectory,
                )
                candidates = _select_controller_evidence(
                    useful_candidates,
                    subproblems,
                    cited_keys,
                    protected_keys=protected_keys,
                )
                surviving_new = {
                    candidate.key for candidate in candidates
                } - round_start_keys

            growth_by_target = {
                target_id: _retained_target_growth(
                    model_probe_additions[target_id],
                    model_action_additions[target_id],
                    fallback_additions[target_id],
                    surviving_new,
                    fallback_used=fallback_used_by_target[target_id],
                )
                for target_id in target_ids
            }

            for target_id in target_ids:
                growth = growth_by_target[target_id]
                state = target_states[target_id]
                state.model_probe_growth = growth.model_probe_growth
                state.model_action_growth = growth.model_action_growth
                state.fallback_growth = growth.fallback_growth
                state.total_growth = growth.total_growth
                state.fallback_used = growth.fallback_used
                state.zero_growth_rounds = (
                    0 if growth.model_growth else state.zero_growth_rounds + 1
                )
                if not growth.model_growth and state.last_probe is not None:
                    failed = state.last_probe.query.strip()
                    if failed and failed not in state.failed_queries:
                        state.failed_queries = (*state.failed_queries, failed)
                if growth.total_growth:
                    state.exhausted = False
                    state.exhausted_action_snapshot = frozenset()
                elif (
                    self.zero_growth_stop_rounds
                    and state.zero_growth_rounds >= self.zero_growth_stop_rounds
                ):
                    state.exhausted = True
                    state.exhausted_action_snapshot = frozenset(
                        _available_expansion_paths(
                            target_id, catalog, refs_at_round_start
                        )
                    )
                    trajectory.controller_errors.append(
                        f"Forced zero-growth early stop for {target_id} in "
                        f"round {round_index} after "
                        f"{state.zero_growth_rounds} stagnant rounds"
                    )
                attempted.add(target_id)
            trajectory.retrieval_calls += retrieval_calls
            previous_support_keys = cited_keys
            pending_expansions = tuple(expansion_records)
            observations, refs = _observations(candidates)
            exhausted = {
                target_id
                for target_id, state in target_states.items()
                if state.exhausted
            }
            statuses = _subproblem_statuses(
                subproblems, assessments, attempted, exhausted
            )
            model_probe_growth = sum(
                item.model_probe_growth for item in growth_by_target.values()
            )
            model_action_growth = sum(
                item.model_action_growth for item in growth_by_target.values()
            )
            fallback_growth = sum(
                item.fallback_growth for item in growth_by_target.values()
            )
            total_growth = sum(item.total_growth for item in growth_by_target.values())
            trajectory.rounds.append(
                RetrievalRound(
                    round_index=round_index,
                    fresh_probe=fresh_probes[0] if fresh_probes else None,
                    fresh_probes=fresh_probes,
                    expansions=tuple(
                        expansion
                        for expansions in expansions_by_target.values()
                        for expansion in expansions
                    ),
                    observations=tuple(observations),
                    missing_facets=tuple(
                        dict.fromkeys(
                            facet
                            for target_id in target_ids
                            for facet in missing_by_target[target_id]
                        )
                    ),
                    root_status=_root_status(root_assessment, attempted, exhausted),
                    subproblem_statuses=statuses,
                    growth=total_growth,
                    model_probe_growth=model_probe_growth,
                    model_action_growth=model_action_growth,
                    fallback_growth=fallback_growth,
                    total_growth=total_growth,
                    fallback_used=any(
                        item.fallback_used for item in growth_by_target.values()
                    ),
                    controller_calls=2,
                    retrieval_calls=retrieval_calls,
                    reason="actions_executed",
                )
            )
        else:
            trajectory.termination_reason = "max_rounds"

        if not trajectory.termination_reason:
            trajectory.termination_reason = "max_rounds"
        all_candidates = self._slim_semantic_candidates(
            question,
            execution,
            all_candidates,
            semantic_slimming_by_key,
            trajectory,
        )
        useful_candidates = [
            candidate
            for candidate in all_candidates
            if usefulness_by_key.get(candidate.key) is True
        ]
        reranked = self._rerank_all(
            question,
            execution,
            useful_candidates,
            trajectory,
            protected_keys=previous_support_keys,
        )
        required_subproblems = tuple(
            item for item in subproblems if item.get("required", True)
        )
        if required_subproblems:
            final_candidates = _coverage_select_v2(
                reranked,
                required_subproblems,
                self.final_top_k,
                protected_keys=previous_support_keys,
            )
        else:
            final_candidates = self._select_final_candidates(
                reranked, protected_keys=previous_support_keys
            )
        delivered = {candidate.key for candidate in final_candidates}
        candidate_by_key = {candidate.key: candidate for candidate in useful_candidates}
        enforceable_support_keys = {
            key
            for key in previous_support_keys
            if key in candidate_by_key and _delivery_enforced(candidate_by_key[key])
        }
        trajectory.protected_support_count = len(previous_support_keys)
        trajectory.delivered_support_count = len(enforceable_support_keys & delivered)
        trajectory.support_delivery_verified = (
            (enforceable_support_keys <= delivered)
            if enforceable_support_keys
            else None
        )
        return _finalize(
            self.backend,
            execution,
            final_candidates,
            question,
            max(1, len(final_candidates)),
            trajectory,
        )

    def _slim_semantic_candidates(
        self,
        question: str,
        execution: RetrievalExecutionContext,
        candidates: Sequence[RuntimeCandidate],
        cache: dict[str, _SemanticSlimmingView | None],
        trajectory: RetrievalTrajectory,
    ) -> list[RuntimeCandidate]:
        slim = getattr(self.controller, "slim_semantic_node", None)
        if not callable(slim):
            return list(candidates)

        output: list[RuntimeCandidate] = []
        for candidate in candidates:
            if not self.semantic_slimming_enabled or candidate.node_type not in {
                "semantic_state",
                "chain_head",
            }:
                output.append(candidate)
                continue
            if candidate.key not in cache:
                items, source_by_ref = _semantic_slimming_items(candidate)
                if not items:
                    cache[candidate.key] = None
                else:
                    trajectory.semantic_slimming_candidates += 1

                    def invoke() -> Mapping[str, Any]:
                        trajectory.semantic_slimming_calls += 1
                        return slim(
                            question,
                            tuple(items),
                            execution.user_context,
                        )

                    try:
                        relevant_refs = self._retry(
                            invoke,
                            lambda value: _parse_semantic_slimming_decisions(
                                value,
                                tuple(source_by_ref),
                            ),
                            trajectory,
                            stage="semantic_node_slimming",
                            unit_id=(
                                f"{candidate.key}-"
                                f"{_stable_digest({'question': question})[:12]}"
                            ),
                        )
                        selected = [
                            source_by_ref[ref]
                            for ref in source_by_ref
                            if ref in relevant_refs
                        ]
                        if not selected:
                            trajectory.semantic_slimming_fallbacks += 1
                            cache[candidate.key] = None
                        else:
                            view = _semantic_slimming_view(
                                selected,
                                total_items=len(source_by_ref),
                            )
                            cache[candidate.key] = view
                            trajectory.semantic_slimming_kept_items += view.kept_items
                            trajectory.semantic_slimming_dropped_items += (
                                view.dropped_items
                            )
                    except Exception as error:
                        trajectory.controller_errors.append(
                            f"Semantic node slimming failed for "
                            f"{candidate.key}: {error}"
                        )
                        trajectory.semantic_slimming_fallbacks += 1
                        cache[candidate.key] = None
            view = cache[candidate.key]
            output.append(
                candidate
                if view is None
                else _apply_semantic_slimming_view(candidate, view)
            )
        return output

    def _filter_useful_candidates(
        self,
        question: str,
        execution: RetrievalExecutionContext,
        candidates: Sequence[RuntimeCandidate],
        usefulness_by_key: dict[str, bool],
        trajectory: RetrievalTrajectory,
    ) -> list[RuntimeCandidate]:
        unseen = [
            candidate
            for candidate in candidates
            if candidate.key not in usefulness_by_key
        ]
        if not self.model_evidence_filter_enabled:
            usefulness_by_key.update((candidate.key, True) for candidate in unseen)
            return list(candidates)
        if unseen:
            filter_batch = getattr(self.controller, "filter_evidence_batch", None)
            if not callable(filter_batch):
                usefulness_by_key.update((candidate.key, True) for candidate in unseen)
            else:
                public_rows = _public_candidate_batch(unseen, ref_prefix="ef")
                trajectory.evidence_filter_candidates += len(public_rows)
                for start in range(0, len(public_rows), _EVIDENCE_FILTER_BATCH_SIZE):
                    public_batch = public_rows[
                        start : start + _EVIDENCE_FILTER_BATCH_SIZE
                    ]
                    candidate_batch = unseen[
                        start : start + _EVIDENCE_FILTER_BATCH_SIZE
                    ]
                    expected_refs = tuple(
                        str(item["result_ref"]) for item in public_batch
                    )
                    batch_index = trajectory.evidence_filter_batches
                    trajectory.evidence_filter_batches += 1

                    def invoke() -> Mapping[str, Any]:
                        trajectory.evidence_filter_calls += 1
                        return filter_batch(
                            question,
                            tuple(public_batch),
                            batch_index,
                            execution.user_context,
                        )

                    try:
                        decisions = self._retry(
                            invoke,
                            lambda value: _parse_evidence_filter_decisions(
                                value, expected_refs
                            ),
                            trajectory,
                            stage="evidence_filter",
                            unit_id=(
                                f"batch-{batch_index}-"
                                f"{_stable_digest({'refs': expected_refs})[:12]}"
                            ),
                        )
                    except Exception:
                        if self.strict_v4:
                            raise
                        decisions = {ref: True for ref in expected_refs}
                    for public, candidate in zip(
                        public_batch, candidate_batch, strict=True
                    ):
                        useful = decisions[str(public["result_ref"])]
                        usefulness_by_key[candidate.key] = useful
                        trajectory.evidence_filter_dropped += int(not useful)

        return [
            candidate
            for candidate in candidates
            if usefulness_by_key.get(candidate.key) is True
        ]

    def _rerank_all(
        self,
        question: str,
        execution: RetrievalExecutionContext,
        candidates: Sequence[RuntimeCandidate],
        trajectory: RetrievalTrajectory,
        *,
        protected_keys: set[str] | None = None,
    ) -> list[RuntimeCandidate]:
        rows = _deduplicate(candidates, rank_fusion=self.rank_fusion)
        if not rows:
            return []
        rows = [
            view
            for candidate in rows
            if (view := _answer_candidate(candidate)) is not None
        ]
        if not rows:
            return []
        # Do not discard an independently cited event just because its wording
        # resembles another event. Both may be needed for a count or timeline.
        protected = [row for row in rows if row.key in (protected_keys or ())]
        others, semantic_dedup_dropped = _semantic_deduplicate(
            [row for row in rows if row.key not in (protected_keys or ())],
            threshold=self.semantic_dedup_threshold,
        )
        rows = [*protected, *others]
        trajectory.semantic_dedup_dropped += semantic_dedup_dropped
        if not rows:
            return []

        public_rows = _public_candidate_batch(rows, ref_prefix="rr")
        public_by_ref = {str(item["result_ref"]): item for item in public_rows}
        all_refs = [str(item["result_ref"]) for item in public_rows]
        checkpoint_key = str(
            execution.metadata.get("rerank_checkpoint_key")
            or _stable_digest(
                {
                    "namespace": execution.namespace,
                    "question": question,
                }
            )
        )
        checkpoint_path = str(
            execution.metadata.get("rerank_checkpoint_path")
            or ":memory:"
        )
        rerank = getattr(self.controller, "rerank_batch", None)
        if not callable(rerank):
            raise TypeError("Multi-round Controller must implement rerank_batch")
        scores_by_ref: dict[str, float] = {}
        attempts_by_ref: dict[str, int] = {ref: 0 for ref in all_refs}
        store = RerankCheckpointStore(checkpoint_path)
        if execution.metadata.get("rerank_reset"):
            store.invalidate(checkpoint_key)
        pending: deque[list[dict[str, Any]]] = deque()
        for start in range(0, len(public_rows), self.rerank_batch_size):
            pending.append(public_rows[start : start + self.rerank_batch_size])

        batch_index = 0
        try:
            while pending:
                batch = pending.popleft()
                expected_refs = tuple(str(item["result_ref"]) for item in batch)
                try:
                    output = store.load_success(
                        checkpoint_key, batch_index
                    )
                    if output is None:
                        output = dict(
                            rerank(
                                question,
                                tuple(batch),
                                batch_index,
                                execution.user_context,
                                allow_partial=True,
                            )
                        )
                        trajectory.rerank_calls += 1
                    valid = _parse_partial_rerank_scores(output, expected_refs)
                    for ref, score in valid.items():
                        scores_by_ref.setdefault(ref, score)
                    missing = [ref for ref in expected_refs if ref not in scores_by_ref]
                    if not missing:
                        store.record_success(
                            checkpoint_key, batch_index, output
                        )
                    else:
                        error = ValueError(
                            f"Rerank output missing {len(missing)} result_ref(s)"
                        )
                        store.record_failure(
                            checkpoint_key, batch_index, error
                        )
                        trajectory.controller_errors.append(str(error))
                except TypeError:
                    raise
                except Exception as error:
                    store.record_failure(
                        checkpoint_key, batch_index, error
                    )
                    trajectory.controller_errors.append(f"Rerank batch failed: {error}")

                retry_candidates: list[dict[str, Any]] = []
                for ref in expected_refs:
                    if ref in scores_by_ref:
                        continue
                    attempts_by_ref[ref] += 1
                    if attempts_by_ref[ref] >= 3:
                        scores_by_ref[ref] = 0.0
                        trajectory.controller_errors.append(
                            f"Rerank gave up on {ref} after 3 attempts; scored 0"
                        )
                    else:
                        retry_candidates.append(public_by_ref[ref])
                if retry_candidates:
                    pending.append(retry_candidates)
                batch_index += 1
        finally:
            store.close()

        for ref in all_refs:
            if ref not in scores_by_ref:
                scores_by_ref[ref] = 0.0
                trajectory.controller_errors.append(
                    f"Rerank missing final score for {ref}; scored 0"
                )

        retrieval_scores = [float(candidate.score) for candidate in rows]
        normalized = _minmax_scores(retrieval_scores)
        for public, candidate, raw_retrieval_score, retrieval_score in zip(
            public_rows, rows, retrieval_scores, normalized, strict=True
        ):
            result_ref = str(public["result_ref"])
            llm_score = scores_by_ref[result_ref]
            fused_score = (
                self.rerank_llm_weight * llm_score
                + self.rerank_retrieval_weight * retrieval_score
            )
            candidate.attributes["rerank_llm_score"] = llm_score
            candidate.attributes["rerank_original_retrieval_score"] = (
                raw_retrieval_score
            )
            candidate.attributes["rerank_retrieval_score"] = retrieval_score
            candidate.attributes["rerank_score"] = fused_score
            candidate.score = fused_score
        trajectory.rerank_candidates = len(rows)
        trajectory.rerank_batches = batch_index
        return sorted(rows, key=lambda candidate: (-candidate.score, candidate.key))

    def _select_final_candidates(
        self,
        reranked: Sequence[RuntimeCandidate],
        *,
        protected_keys: set[str] | None = None,
    ) -> list[RuntimeCandidate]:
        rows = [
            view
            for candidate in reranked
            if (view := _answer_candidate(candidate)) is not None
        ]
        top = list(rows[: self.final_top_k])
        protected = protected_keys or set()
        top_keys = {row.key for row in top}
        missing = [
            row for row in rows if row.key in protected and row.key not in top_keys
        ]
        if not missing:
            return top
        # Preserve rerank order for protected candidates that already score into
        # the budget. Only rescue cited candidates that fell outside it, replacing
        # the lowest-ranked unprotected tail so the top of the ranking is stable.
        replaceable = [
            index for index, row in enumerate(top) if row.key not in protected
        ]
        for row in missing:
            if not replaceable:
                break
            top[replaceable.pop()] = row
        return top

    def _check_support_delivery(
        self,
        root: dict[str, Any],
        assessments: Mapping[str, dict[str, Any]],
        refs: Mapping[str, RuntimeCandidate],
        trajectory: RetrievalTrajectory,
    ) -> set[str]:
        sufficient = [
            item for item in (root, *assessments.values()) if item["sufficient"]
        ]
        if not sufficient:
            return set()
        trajectory.support_delivery_checks += 1
        support = {ref for item in sufficient for ref in item["supporting_result_refs"]}
        missing: set[str] = set()
        for ref in support:
            if ref not in refs:
                missing.add(ref)
                continue
            view = _answer_candidate(refs[ref])
            if view is not None:
                continue
            # A chain_head without a native node payload is a test/legacy
            # candidate that cannot be projected and has no source evidence to
            # recover. Enforce delivery only when the backend can materialize
            # the cited facts; otherwise keep the pre-P0 behavior (the head is
            # simply excluded from the final context).
            if not _delivery_enforced(refs[ref]):
                continue
            missing.add(ref)
        if not missing:
            return set()
        trajectory.support_delivery_failures += 1
        for item in (root, *assessments.values()):
            if item is root or missing.intersection(item["supporting_result_refs"]):
                item["sufficient"] = False
                item["missing_facets"] = list(
                    dict.fromkeys([*item["missing_facets"], "object"])
                )
        trajectory.controller_errors.append(
            "Support not deliverable within final evidence budget: "
            + ", ".join(sorted(missing))
        )
        return missing

    def _initial_plan(
        self,
        question: str,
        execution: RetrievalExecutionContext,
        trajectory: RetrievalTrajectory,
    ) -> RetrievalPlan:
        def parse(raw: Any) -> RetrievalPlan:
            if not isinstance(raw, Mapping):
                raise ValueError("V4 initial Controller output must be an object")
            if set(raw) != {"subproblems"}:
                raise ValueError(
                    "V4 initial Controller output must contain only subproblems"
                )
            values = normalize_subproblems(
                raw.get("subproblems"),
                original_question=question,
                max_subproblems=self.max_subproblems,
            )
            for value in values:
                if not value["facets"]:
                    value["facets"] = list(
                        self.analyzer.analyze(value["query"]).required_facets
                    )
            if values:
                budgets = tuple(
                    _initial_probe_budget(
                        value["facets"],
                        self.simple_probe_budget,
                        self.complex_probe_budget,
                    )
                    for value in values
                )
                baseline_budget = max(budgets)
            else:
                baseline_budget = _initial_probe_budget(
                    self.analyzer.analyze(question).required_facets,
                    self.simple_probe_budget,
                    self.complex_probe_budget,
                )
            baseline = Probe(
                id="baseline",
                query=question,
                budget=baseline_budget,
                lane="baseline",
            )
            model_probes = tuple(
                Probe(
                    id=f"p{index}",
                    lane="model",
                    required=bool(value["required"]),
                    subproblem_id=f"s{index}",
                    query=value["query"],
                    must_terms=tuple(value["must_terms"]),
                    should_terms=tuple(value["should_terms"]),
                    entities=tuple(value["entities"]),
                    facets=tuple(value["facets"]),
                    time_constraint=value["time_constraint"],
                    budget=budgets[index - 1],
                )
                for index, value in enumerate(values, 1)
            )
            plan = RetrievalPlan(
                backend_id=self.backend.capabilities.backend_id,
                schema_version=self.backend.capabilities.plan_schema_version,
                probes=(baseline, *model_probes),
                final_top_k=self.final_top_k,
            )
            validate_plan(plan, self.backend.capabilities)
            return plan

        return self._retry(
            lambda: self.controller.initial_plan(question, execution.user_context),
            parse,
            trajectory,
            stage="controller_initial",
            unit_id=question,
        )

    def _followup_decision(
        self,
        question: str,
        subproblems: Sequence[Mapping[str, Any]],
        observations: Sequence[Mapping[str, Any]],
        round_index: int,
        execution: RetrievalExecutionContext,
        catalog: (Sequence[RetrievalAction] | Mapping[str, Sequence[RetrievalAction]]),
        trajectory: RetrievalTrajectory,
    ) -> Mapping[str, Any]:
        expected_ids = {str(item["subproblem_id"]) for item in subproblems}
        target_ids = {"root", *expected_ids}
        if isinstance(catalog, Mapping):
            by_target = {
                target_id: {
                    action.action_id: action for action in catalog.get(target_id, ())
                }
                for target_id in target_ids
            }
        else:
            common = {action.action_id: action for action in catalog}
            by_target = {target_id: common for target_id in target_ids}
        required_ids = {
            str(item["subproblem_id"])
            for item in subproblems
            if item.get("required", True)
        }
        result_refs = {str(item["result_ref"]) for item in observations}
        refs_by_subproblem = {
            subproblem_id: {
                str(item["result_ref"])
                for item in observations
                if subproblem_id in item.get("subproblem_ids", ())
            }
            for subproblem_id in expected_ids
        }

        def parse(raw: Any) -> Mapping[str, Any]:
            if not isinstance(raw, Mapping):
                raise ValueError("V4 follow-up Controller output must be an object")
            decision = dict(raw)
            required = {"root_assessment", "assessments", "target_ids"}
            if set(decision) != required:
                raise ValueError("V4 Controller output uses removed protocol fields")
            if _contains_forbidden_controller_value(decision):
                raise ValueError("Controller output contains an internal identifier")
            root = self._parse_assessment(
                decision.get("root_assessment"),
                expected_fields={
                    "sufficient",
                    "missing_facets",
                    "supporting_result_refs",
                    "followup_probe",
                    "action_ids",
                },
                valid_refs=result_refs,
                question=question,
                round_index=round_index,
                actions_by_id=by_target["root"],
                trajectory=trajectory,
                label="root",
            )
            assessments = decision.get("assessments")
            if not isinstance(assessments, list) or len(assessments) != len(
                expected_ids
            ):
                raise ValueError("V4 Controller must assess every subproblem")
            parsed: list[dict[str, Any]] = []
            seen_subproblems: set[str] = set()
            fields = {
                "subproblem_id",
                "sufficient",
                "missing_facets",
                "supporting_result_refs",
                "followup_probe",
                "action_ids",
            }
            for raw_assessment in assessments:
                if not isinstance(raw_assessment, Mapping):
                    raise ValueError("V4 Controller assessment must be an object")
                assessment = dict(raw_assessment)
                assessment.setdefault("supporting_result_refs", [])
                assessment.setdefault("followup_probe", None)
                assessment.setdefault("action_ids", [])
                if set(assessment) != fields:
                    raise ValueError("V4 Controller assessment uses invalid fields")
                subproblem_id = str(assessment.get("subproblem_id") or "")
                if (
                    subproblem_id not in expected_ids
                    or subproblem_id in seen_subproblems
                ):
                    raise ValueError(
                        "V4 Controller assessment has invalid subproblem_id"
                    )
                seen_subproblems.add(subproblem_id)
                parsed.append(
                    self._parse_assessment(
                        assessment,
                        expected_fields=fields,
                        valid_refs=refs_by_subproblem[subproblem_id],
                        question=question,
                        round_index=round_index,
                        actions_by_id=by_target[subproblem_id],
                        trajectory=trajectory,
                        label=subproblem_id,
                    )
                )
            targets = decision.get("target_ids")
            if (
                not isinstance(targets, list)
                or not targets
                or any(not isinstance(item, str) for item in targets)
                or len(targets) != len(set(targets))
            ):
                raise ValueError("V4 Controller target_ids is invalid")
            by_subproblem = {
                assessment["subproblem_id"]: assessment for assessment in parsed
            }
            actionable = {
                *(("root",) if root["sufficient"] is False else ()),
                *(
                    target_id
                    for target_id in required_ids
                    if not by_subproblem[target_id]["sufficient"]
                ),
            }
            if actionable:
                if any(target_id not in actionable for target_id in targets):
                    raise ValueError(
                        "V4 Controller target_ids must select insufficient targets"
                    )
            elif targets != ["root"]:
                raise ValueError(
                    "V4 Controller target_ids must use root after sufficiency"
                )
            decision["root_assessment"] = root
            decision["assessments"] = parsed
            self._cap_selected_actions_for_round(
                decision,
                targets,
                round_index=round_index,
                trajectory=trajectory,
            )
            return decision

        return self._retry(
            lambda: self.controller.assess(
                question,
                tuple(subproblems),
                tuple(observations),
                round_index,
                execution.user_context,
            ),
            parse,
            trajectory,
            stage="controller_followup",
            unit_id=f"round-{round_index}",
        )

    def _select_expansion_sources(
        self,
        question: str,
        observations: Sequence[Mapping[str, Any]],
        round_index: int,
        execution: RetrievalExecutionContext,
        trajectory: RetrievalTrajectory,
    ) -> tuple[str, ...]:
        available_refs = tuple(str(item["result_ref"]) for item in observations)
        selector = getattr(self.controller, "select_expansion_sources", None)
        if not callable(selector):
            return available_refs[:_MAX_EXPANSION_SOURCES]
        available = set(available_refs)

        def parse(raw: Any) -> tuple[str, ...]:
            if not isinstance(raw, Mapping) or set(raw) != {"source_refs"}:
                raise ValueError(
                    "V4 expansion source output must contain only source_refs"
                )
            source_refs = raw["source_refs"]
            if (
                not isinstance(source_refs, list)
                or any(not isinstance(item, str) for item in source_refs)
                or len(source_refs) != len(set(source_refs))
            ):
                raise ValueError(
                    "V4 expansion source_refs must be a unique string list"
                )
            if len(source_refs) > _MAX_EXPANSION_SOURCES:
                raise ValueError(
                    "V4 expansion source selection allows at most five refs"
                )
            if any(item not in available for item in source_refs):
                raise ValueError(
                    "V4 expansion source selection contains an unavailable result_ref"
                )
            return tuple(source_refs)

        return self._retry(
            lambda: selector(
                question,
                tuple(observations),
                round_index,
                execution.user_context,
            ),
            parse,
            trajectory,
            stage="controller_expansion_sources",
            unit_id=f"round-{round_index}",
        )

    def _cap_selected_actions_for_round(
        self,
        decision: Mapping[str, Any],
        target_ids: Sequence[str],
        *,
        round_index: int,
        trajectory: RetrievalTrajectory,
    ) -> None:
        root = decision["root_assessment"]
        by_target = {
            "root": root,
            **{str(item["subproblem_id"]): item for item in decision["assessments"]},
        }
        target_set = set(target_ids)
        for target_id, assessment in by_target.items():
            if target_id not in target_set:
                assessment["selected_actions"] = []

        selected_ids: set[str] = set()
        selected_count = 0
        for target_id in target_ids:
            assessment = by_target[target_id]
            kept: list[RetrievalAction] = []
            for action in assessment.get("selected_actions", ()):
                if action.action_id in selected_ids:
                    trajectory.controller_errors.append(
                        "Dropped duplicate cross-target action ID in round "
                        f"{round_index}: {action.action_id}"
                    )
                    continue
                if selected_count >= self.max_actions_per_round:
                    trajectory.controller_errors.append(
                        "Dropped action beyond global round action limit in round "
                        f"{round_index}: {action.action_id}"
                    )
                    continue
                selected_ids.add(action.action_id)
                selected_count += 1
                kept.append(action)
            assessment["selected_actions"] = kept

    def _parse_assessment(
        self,
        raw: Any,
        *,
        expected_fields: set[str],
        valid_refs: set[str],
        question: str,
        round_index: int,
        actions_by_id: Mapping[str, RetrievalAction],
        trajectory: RetrievalTrajectory,
        label: str,
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ValueError(f"V4 Controller {label} assessment must be an object")
        assessment = dict(raw)
        assessment.setdefault("supporting_result_refs", [])
        assessment.setdefault("followup_probe", None)
        assessment.setdefault("action_ids", [])
        if set(assessment) != expected_fields:
            raise ValueError(f"V4 Controller {label} assessment uses invalid fields")
        if not isinstance(assessment.get("sufficient"), bool):
            raise ValueError("V4 Controller sufficient must be boolean")
        missing_facets = assessment.get("missing_facets")
        if not isinstance(missing_facets, list) or any(
            not isinstance(item, str) or item not in FACETS for item in missing_facets
        ):
            raise ValueError("V4 Controller missing_facets is invalid")
        assessment["missing_facets"] = list(dict.fromkeys(missing_facets))
        supporting_refs = assessment.get("supporting_result_refs")
        if not isinstance(supporting_refs, list) or any(
            not isinstance(item, str) for item in supporting_refs
        ):
            raise ValueError("V4 Controller supporting_result_refs is invalid")
        filtered_refs = list(
            dict.fromkeys(item for item in supporting_refs if item in valid_refs)
        )
        if len(filtered_refs) != len(dict.fromkeys(supporting_refs)):
            trajectory.controller_errors.append(
                f"Dropped invalid supporting result refs for {label} in round {round_index}"
            )
        assessment["supporting_result_refs"] = filtered_refs
        if assessment["sufficient"] and not filtered_refs:
            assessment["sufficient"] = False
            trajectory.controller_errors.append(
                f"Downgraded unsupported sufficient assessment for {label}"
            )

        followup = assessment.get("followup_probe")
        if followup is not None:
            if not isinstance(followup, Mapping):
                raise ValueError("V4 Controller followup_probe is invalid")
            followup = dict(followup)
            probe_query = str(followup.get("query") or question).strip()
            hints = derive_probe_hints(probe_query)
            followup.setdefault("must_terms", hints["must_terms"])
            followup.setdefault("should_terms", hints["should_terms"])
            followup.setdefault("entities", hints["entities"])
            followup.setdefault("time_constraint", None)
            followup.setdefault(
                "facets", list(self.analyzer.analyze(probe_query).required_facets)
            )
            followup = normalize_subproblem(
                followup, original_question=question, include_required=False
            )
            assessment["followup_probe"] = followup

        action_ids = assessment.get("action_ids")
        if not isinstance(action_ids, list) or any(
            not isinstance(item, str) for item in action_ids
        ):
            raise ValueError("V4 Controller action_ids must be a string list")
        selected: list[RetrievalAction] = []
        seen_actions: set[str] = set()
        selected_by_source: dict[str, int] = {}
        for action_id in action_ids:
            if action_id in seen_actions:
                trajectory.controller_errors.append(
                    f"Dropped duplicate action ID in round {round_index}: {action_id}"
                )
                continue
            seen_actions.add(action_id)
            action = actions_by_id.get(action_id)
            if action is None:
                trajectory.controller_errors.append(
                    f"Dropped unavailable action ID in round {round_index}: {action_id}"
                )
                continue
            source_count = selected_by_source.get(action.source_ref, 0)
            if source_count >= self.max_actions_per_source:
                trajectory.controller_errors.append(
                    f"Dropped over-limit action ID in round {round_index}: "
                    f"{action_id} (source {action.source_ref} already has "
                    f"{source_count} selected actions)"
                )
                continue
            selected_by_source[action.source_ref] = source_count + 1
            selected.append(action)
        assessment["selected_actions"] = selected
        return assessment

    def _action_catalog(
        self,
        question: str,
        observations: Sequence[Mapping[str, Any]],
    ) -> tuple[RetrievalAction, ...]:
        getter = getattr(self.controller, "action_catalog", None)
        if callable(getter):
            return tuple(getter(question, observations))
        return self.analyzer.action_catalog(
            self.analyzer.analyze(question), observations
        )

    def _fresh_probe_for_target(
        self,
        value: Any,
        round_index: int,
        target_id: str,
        state: _TargetState,
        trajectory: RetrievalTrajectory,
        *,
        missing_facets: Sequence[str] = (),
        allow_reformulation: bool = False,
    ) -> Probe | None:
        normalized = dict(value) if isinstance(value, Mapping) else None
        duplicate_exhausted = False
        if normalized is not None:
            query_text = _normalized_query(str(normalized.get("query") or ""))
            if query_text and query_text in state.attempted_query_texts:
                state.duplicate_query_events += 1
                trajectory.controller_errors.append(
                    f"Reformulated duplicate followup_probe query text for "
                    f"{target_id} in round {round_index} (duplicate event "
                    f"{state.duplicate_query_events}): {normalized['query']}"
                )
                if (
                    allow_reformulation
                    and state.last_probe is not None
                    and state.reformulation_stage < _DUPLICATE_QUERY_REFORMULATIONS
                ):
                    normalized = _diversified_probe_value(
                        normalized,
                        state.last_probe,
                        missing_facets,
                        stage=state.duplicate_query_events,
                    )
                    state.reformulation_stage = max(
                        state.reformulation_stage,
                        state.duplicate_query_events,
                    )
                else:
                    normalized = None
                    duplicate_exhausted = True
                    state.exhausted = True
                    state.reformulation_exhausted = True
                    trajectory.controller_errors.append(
                        f"Exhausted duplicate-query reformulations for {target_id} "
                        f"in round {round_index}"
                    )
        if normalized is not None and state.zero_growth_rounds and state.last_probe:
            normalized = _preserve_failed_probe_constraints(
                normalized, state.last_probe
            )
        if normalized is not None:
            fingerprint = _probe_fingerprint(normalized)
            if fingerprint in state.attempted_probe_fingerprints:
                trajectory.controller_errors.append(
                    f"Dropped duplicate followup_probe for {target_id} "
                    f"in round {round_index}: {normalized['query']}"
                )
                normalized = None

        if (
            normalized is None
            and not duplicate_exhausted
            and allow_reformulation
            and state.zero_growth_rounds
            and state.last_probe
            and state.reformulation_stage < _DUPLICATE_QUERY_REFORMULATIONS
        ):
            normalized = _relaxed_probe_value(state.last_probe)
            if normalized is not None:
                fingerprint = _probe_fingerprint(normalized)
                if fingerprint in state.attempted_probe_fingerprints:
                    normalized = None
                else:
                    state.reformulation_stage = _DUPLICATE_QUERY_REFORMULATIONS

        if normalized is None:
            return None
        probe = Probe(
            id=f"fresh_{round_index}_{target_id}",
            lane="model",
            required=True,
            subproblem_id="" if target_id == "root" else target_id,
            query=str(normalized["query"]),
            must_terms=tuple(normalized["must_terms"]),
            should_terms=tuple(normalized["should_terms"]),
            entities=tuple(normalized["entities"]),
            facets=tuple(normalized.get("facets") or ()),
            time_constraint=normalized["time_constraint"],
            budget=self.followup_probe_budget,
        )
        state.attempted_probe_fingerprints.add(_probe_fingerprint(probe))
        state.attempted_query_texts.add(_normalized_query(probe.query))
        state.last_probe = probe
        return probe

    def _rule_expansion_for_target(
        self,
        target_id: str,
        state: _TargetState,
        catalog: Sequence[RetrievalAction],
        refs: Mapping[str, RuntimeCandidate],
        missing_facets: Sequence[str],
        round_index: int,
    ) -> Expansion | None:
        """Prefer target-local paths, then explicitly try cross-target bridges."""
        valid_catalog = tuple(action for action in catalog if action.source_ref in refs)
        if target_id == "root":
            stages = (valid_catalog,)
        else:
            target_local = tuple(
                action
                for action in valid_catalog
                if target_id in refs[action.source_ref].subproblem_ids
            )
            cross_target_bridges = tuple(
                action
                for action in valid_catalog
                if target_id not in refs[action.source_ref].subproblem_ids
            )
            stages = (target_local, cross_target_bridges)

        for stage in stages:
            # Recover exact source wording before walking broad causal/topic
            # links when a name, relation, time or event-set facet is missing.
            prefer_source = bool(set(missing_facets) & FACETS)
            ordered = sorted(
                stage,
                key=lambda action: (
                    0
                    if prefer_source and action.action == "SOURCE_EVIDENCE_LOOKUP"
                    else 1
                ),
            )
            for action in ordered:
                path = _expansion_path_key(
                    target_id,
                    refs[action.source_ref].key,
                    action.action,
                    action.direction,
                    action.constraints,
                )
                if path in state.attempted_expansion_paths:
                    continue
                expansions = self._parse_actions(
                    (action,),
                    round_index,
                    refs,
                    missing_facets,
                    target_id,
                    state.attempted_expansion_paths,
                )
                if expansions:
                    return expansions[0]
        return None

    def _parse_actions(
        self,
        actions: Sequence[RetrievalAction],
        round_index: int,
        refs: Mapping[str, RuntimeCandidate],
        missing_facets: Sequence[str],
        subproblem_id: str,
        attempted_paths: set[tuple[str, ...]],
    ) -> list[Expansion]:
        output: list[Expansion] = []
        missing_intent = _standard_expansion_intent(missing_facets)
        for index, action in enumerate(actions, 1):
            if action.source_ref not in refs:
                continue
            if (
                _expansion_path_key(
                    subproblem_id,
                    refs[action.source_ref].key,
                    action.action,
                    action.direction,
                    action.constraints,
                )
                in attempted_paths
            ):
                continue
            query = str(action.constraints.get("query") or "").strip()
            intent = f"question: {query}; {missing_intent}" if query else missing_intent
            expansion = Expansion(
                id=f"x{round_index}_{subproblem_id}_{index}",
                source=action.source_ref,
                edge_type=action.action,
                direction=action.direction,
                hops=1,
                budget=self.expansion_budget_per_anchor,
                subproblem_id=subproblem_id,
                selector=Selector(attributes=dict(action.constraints)),
                evidence_facet=intent,
            )
            validate_runtime_expansion(expansion, self.backend.capabilities)
            output.append(expansion)
        return output

    def _retry(
        self,
        invoke: Callable[[], Any],
        parse: Callable[[Any], Any],
        trajectory: RetrievalTrajectory,
        *,
        stage: str,
        unit_id: str,
    ) -> Any:
        if self.strict_v4:

            def operation() -> Any:
                trajectory.controller_calls += 1
                return parse(invoke())

            return retry_v4_call(
                operation,
                policy=self.failure_policy,
                context=V4OperationContext(
                    stage=stage,
                    unit_id=unit_id,
                    checkpoint_key=f"retrieval:{stage}:{unit_id}",
                ),
                error_type=V4RetrievalError,
                validation_errors=(ValueError, TypeError, KeyError),
                on_failure=lambda attempt, error: trajectory.controller_errors.append(
                    str(error)
                ),
            )
        last_error: Exception | None = None
        for _ in range(self.controller_max_retries + 1):
            trajectory.controller_calls += 1
            try:
                return parse(invoke())
            except Exception as error:
                last_error = error
                trajectory.controller_errors.append(str(error))
        assert last_error is not None
        raise last_error


def _standard_expansion_intent(missing_facets: Sequence[str]) -> str:
    values = [item for item in dict.fromkeys(missing_facets) if item in FACETS]
    return "missing facets: " + ", ".join(values or ["evidence"])


def _expansion_path_key(
    target_id: str,
    candidate_key: str,
    action: str,
    direction: str,
    constraints: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    base = (target_id, candidate_key, action, direction)
    if not constraints:
        return base
    constraint_key = repr(_fingerprint_value(dict(constraints)))
    return (*base, constraint_key)


def _probe_fingerprint(value: Probe | Mapping[str, Any]) -> tuple[Any, ...]:
    if isinstance(value, Probe):
        query = value.query
        must_terms = value.must_terms
        should_terms = value.should_terms
        entities = value.entities
        time_constraint = value.time_constraint
    else:
        query = str(value.get("query") or "")
        must_terms = value.get("must_terms") or ()
        should_terms = value.get("should_terms") or ()
        entities = value.get("entities") or ()
        time_constraint = value.get("time_constraint")
    normalized_time = tuple(
        sorted(
            (str(key), _fingerprint_value(item))
            for key, item in (time_constraint or {}).items()
        )
    )
    return (
        _normalized_query(query),
        tuple(_normalized_query(item) for item in must_terms),
        tuple(_normalized_query(item) for item in should_terms),
        tuple(_normalized_query(item) for item in entities),
        normalized_time,
    )


def _fingerprint_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _fingerprint_value(item)) for key, item in value.items())
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_fingerprint_value(item) for item in value)
    return value


def _initial_target_states(probes: Sequence[Probe]) -> dict[str, _TargetState]:
    output: dict[str, _TargetState] = {}
    for probe in probes:
        target_id = probe.subproblem_id or "root"
        output[target_id] = _TargetState(
            attempted_probe_fingerprints={_probe_fingerprint(probe)},
            attempted_query_texts={_normalized_query(probe.query)},
            last_probe=probe,
        )
    return output


def _preserve_failed_probe_constraints(
    value: Mapping[str, Any], previous: Probe
) -> dict[str, Any]:
    output = dict(value)
    output["entities"] = list(previous.entities)
    if previous.time_constraint and previous.time_constraint.get("hard") is True:
        output["time_constraint"] = dict(previous.time_constraint)
    return output


def _diversified_probe_value(
    value: Mapping[str, Any],
    previous: Probe,
    missing_facets: Sequence[str],
    *,
    stage: int,
) -> dict[str, Any]:
    """Turn an exact repeated probe into a bounded facet-oriented alternative."""
    output = _preserve_failed_probe_constraints(value, previous)
    entities = list(
        dict.fromkeys(
            str(item).strip()
            for item in output.get("entities", ())
            if str(item).strip()
        )
    )
    facets = list(
        dict.fromkeys(
            str(item).strip()
            for item in (*missing_facets, *(output.get("facets") or ()))
            if str(item).strip()
        )
    )
    raw_query = str(output.get("query") or previous.query).strip()
    facet_text = " ".join(facets) or "related"
    suffix = (
        f"{facet_text} supporting evidence"
        if stage == 1
        else f"{facet_text} alternative cause outcome time evidence"
    )
    entity_text = " ".join(entities)
    output["query"] = " ".join(
        item for item in (entity_text, raw_query, suffix) if item
    )

    entity_markers = {_normalized_query(item) for item in entities}
    retained_must_terms: list[str] = []
    relaxed_terms: list[str] = []
    for term in output.get("must_terms", ()):
        marker = _normalized_query(str(term))
        if marker and any(
            marker == entity or marker in entity for entity in entity_markers
        ):
            retained_must_terms.append(str(term))
        else:
            relaxed_terms.append(str(term))
    output["must_terms"] = retained_must_terms
    output["should_terms"] = list(
        dict.fromkeys(
            str(item)
            for item in (*output.get("should_terms", ()), *relaxed_terms, *facets)
            if str(item).strip()
        )
    )
    output["entities"] = entities
    output["facets"] = facets
    output.setdefault("time_constraint", None)
    return output


def _relaxed_probe_value(previous: Probe) -> dict[str, Any] | None:
    entity_markers = {_normalized_query(item) for item in previous.entities}
    retained: list[str] = []
    relaxed: list[str] = []
    for term in previous.must_terms:
        marker = _normalized_query(term)
        if any(marker == entity or marker in entity for entity in entity_markers):
            retained.append(term)
        else:
            relaxed.append(term)
    if not relaxed:
        return None
    should_terms = list(dict.fromkeys((*previous.should_terms, *relaxed)))
    return {
        "query": previous.query,
        "must_terms": retained,
        "should_terms": should_terms,
        "entities": list(previous.entities),
        "facets": list(previous.facets),
        "time_constraint": (
            dict(previous.time_constraint)
            if previous.time_constraint is not None
            else None
        ),
    }


def _has_reformulation_opportunity(
    state: _TargetState, *, allow_reformulation: bool
) -> bool:
    return bool(
        allow_reformulation
        and not state.reformulation_exhausted
        and state.zero_growth_rounds
        and state.last_probe is not None
        and state.reformulation_stage < _DUPLICATE_QUERY_REFORMULATIONS
        and _relaxed_probe_value(state.last_probe) is not None
    )


def _available_expansion_paths(
    target_id: str,
    catalog: Sequence[RetrievalAction],
    refs: Mapping[str, RuntimeCandidate],
) -> set[tuple[str, ...]]:
    return {
        _expansion_path_key(
            target_id,
            refs[action.source_ref].key,
            action.action,
            action.direction,
            action.constraints,
        )
        for action in catalog
        if action.source_ref in refs
    }


def _filter_action_catalogs_by_target(
    catalog: Sequence[RetrievalAction],
    refs: Mapping[str, RuntimeCandidate],
    states: Mapping[str, _TargetState],
) -> dict[str, tuple[RetrievalAction, ...]]:
    return {
        target_id: tuple(
            action
            for action in catalog
            if action.source_ref in refs
            and _expansion_path_key(
                target_id,
                refs[action.source_ref].key,
                action.action,
                action.direction,
                action.constraints,
            )
            not in state.attempted_expansion_paths
        )
        for target_id, state in states.items()
    }


def _prepare_action_catalogs_by_target(
    catalog: Sequence[RetrievalAction],
    refs: Mapping[str, RuntimeCandidate],
    states: Mapping[str, _TargetState],
) -> dict[str, tuple[RetrievalAction, ...]]:
    filtered = _filter_action_catalogs_by_target(catalog, refs, states)
    return {
        target_id: tuple(
            replace(
                action,
                action_id="a_"
                + _stable_digest(
                    {
                        "target_id": target_id,
                        "candidate_key": refs[action.source_ref].key,
                        "action": action.action,
                        "direction": action.direction,
                        "constraints": dict(action.constraints),
                    }
                )[:12],
            )
            for action in actions
        )
        for target_id, actions in filtered.items()
    }


def _refresh_exhausted_targets(
    states: Mapping[str, _TargetState],
    catalog: Sequence[RetrievalAction],
    refs: Mapping[str, RuntimeCandidate],
) -> None:
    for target_id, state in states.items():
        if not state.exhausted or state.reformulation_exhausted:
            continue
        untried = (
            _available_expansion_paths(target_id, catalog, refs)
            - state.attempted_expansion_paths
        )
        if untried - set(state.exhausted_action_snapshot):
            state.exhausted = False
            state.exhausted_action_snapshot = frozenset()


def _cited_candidate_keys(
    decision: Mapping[str, Any], refs: Mapping[str, RuntimeCandidate]
) -> set[str]:
    assessments = [decision.get("root_assessment"), *decision.get("assessments", ())]
    return {
        refs[result_ref].key
        for assessment in assessments
        if isinstance(assessment, Mapping)
        for result_ref in assessment.get("supporting_result_refs", ())
        if result_ref in refs
    }


def _reinforce_cited_expansion_anchors(
    candidates: Sequence[RuntimeCandidate],
    expansions: Sequence[_ExpansionRecord],
    cited_keys: set[str],
    rewarded_expansions: set[str],
) -> None:
    by_key = {candidate.key: candidate for candidate in candidates}
    for expansion in expansions:
        if (
            expansion.expansion_id in rewarded_expansions
            or not expansion.result_keys.intersection(cited_keys)
        ):
            continue
        anchor = by_key.get(expansion.anchor_key)
        if anchor is None:
            continue
        anchor.anchor_reinforcement = min(0.06, anchor.anchor_reinforcement + 0.03)
        rewarded_expansions.add(expansion.expansion_id)


def _select_controller_evidence(
    rows: Sequence[RuntimeCandidate],
    subproblems: Sequence[Mapping[str, Any]],
    previous_support_keys: set[str],
    limit: int = 32,
    protected_keys: set[str] | None = None,
) -> list[RuntimeCandidate]:
    ranked = sorted(rows, key=lambda item: (-item.score, item.key))
    selected: list[RuntimeCandidate] = []
    selected_keys: set[str] = set()

    def add(candidate: RuntimeCandidate) -> None:
        if candidate.key not in selected_keys and len(selected) < limit:
            selected.append(candidate)
            selected_keys.add(candidate.key)

    for subproblem in subproblems:
        if not subproblem.get("required", True):
            continue
        target_id = str(subproblem["subproblem_id"])
        for candidate in (item for item in ranked if target_id in item.subproblem_ids):
            add(candidate)
            if sum(target_id in item.subproblem_ids for item in selected) >= 2:
                break
    retained_previous = 0
    for candidate in ranked:
        if candidate.key not in previous_support_keys:
            continue
        before = len(selected)
        add(candidate)
        retained_previous += int(len(selected) > before)
        if retained_previous == 8:
            break
    for candidate in ranked:
        if candidate.key in (protected_keys or ()):
            add(candidate)
    for candidate in ranked:
        add(candidate)
        if len(selected) == limit:
            break
    return selected


def _initial_probe_budget(
    facets: Sequence[str], simple_budget: int = 8, complex_budget: int = 12
) -> int:
    values = tuple(dict.fromkeys(item for item in facets if item in FACETS))
    if len(values) >= 2 or set(values).intersection(
        {
            "cause",
            "outcome",
            "conflict",
            "count",
        }
    ):
        return int(complex_budget)
    return int(simple_budget)


def _subproblem_statuses(
    subproblems: Sequence[Mapping[str, Any]],
    assessments: Mapping[str, Mapping[str, Any]],
    attempted: set[str],
    exhausted: set[str],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "subproblem_id": str(item["subproblem_id"]),
            "required": bool(item.get("required", True)),
            "sufficient": bool(assessments[str(item["subproblem_id"])]["sufficient"]),
            "attempted": str(item["subproblem_id"]) in attempted,
            "exhausted": str(item["subproblem_id"]) in exhausted,
            "missing_facets": list(
                assessments[str(item["subproblem_id"])]["missing_facets"]
            ),
            "supporting_result_refs": list(
                assessments[str(item["subproblem_id"])]["supporting_result_refs"]
            ),
        }
        for item in subproblems
    )


def _root_status(
    assessment: Mapping[str, Any], attempted: set[str], exhausted: set[str]
) -> Mapping[str, Any]:
    return {
        "target_id": "root",
        "sufficient": bool(assessment["sufficient"]),
        "attempted": "root" in attempted,
        "exhausted": "root" in exhausted,
        "missing_facets": list(assessment["missing_facets"]),
        "supporting_result_refs": list(assessment["supporting_result_refs"]),
    }


def _root_and_required_sufficient(
    root: Mapping[str, Any],
    subproblems: Sequence[Mapping[str, Any]],
    assessments: Mapping[str, Mapping[str, Any]],
) -> bool:
    return bool(root["sufficient"]) and all(
        not item.get("required", True)
        or assessments[str(item["subproblem_id"])]["sufficient"]
        for item in subproblems
    )


def _actionable_targets(
    root: Mapping[str, Any],
    subproblems: Sequence[Mapping[str, Any]],
    assessments: Mapping[str, Mapping[str, Any]],
    exhausted: set[str],
) -> list[str]:
    output: list[str] = []
    if not root["sufficient"] and "root" not in exhausted:
        output.append("root")
    output.extend(
        subproblem_id
        for item in subproblems
        if item.get("required", True)
        and (subproblem_id := str(item["subproblem_id"])) not in exhausted
        and not assessments[subproblem_id]["sufficient"]
    )
    return output


def _coverage_select_v2(
    rows: Sequence[RuntimeCandidate],
    subproblems: Sequence[Mapping[str, Any]],
    limit: int,
    *,
    protected_keys: set[str] | None = None,
) -> list[RuntimeCandidate]:
    """Reserve required-hop evidence while preserving the global rerank order."""
    ranked = _deduplicate(
        [
            view
            for candidate in rows
            if (view := _answer_candidate(candidate)) is not None
        ]
    )
    required_ids = [
        str(item["subproblem_id"]) for item in subproblems if item.get("required", True)
    ]
    if not required_ids:
        return ranked[:limit]

    selected_keys: set[str] = set()

    def reserve(candidate: RuntimeCandidate) -> bool:
        if candidate.key in selected_keys or len(selected_keys) >= limit:
            return False
        selected_keys.add(candidate.key)
        return True

    # Guarantee one result for every required hop before preserving additional
    # Controller-cited support. A second pass raises the quota to two when the
    # fixed final budget has room.
    for subproblem_id in required_ids:
        candidate = next(
            (
                row
                for row in ranked
                if subproblem_id in row.subproblem_ids and row.key not in selected_keys
            ),
            None,
        )
        if candidate is not None:
            reserve(candidate)

    protected = protected_keys or set()
    for candidate in ranked:
        if candidate.key in protected:
            reserve(candidate)

    for subproblem_id in required_ids:
        represented = sum(
            subproblem_id in row.subproblem_ids and row.key in selected_keys
            for row in ranked
        )
        for candidate in ranked:
            if represented >= 2 or len(selected_keys) >= limit:
                break
            if subproblem_id in candidate.subproblem_ids and reserve(candidate):
                represented += 1

    for candidate in ranked:
        reserve(candidate)
        if len(selected_keys) >= limit:
            break
    return [row for row in ranked if row.key in selected_keys]


def _execute_probe_batch(
    backend: RetrievalBackend,
    execution: RetrievalExecutionContext,
    probes: Sequence[Probe],
) -> tuple[list[RuntimeCandidate], dict[str, list[RuntimeCandidate]]]:
    rows = list(backend.global_search(execution, probes))
    by_source: dict[str, list[RuntimeCandidate]] = {probe.id: [] for probe in probes}
    probes_by_id = {probe.id: probe for probe in probes}
    for row in rows:
        for source in row.sources:
            if source in by_source:
                by_source[source].append(row)
                probe = probes_by_id[source]
                if probe.subproblem_id:
                    row.subproblem_ids.add(probe.subproblem_id)
    if len(probes) == 1 and not by_source[probes[0].id]:
        by_source[probes[0].id] = rows
        for row in rows:
            row.sources.add(probes[0].id)
            if probes[0].subproblem_id:
                row.subproblem_ids.add(probes[0].subproblem_id)
    return rows, by_source


def _observations(
    rows: Sequence[RuntimeCandidate],
) -> tuple[list[Mapping[str, Any]], dict[str, RuntimeCandidate]]:
    observations: list[Mapping[str, Any]] = []
    refs: dict[str, RuntimeCandidate] = {}
    for index, row in enumerate(_deduplicate(rows), 1):
        ref = f"r{index}"
        observations.append(row.public_observation(ref))
        refs[ref] = row
    return observations, refs


def _public_candidate_batch(
    rows: Sequence[RuntimeCandidate], *, ref_prefix: str
) -> list[dict[str, Any]]:
    fields = (
        "result_ref",
        "node_type",
        "summary",
        "salient_facts",
        "exact_mentions",
        "event_time_start",
        "event_time_end",
        "valid_from",
        "valid_to",
        "location",
        "entities",
        "source_steps",
        "subproblem_ids",
    )
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        observation = row.public_observation(f"{ref_prefix}{index}")
        result.append({key: observation.get(key) for key in fields})
    return result


def _semantic_slimming_items(
    candidate: RuntimeCandidate,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    source_by_ref: dict[str, dict[str, Any]] = {}
    seen_facts: set[str] = set()
    facts = candidate.attributes.get("salient_facts") or ()
    if isinstance(facts, Sequence) and not isinstance(facts, (str, bytes)):
        for raw_fact in facts:
            fact = (
                dict(raw_fact) if isinstance(raw_fact, Mapping) else {"value": raw_fact}
            )
            fingerprint = _stable_digest(
                {
                    key: fact.get(key)
                    for key in (
                        "subject",
                        "predicate",
                        "dimension",
                        "aspect",
                        "value",
                        "valid_from",
                        "valid_to",
                    )
                }
            )
            if fingerprint in seen_facts:
                continue
            seen_facts.add(fingerprint)
            item_ref = f"f{len(seen_facts)}"
            public = {
                "item_ref": item_ref,
                "kind": "fact",
                "subject": fact.get("subject"),
                "predicate": fact.get("predicate") or fact.get("key"),
                "dimension": fact.get("dimension"),
                "aspect": fact.get("aspect"),
                "value": fact.get("value"),
                "valid_from": fact.get("valid_from"),
                "valid_to": fact.get("valid_to"),
            }
            items.append(public)
            source_by_ref[item_ref] = {"kind": "fact", "value": fact}

    evidence = candidate.attributes.get("_semantic_evidence") or ()
    seen_evidence: set[str] = set()
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
        for raw_evidence in evidence:
            if not isinstance(raw_evidence, Mapping):
                continue
            source = dict(raw_evidence)
            fingerprint = _stable_digest(
                {
                    "session": source.get("session_id"),
                    "turn": source.get("turn_id"),
                    "participant": source.get("participant_id"),
                    "observed_at": source.get("observed_at"),
                    "content": source.get("content"),
                }
            )
            if fingerprint in seen_evidence:
                continue
            seen_evidence.add(fingerprint)
            item_ref = f"e{len(seen_evidence)}"
            public = {
                "item_ref": item_ref,
                "kind": "source_evidence",
                "session": source.get("session_id"),
                "turn": source.get("turn_id"),
                "speaker": source.get("role"),
                "participant": source.get("participant_id"),
                "observed_at": source.get("observed_at"),
                "content": source.get("content"),
            }
            items.append(public)
            source_by_ref[item_ref] = {"kind": "source_evidence", "value": source}
    return items, source_by_ref


def _parse_semantic_slimming_decisions(
    output: Mapping[str, Any], expected_refs: Sequence[str]
) -> set[str]:
    if not isinstance(output, Mapping) or set(output) != {"decisions"}:
        raise ValueError("Semantic slimming output must contain only decisions")
    decisions = output.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Semantic slimming decisions must be a list")
    expected = set(expected_refs)
    by_ref: dict[str, bool] = {}
    for item in decisions:
        if not isinstance(item, Mapping) or set(item) != {"item_ref", "relevant"}:
            raise ValueError(
                "Semantic slimming decisions require item_ref and relevant"
            )
        item_ref = str(item["item_ref"])
        relevant = item["relevant"]
        if item_ref not in expected or item_ref in by_ref:
            raise ValueError("Semantic slimming returned an invalid item_ref")
        if not isinstance(relevant, bool):
            raise ValueError("Semantic slimming relevant must be boolean")
        by_ref[item_ref] = relevant
    if set(by_ref) != expected:
        raise ValueError("Semantic slimming must classify every supplied item_ref")
    return {item_ref for item_ref, relevant in by_ref.items() if relevant}


def _semantic_slimming_view(
    selected: Sequence[Mapping[str, Any]], *, total_items: int
) -> _SemanticSlimmingView:
    facts: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    lines: list[str] = []
    for item in selected:
        kind = str(item.get("kind") or "")
        value = item.get("value")
        if not isinstance(value, Mapping):
            continue
        source = dict(value)
        if kind == "fact":
            facts.append(source)
            rendered_value = source.get("value")
            if isinstance(rendered_value, (dict, list)):
                rendered_value = json.dumps(rendered_value, ensure_ascii=False)
            label = str(source.get("predicate") or source.get("key") or "fact")
            subject = str(source.get("subject") or "").strip()
            lines.append(
                f"{subject + ': ' if subject else ''}{label} = {rendered_value}"
            )
        elif kind == "source_evidence":
            evidence.append(source)
            content = str(source.get("content") or "").strip()
            if content:
                speaker = str(
                    source.get("role") or source.get("participant_id") or ""
                ).strip()
                observed_at = str(source.get("observed_at") or "").strip()
                prefix = ", ".join(part for part in (speaker, observed_at) if part)
                lines.append(f"[{prefix}] {content}" if prefix else content)
    summary = "\n".join(dict.fromkeys(line for line in lines if line)).strip()
    return _SemanticSlimmingView(
        summary=summary,
        facts=tuple(facts),
        evidence=tuple(evidence),
        kept_items=len(facts) + len(evidence),
        dropped_items=max(0, total_items - len(facts) - len(evidence)),
    )


def _delivery_enforced(candidate: RuntimeCandidate) -> bool:
    """Whether P0 evidence delivery should be enforced for a cited candidate.

    Chain heads from test/legacy backends have no native node payload, so they
    cannot be projected and have no recoverable source evidence. They keep the
    pre-P0 behavior and are excluded from delivery verification.
    """
    if _answer_candidate(candidate) is not None:
        return True
    return not (
        candidate.node_type == "chain_head"
        and getattr(candidate.payload, "node", None) is None
    )


def _answer_candidate(candidate: RuntimeCandidate) -> RuntimeCandidate | None:
    """Project a selected head view without changing stored navigation nodes."""
    if candidate.node_type != "chain_head":
        return candidate
    if not candidate.attributes.get("answer_view_ready"):
        return None
    attributes = {**candidate.attributes, "materialized_from_chain_head": True}
    # A chain spans many events. Its creation/epoch date must not masquerade
    # as the event date of every selected fact. Individual fact bounds and
    # utterance timestamps remain available in the view.
    for key in ("event_time_start", "event_time_end", "valid_from", "valid_to"):
        attributes[key] = None
    observation = {**candidate.observation, "node_type": "semantic_state"}
    for key in ("event_time_start", "event_time_end", "valid_from", "valid_to"):
        observation[key] = None
    payload = candidate.payload
    node = getattr(payload, "node", None)
    if node is not None:
        metadata = dict(node.metadata)
        for key in (
            "absolute_time_start",
            "absolute_time_end",
            "time_anchor",
            "raw_time_expression",
        ):
            metadata.pop(key, None)
        metadata["materialized_from_chain_head"] = True
        node = replace(
            node,
            node_type="semantic_state",
            title="Retrieved facts and source evidence",
            event_time_start=None,
            event_time_end=None,
            valid_from=None,
            valid_to=None,
            observed_at=None,
            time_precision=None,
            metadata=metadata,
        )
        payload = replace(payload, node=node)
    return replace(
        candidate,
        node_type="semantic_state",
        attributes=attributes,
        observation=observation,
        payload=payload,
    )


def _apply_semantic_slimming_view(
    candidate: RuntimeCandidate, view: _SemanticSlimmingView
) -> RuntimeCandidate:
    facts = [dict(item) for item in view.facts]
    evidence = [dict(item) for item in view.evidence]
    attributes = dict(candidate.attributes)
    attributes.update(
        {
            "summary": view.summary,
            "salient_facts": facts,
            "_semantic_evidence": evidence,
            "semantic_slimming_applied": True,
            "semantic_slimming_kept_items": view.kept_items,
            "semantic_slimming_dropped_items": view.dropped_items,
        }
    )
    if candidate.node_type == "chain_head":
        attributes["answer_view_ready"] = bool(view.summary)
    observation = dict(candidate.observation)
    observation.update(
        {
            "summary": view.summary,
            "salient_facts": facts,
            "semantic_slimming_applied": True,
        }
    )
    payload = candidate.payload
    native_node = getattr(payload, "node", None)
    if native_node is not None:
        try:
            metadata = dict(getattr(native_node, "metadata", {}) or {})
            metadata.update(
                {
                    "state_summary": view.summary,
                    "salient_facts": facts,
                    "semantic_slimming_applied": True,
                    "semantic_slimming_kept_items": view.kept_items,
                    "semantic_slimming_dropped_items": view.dropped_items,
                }
            )
            slimmed_node = replace(
                native_node,
                summary=view.summary,
                text=view.summary,
                facts=facts,
                evidence_refs=evidence,
                metadata=metadata,
            )
            payload = replace(payload, node=slimmed_node)
        except (TypeError, ValueError):
            payload = candidate.payload
    return replace(
        candidate,
        attributes=attributes,
        observation=observation,
        payload=payload,
    )


def _parse_partial_rerank_scores(
    output: Mapping[str, Any], expected_refs: Sequence[str]
) -> dict[str, float]:
    """Collect only valid, in-batch scores; ignore malformed/extra items.

    The controller is still responsible for its strict protocol checks when
    ``allow_partial=False``. This runtime helper is intentionally tolerant so
    a few malformed items can be retried instead of failing the question.
    """
    if not isinstance(output, Mapping):
        return {}
    scores = output.get("scores")
    if not isinstance(scores, list):
        return {}
    expected = set(expected_refs)
    parsed: dict[str, float] = {}
    for item in scores:
        if not isinstance(item, Mapping) or set(item) != {"result_ref", "score"}:
            continue
        result_ref = str(item["result_ref"])
        if result_ref not in expected:
            continue
        raw_score = item["score"]
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            continue
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            continue
        if result_ref in parsed:
            continue
        parsed[result_ref] = score
    return parsed


def _parse_evidence_filter_decisions(
    value: Any,
    expected_refs: Sequence[str],
) -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != {"decisions"}:
        raise ValueError("V4 evidence filter output must contain only a decisions list")
    decisions = value.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("V4 evidence filter decisions must be a list")
    expected = set(expected_refs)
    output: dict[str, bool] = {}
    for item in decisions:
        if not isinstance(item, Mapping) or set(item) != {"result_ref", "useful"}:
            raise ValueError(
                "V4 evidence filter decisions require result_ref and useful"
            )
        result_ref = str(item["result_ref"])
        useful = item["useful"]
        if result_ref not in expected or result_ref in output:
            raise ValueError("V4 evidence filter output contains an invalid result_ref")
        if not isinstance(useful, bool):
            raise ValueError("V4 evidence filter useful must be boolean")
        output[result_ref] = useful
    if set(output) != expected:
        raise ValueError("V4 evidence filter must classify every supplied result_ref")
    return output


def _minmax_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high, abs_tol=1e-12):
        return [0.0 for _ in scores]
    scale = high - low
    return [(score - low) / scale for score in scores]


def _stable_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()






def _deduplicate(
    rows: Sequence[RuntimeCandidate], *, rank_fusion: bool = False
) -> list[RuntimeCandidate]:
    merged: dict[str, RuntimeCandidate] = {}
    for row in rows:
        current = merged.get(row.key)
        if current is None:
            merged[row.key] = row
            continue
        current.sources.update(row.sources)
        current.subproblem_ids.update(row.subproblem_ids)
        current.anchor_reinforcement = max(
            current.anchor_reinforcement, row.anchor_reinforcement
        )
        for action_id, action_score in row.action_scores.items():
            current.action_scores[action_id] = max(
                action_score,
                current.action_scores.get(action_id, float("-inf")),
            )
        for action_id, contribution in row.rank_contributions.items():
            current.rank_contributions[action_id] = max(
                contribution,
                current.rank_contributions.get(action_id, 0.0),
            )
        if row.score > current.score:
            current.score = row.score
            current.payload = row.payload
            current.attributes = row.attributes
            current.observation = row.observation
    if rank_fusion and merged:
        for candidate in merged.values():
            contributions = candidate.rank_contributions.values()
            best = max(contributions, default=candidate.score)
            extra_actions = max(0, len(candidate.rank_contributions) - 1)
            consistency = min(0.08, 0.02 * extra_actions)
            candidate.score = best + consistency + candidate.anchor_reinforcement
            candidate.near_duplicate_penalty = 0.0
        _apply_near_duplicate_penalties(merged.values())
    return sorted(merged.values(), key=lambda item: (-item.score, item.key))


def _record_action_ranks(
    rows: Sequence[RuntimeCandidate], action_id: str, constant: int
) -> None:
    ranked = sorted(
        _deduplicate(rows),
        key=lambda row: (
            -row.action_scores.get(action_id, row.score),
            row.key,
        ),
    )
    for rank, row in enumerate(ranked, 1):
        row.rank_contributions[action_id] = max(
            row.rank_contributions.get(action_id, 0.0),
            (constant + 1.0) / (constant + rank),
        )


def _semantic_deduplicate(
    candidates: Sequence[RuntimeCandidate], *, threshold: float
) -> tuple[list[RuntimeCandidate], int]:
    ranked = sorted(candidates, key=lambda item: (-item.score, item.key))
    accepted: list[RuntimeCandidate] = []
    dropped = 0
    for candidate in ranked:
        vector = _candidate_embedding(candidate)
        duplicate_of: RuntimeCandidate | None = None
        if vector is not None:
            for previous in accepted:
                previous_vector = _candidate_embedding(previous)
                if previous_vector is None:
                    continue
                similarity = _vector_cosine(vector, previous_vector)
                if similarity > threshold and not _times_explicitly_disjoint(
                    candidate, previous
                ):
                    duplicate_of = previous
                    break
        if duplicate_of is None:
            accepted.append(candidate)
            continue
        duplicate_of.sources.update(candidate.sources)
        duplicate_of.subproblem_ids.update(candidate.subproblem_ids)
        dropped += 1
    return accepted, dropped


def _apply_near_duplicate_penalties(
    candidates: Iterable[RuntimeCandidate],
) -> None:
    ranked = sorted(candidates, key=lambda item: (-item.score, item.key))
    accepted: list[RuntimeCandidate] = []
    for candidate in ranked:
        penalty = max(
            (_near_duplicate_penalty(candidate, previous) for previous in accepted),
            default=0.0,
        )
        candidate.near_duplicate_penalty = penalty
        candidate.score -= penalty
        accepted.append(candidate)


def _near_duplicate_penalty(first: RuntimeCandidate, second: RuntimeCandidate) -> float:
    first_vector = _candidate_embedding(first)
    second_vector = _candidate_embedding(second)
    if first_vector is None or second_vector is None:
        return 0.0
    similarity = _vector_cosine(first_vector, second_vector)
    if similarity < 0.92 or _times_explicitly_disjoint(first, second):
        return 0.0
    source_overlap = bool(
        _candidate_evidence_sources(first) & _candidate_evidence_sources(second)
    )
    entity_overlap = bool(_candidate_entities(first) & _candidate_entities(second))
    if not source_overlap and not (
        entity_overlap and _candidate_times_consistent(first, second)
    ):
        return 0.0
    return min(0.08, max(0.0, similarity - 0.92))


def _candidate_embedding(candidate: RuntimeCandidate) -> tuple[float, ...] | None:
    node = getattr(candidate.payload, "node", None)
    raw = getattr(node, "embedding", None)
    if raw is None:
        raw = candidate.attributes.get("embedding")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        return None
    try:
        vector = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    return vector if any(value != 0.0 for value in vector) else None


def _vector_cosine(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        return 0.0
    first_norm = sum(value * value for value in first) ** 0.5
    second_norm = sum(value * value for value in second) ** 0.5
    if not first_norm or not second_norm:
        return 0.0
    return max(
        0.0,
        min(
            1.0,
            sum(left * right for left, right in zip(first, second))
            / (first_norm * second_norm),
        ),
    )


def _candidate_evidence_sources(candidate: RuntimeCandidate) -> set[str]:
    node = getattr(candidate.payload, "node", None)
    raw = getattr(node, "evidence_refs", None)
    if raw is None:
        raw = (
            candidate.attributes.get("evidence_refs")
            or candidate.attributes.get("source_refs")
            or ()
        )
    markers = {repr(_fingerprint_value(item)) for item in raw}
    return markers or set(candidate.sources)


def _candidate_entities(candidate: RuntimeCandidate) -> set[str]:
    node = getattr(candidate.payload, "node", None)
    raw = getattr(node, "entities", None)
    if raw is None:
        raw = candidate.attributes.get("entities") or ()
    return {_normalized_query(item) for item in raw if str(item).strip()}


def _candidate_interval(
    candidate: RuntimeCandidate,
) -> tuple[str | None, str | None]:
    node = getattr(candidate.payload, "node", None)
    start = (
        getattr(node, "event_time_start", None)
        or getattr(node, "valid_from", None)
        or candidate.attributes.get("event_time_start")
        or candidate.attributes.get("valid_from")
    )
    end = (
        getattr(node, "event_time_end", None)
        or getattr(node, "valid_to", None)
        or candidate.attributes.get("event_time_end")
        or candidate.attributes.get("valid_to")
        or start
    )
    return (
        str(start) if start else None,
        str(end) if end else None,
    )


def _times_explicitly_disjoint(
    first: RuntimeCandidate, second: RuntimeCandidate
) -> bool:
    first_start, first_end = _candidate_interval(first)
    second_start, second_end = _candidate_interval(second)
    return bool(
        first_start
        and first_end
        and second_start
        and second_end
        and (first_end < second_start or second_end < first_start)
    )


def _candidate_times_consistent(
    first: RuntimeCandidate, second: RuntimeCandidate
) -> bool:
    first_start, _ = _candidate_interval(first)
    second_start, _ = _candidate_interval(second)
    return bool(
        first_start and second_start and not _times_explicitly_disjoint(first, second)
    )


def _finalize(
    backend: RetrievalBackend,
    execution: RetrievalExecutionContext,
    candidates: Sequence[RuntimeCandidate],
    question: str,
    final_top_k: int,
    trajectory: RetrievalTrajectory,
) -> Any:
    result = backend.finalize(execution, candidates, question, final_top_k, trajectory)
    try:
        result.trajectory = trajectory
    except (AttributeError, TypeError):
        pass
    return result


def _normalized_query(value: str) -> str:
    return " ".join(str(value).casefold().split())


_INTERNAL_VALUE = re.compile(
    r"\b(?:node|event|state|chain|head|scope|turn|gold)[_-][0-9a-f]{8,}\b|"
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


def _contains_forbidden_controller_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        forbidden = {"node_id", "payload", "fact_key", "internal_id"}
        if forbidden & {str(key).casefold() for key in value}:
            return True
        return any(
            _contains_forbidden_controller_value(item) for item in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_forbidden_controller_value(item) for item in value)
    return isinstance(value, str) and bool(_INTERNAL_VALUE.search(value))


__all__ = ["MultiRoundRetriever"]
