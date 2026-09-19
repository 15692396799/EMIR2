from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from memory.retrieve.base import RetrievalExecutionContext


_INTERNAL_ID = re.compile(
    r"\b(?:node|event|state|chain|head|scope|turn|result|gold)[_-][0-9a-f]{8,}\b|"
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BackendCapabilities:
    backend_id: str
    backend_version: str
    node_types: frozenset[str]
    edge_directions: Mapping[str, frozenset[str]]
    max_hops: int
    max_total_budget: int
    plan_schema_version: int = 2
    virtual_actions: frozenset[str] = frozenset()
    time_search: bool = False
    graph_expansion: bool = True
    static_plans: bool = True
    absolute_time_correction: bool = False
    intent_aware_multi_round: bool = False

    def allows_edge(self, edge_type: str, direction: str) -> bool:
        return direction in self.edge_directions.get(edge_type, frozenset())

    def allows_action(self, action: str, direction: str) -> bool:
        if action in self.virtual_actions:
            return direction == "lookup"
        return self.allows_edge(action, direction)


@dataclass(frozen=True)
class Probe:
    id: str
    query: str
    budget: int
    lane: str = "model"
    required: bool = True
    subproblem_id: str = ""
    must_terms: tuple[str, ...] = ()
    should_terms: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    facets: tuple[str, ...] = ()
    time_constraint: Mapping[str, Any] | None = None
    node_types: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def public_subproblem(self) -> dict[str, Any]:
        return {
            "subproblem_id": self.subproblem_id,
            "lane": self.lane,
            "required": self.required,
            "query": self.query,
            "must_terms": list(self.must_terms),
            "should_terms": list(self.should_terms),
            "entities": list(self.entities),
            "facets": list(self.facets),
            "time_constraint": (
                dict(self.time_constraint) if self.time_constraint is not None else None
            ),
        }


@dataclass(frozen=True)
class Selector:
    rank_start: int = 1
    rank_end: int = 8
    node_types: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Expansion:
    id: str
    source: str
    edge_type: str
    direction: str
    hops: int
    budget: int
    subproblem_id: str = ""
    selector: Selector = field(default_factory=Selector)
    evidence_facet: str = ""
    reason: str = ""


@dataclass(frozen=True)
class RetrievalPlan:
    backend_id: str
    schema_version: int
    probes: tuple[Probe, ...]
    expansions: tuple[Expansion, ...] = ()
    final_top_k: int = 16

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RetrievalPlan":
        probes = tuple(
            Probe(
                id=str(item.get("id") or ""),
                query=str(item.get("query") or "").strip(),
                budget=int(item.get("budget", 1)),
                lane=str(item.get("lane") or "model"),
                required=bool(item.get("required", True)),
                subproblem_id=str(item.get("subproblem_id") or ""),
                must_terms=tuple(str(term) for term in item.get("must_terms", ())),
                should_terms=tuple(str(term) for term in item.get("should_terms", ())),
                entities=tuple(str(entity) for entity in item.get("entities", ())),
                facets=tuple(str(facet) for facet in item.get("facets", ())),
                time_constraint=(
                    dict(item["time_constraint"])
                    if isinstance(item.get("time_constraint"), Mapping)
                    else None
                ),
                node_types=tuple(
                    str(node_type) for node_type in item.get("node_types", ())
                ),
                attributes=dict(item.get("attributes") or {}),
            )
            for item in value.get("probes", ())
        )
        expansions = []
        for item in value.get("expansions", ()):
            selector = item.get("selector") or {}
            expansions.append(
                Expansion(
                    id=str(item.get("id") or ""),
                    source=str(item.get("source") or ""),
                    edge_type=str(item.get("edge_type") or ""),
                    direction=str(item.get("direction") or ""),
                    hops=int(item.get("hops", 1)),
                    budget=int(item.get("budget", 1)),
                    subproblem_id=str(item.get("subproblem_id") or ""),
                    selector=Selector(
                        rank_start=int(selector.get("rank_start", 1)),
                        rank_end=int(selector.get("rank_end", 8)),
                        node_types=tuple(
                            str(node_type)
                            for node_type in selector.get("node_types", ())
                        ),
                        attributes=dict(selector.get("attributes") or {}),
                    ),
                    evidence_facet=str(item.get("evidence_facet") or "").strip(),
                    reason=str(item.get("reason") or "").strip(),
                )
            )
        return cls(
            backend_id=str(value.get("backend_id") or ""),
            schema_version=int(value.get("schema_version", 0)),
            probes=probes,
            expansions=tuple(expansions),
            final_top_k=int(value.get("final_top_k", 16)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeCandidate:
    key: str
    node_type: str
    score: float
    attributes: dict[str, Any] = field(default_factory=dict)
    observation: Mapping[str, Any] = field(default_factory=dict)
    payload: Any = None
    sources: set[str] = field(default_factory=set)
    action_scores: dict[str, float] = field(default_factory=dict)
    rank_contributions: dict[str, float] = field(default_factory=dict)
    subproblem_ids: set[str] = field(default_factory=set)
    anchor_reinforcement: float = 0.0
    near_duplicate_penalty: float = 0.0

    def public_observation(self, result_ref: str) -> dict[str, Any]:
        value = dict(self.observation)
        value.setdefault("summary", str(self.attributes.get("summary") or ""))
        value.setdefault(
            "salient_facts", list(self.attributes.get("salient_facts") or ())
        )
        value.setdefault(
            "exact_mentions", list(self.attributes.get("exact_mentions") or ())
        )
        value.setdefault("event_time_start", self.attributes.get("event_time_start"))
        value.setdefault("event_time_end", self.attributes.get("event_time_end"))
        value.setdefault("valid_from", self.attributes.get("valid_from"))
        value.setdefault("valid_to", self.attributes.get("valid_to"))
        value.setdefault("location", self.attributes.get("location"))
        value.setdefault(
            "confidence", float(self.attributes.get("confidence", self.score))
        )
        value.setdefault("node_type", self.node_type)
        value.setdefault(
            "expandable_edges", list(self.attributes.get("expandable_edges") or ())
        )
        value["source_steps"] = sorted(self.sources)
        value["subproblem_ids"] = sorted(self.subproblem_ids)
        value["result_ref"] = result_ref
        if _contains_internal_id(value):
            value = _scrub_internal_ids(value)
        return value


@dataclass
class RetrievalRound:
    round_index: int
    initial_probes: tuple[Probe, ...] = ()
    fresh_probe: Probe | None = None
    fresh_probes: tuple[Probe, ...] = ()
    expansions: tuple[Expansion, ...] = ()
    observations: tuple[Mapping[str, Any], ...] = ()
    covered_facets: tuple[str, ...] = ()
    missing_facets: tuple[str, ...] = ()
    growth: int = 0
    model_probe_growth: int = 0
    model_action_growth: int = 0
    fallback_growth: int = 0
    total_growth: int = 0
    fallback_used: bool = False
    controller_calls: int = 0
    retrieval_calls: int = 0
    sufficient: bool = False
    root_status: Mapping[str, Any] = field(default_factory=dict)
    subproblem_statuses: tuple[Mapping[str, Any], ...] = ()
    reason: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalTrajectory:
    question: str
    backend_id: str
    rounds: list[RetrievalRound] = field(default_factory=list)
    termination_reason: str = ""
    controller_calls: int = 0
    retrieval_calls: int = 0
    evidence_filter_candidates: int = 0
    evidence_filter_batches: int = 0
    evidence_filter_calls: int = 0
    evidence_filter_dropped: int = 0
    semantic_slimming_candidates: int = 0
    semantic_slimming_calls: int = 0
    semantic_slimming_kept_items: int = 0
    semantic_slimming_dropped_items: int = 0
    semantic_slimming_fallbacks: int = 0
    rerank_candidates: int = 0
    rerank_batches: int = 0
    rerank_calls: int = 0
    semantic_dedup_dropped: int = 0
    support_delivery_checks: int = 0
    support_delivery_failures: int = 0
    protected_support_count: int = 0
    delivered_support_count: int = 0
    support_delivery_verified: bool | None = None
    evidence_backtrack_calls: int = 0
    controller_errors: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return bool(
            self.rounds
            and self.rounds[-1].sufficient
            and self.termination_reason != "controller_error"
            and self.support_delivery_verified is not False
        )

    @property
    def final_evidence(self) -> list[Mapping[str, Any]]:
        return list(self.rounds[-1].observations) if self.rounds else []

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["accepted"] = self.accepted
        return value


class RetrievalBackend(Protocol):
    backend_id: str
    capabilities: BackendCapabilities


    def global_search(
        self, execution: RetrievalExecutionContext, probes: Sequence[Probe]
    ) -> Sequence[RuntimeCandidate]: ...

    def expand(
        self,
        execution: RetrievalExecutionContext,
        anchors: Sequence[RuntimeCandidate],
        expansion: Expansion,
    ) -> Sequence[RuntimeCandidate]: ...

    def finalize(
        self,
        execution: RetrievalExecutionContext,
        candidates: Sequence[RuntimeCandidate],
        question: str,
        final_top_k: int,
        trace: RetrievalTrajectory,
    ) -> Any: ...


class MultiRoundController(Protocol):
    def set_bootstrap_evidence(self, evidence: Sequence[Mapping[str, Any]]) -> None: ...

    def set_expansion_source_refs(self, source_refs: Sequence[str]) -> None: ...

    def set_available_actions_by_target(
        self,
        actions: Mapping[str, Sequence[Any]],
    ) -> None: ...

    def initial_plan(
        self, question: str, context: Any = None
    ) -> RetrievalPlan | Mapping[str, Any]: ...

    def assess(
        self,
        question: str,
        subproblems: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
        round_index: int,
        context: Any = None,
    ) -> Mapping[str, Any]: ...

    def filter_evidence_batch(
        self,
        question: str,
        candidates: Sequence[Mapping[str, Any]],
        batch_index: int,
        context: Any = None,
    ) -> Mapping[str, Any]: ...

    def slim_semantic_node(
        self,
        question: str,
        items: Sequence[Mapping[str, Any]],
        context: Any = None,
    ) -> Mapping[str, Any]: ...

    def select_expansion_sources(
        self,
        question: str,
        candidates: Sequence[Mapping[str, Any]],
        round_index: int,
        context: Any = None,
    ) -> Mapping[str, Any]: ...

    def rerank_batch(
        self,
        question: str,
        candidates: Sequence[Mapping[str, Any]],
        batch_index: int,
        context: Any = None,
        *,
        allow_partial: bool = False,
    ) -> Mapping[str, Any]: ...


def validate_plan(plan: RetrievalPlan, capabilities: BackendCapabilities) -> None:
    if plan.backend_id != capabilities.backend_id:
        raise ValueError(
            f"Plan backend_id {plan.backend_id!r} does not match {capabilities.backend_id!r}"
        )
    if plan.schema_version != capabilities.plan_schema_version:
        raise ValueError(
            f"Plan schema_version {plan.schema_version} does not match "
            f"{capabilities.plan_schema_version} for {capabilities.backend_id}"
        )
    if not plan.probes:
        raise ValueError("RetrievalPlan must contain at least one probe")
    if plan.final_top_k < 1:
        raise ValueError("final_top_k must be positive")
    identifiers: set[str] = set()
    total_budget = 0
    for probe in plan.probes:
        _validate_identifier(probe.id, "probe")
        if probe.id in identifiers:
            raise ValueError(f"Duplicate plan step id: {probe.id}")
        identifiers.add(probe.id)
        if not probe.query or probe.budget < 1:
            raise ValueError(
                f"Probe {probe.id!r} must have a query and positive budget"
            )
        if plan.schema_version >= 6 and probe.lane not in {"baseline", "model"}:
            raise ValueError(f"Probe {probe.id!r} has an invalid lane")
        if (
            plan.schema_version >= 6
            and probe.lane == "baseline"
            and (
                probe.subproblem_id
                or probe.must_terms
                or probe.should_terms
                or probe.entities
                or probe.time_constraint is not None
            )
        ):
            raise ValueError("Baseline probes cannot contain structured hints")
        if (
            plan.schema_version >= 6
            and probe.lane == "model"
            and not probe.subproblem_id
        ):
            raise ValueError("Model probes require a subproblem_id")
        if probe.subproblem_id and not re.fullmatch(r"s[1-4]", probe.subproblem_id):
            raise ValueError(f"Probe {probe.id!r} has an invalid subproblem_id")
        if len(probe.must_terms) > 4 or len(probe.should_terms) > 8:
            raise ValueError(f"Probe {probe.id!r} exceeds keyword limits")
        if len(probe.entities) > 4:
            raise ValueError(f"Probe {probe.id!r} exceeds entity limits")
        if probe.time_constraint is not None:
            _validate_query_grounded_mapping(
                probe.time_constraint, f"probe {probe.id} time_constraint"
            )
        _validate_node_types(probe.node_types, capabilities)
        _validate_query_grounded_mapping(
            probe.attributes, f"probe {probe.id} attributes"
        )
        total_budget += probe.budget
    for expansion in plan.expansions:
        _validate_identifier(expansion.id, "expansion")
        if expansion.id in identifiers:
            raise ValueError(f"Duplicate plan step id: {expansion.id}")
        if expansion.source not in identifiers:
            raise ValueError(
                f"Expansion {expansion.id!r} has unknown or forward source {expansion.source!r}"
            )
        validate_runtime_expansion(expansion, capabilities)
        identifiers.add(expansion.id)
        total_budget += expansion.budget
    if (
        not capabilities.intent_aware_multi_round
        and total_budget > capabilities.max_total_budget
    ):
        raise ValueError(
            f"Plan candidate budget {total_budget} exceeds backend limit {capabilities.max_total_budget}"
        )
    if _contains_internal_id(plan.to_dict()):
        raise ValueError("RetrievalPlan contains an internal/result/gold identifier")


def validate_runtime_expansion(
    expansion: Expansion, capabilities: BackendCapabilities
) -> None:
    if not capabilities.allows_action(expansion.edge_type, expansion.direction):
        raise ValueError(
            f"Illegal native edge/direction for {capabilities.backend_id}: "
            f"{expansion.edge_type}/{expansion.direction}"
        )
    if expansion.hops < 1 or expansion.hops > capabilities.max_hops:
        raise ValueError(f"Expansion {expansion.id!r} exceeds backend hop capability")
    if expansion.budget < 1:
        raise ValueError(f"Expansion {expansion.id!r} must have a positive budget")
    if (
        expansion.selector.rank_start < 1
        or expansion.selector.rank_end < expansion.selector.rank_start
    ):
        raise ValueError(
            f"Expansion {expansion.id!r} has an invalid selector rank range"
        )
    _validate_node_types(expansion.selector.node_types, capabilities)
    _validate_query_grounded_mapping(
        expansion.selector.attributes, f"expansion {expansion.id} selector attributes"
    )


def _validate_identifier(value: str, label: str) -> None:
    if not value or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError(f"Invalid {label} id: {value!r}")


def _validate_node_types(
    node_types: Sequence[str], capabilities: BackendCapabilities
) -> None:
    unknown = sorted(set(node_types) - set(capabilities.node_types))
    if unknown:
        raise ValueError(
            f"Unknown native node types for {capabilities.backend_id}: {', '.join(unknown)}"
        )


def _validate_query_grounded_mapping(value: Mapping[str, Any], label: str) -> None:
    forbidden = {
        "node_id",
        "node_ids",
        "result_ref",
        "teacher_result_ref",
        "gold_evidence",
        "answer",
    }
    overlap = sorted({str(key).lower() for key in value} & forbidden)
    if overlap:
        raise ValueError(f"{label} contains forbidden keys: {', '.join(overlap)}")


def _contains_internal_id(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_internal_id(key) or _contains_internal_id(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_internal_id(item) for item in value)
    return isinstance(value, str) and bool(_INTERNAL_ID.search(value))


def _scrub_internal_ids(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _scrub_internal_ids(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_scrub_internal_ids(item) for item in value]
    if isinstance(value, str):
        return _INTERNAL_ID.sub("[redacted]", value)
    return value


__all__ = [
    "BackendCapabilities",
    "Expansion",
    "MultiRoundController",
    "Probe",
    "RetrievalBackend",
    "RetrievalPlan",
    "RetrievalRound",
    "RetrievalTrajectory",
    "RuntimeCandidate",
    "Selector",
    "validate_plan",
    "validate_runtime_expansion",
]
