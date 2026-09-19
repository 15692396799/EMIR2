from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


SCHEMA_VERSION = 4
NODE_TYPES = {"chain_head", "event", "semantic_state"}

# exact_mentions entries longer than this are dropped at build time instead of
# being truncated: truncation would break the character-for-character guarantee,
# and long quotes/captions are already preserved in evidence refs.
EXACT_MENTION_MAX_LENGTH = 120

# Vertical/structural edges: chain-internal. Built by _structural_edges and
# _apply_semantic_updates. Always scope-local; wiped and re-derived on rebuild.
STRUCTURAL_EDGE_TYPES = {
    "FIRST_TIMELINE_NODE", "TEMPORAL_NEXT", "INITIAL_STATE", "CURRENT_STATE",
    "SEMANTIC_NEXT", "SUPERSEDES", "SUPPORTED_BY",
}

# Event-to-event lateral edges, mostly cross-chain. Built by _explicit_edges
# (LLM within-window relations) and _cross_window_edges (deterministic candidates
# + adjudication LLM, across windows). Cross-scope EVENT edges persist in
# v4_cross_scope_edges.
EVENT_EDGE_TYPES = {
    "CAUSES", "CONTRIBUTES_TO", "CONTEXT_FOR", "ENABLES", "PART_OF",
    "FOLLOW_UP_OF", "SAME_EVENT",
}

# Event<->state and state<->state edges. Built by _apply_semantic_updates
# (TRIGGERS_STATE_CHANGE / SUPERSEDES / CONTRADICTS) and _apply_semantic_updates
# + _cross_window_edges (SUPPORTS_STATE / AFFECTS_TOPIC). Scope-local only.
STATE_EDGE_TYPES = {
    "TRIGGERS_STATE_CHANGE", "SUPPORTS_STATE", "CONTRADICTS", "AFFECTS_TOPIC",
}

EDGE_TYPES = STRUCTURAL_EDGE_TYPES | EVENT_EDGE_TYPES | STATE_EDGE_TYPES
EDGE_TYPE_DESCRIPTIONS = {
    "AFFECTS_TOPIC": "Directed event/state influence on a broader topic; use when evidence must connect an occurrence to a changed topic.",
    "CAUSES": "Directed causal link from cause to effect; use for why/how questions that need the cause or consequence node.",
    "CONTEXT_FOR": "Directed background-to-focus link; use when an event supplies context needed to interpret another event.",
    "CONTRADICTS": "Directed conflict between semantic states; use when the answer depends on resolving inconsistent states.",
    "CONTRIBUTES_TO": "Directed partial-cause link toward an outcome; use when several events may jointly explain a result.",
    "CURRENT_STATE": "Directed chain-to-current-state pointer; use to find the latest semantic state for a topic chain.",
    "ENABLES": "Directed prerequisite link; use when one event made a later event or state possible.",
    "FIRST_TIMELINE_NODE": "Directed chain-to-first-event pointer; use for the earliest event in a topic timeline.",
    "FOLLOW_UP_OF": "Directed follow-up to earlier event; use for later actions or outcomes tied to an earlier event.",
    "INITIAL_STATE": "Directed chain-to-initial-state pointer; use to find the first known semantic state for a topic.",
    "PART_OF": "Directed part-to-whole relation; use when a detail belongs to a larger event or plan.",
    "SAME_EVENT": "Bidirectional duplicate/alias event relation; use to merge evidence about the same occurrence.",
    "SEMANTIC_NEXT": "Directed previous-to-next semantic state version; use to move through state history.",
    "SUPERSEDES": "Directed newer-state replaces older-state relation; use to find the current superseding fact.",
    "SUPPORTED_BY": "Directed claim/state-to-support relation; use to fetch evidence supporting a summary or state.",
    "SUPPORTS_STATE": "Directed event-to-state support relation; use when an event justifies a semantic state.",
    "TEMPORAL_NEXT": "Directed chronological next-event relation; use for before/after sequence questions.",
    "TRIGGERS_STATE_CHANGE": "Directed event-to-state-change relation; use when an event caused a preference/status update.",
}
EDGE_TRUST_LEVELS = {"explicit", "inferred", "derived"}


def stable_id(prefix: str, *parts: Any) -> str:
    encoded = "\u241f".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:24]}"


def _canonical(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


@dataclass(frozen=True, order=True)
class Participant:
    participant_id: str
    role: str

    @classmethod
    def from_value(cls, value: Any) -> "Participant":
        if isinstance(value, Participant):
            return value
        if not isinstance(value, dict):
            raise ValueError("participants must contain objects with participant_id and role")
        participant_id = str(value.get("participant_id") or value.get("id") or "").strip()
        role = str(value.get("role") or "").strip().lower()
        if not participant_id or not role:
            raise ValueError("participant_id and role are required")
        return cls(participant_id=participant_id, role=role)

    def to_dict(self) -> dict[str, str]:
        return {"participant_id": self.participant_id, "role": self.role}


def canonical_participants(values: Iterable[Any]) -> list[Participant]:
    by_key: dict[tuple[str, str], Participant] = {}
    for value in values:
        participant = Participant.from_value(value)
        by_key[(participant.role.casefold(), participant.participant_id.casefold())] = participant
    if not by_key:
        raise ValueError("at least one participant is required")
    return [by_key[key] for key in sorted(by_key)]


def participant_scope_id(namespace: str, participants: Iterable[Any]) -> str:
    canonical = canonical_participants(participants)
    return stable_id("scope", namespace, [value.to_dict() for value in canonical])


@dataclass
class ParticipantScope:
    id: str
    namespace: str
    participants: list[Participant]
    visibility_policy: str = "scope_only"
    metadata: dict[str, Any] = field(default_factory=dict)
    revision: int = 0

    @classmethod
    def create(
        cls, namespace: str, participants: Iterable[Any], metadata: dict[str, Any] | None = None
    ) -> "ParticipantScope":
        canonical = canonical_participants(participants)
        return cls(participant_scope_id(namespace, canonical), namespace, canonical, metadata=dict(metadata or {}))


@dataclass
class TopicChain:
    id: str
    namespace: str
    scope_id: str
    topic_id: str
    name: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    rolling_summary: str = ""
    representative_entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryNode:
    id: str
    namespace: str
    scope_id: str
    chain_id: str
    topic_id: str
    node_type: str
    title: str
    summary: str
    text: str = ""
    actors: list[str] = field(default_factory=list)
    action: str = ""
    objects: list[str] = field(default_factory=list)
    location: str | None = None
    event_time_start: str | None = None
    event_time_end: str | None = None
    time_precision: str | None = None
    observed_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    facts: list[dict[str, Any]] = field(default_factory=list)
    changed_keys: list[str] = field(default_factory=list)
    previous_state_id: str | None = None
    trigger_event_ids: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 1.0
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"unsupported V4 node_type: {self.node_type}")
        if not self.entity_ids and self.entities:
            self.entity_ids = list(self.entities)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # Backend-neutral result adapters and evaluation code use these names.
    @property
    def absolute_time_start(self) -> str | None:
        return self.event_time_start or self.valid_from

    @property
    def absolute_time_end(self) -> str | None:
        return self.event_time_end or self.valid_to

    @property
    def source_refs(self) -> list[dict[str, Any]]:
        return self.evidence_refs

    @property
    def keywords(self) -> list[str]:
        return []


@dataclass
class MemoryEdge:
    id: str
    namespace: str
    scope_id: str
    source_id: str
    target_id: str
    edge_type: str
    # "directed" by default. SAME_EVENT is stored once with source_id<target_id;
    # the retriever reads both directions via adjacent(direction="both").
    direction: str = "directed"
    confidence: float = 1.0
    trust: str = "explicit"
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    created_method: str = "builder"
    valid_from: str | None = None
    valid_to: str | None = None
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.edge_type = str(self.edge_type).upper()
        if self.edge_type not in EDGE_TYPES:
            raise ValueError(f"unsupported V4 edge_type: {self.edge_type}")
        if self.trust not in EDGE_TRUST_LEVELS:
            raise ValueError(f"unsupported edge trust: {self.trust}")
        if self.source_id == self.target_id:
            raise ValueError("self edges are not allowed")


@dataclass
class SemanticFact:
    id: str
    namespace: str
    scope_id: str
    chain_id: str
    state_node_id: str
    key: str
    subject: str
    predicate: str
    value: Any
    dimension: str = "state"
    aspect: str = "value"
    confidence: float = 1.0
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    valid_from: str | None = None
    valid_to: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    history_origin: str | None = None
    status: str = "confirmed"


@dataclass
class TimeConstraint:
    operator: str = "any"
    start: str | None = None
    end: str | None = None
    precision: str | None = None
    hard: bool = False

    @classmethod
    def from_dict(cls, value: Any) -> "TimeConstraint":
        if not isinstance(value, dict):
            return cls()
        if "hard" in value and not isinstance(value["hard"], bool):
            raise ValueError("time.hard must be a JSON boolean")
        operator = str(value.get("operator") or "any").lower()
        if operator not in {"any", "at", "before", "after", "between"}:
            operator = "any"
        return cls(
            operator=operator,
            start=_optional_string(value.get("start") or value.get("value")),
            end=_optional_string(value.get("end")),
            precision=_optional_string(value.get("precision")),
            hard=value.get("hard", False),
        )


@dataclass
class RetrievalProbe:
    id: str
    query: str
    node_types: list[str] = field(default_factory=list)
    topic_hints: list[str] = field(default_factory=list)
    entity_hints: list[str] = field(default_factory=list)
    must_terms: list[str] = field(default_factory=list)
    should_terms: list[str] = field(default_factory=list)
    facets: list[str] = field(default_factory=list)
    time: TimeConstraint = field(default_factory=TimeConstraint)
    budget: int = 8

    @classmethod
    def from_dict(cls, value: dict[str, Any], index: int, max_budget: int) -> "RetrievalProbe":
        node_types = [str(item) for item in _list(value, "node_types") if str(item) in NODE_TYPES]
        return cls(
            id=str(value.get("id") or f"p{index}"),
            query=str(value.get("query") or "").strip(),
            node_types=node_types,
            topic_hints=_strings(_list(value, "topic_hints")),
            entity_hints=_strings(_list(value, "entity_hints")),
            must_terms=_strings(_list(value, "must_terms"))[:4],
            should_terms=_strings(_list(value, "should_terms"))[:8],
            facets=[
                str(item) for item in _list(value, "facets")
                if str(item) in {
                    "actor", "action", "object", "time", "location", "cause",
                    "outcome", "state", "conflict", "count",
                }
            ],
            time=TimeConstraint.from_dict(value.get("time")),
            budget=max(1, min(max_budget, int(value.get("budget", min(8, max_budget))))),
        )


@dataclass
class EdgeExpansion:
    from_probe: str
    edge_types: list[str]
    direction: str = "both"
    max_hops: int = 1
    budget: int = 4
    anchor_selector: AnchorSelector | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], max_budget: int) -> "EdgeExpansion":
        direction = str(value.get("direction") or "both").lower()
        if direction not in {"in", "out", "both"}:
            direction = "both"
        return cls(
            from_probe=str(value.get("from_probe") or ""),
            edge_types=[str(item).upper() for item in _list(value, "edge_types") if str(item).upper() in EDGE_TYPES],
            direction=direction,
            max_hops=1,
            budget=max(1, min(max_budget, int(value.get("budget", min(4, max_budget))))),
            anchor_selector=AnchorSelector.from_dict(value.get("anchor_selector")),
        )


@dataclass
class AnchorSelector:
    rank_window: list[int] = field(default_factory=lambda: [1, 8])
    node_types: list[str] = field(default_factory=list)
    topic_hints: list[str] = field(default_factory=list)
    entity_hints: list[str] = field(default_factory=list)
    time: TimeConstraint = field(default_factory=TimeConstraint)

    @classmethod
    def from_dict(cls, value: Any) -> "AnchorSelector | None":
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("anchor_selector must be an object")
        rank_window = value.get("rank_window", [1, 8])
        if not isinstance(rank_window, list) or len(rank_window) != 2:
            raise ValueError("anchor_selector.rank_window must be [start, end]")
        start, end = rank_window
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 1
            or end < start
        ):
            raise ValueError("anchor_selector.rank_window must contain positive ascending integers")
        return cls(
            rank_window=[start, end],
            node_types=[str(item) for item in _list(value, "node_types") if str(item) in NODE_TYPES],
            topic_hints=_strings(_list(value, "topic_hints")),
            entity_hints=_strings(_list(value, "entity_hints")),
            time=TimeConstraint.from_dict(value.get("time")),
        )


@dataclass
class RetrievalPlan:
    probes: list[RetrievalProbe]
    evidence_facets: list[str] = field(default_factory=list)
    edge_expansions: list[EdgeExpansion] = field(default_factory=list)
    final_top_k: int = 16
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        edge_expansions = []
        for expansion in self.edge_expansions:
            value = asdict(expansion)
            if expansion.anchor_selector is None:
                value.pop("anchor_selector", None)
            edge_expansions.append(value)
        return {
            "version": 1,
            "evidence_facets": list(self.evidence_facets),
            "probes": [asdict(probe) for probe in self.probes],
            "edge_expansions": edge_expansions,
            "final_top_k": self.final_top_k,
        }


@dataclass
class TeacherTrajectory:
    query: str
    rounds: list[dict[str, Any]] = field(default_factory=list)
    final_evidence: list[dict[str, Any]] = field(default_factory=list)
    accepted: bool = False
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if _contains_forbidden_trajectory_key(value) or contains_internal_id(value):
            raise ValueError("TeacherTrajectory contains an internal identifier")
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherTrajectory":
        return cls(
            query=str(value.get("query") or ""),
            rounds=list(value.get("rounds") or []),
            final_evidence=list(value.get("final_evidence") or []),
            accepted=bool(value.get("accepted", False)),
            version=1,
        )


INTERNAL_ID_PATTERN = re.compile(
    r"\b(?:node|event|state|chain|head|scope|turn|v2node)_[0-9a-f]{8,}\b|"
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


def contains_internal_id(value: Any) -> bool:
    if isinstance(value, dict):
        return any(contains_internal_id(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_internal_id(item) for item in value)
    return isinstance(value, str) and bool(INTERNAL_ID_PATTERN.search(value))


def _contains_forbidden_trajectory_key(value: Any) -> bool:
    forbidden = {
        "node_id", "node_ids", "turn_id", "turn_ids", "scope_id", "chain_id",
        "evidence_refs", "gold_evidence", "gold_evidence_ids",
    }
    if isinstance(value, dict):
        return any(
            str(key).casefold() in forbidden or _contains_forbidden_trajectory_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_trajectory_key(child) for child in value)
    return False


def _optional_string(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def _list(value: dict[str, Any], key: str) -> list[Any]:
    raw = value.get(key, [])
    if not isinstance(raw, list):
        raise ValueError(f"RetrievalPlan field {key!r} must be a list")
    return raw


def _strings(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
