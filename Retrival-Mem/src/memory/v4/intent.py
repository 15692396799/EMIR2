from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


FACETS = frozenset({
    "actor",
    "action",
    "object",
    "time",
    "location",
    "cause",
    "outcome",
    "state",
    "conflict",
    "count",
})

VIRTUAL_ACTIONS = frozenset({
    "SOURCE_EVIDENCE_LOOKUP",
    "ENTITY_LOOKUP",
    "CROSS_ENTITY_LOOKUP",
    "TIME_OVERLAP_LOOKUP",
    "TEMPORAL_BEFORE_LOOKUP",
    "TEMPORAL_AFTER_LOOKUP",
    "NEAREST_TIME_LOOKUP",
})

CAUSAL_EDGES = (
    "CAUSES",
    "CONTRIBUTES_TO",
    "ENABLES",
    "CONTEXT_FOR",
)
TEMPORAL_EDGES = ("TEMPORAL_NEXT",)
CURRENT_EDGES = ("CURRENT_STATE",)
STATE_HISTORY_EDGES = ("SEMANTIC_NEXT", "INITIAL_STATE")
SUPPORT_EDGES = ("SUPPORTED_BY", "SUPPORTS_STATE")

DEFAULT_EDGE_PRIORITY = (
    *CAUSAL_EDGES,
    *TEMPORAL_EDGES,
    *CURRENT_EDGES,
    *STATE_HISTORY_EDGES,
    *SUPPORT_EDGES,
    "TRIGGERS_STATE_CHANGE",
    "SUPERSEDES",
    "CONTRADICTS",
    "FOLLOW_UP_OF",
    "SAME_EVENT",
    "PART_OF",
    "AFFECTS_TOPIC",
)


@dataclass(frozen=True)
class QuestionIntent:
    question_types: frozenset[str]
    required_facets: tuple[str, ...]
    complexity: str
    edge_priorities: tuple[str, ...]
    initial_probe_budget: int
    followup_probe_budget: int = 6
    expansion_budget: int = 3
    allow_entity_lookup: bool = False
    allow_time_overlap_lookup: bool = False
    normalized_question: str = ""


@dataclass(frozen=True)
class RetrievalAction:
    action_id: str
    source_ref: str
    action: str
    direction: str = ""
    constraints: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_virtual(self) -> bool:
        return self.action in VIRTUAL_ACTIONS

    def protocol_value(self) -> list[str]:
        if self.is_virtual:
            return [self.source_ref, self.action]
        return [self.source_ref, self.action, self.direction]


class QuestionIntentAnalyzer:
    """Deterministic V4 query classification, budgeting, and action cataloging."""

    _PATTERNS = {
        "when": (
            r"\bwhen\b", r"\bwhat (?:time|date|day|year|month)\b",
            r"什么时候", r"何时", r"哪天", r"哪一年", r"几点",
        ),
        "where": (
            r"\bwhere\b", r"\b(?:which|what) (?:place|location)\b",
            r"哪里", r"哪儿", r"何处", r"什么地方", r"地点",
        ),
        "who": (
            r"\bwho(?:m|se)?\b", r"\bwhich (?:person|people|participant)\b",
            r"谁", r"哪位", r"哪些人", r"参与者",
        ),
        "current_latest": (
            r"\bcurrent(?:ly)?\b", r"\blatest\b", r"\bmost recent\b", r"\bnow\b",
            r"当前", r"现在", r"最新", r"最近", r"目前",
        ),
        "before_after": (
            r"\bbefore\b", r"\bafter\b", r"\bnext\b", r"\bprevious(?:ly)?\b",
            r"之前", r"之后", r"前后", r"后来", r"此前", r"随后", r"接下来",
        ),
        "why_how": (
            r"\bwhy\b", r"\bhow\b", r"\breason\b", r"\bcause[ds]?\b",
            r"为什么", r"为何", r"原因", r"怎么", r"如何",
        ),
        "count_list": (
            r"\bhow many\b", r"\blist\b", r"\bwhich\b", r"\ball\b", r"\bname (?:the|all)\b",
            r"多少", r"几个", r"列出", r"哪些", r"所有", r"全部", r"清单",
        ),
        "simultaneous": (
            r"\bsimultaneous(?:ly)?\b", r"\bat the same time\b", r"\bwhile\b", r"\boverlap(?:ping)?\b",
            r"同时", r"同期", r"同一时间", r"重叠", r"期间",
        ),
        "nearest_time": (
            r"\bnearest\b", r"\bclosest\b", r"\baround (?:that|this) time\b",
            r"最接近", r"最近的时间", r"前后发生",
        ),
        "cross_entity": (
            r"\bbetween\b.+\band\b", r"\brelationship between\b",
            r"\bcompare\b.+\b(?:and|with)\b", r"之间", r"关系", r"比较",
        ),
        "hypothetical_reasoning": (
            r"\bwould\b", r"\bmight\b", r"\bcould\b", r"\blikely\b",
            r"\bconsidered\b", r"\battributes\b", r"\bprefer(?:s|red)?\b",
            r"\bpersonality\b", r"\bopen to\b", r"\binterested in\b",
            r"会", r"可能", r"也许", r"会不会",
        ),
        "comparison": (
            r"\bor\b", r"\bversus\b", r"\bvs\.?\b", r"\bcompare[sd]?\b",
            r"还是", r"或者", r"比较",
        ),
    }

    _FACETS = {
        "when": ("time",),
        "where": ("location",),
        "who": ("actor",),
        "current_latest": ("state", "time"),
        "before_after": ("time", "action"),
        "why_how": ("cause", "outcome"),
        "count_list": ("count",),
        "simultaneous": ("time",),
        "nearest_time": ("time", "action"),
        "cross_entity": ("actor", "object"),
        "hypothetical_reasoning": ("state", "cause"),
        "comparison": ("state", "object"),
    }

    def analyze(self, question: str) -> QuestionIntent:
        normalized = self.normalize_query(question)
        lowered = normalized.casefold()
        detected = {
            label
            for label, patterns in self._PATTERNS.items()
            if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in patterns)
        }
        explicit_complex = bool(re.search(
            r"\b(?:and|then|also|compare|relationship|across)\b|并且|并|以及|然后|比较|关系|多跳",
            lowered,
        ))
        if len(detected) >= 2 or explicit_complex:
            detected.add("complex")
        unrecognized = not detected

        facets: list[str] = []
        for label in (
            "who", "where", "when", "current_latest", "before_after",
            "why_how", "count_list", "hypothetical_reasoning", "comparison",
            "simultaneous", "nearest_time", "cross_entity",
        ):
            if label not in detected:
                continue
            for facet in self._FACETS[label]:
                if facet not in facets:
                    facets.append(facet)
        if not facets:
            # Unrecognized questions default to a high-recall complex profile
            # instead of the previous simple profile: a wider initial budget and
            # state/cause facets give the Controller explicit coverage targets
            # for would/might/considered/attributes-style reasoning questions.
            facets = ["state", "cause"]

        priorities = self._edge_priorities(detected, lowered)
        complex_question = bool(
            unrecognized
            or detected.intersection({
                "count_list", "why_how", "complex",
                "hypothetical_reasoning", "comparison",
            })
        )
        return QuestionIntent(
            question_types=frozenset(detected),
            required_facets=tuple(facets),
            complexity="complex" if complex_question else "simple",
            edge_priorities=priorities,
            initial_probe_budget=12 if complex_question else 8,
            allow_entity_lookup="count_list" in detected,
            allow_time_overlap_lookup="simultaneous" in detected,
            normalized_question=normalized,
        )

    def action_catalog(
        self,
        intent: QuestionIntent,
        results: Sequence[Mapping[str, Any]],
    ) -> tuple[RetrievalAction, ...]:
        priorities = {edge: index for index, edge in enumerate(intent.edge_priorities)}
        raw: list[tuple[int, int, str, str, str, Mapping[str, Any]]] = []
        seen: set[tuple[str, str, str]] = set()

        for result in results:
            source_ref = str(result.get("result_ref") or "")
            if not source_ref:
                continue
            for option in result.get("available_edges") or ():
                if not isinstance(option, Mapping):
                    continue
                edge_type = str(option.get("edge_type") or "").upper()
                direction = str(option.get("direction") or "").lower()
                if not edge_type or direction not in {"in", "out"}:
                    continue
                key = (source_ref, edge_type, direction)
                if key in seen:
                    continue
                seen.add(key)
                direction_rank = self._direction_rank(
                    intent.question_types, intent.normalized_question, edge_type, direction
                )
                raw.append((
                    priorities.get(edge_type, 999), direction_rank,
                    source_ref, edge_type, direction, {},
                ))

            node_type = str(result.get("node_type") or "")
            has_event_start = bool(
                node_type == "event" and result.get("event_time_start")
            )
            has_event_interval = bool(
                has_event_start and result.get("event_time_end")
            )
            base_constraints = {
                "query": intent.normalized_question,
            }
            event_constraints = {
                **base_constraints,
                "node_types": ["event"],
            }

            def add_virtual(
                priority: int,
                action: str,
                constraints: Mapping[str, Any],
            ) -> None:
                key = (source_ref, action, "lookup")
                if key in seen:
                    return
                seen.add(key)
                raw.append((
                    priority, 0, source_ref, action, "lookup", constraints,
                ))

            if result.get("has_source_evidence"):
                add_virtual(880, "SOURCE_EVIDENCE_LOOKUP", {
                    **base_constraints,
                    "adjacent_turns": 1,
                })

            asks_before, asks_after = self._temporal_direction(
                intent.normalized_question
            )
            if has_event_start and "before_after" in intent.question_types:
                if asks_before:
                    add_virtual(890, "TEMPORAL_BEFORE_LOOKUP", {
                        **event_constraints,
                        "temporal_relation": "before",
                        "max_distance_days": 3650,
                    })
                if asks_after:
                    add_virtual(891, "TEMPORAL_AFTER_LOOKUP", {
                        **event_constraints,
                        "temporal_relation": "after",
                        "max_distance_days": 3650,
                    })
            if has_event_start and "nearest_time" in intent.question_types:
                add_virtual(892, "NEAREST_TIME_LOOKUP", {
                    **event_constraints,
                    "temporal_relation": "nearest",
                    "max_distance_days": 3650,
                })
            if node_type in {"event", "semantic_state"} and (
                "cross_entity" in intent.question_types
            ):
                add_virtual(899, "CROSS_ENTITY_LOOKUP", {
                    **event_constraints,
                    "entity_match": "all",
                    "min_entities": 2,
                })

            if intent.allow_entity_lookup and node_type in {
                "event", "semantic_state"
            }:
                add_virtual(900, "ENTITY_LOOKUP", {
                    **base_constraints,
                    "entity_match": "any",
                })

            if intent.allow_time_overlap_lookup and has_event_interval:
                add_virtual(901, "TIME_OVERLAP_LOOKUP", {
                    **event_constraints,
                    "temporal_relation": "overlap",
                })

        raw.sort(key=lambda item: item[:5])
        return tuple(
            RetrievalAction(
                action_id=f"a{index}",
                source_ref=source_ref,
                action=action,
                direction=direction,
                constraints=dict(constraints),
            )
            for index, (
                _, _, source_ref, action, direction, constraints
            ) in enumerate(raw, 1)
        )

    @staticmethod
    def normalize_query(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(value or ""))
        return " ".join(normalized.split()).strip()

    @staticmethod
    def _edge_priorities(types: set[str], question: str) -> tuple[str, ...]:
        ordered: list[str] = []
        if (
            not types
            or types.intersection({
                "why_how", "hypothetical_reasoning", "comparison",
            })
        ):
            ordered.extend(CAUSAL_EDGES)
        if "before_after" in types:
            ordered.extend(TEMPORAL_EDGES)
        if "current_latest" in types:
            ordered.extend(CURRENT_EDGES)
        if types.intersection({"current_latest", "before_after"}):
            ordered.extend(STATE_HISTORY_EDGES)
        ordered.extend(SUPPORT_EDGES)
        ordered.extend(DEFAULT_EDGE_PRIORITY)
        return tuple(dict.fromkeys(ordered))

    @staticmethod
    def _direction_rank(
        types: frozenset[str], question: str, edge_type: str, direction: str
    ) -> int:
        if edge_type != "TEMPORAL_NEXT" or "before_after" not in types:
            return 0
        asks_before, asks_after = QuestionIntentAnalyzer._temporal_direction(
            question
        )
        preferred = "in" if asks_before and not asks_after else "out"
        return 0 if direction == preferred else 1

    @staticmethod
    def _temporal_direction(question: str) -> tuple[bool, bool]:
        lowered = question.casefold()
        asks_before = bool(re.search(
            r"\bbefore\b|\bprevious(?:ly)?\b|之前|此前|前面|先前|以前",
            lowered,
        ))
        asks_after = bool(re.search(
            r"\bafter\b|\bnext\b|之后|后来|随后|接下来|以后",
            lowered,
        ))
        return asks_before, asks_after


__all__ = [
    "FACETS",
    "QuestionIntent",
    "QuestionIntentAnalyzer",
    "RetrievalAction",
    "VIRTUAL_ACTIONS",
]
