"""Version-neutral public memory result dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


_ANSWER_CONTEXT_MODES = {"summary only", "content aware"}


def content_aware_prompt_lines(
    item: "EpisodicMemoryItem", indent: str = "   "
) -> list[str]:
    """Render full memory content and its exact source conversation records."""
    lines: list[str] = []
    if item.text and item.text != item.summary:
        lines.append(f"{indent}memory content: {item.text}")

    records = item.metadata.get("source_evidence") or []
    if not isinstance(records, list) or not records:
        return lines
    lines.append(f"{indent}associated original evidence:")
    seen: set[tuple[str, ...]] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        content = str(record.get("content") or "")
        if not content:
            continue
        identity = tuple(
            str(record.get(key) or "")
            for key in (
                "session_id", "turn_id", "role", "participant_id",
                "observed_at", "content",
            )
        )
        if identity in seen:
            continue
        seen.add(identity)
        provenance = "; ".join(
            f"{label}={record.get(key)}"
            for label, key in (
                ("session", "session_id"),
                ("turn", "turn_id"),
                ("speaker", "role"),
                ("participant", "participant_id"),
                ("observed_at", "observed_at"),
            )
            if record.get(key) not in (None, "")
        )
        prefix = f" [{provenance}]" if provenance else ""
        lines.append(f"{indent}  -{prefix} {content}")
    if lines[-1] == f"{indent}associated original evidence:":
        lines.pop()
    return lines


@dataclass
class SemanticMemoryItem:
    id: str
    namespace: str
    subject: str
    predicate: str
    object: str
    qualifiers: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    evidence_node_ids: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    updated_at: str | None = None
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodicMemoryItem:
    node_id: str
    node_type: str
    depth: int
    title: str
    summary: str
    text: str
    timestamp_start: str | None = None
    timestamp_end: str | None = None
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    source_refs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    semantic_memories: list[SemanticMemoryItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["semantic_memories"] = [
            item.to_dict() for item in self.semantic_memories
        ]
        return value


@dataclass
class NavigationItem:
    navigation_id: str
    action: str
    target_node_id: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    allowed_depth_range: tuple[int, int] = (0, 4)
    label: str = ""
    reason: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_depth_range"] = list(self.allowed_depth_range)
        return value


@dataclass
class RetrievalRoundTrace:
    round_index: int
    budget: int
    action: str
    query_plan: dict[str, Any] | None = None
    selected_navigation_ids: list[str] = field(default_factory=list)
    forced_constraints: dict[str, Any] = field(default_factory=dict)
    returned_node_ids: list[str] = field(default_factory=list)
    navigation_bar: list[NavigationItem] = field(default_factory=list)
    sufficient: bool | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["navigation_bar"] = [
            item.to_dict() for item in self.navigation_bar
        ]
        return value


@dataclass
class RetrievalBundle:
    question: str
    namespace: str
    episodic_memories: list[EpisodicMemoryItem] = field(default_factory=list)
    trace: list[RetrievalRoundTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "namespace": self.namespace,
            "episodic_memories": [
                item.to_dict() for item in self.episodic_memories
            ],
            "trace": [item.to_dict() for item in self.trace],
        }

    def to_prompt_context(
        self,
        include_trace: bool = False,
        answer_context_mode: str = "summary only",
    ) -> str:
        answer_context_mode = str(answer_context_mode or "summary only").strip().lower()
        if answer_context_mode not in _ANSWER_CONTEXT_MODES:
            raise ValueError(
                'answer_context_mode must be "summary only" or "content aware"'
            )
        lines = ["Retrieved memories:"]
        for index, item in enumerate(self.episodic_memories, 1):
            lines.append(
                f"{index}. [{item.node_type}] {item.title}: "
                f"{item.summary or item.text}"
            )
            if answer_context_mode == "content aware":
                lines.extend(content_aware_prompt_lines(item))
        if not self.episodic_memories:
            lines.append("- None")
        if include_trace:
            lines.append("Retrieval trace:")
            for trace in self.trace:
                lines.append(
                    f"- round {trace.round_index}: {trace.action}, returned: "
                    f"{', '.join(trace.returned_node_ids)}"
                )
        return "\n".join(lines)


__all__ = [
    "EpisodicMemoryItem",
    "NavigationItem",
    "RetrievalBundle",
    "RetrievalRoundTrace",
    "SemanticMemoryItem",
]
