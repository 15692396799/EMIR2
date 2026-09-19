from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from memory.time_utils import format_time_display
from memory.results import RetrievalBundle, content_aware_prompt_lines


class RetrievalResult(RetrievalBundle):
    """Backend-neutral result with the v1 serialization and prompt contract."""

    def __init__(self, *args: Any, trajectory: Any = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.trajectory = trajectory

    @classmethod
    def from_bundle(cls, bundle: RetrievalBundle) -> "RetrievalResult":
        if isinstance(bundle, cls):
            return bundle
        return cls(
            question=bundle.question,
            namespace=bundle.namespace,
            episodic_memories=bundle.episodic_memories,
            trace=bundle.trace,
            trajectory=getattr(bundle, "trajectory", None),
        )

    def to_prompt_context(
        self, include_trace: bool = False, answer_context_mode: str = "summary only"
    ) -> str:
        answer_context_mode = str(answer_context_mode or "summary only").strip().lower()
        if answer_context_mode not in {"summary only", "content aware"}:
            raise ValueError('answer_context_mode must be "summary only" or "content aware"')
        lines = ["[Time Evidence]"]
        time_lines = _time_evidence_lines(self.episodic_memories)
        lines.extend(time_lines or ["- None"])
        lines.extend(["", "[Memories]"])
        if not self.episodic_memories:
            lines.append("- None")
        for index, item in enumerate(self.episodic_memories, start=1):
            text = item.summary or item.text
            conflict = " [CONFLICTED]" if item.metadata.get("unresolved_conflict") else ""
            lines.append(f"{index}. [{item.node_type}]{conflict} {item.title}: {text}")
            displayed_time = _display_memory_time(item.metadata)
            if displayed_time:
                lines.append(f"   absolute time: {displayed_time}")
            elif item.timestamp_start or item.timestamp_end:
                lines.append(f"   time: {item.timestamp_start or ''} - {item.timestamp_end or ''}")
            if item.semantic_memories:
                semantic = "; ".join(
                    f"{memory.subject} {memory.predicate} {memory.object}"
                    for memory in item.semantic_memories
                )
                lines.append(f"   semantic: {semantic}")
            if answer_context_mode == "content aware":
                lines.extend(content_aware_prompt_lines(item))
        if include_trace:
            lines.extend(["", "Retrieval trace:"])
            for trace in self.trace:
                lines.append(
                    f"- round {trace.round_index}: {trace.action}, returned: {', '.join(trace.returned_node_ids)}"
                )
        return "\n".join(lines)


_UNRELIABLE_TIME_STATUSES = frozenset({
    "unresolved",
    "resolved_relative_approximate",
    "resolved_relative_defaulted",
})


def _time_evidence_lines(items: list[Any]) -> list[str]:
    lines = []
    seen = set()
    for item in items:
        metadata = item.metadata
        raw = str(metadata.get("raw_time_expression") or "").strip()
        status = str(metadata.get("normalization_status") or "")
        if status in _UNRELIABLE_TIME_STATUSES:
            if not raw:
                continue
            line = (
                f"- temporal expression: {json_string(raw)} "
                "(approximate; no reliable absolute date)"
            )
        else:
            displayed = _display_memory_time(metadata)
            if not displayed:
                continue
            precision = metadata.get("time_precision") or "unknown"
            label = json_string(raw) if raw else json_string(item.title or item.node_id)
            line = f"- {label} -> {displayed} (precision: {precision})"
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def json_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _display_memory_time(metadata: dict[str, Any]) -> str | None:
    return format_time_display(
        metadata.get("absolute_time_start"),
        metadata.get("absolute_time_end"),
        metadata.get("time_precision"),
    )


@runtime_checkable
class MemoryBackend(Protocol):
    backend_id: str
    capabilities: Any

    def ingest(
        self, namespace: str, conversation: Any, metadata: dict[str, Any] | None = None,
        *, participants: list[dict[str, str]] | None = None,
    ) -> None:
        ...

    def ingest_conversation(
        self,
        namespace: str,
        conversation: Any,
        metadata: dict[str, Any] | None = None,
        *,
        participants: list[dict[str, str]] | None = None,
    ) -> None:
        ...

    def global_search(self, execution: Any, probes: Any) -> Any:
        ...

    def expand(self, execution: Any, anchors: Any, expansion: Any) -> Any:
        ...

    def finalize(
        self, execution: Any, candidates: Any, question: str,
        final_top_k: int, trace: Any,
    ) -> RetrievalResult:
        ...


    def is_namespace_ready(
        self, namespace: str, *, participants: list[dict[str, str]] | None = None
    ) -> bool:
        ...

    def is_ready(self, namespace: str, *, participants: list[dict[str, str]] | None = None) -> bool:
        ...

    def close(self) -> None:
        ...
