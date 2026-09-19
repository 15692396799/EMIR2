from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol

if TYPE_CHECKING:
    from memory.base import MemoryBackend, RetrievalResult


@dataclass(frozen=True)
class RetrievalRequest:
    question: str
    namespace: str
    context: Any = None
    participants: tuple[dict[str, str], ...] = ()
    strategy: str | None = None
    controller_profile: str | None = None
    capture_trajectory: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def execution_context(self) -> "RetrievalExecutionContext":
        return RetrievalExecutionContext(
            namespace=self.namespace,
            participants=self.participants,
            user_context=self.context,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class RetrievalExecutionContext:
    namespace: str
    participants: tuple[dict[str, str], ...] = ()
    user_context: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RetrievalStrategy(Protocol):
    strategy_id: str

    def retrieve(
        self,
        backend: MemoryBackend,
        request: RetrievalRequest,
        config: Mapping[str, Any],
    ) -> RetrievalResult:
        ...
