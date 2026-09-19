from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory.v4.memory_system import MemorySystem
from memory.v4.config import FailurePolicyConfig, V4MemoryConfig
from memory.v4.errors import MemoryNotReadyError
from memory.v4.failure import (
    V4AnswerError,
    V4BuildStageError,
    V4RetrievalError,
    V4TemporalCorrectionError,
)
from memory.v4.schemas import (
    MemoryEdge,
    MemoryNode,
    Participant,
    ParticipantScope,
    SemanticFact,
    TimeConstraint,
    TopicChain,
)
from memory.v4.time_normalizer import TimeNormalization, TimeNormalizer
from memory.v4.semantic import (
    PendingConflict,
    SemanticEpoch,
    SemanticEpochMachine,
    SemanticFactRecord,
    SemanticReducer,
    SemanticTransition,
    SemanticTimeline,
    SemanticValidationError,
    fact_key,
    revision_state_id,
)

__all__ = [
    "MemoryEdge",
    "MemoryNode",
    "MemoryNotReadyError",
    "MemorySystem",
    "FailurePolicyConfig",
    "V4AnswerError",
    "V4BuildStageError",
    "V4RetrievalError",
    "V4TemporalCorrectionError",
    "V4MemoryConfig",
    "Participant",
    "ParticipantScope",
    "SemanticFact",
    "TimeConstraint",
    "TopicChain",
    "TimeNormalization",
    "TimeNormalizer",
    "PendingConflict",
    "SemanticEpoch",
    "SemanticEpochMachine",
    "SemanticFactRecord",
    "SemanticReducer",
    "SemanticTransition",
    "SemanticTimeline",
    "SemanticValidationError",
    "fact_key",
    "revision_state_id",
]


def __getattr__(name: str) -> Any:
    if name == "MemorySystem":
        from memory.v4.memory_system import MemorySystem

        return MemorySystem
    raise AttributeError(name)
