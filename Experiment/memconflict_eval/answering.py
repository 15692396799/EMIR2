"""Answer generation for MemConflict questions (point 8, answer side).

Mirrors ``agent.agent.generate_answer`` but routes on ``conflict_type`` instead
of LoCoMo category and uses the MemConflict answer prompts. The chat client is
the memory system's configured ``answer_model``, built by Retrival-Mem's own
``make_chat_client`` so no code in that checkout needs changing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from . import runtime
from .data import Question
from .memory import RetrievedMemory
from .prompts import build_answer_messages, format_memory_context


@dataclass
class AnswerResult:
    text: str
    duration_ms: float
    context: str

    def to_dict(self) -> dict[str, Any]:
        return {"Model_Answer": self.text, "Answer_Duration_ms": self.duration_ms}


class MemConflictAnswerer:
    """Wraps the Retrival-Mem answer model with the MemConflict prompt."""

    def __init__(self, config: Any = None) -> None:
        self.config = config if config is not None else runtime.load_memory_config()
        # runtime.build_chat_client handles the azure provider; everything else
        # is the unmodified Retrival-Mem factory.
        self.client = runtime.build_chat_client(self.config.answer_model)

    def answer(
        self,
        question: Question,
        memories: list[RetrievedMemory],
        *,
        namespace: str = "",
        memory_context: str | None = None,
    ) -> AnswerResult:
        context = (
            memory_context
            if memory_context is not None
            else format_memory_context([memory.to_dict() for memory in memories], namespace)
        )
        messages = build_answer_messages(question.question, context, question.conflict_type)
        start = time.perf_counter()
        text = self.client.chat(messages)
        duration_ms = (time.perf_counter() - start) * 1000.0
        return AnswerResult(text=str(text).strip(), duration_ms=duration_ms, context=context)
