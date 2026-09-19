"""LLM judge for MemConflict answers (point 8, judge side).

Replaces the LoCoMo ``MEM0_ACCURACY_PROMPT`` binary CORRECT/WRONG judgement with
a graded answer score, the conflict-handling diagnostic, and the white-box
support rank that the benchmark's AA / SEH@K / SRS metrics need.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from . import runtime
from .data import Question
from .memory import RetrievedMemory
from .prompts import (
    build_judge_messages,
    extract_json_object,
    judge_json_schema,
)


@dataclass
class JudgeResult:
    answer_accuracy: float
    conflict_handling: int
    support_rank: int
    reasoning: str
    duration_ms: float
    raw_response: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "Answer_Accuracy": self.answer_accuracy,
            "Conflict_Handling": self.conflict_handling,
            "Support_Rank": self.support_rank,
            "Reasoning": self.reasoning,
            "Judge_Duration_ms": self.duration_ms,
            "Judge_Error": self.error,
        }


def _coerce_accuracy(value: Any, conflict_type: str) -> float:
    """Mirror the benchmark's trinary collapse and its per-type granularity."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if conflict_type == "conditional_conflict":
        # Conditional answers are scored all-or-nothing.
        return 1.0 if number >= 0.75 else 0.0
    if number >= 0.75:
        return 1.0
    if number >= 0.25:
        return 0.5
    return 0.0


def _coerce_flag(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return 1 if int(value) != 0 else 0
    except (TypeError, ValueError):
        return 0


def _coerce_rank(value: Any, top_k: int) -> int:
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return 0
    return rank if 1 <= rank <= top_k else 0


class MemConflictJudge:
    """Wraps Retrival-Mem's configured judge model with the MemConflict prompt."""

    def __init__(self, config: Any = None, *, top_k: int = 3) -> None:
        client = runtime.import_retrival_mem()
        self.config = config if config is not None else runtime.load_memory_config()
        self.client = client.make_chat_client(self.config.judge_model)
        self.top_k = int(top_k)

    def judge(
        self,
        question: Question,
        model_answer: str,
        memories: list[RetrievedMemory],
    ) -> JudgeResult:
        messages = build_judge_messages(
            question=question.question,
            gold_answer=question.answer,
            model_answer=model_answer,
            conflict_type=question.conflict_type,
            memories=[memory.to_dict() for memory in memories],
            top_k=self.top_k,
        )
        start = time.perf_counter()
        try:
            raw = self.client.chat(
                messages, json_mode=True, json_schema=judge_json_schema(self.top_k)
            )
            duration_ms = (time.perf_counter() - start) * 1000.0
            payload = extract_json_object(raw)
        except Exception as error:  # keep the run alive, record the failure
            duration_ms = (time.perf_counter() - start) * 1000.0
            return JudgeResult(
                answer_accuracy=0.0,
                conflict_handling=0,
                support_rank=0,
                reasoning="",
                duration_ms=duration_ms,
                raw_response=str(locals().get("raw", "")),
                error=f"{type(error).__name__}: {error}",
            )
        return JudgeResult(
            answer_accuracy=_coerce_accuracy(
                payload.get("answer_accuracy"), question.conflict_type
            ),
            conflict_handling=_coerce_flag(payload.get("conflict_handling")),
            support_rank=_coerce_rank(payload.get("support_rank"), self.top_k),
            reasoning=str(payload.get("reasoning") or ""),
            duration_ms=duration_ms,
            raw_response=str(raw),
        )
