from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from evaluation.scoring import score_references


@runtime_checkable
class BenchmarkEvaluator(Protocol):
    name: str

    def load_examples(self, source: str | Path) -> list[dict[str, Any]]: ...
    def build_memory_inputs(self, example: dict[str, Any]) -> Any: ...
    def build_queries(self, example: dict[str, Any]) -> list[dict[str, Any]]: ...
    def score_answer(self, prediction: str, references: Iterable[str]) -> dict[str, float]: ...
    def aggregate(self, scores: Iterable[dict[str, float]]) -> dict[str, float]: ...


class LocomoEvaluator:
    name = "locomo"

    def load_examples(self, source: str | Path) -> list[dict[str, Any]]:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("LoCoMo source must be a JSON array")
        return [dict(item) for item in value]

    def build_memory_inputs(self, example: dict[str, Any]) -> Any:
        return example.get("conversation") or example.get("conversations") or example.get("dialogue") or []

    def build_queries(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        return [dict(item) for item in (example.get("qa") or example.get("questions") or [])]

    def score_answer(self, prediction: str, references: Iterable[str]) -> dict[str, float]:
        return score_references(prediction, tuple(references))

    def aggregate(self, scores: Iterable[dict[str, float]]) -> dict[str, float]:
        rows = list(scores)
        if not rows:
            return {"exact_match": 0.0, "token_f1": 0.0, "count": 0}
        return {
            "exact_match": sum(row.get("exact_match", 0.0) for row in rows) / len(rows),
            "token_f1": sum(row.get("token_f1", 0.0) for row in rows) / len(rows),
            "count": len(rows),
        }


class ProactiveMemBenchEvaluator(LocomoEvaluator):
    name = "proactive_membench"


_EVALUATORS: dict[str, BenchmarkEvaluator] = {
    "locomo": LocomoEvaluator(),
    "proactive_membench": ProactiveMemBenchEvaluator(),
}


def get_benchmark_evaluator(name: str) -> BenchmarkEvaluator:
    try:
        return _EVALUATORS[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark: {name}") from exc
