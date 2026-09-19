"""MemConflict metrics (point 9).

The experiment reports the conflict-aware table required for EMIR2: for each of
the three conflict types an answer-accuracy column (AA) and a white-box
retrieval-hit column (SEH@3), plus the mean AA across conflict types.

    Method | Dynamic AA | Dynamic SEH@3 | Static AA | Static SEH@3
           | Conditional AA | Conditional SEH@3 | Average AA

Definitions used here:

* AA  = mean judged answer accuracy (0 / 0.5 / 1 for dynamic and static,
        0 / 1 for conditional).
* SEH@3 = share of questions whose supporting memory appears within the top-3
        retrieved memories (the judge returns the 1-based rank of the first
        supporting memory).
* SRS = mean of ``1 / log2(rank + 1)``, the rank-weighted retrieval score.
* UOCS / CRS = the dynamic / static conflict-handling diagnostics. They are
        reported as extra columns because the same judge call produces them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


CONFLICT_TYPES = ("dynamic_conflict", "static_conflict", "conditional_conflict")

CONFLICT_LABELS = {
    "dynamic_conflict": "Dynamic",
    "static_conflict": "Static",
    "conditional_conflict": "Conditional",
}


def srs_from_rank(rank: int) -> float:
    """Rank-weighted retrieval score for one question (0 when not retrieved)."""
    if rank < 1:
        return 0.0
    return 1.0 / math.log2(float(rank) + 1.0)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


@dataclass
class ConflictMetrics:
    conflict_type: str
    question_count: int = 0
    answer_accuracy: float | None = None
    seh_at_3: float | None = None
    srs: float | None = None
    conflict_handling: float | None = None
    judge_error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "Question_Count": self.question_count,
            "Answer_Accuracy": self.answer_accuracy,
            "SEH@3": self.seh_at_3,
            "SRS": self.srs,
            "Conflict_Handling": self.conflict_handling,
            "Judge_Error_Count": self.judge_error_count,
        }


@dataclass
class BenchmarkMetrics:
    by_conflict_type: dict[str, ConflictMetrics] = field(default_factory=dict)
    question_count: int = 0
    judge_error_count: int = 0

    @property
    def average_aa(self) -> float | None:
        return _mean(
            [
                metrics.answer_accuracy
                for metrics in self.by_conflict_type.values()
                if metrics.answer_accuracy is not None
            ]
        )

    @property
    def average_seh_at_3(self) -> float | None:
        return _mean(
            [
                metrics.seh_at_3
                for metrics in self.by_conflict_type.values()
                if metrics.seh_at_3 is not None
            ]
        )

    @property
    def average_srs(self) -> float | None:
        return _mean(
            [
                metrics.srs
                for metrics in self.by_conflict_type.values()
                if metrics.srs is not None
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "Question_Count": self.question_count,
            "Judge_Error_Count": self.judge_error_count,
            "Average_AA": self.average_aa,
            "Average_SEH@3": self.average_seh_at_3,
            "Average_SRS": self.average_srs,
            "By_Conflict_Type": {
                conflict_type: metrics.to_dict()
                for conflict_type, metrics in self.by_conflict_type.items()
            },
        }


def aggregate(scored_questions: Iterable[dict[str, Any]]) -> BenchmarkMetrics:
    """Aggregate per-question judge output into the benchmark table.

    Each item must provide ``conflict_type`` plus the judge fields
    ``answer_accuracy``, ``support_rank`` and ``conflict_handling``.
    """
    buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in CONFLICT_TYPES}
    total = 0
    errors = 0
    for item in scored_questions:
        conflict_type = str(item.get("conflict_type") or "")
        if conflict_type not in buckets:
            continue
        buckets[conflict_type].append(item)
        total += 1
        if item.get("judge_error"):
            errors += 1

    result = BenchmarkMetrics(question_count=total, judge_error_count=errors)
    for conflict_type in CONFLICT_TYPES:
        rows = buckets[conflict_type]
        if not rows:
            result.by_conflict_type[conflict_type] = ConflictMetrics(conflict_type)
            continue
        accuracies = [float(row.get("answer_accuracy") or 0.0) for row in rows]
        ranks = [int(row.get("support_rank") or 0) for row in rows]
        handling = [int(row.get("conflict_handling") or 0) for row in rows]
        hits = [1.0 if 1 <= rank <= 3 else 0.0 for rank in ranks]
        result.by_conflict_type[conflict_type] = ConflictMetrics(
            conflict_type=conflict_type,
            question_count=len(rows),
            answer_accuracy=_mean(accuracies),
            seh_at_3=_mean(hits),
            srs=_mean([srs_from_rank(rank) for rank in ranks]),
            conflict_handling=_mean([float(value) for value in handling]),
            judge_error_count=sum(1 for row in rows if row.get("judge_error")),
        )
    return result


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.4f}"


def render_table3(metrics: BenchmarkMetrics, method_name: str = "EMIR²") -> str:
    """Render the conflict-aware table in the paper's Table 3 layout."""
    dynamic = metrics.by_conflict_type.get("dynamic_conflict", ConflictMetrics("dynamic_conflict"))
    static = metrics.by_conflict_type.get("static_conflict", ConflictMetrics("static_conflict"))
    conditional = metrics.by_conflict_type.get(
        "conditional_conflict", ConflictMetrics("conditional_conflict")
    )
    lines = [
        "| Method | Dynamic AA↑ | Dynamic SEH@3↑ | Static AA↑ | Static SEH@3↑ | "
        "Conditional AA↑ | Conditional SEH@3↑ | Average AA↑ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        "| {name} | {daa} | {dseh} | {saa} | {sseh} | {caa} | {cseh} | {avg} |".format(
            name=method_name,
            daa=_fmt(dynamic.answer_accuracy),
            dseh=_fmt(dynamic.seh_at_3),
            saa=_fmt(static.answer_accuracy),
            sseh=_fmt(static.seh_at_3),
            caa=_fmt(conditional.answer_accuracy),
            cseh=_fmt(conditional.seh_at_3),
            avg=_fmt(metrics.average_aa),
        ),
    ]
    return "\n".join(lines)


def render_detail_table(metrics: BenchmarkMetrics) -> str:
    """Extra columns produced by the same judge call (Tables 5/6 material)."""
    lines = [
        "| Conflict Type | Questions | AA↑ | SEH@3↑ | SRS↑ | Conflict Handling↑ | Judge Errors |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for conflict_type in CONFLICT_TYPES:
        item = metrics.by_conflict_type.get(conflict_type, ConflictMetrics(conflict_type))
        lines.append(
            "| {label} | {count} | {aa} | {seh} | {srs} | {handling} | {errors} |".format(
                label=CONFLICT_LABELS[conflict_type],
                count=item.question_count,
                aa=_fmt(item.answer_accuracy),
                seh=_fmt(item.seh_at_3),
                srs=_fmt(item.srs),
                handling=_fmt(item.conflict_handling),
                errors=item.judge_error_count,
            )
        )
    lines.append(
        "| **Average** | {count} | {aa} | {seh} | {srs} | – | {errors} |".format(
            count=metrics.question_count,
            aa=_fmt(metrics.average_aa),
            seh=_fmt(metrics.average_seh_at_3),
            srs=_fmt(metrics.average_srs),
            errors=metrics.judge_error_count,
        )
    )
    return "\n".join(lines)
