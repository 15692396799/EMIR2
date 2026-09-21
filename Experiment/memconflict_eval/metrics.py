"""MemConflict metrics (point 9).

The experiment reports the three benchmark tables required for EMIR2. All three
are *projections of one judge pass*: the judge is asked once per question and
returns the graded answer score, the conflict-handling flag and the 1-based
rank of the first supporting memory. Nothing below issues a model call.

Table 3 (conflict-aware, the row EMIR2 reports):

    Method | Dynamic AA | Dynamic SEH@K | Static AA | Static SEH@K
           | Conditional AA | Conditional SEH@K | Average AA

Table 5 (black-box performance by conflict type):

    Method | Dynamic AA | Dynamic UOCS | Static AA | Static CRS
           | Conditional AA | Average AA

Table 6 (white-box retrieval and ranking by conflict type):

    Method | Dynamic SEH@K | Dynamic SRS | Static SEH@K | Static SRS
           | Conditional SEH@K | Conditional SRS | Average SEH@K | Average SRS

Definitions used here (identical to the upstream scorer in
``MemConflict/Evaluation/eval_scoring.py``):

* AA  = mean judged answer accuracy (0 / 0.5 / 1 for dynamic and static,
        0 / 1 for conditional).
* SEH@K = share of questions whose supporting memory appears within the top-K
        retrieved memories (the judge returns the 1-based rank of the first
        supporting memory). Out-of-window ranks count as a miss.
* SRS = mean of ``1 / log2(rank + 1)`` inside the same Top-K window; it
        collapses to 0 when the supporting memory is outside the window.
* UOCS / CRS = the dynamic / static conflict-handling diagnostics
        (``update_awareness_and_order_consistency_score`` /
        ``conflict_recognition_score`` upstream). They are binary per question
        and come from the same judge call as AA and the support rank.

``WHITE_BOX_K_VALUES`` mirrors the upstream scorer: the judge is called once
with a Top-5 window and SEH@2 / SEH@3 / SEH@5 are all derived from that single
rank, which is why the columns stay mutually consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


CONFLICT_TYPES = ("dynamic_conflict", "static_conflict", "conditional_conflict")

CONFLICT_LABELS = {
    "dynamic_conflict": "Dynamic",
    "static_conflict": "Static",
    "conditional_conflict": "Conditional",
}

# Upstream (``eval_scoring.WHITE_BOX_TOP_K_VALUES``) derives every white-box
# column from one Top-5 ranking pass. Keep the same window set so a re-scored
# run can report @2 / @3 / @5 without a second judge call.
WHITE_BOX_K_VALUES = (2, 3, 5)


def srs_from_rank(rank: int) -> float:
    """Rank-weighted retrieval score for one question (0 when not retrieved)."""
    if rank < 1:
        return 0.0
    return 1.0 / math.log2(float(rank) + 1.0)


def srs_at_k(rank: int, top_k: int) -> float:
    """``srs_from_rank`` inside one Top-K window (0 when the rank is outside)."""
    if top_k < 1 or rank < 1 or rank > top_k:
        return 0.0
    return srs_from_rank(rank)


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
    white_box_k: int = 3

    @property
    def uocs(self) -> float | None:
        """Table 5's dynamic column (update awareness and order consistency)."""
        return self.conflict_handling if self.conflict_type == "dynamic_conflict" else None

    @property
    def crs(self) -> float | None:
        """Table 5's static column (conflict recognition score)."""
        return self.conflict_handling if self.conflict_type == "static_conflict" else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "Question_Count": self.question_count,
            "Answer_Accuracy": self.answer_accuracy,
            f"SEH@{self.white_box_k}": self.seh_at_3,
            "SRS": self.srs,
            "Conflict_Handling": self.conflict_handling,
            "White_Box_K": self.white_box_k,
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
            f"Average_SEH@{self.white_box_k}": self.average_seh_at_3,
            "Average_SRS": self.average_srs,
            "By_Conflict_Type": {
                conflict_type: metrics.to_dict()
                for conflict_type, metrics in self.by_conflict_type.items()
            },
        }

    @property
    def white_box_k(self) -> int:
        """The Top-K window every SEH/SRS number in this aggregate was cut at."""
        for metrics in self.by_conflict_type.values():
            return metrics.white_box_k
        return 3


def aggregate(
    scored_questions: Iterable[dict[str, Any]],
    white_box_k: int = 3,
) -> BenchmarkMetrics:
    """Aggregate per-question judge output into the benchmark table.

    Each item must provide ``conflict_type`` plus the judge fields
    ``answer_accuracy``, ``support_rank`` and ``conflict_handling``.

    ``white_box_k`` is the Top-K window of the white-box columns; the judge's
    own ranking window must be at least this large (see
    ``run_scoring.py --judge-top-k``) or the ranks were already truncated.
    """
    white_box_k = int(white_box_k)
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
            result.by_conflict_type[conflict_type] = ConflictMetrics(
                conflict_type, white_box_k=white_box_k
            )
            continue
        accuracies = [float(row.get("answer_accuracy") or 0.0) for row in rows]
        ranks = [int(row.get("support_rank") or 0) for row in rows]
        handling = [int(row.get("conflict_handling") or 0) for row in rows]
        hits = [1.0 if 1 <= rank <= white_box_k else 0.0 for rank in ranks]
        handling_mean = _mean([float(value) for value in handling])
        if conflict_type == "conditional_conflict":
            # Table 5 defines no diagnostic for conditional questions and the
            # judge prompt asks for none either, so reporting the flag the
            # judge emits anyway would be noise.
            handling_mean = None
        result.by_conflict_type[conflict_type] = ConflictMetrics(
            conflict_type=conflict_type,
            question_count=len(rows),
            answer_accuracy=_mean(accuracies),
            seh_at_3=_mean(hits),
            srs=_mean([srs_at_k(rank, white_box_k) for rank in ranks]),
            conflict_handling=handling_mean,
            judge_error_count=sum(1 for row in rows if row.get("judge_error")),
            white_box_k=white_box_k,
        )
    return result


def aggregate_by_k(
    scored_questions: Iterable[dict[str, Any]],
    k_values: Sequence[int] = WHITE_BOX_K_VALUES,
) -> dict[str, BenchmarkMetrics]:
    """White-box columns for several Top-K windows from one judging pass.

    Upstream calls the judge once with the largest window and derives the
    smaller windows from the same support rank, so ``aggregate_by_k`` is a pure
    re-projection: it never re-asks the model and never changes AA / UOCS / CRS.
    """
    rows = list(scored_questions)
    return {
        str(int(top_k)): aggregate(rows, white_box_k=int(top_k))
        for top_k in k_values
    }


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.4f}"


def _conflict_rows(
    metrics: BenchmarkMetrics,
) -> tuple[ConflictMetrics, ConflictMetrics, ConflictMetrics]:
    """The Dynamic / Static / Conditional buckets, with empty defaults."""
    return tuple(  # type: ignore[return-value]
        metrics.by_conflict_type.get(
            conflict_type, ConflictMetrics(conflict_type, white_box_k=metrics.white_box_k)
        )
        for conflict_type in CONFLICT_TYPES
    )


def render_table3(metrics: BenchmarkMetrics, method_name: str = "EMIR²") -> str:
    """Render the conflict-aware table in the paper's Table 3 layout."""
    dynamic, static, conditional = _conflict_rows(metrics)
    seh = f"SEH@{metrics.white_box_k}↑"
    lines = [
        f"| Method | Dynamic AA↑ | Dynamic {seh} | Static AA↑ | Static {seh} | "
        f"Conditional AA↑ | Conditional {seh} | Average AA↑ |",
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


def render_table5(metrics: BenchmarkMetrics, method_name: str = "EMIR²") -> str:
    """Table 5: black-box performance by conflict type.

    Dynamic questions report AA and UOCS, static questions AA and CRS, and
    conditional questions AA only, followed by the mean AA. Every cell is read
    off the same aggregate as Table 3, so the AA columns are identical by
    construction.
    """
    dynamic, static, conditional = _conflict_rows(metrics)
    lines = [
        "| Method | Dynamic AA↑ | Dynamic UOCS↑ | Static AA↑ | Static CRS↑ | "
        "Conditional AA↑ | Average AA↑ |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| {name} | {daa} | {duocs} | {saa} | {scrs} | {caa} | {avg} |".format(
            name=method_name,
            daa=_fmt(dynamic.answer_accuracy),
            duocs=_fmt(dynamic.uocs),
            saa=_fmt(static.answer_accuracy),
            scrs=_fmt(static.crs),
            caa=_fmt(conditional.answer_accuracy),
            avg=_fmt(metrics.average_aa),
        ),
    ]
    return "\n".join(lines)


def render_table6(metrics: BenchmarkMetrics, method_name: str = "EMIR²") -> str:
    """Table 6: white-box retrieval and ranking by conflict type.

    SEH@K and SRS come from the same support rank the judge returned for
    Table 3, so the SEH@K column is identical to Table 3's by construction.
    """
    dynamic, static, conditional = _conflict_rows(metrics)
    seh = f"SEH@{metrics.white_box_k}↑"
    lines = [
        f"| Method | Dynamic {seh} | Dynamic SRS↑ | Static {seh} | Static SRS↑ | "
        f"Conditional {seh} | Conditional SRS↑ | Average {seh} | Average SRS↑ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        "| {name} | {dseh} | {dsrs} | {sseh} | {ssrs} | {cseh} | {csrs} | "
        "{aseh} | {asrs} |".format(
            name=method_name,
            dseh=_fmt(dynamic.seh_at_3),
            dsrs=_fmt(dynamic.srs),
            sseh=_fmt(static.seh_at_3),
            ssrs=_fmt(static.srs),
            cseh=_fmt(conditional.seh_at_3),
            csrs=_fmt(conditional.srs),
            aseh=_fmt(metrics.average_seh_at_3),
            asrs=_fmt(metrics.average_srs),
        ),
    ]
    return "\n".join(lines)


def render_white_box_by_k(
    by_k: Mapping[str, BenchmarkMetrics],
    method_name: str = "EMIR²",
) -> str:
    """One row per Top-K window, mirroring upstream's ``White_Box_By_K``."""
    lines = [
        "| Method | Top-K | Dynamic SEH↑ | Dynamic SRS↑ | Static SEH↑ | Static SRS↑ | "
        "Conditional SEH↑ | Conditional SRS↑ | Average SEH↑ | Average SRS↑ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in sorted(by_k, key=lambda value: int(value)):
        metrics = by_k[key]
        dynamic, static, conditional = _conflict_rows(metrics)
        lines.append(
            "| {name} | @{k} | {dseh} | {dsrs} | {sseh} | {ssrs} | {cseh} | {csrs} | "
            "{aseh} | {asrs} |".format(
                name=method_name,
                k=int(key),
                dseh=_fmt(dynamic.seh_at_3),
                dsrs=_fmt(dynamic.srs),
                sseh=_fmt(static.seh_at_3),
                ssrs=_fmt(static.srs),
                cseh=_fmt(conditional.seh_at_3),
                csrs=_fmt(conditional.srs),
                aseh=_fmt(metrics.average_seh_at_3),
                asrs=_fmt(metrics.average_srs),
            )
        )
    return "\n".join(lines)


def render_all_tables(
    metrics: BenchmarkMetrics,
    method_name: str = "EMIR²",
    by_k: Mapping[str, BenchmarkMetrics] | None = None,
) -> str:
    """The three paper tables in one markdown document."""
    sections = [
        "# MemConflict tables 3 / 5 / 6 (one judge pass)\n",
        f"> Method: {method_name}. White-box window: Top-{metrics.white_box_k}.\n",
        "## Table 3: conflict-aware evaluation\n",
        render_table3(metrics, method_name),
        "\n## Table 5: black-box performance\n",
        render_table5(metrics, method_name),
        "\n## Table 6: white-box retrieval and ranking\n",
        render_table6(metrics, method_name),
        "\n## Table 6 detail: white-box by Top-K\n",
        render_white_box_by_k(by_k or {str(metrics.white_box_k): metrics}, method_name),
        "\n## Detail behind the rows\n",
        render_detail_table(metrics),
    ]
    return "\n".join(sections) + "\n"


def render_detail_table(metrics: BenchmarkMetrics) -> str:
    """Extra columns produced by the same judge call (Tables 5/6 material)."""
    lines = [
        f"| Conflict Type | Questions | AA↑ | SEH@{metrics.white_box_k}↑ | SRS↑ | "
        "Conflict Handling↑ | Judge Errors |",
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
