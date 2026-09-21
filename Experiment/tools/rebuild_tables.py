"""Re-project an already-scored run into Tables 3 / 5 / 6, without any LLM call.

``run_scoring.py`` writes one ``Evaluation`` object per question containing the
graded answer score, the conflict-handling flag and the support rank. Every
cell of the three paper tables is a function of those three numbers, so the
tables can be rebuilt offline from ``scores.jsonl``:

* Table 3 - AA and SEH@K per conflict type, plus Average AA;
* Table 5 - AA with the UOCS (dynamic) / CRS (static) diagnostics;
* Table 6 - SEH@K and SRS per conflict type, plus the two averages.

This is the tool to use when the *presentation* changes but the judging does
not: no judge call, no credentials, no GPU. It is also the cheapest way to
check that a code change did not silently alter the metrics.

The guarantee and its boundary:

* every column is a function of ``{Answer_Accuracy, Conflict_Handling,
  Support_Rank, conflict_type}``, so all three tables rebuild exactly;
* a white-box window can only be rebuilt inside the window the judge already
  saw. Ranks past it were recorded as 0, so a wider window is a lower bound,
  not a measurement.

Usage::

    python Experiment/tools/rebuild_tables.py --run-dir Experiment/runs/<timestamp>
    python Experiment/tools/rebuild_tables.py --scores <scores.jsonl> --white-box-k 2,3,5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memconflict_eval.metrics import (  # noqa: E402
    CONFLICT_TYPES,
    aggregate,
    aggregate_by_k,
    render_all_tables,
    render_table3,
    render_table5,
    render_table6,
    render_white_box_by_k,
)


def read_scores(path: Path) -> list[dict[str, Any]]:
    """Load the per-persona rows of a ``scores.jsonl`` file."""
    personas: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                personas.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
    return personas


def rows_from_scored_personas(
    personas: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten scored personas into the rows ``metrics.aggregate`` consumes."""
    rows: list[dict[str, Any]] = []
    for persona in personas:
        for session in persona.get("Sessions") or []:
            for question in session.get("Questions") or []:
                evaluation = question.get("Evaluation")
                present = isinstance(evaluation, dict)
                if not isinstance(evaluation, dict):
                    evaluation = {}
                    error: Any = "missing_evaluation"
                else:
                    error = evaluation.get("Judge_Error")
                rows.append(
                    {
                        "conflict_type": str(question.get("conflict_type") or ""),
                        "answer_accuracy": evaluation.get("Answer_Accuracy") or 0.0,
                        "conflict_handling": evaluation.get("Conflict_Handling") or 0,
                        "support_rank": evaluation.get("Support_Rank") or 0,
                        "judge_error": error,
                        "evaluation_present": present,
                        "answer_error": question.get("Answer_Error") or None,
                    }
                )
    return rows


def recorded_judge_window(scores_path: Path) -> int | None:
    """The judge window ``run_scoring.py`` recorded next to ``scores.jsonl``.

    Newer runs write ``Judge_Top_K``; older ones only have ``White_Box_Top_K``,
    which was the same number because judging and reporting shared one window.
    """
    path = scores_path.parent / "metrics.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("Judge_Top_K", "White_Box_Top_K"):
        value = data.get(key)
        if isinstance(value, int) and value >= 1:
            return value
    return None


def parse_k_values(raw: str) -> list[int]:
    values: list[int] = []
    for piece in str(raw).replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 1:
            raise ValueError(f"Top-K must be >= 1, got {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("no Top-K values given")
    return sorted(values)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild MemConflict Tables 3 / 5 / 6 from a scored run."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Directory holding scores.jsonl.")
    source.add_argument("--scores", type=Path, help="Explicit scores.jsonl path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the tables (default: the run directory).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Primary white-box K for the Table 3 / Table 6 columns (default 3).",
    )
    parser.add_argument(
        "--white-box-k",
        type=str,
        default=None,
        help="Extra white-box windows, e.g. 2,3,5. Default: only --top-k.",
    )
    parser.add_argument("--method-name", type=str, default="EMIR²")
    return parser


def main(argv: list[str] | None = None) -> int:
    from memconflict_eval import runtime

    runtime.ensure_utf8_console()
    args = build_arg_parser().parse_args(argv)
    scores_path = args.scores if args.scores else (args.run_dir / "scores.jsonl")
    if not scores_path.is_file():
        print(f"[error] scores file not found: {scores_path}", file=sys.stderr)
        return 2
    output_dir = args.output_dir or scores_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_k = int(args.top_k)
    try:
        white_box_ks = (
            parse_k_values(args.white_box_k) if args.white_box_k else [primary_k]
        )
    except ValueError as error:
        print(f"[error] --white-box-k: {error}", file=sys.stderr)
        return 2
    if primary_k not in white_box_ks:
        white_box_ks = sorted(white_box_ks + [primary_k])

    personas = read_scores(scores_path)
    rows = rows_from_scored_personas(personas)
    if not rows:
        print("[error] no scored questions found", file=sys.stderr)
        return 2

    # Two ways a rebuild could quietly produce wrong numbers: a question with no
    # conflict type would be dropped from every column (deflating the
    # denominator), and a question with no Evaluation would be scored as a
    # failure. Report both instead of hiding them.
    unknown = [row for row in rows if row["conflict_type"] not in CONFLICT_TYPES]
    if unknown:
        print(
            f"[error] {len(unknown)}/{len(rows)} question(s) carry no usable "
            "conflict_type; refusing to emit tables over a partial denominator",
            file=sys.stderr,
        )
        return 2
    missing = [row for row in rows if not row.get("evaluation_present", True)]
    judge_errors = [row for row in rows if row.get("judge_error")]
    if missing:
        print(
            f"[warn] {len(missing)}/{len(rows)} question(s) have no Evaluation "
            "block and count as judged failures; the tables below are a lower "
            "bound. Re-run Experiment/run_scoring.py for those personas.",
            file=sys.stderr,
        )
    if judge_errors:
        print(
            f"[warn] {len(judge_errors)}/{len(rows)} question(s) failed judging; "
            "they count as 0 in every column.",
            file=sys.stderr,
        )

    judge_window = recorded_judge_window(scores_path)
    if judge_window:
        print(f"[info] judge window (metrics.json): Top-{judge_window}")
        if max(white_box_ks) > judge_window:
            print(
                f"[warn] requested windows {white_box_ks} exceed the judge window "
                f"Top-{judge_window}: ranks beyond {judge_window} were never "
                "recorded, so those columns are lower bounds. Re-score with "
                f"--judge-top-k {max(white_box_ks)} for real values.",
                file=sys.stderr,
            )

    metrics = aggregate(rows, white_box_k=primary_k)
    by_k = aggregate_by_k(rows, white_box_ks)

    table3 = render_table3(metrics, args.method_name)
    table5 = render_table5(metrics, args.method_name)
    table6 = render_table6(metrics, args.method_name)
    by_k_table = render_white_box_by_k(by_k, args.method_name)

    (output_dir / "table3.md").write_text(
        "# Table 3: conflict-aware evaluation\n\n"
        f"{table3}\n\n"
        f"<!-- rebuilt offline from {scores_path.name}; no judge call -->\n",
        encoding="utf-8",
    )
    (output_dir / "table5.md").write_text(
        "# Table 5: black-box performance by conflict type\n\n"
        f"{table5}\n\n"
        f"<!-- rebuilt offline from {scores_path.name}; no judge call -->\n",
        encoding="utf-8",
    )
    (output_dir / "table6.md").write_text(
        "# Table 6: white-box retrieval and ranking by conflict type\n\n"
        f"{table6}\n\n## White-box by Top-K\n\n{by_k_table}\n\n"
        f"<!-- rebuilt offline from {scores_path.name}; no judge call -->\n",
        encoding="utf-8",
    )
    (output_dir / "tables.md").write_text(
        render_all_tables(metrics, args.method_name, by_k),
        encoding="utf-8",
    )

    print(f"[input] scores  : {scores_path}")
    print(f"[info] questions: {len(rows)}")
    print()
    print(table3)
    print()
    print(table5)
    print()
    print(table6)
    print()
    print(f"[done] tables   : {output_dir / 'tables.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
