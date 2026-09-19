"""Points 8+9: judge the answers and produce the conflict-aware table.

Reads a ``results.jsonl`` produced by ``run_experiment.py``, judges every answer
with the MemConflict judge prompt, and writes:

* ``scores.jsonl``  - one persona per line, each question annotated with the
  judged accuracy, conflict handling, support rank and reasoning;
* ``metrics.json``  - the aggregated numbers;
* ``table3.md``     - the Table 3 row for EMIR2.

Usage::

    python Experiment/run_scoring.py --run-dir Experiment/runs/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from memconflict_eval import runtime  # noqa: E402
from memconflict_eval.data import Question  # noqa: E402
from memconflict_eval.judging import MemConflictJudge  # noqa: E402
from memconflict_eval.memory import RetrievedMemory  # noqa: E402
from memconflict_eval.metrics import (  # noqa: E402
    aggregate,
    render_detail_table,
    render_table3,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge MemConflict answers and build the conflict table."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Directory holding results.jsonl.")
    source.add_argument("--results", type=Path, help="Explicit results.jsonl path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write scores/metrics (default: the run directory).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="White-box K for SEH@K (Table 3 uses 3).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=runtime.default_config_path(),
        help="Retrival-Mem config supplying the judge model.",
    )
    parser.add_argument("--method-name", type=str, default="EMIR²")
    return parser


def _question_from_record(raw: dict[str, Any]) -> Question:
    return Question(
        question_id=str(raw.get("question_id") or ""),
        question=str(raw.get("question") or ""),
        answer=str(raw.get("answer") or ""),
        conflict_type=str(raw.get("conflict_type") or ""),
        ability_target=str(raw.get("ability_target") or ""),
        difficulty=str(raw.get("difficulty") or ""),
    )


def _memories_from_record(raw: dict[str, Any], top_k: int) -> list[RetrievedMemory]:
    memories: list[RetrievedMemory] = []
    for index, item in enumerate((raw.get("Retrieved_Memories") or [])[:top_k], start=1):
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        memories.append(
            RetrievedMemory(
                rank=int(item.get("rank") or index),
                memory=str(item.get("memory") or ""),
                created_at=str(item.get("created_at") or "Unknown Time"),
                score=float(score) if isinstance(score, (int, float)) else None,
                node_type=str(item.get("node_type") or ""),
            )
        )
    return memories


def score_persona(
    persona: dict[str, Any],
    *,
    judge: MemConflictJudge,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flat: list[dict[str, Any]] = []
    sessions_out: list[dict[str, Any]] = []
    for session in persona.get("Sessions") or []:
        questions_out: list[dict[str, Any]] = []
        for raw in session.get("Questions") or []:
            question = _question_from_record(raw)
            memories = _memories_from_record(raw, top_k)
            result = judge.judge(
                question=question,
                model_answer=str(raw.get("Model_Answer") or ""),
                memories=memories,
            )
            scored = dict(raw)
            scored["Evaluation"] = result.to_dict()
            scored["question_id"] = question.question_id
            scored["conflict_type"] = question.conflict_type
            questions_out.append(scored)
            flat.append(
                {
                    "conflict_type": question.conflict_type,
                    "answer_accuracy": result.answer_accuracy,
                    "support_rank": result.support_rank,
                    "conflict_handling": result.conflict_handling,
                    "judge_error": result.error,
                }
            )
        session_out = dict(session)
        session_out["Questions"] = questions_out
        sessions_out.append(session_out)

    scored_persona = dict(persona)
    scored_persona["Sessions"] = sessions_out
    return scored_persona, flat


def main(argv: list[str] | None = None) -> int:
    runtime.ensure_utf8_console()
    args = build_arg_parser().parse_args(argv)
    run_dir = args.run_dir
    results_path = args.results if args.results else (run_dir / "results.jsonl")
    if not results_path.is_file():
        print(f"[error] results file not found: {results_path}", file=sys.stderr)
        return 2
    output_dir = args.output_dir or (run_dir if run_dir else results_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    personas = []
    with open(results_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                personas.append(json.loads(line))
    if not personas:
        print("[error] no persona records found", file=sys.stderr)
        return 2

    config = runtime.load_memory_config(args.config)
    missing = runtime.missing_env_names(config, runtime.SCORING_ROLES)
    if missing:
        print("[error] missing credentials: " + ", ".join(missing), file=sys.stderr)
        print("        " + runtime.credential_hint(), file=sys.stderr)
        return 3

    print(f"[judge] model    : {config.judge_model.model}")
    print(f"[judge] top-k    : {args.top_k}")
    print(f"[input] results  : {results_path}")

    try:
        judge = MemConflictJudge(config, top_k=args.top_k)
    except Exception as error:
        print(f"[error] could not build the judge model: {type(error).__name__}: {error}", file=sys.stderr)
        print("        " + runtime.credential_hint(), file=sys.stderr)
        return 3
    scored_rows: list[dict[str, Any]] = []
    all_questions: list[dict[str, Any]] = []

    for index, persona in enumerate(personas, start=1):
        scored_persona, flat = score_persona(persona, judge=judge, top_k=args.top_k)
        scored_persona["Evaluated_Question_Count"] = len(flat)
        scored_rows.append(scored_persona)
        all_questions.extend(flat)
        print(
            f"[persona {index}/{len(personas)}] {persona.get('Persona_ID')} "
            f"judged {len(flat)} questions"
        )

    metrics = aggregate(all_questions)
    scores_path = output_dir / "scores.jsonl"
    with open(scores_path, "w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "Memory_System": personas[0].get("Memory_System"),
                "Method": args.method_name,
                "White_Box_Top_K": args.top_k,
                "Scored_At": datetime.now().isoformat(timespec="seconds"),
                "Metrics": metrics.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    table3 = render_table3(metrics, args.method_name)
    detail = render_detail_table(metrics)
    (output_dir / "table3.md").write_text(
        "# Table 3: conflict-aware evaluation\n\n"
        f"{table3}\n\n"
        "## Detail behind the row\n\n"
        f"{detail}\n",
        encoding="utf-8",
    )

    print()
    print(table3)
    print()
    print(detail)
    print()
    print(f"[done] scores  : {scores_path}")
    print(f"[done] metrics : {output_dir / 'metrics.json'}")
    print(f"[done] table   : {output_dir / 'table3.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
