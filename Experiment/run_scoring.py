"""Points 8+9 (+16/17): judge the answers and produce Tables 3 / 5 / 6.

Reads a ``results.jsonl`` produced by ``run_experiment.py``, judges every answer
with the MemConflict judge prompt, and writes:

* ``scores.jsonl``  - one persona per line, each question annotated with the
  judged accuracy, conflict handling, support rank and reasoning;
* ``metrics.json``  - the aggregated numbers (including white-box by Top-K);
* ``table3.md``     - conflict-aware evaluation (AA + SEH@K per conflict type);
* ``table5.md``     - black-box performance (AA + UOCS / CRS per conflict type);
* ``table6.md``     - white-box retrieval and ranking (SEH@K + SRS per type);
* ``tables.md``     - the three tables plus the by-K and detail breakdowns.

All three tables are projections of the *same* judge pass: the judge returns
the graded answer score, the conflict-handling flag and the support rank in one
call, so Table 5 and Table 6 add no model traffic beyond Table 3. The judge
window (``--judge-top-k``) only has to be at least as large as the largest
white-box window (``--white-box-k``), because the smaller windows are derived
from the same rank.

Personas are judged independently, so ``--judge-workers N`` scores N personas at
once (point 17); an Ollama judge is additionally routed over the containers
listed in ``OLLAMA_BASE_URLS`` (point 16), one persona per unit.

Usage::

    python Experiment/run_scoring.py --run-dir Experiment/runs/<timestamp>
    python Experiment/run_scoring.py --run-dir <dir> --judge-workers 4
    python Experiment/run_scoring.py --run-dir <dir> --white-box-k 2,3,5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from memconflict_eval import runtime  # noqa: E402
from memconflict_eval import ollama_units, parallel  # noqa: E402
from memconflict_eval.data import Question  # noqa: E402
from memconflict_eval.judging import MemConflictJudge  # noqa: E402
from memconflict_eval.memory import RetrievedMemory  # noqa: E402
from memconflict_eval.metrics import (  # noqa: E402
    WHITE_BOX_K_VALUES,
    aggregate,
    aggregate_by_k,
    render_all_tables,
    render_detail_table,
    render_table3,
    render_table5,
    render_table6,
    render_white_box_by_k,
)
from memconflict_eval.progress import ProgressReporter  # noqa: E402


def parse_k_values(raw: str) -> list[int]:
    """Parse ``2,3,5`` into ``[2, 3, 5]`` (deduplicated, ascending)."""
    values: list[int] = []
    for piece in str(raw).replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = int(piece)
        except ValueError as error:
            raise ValueError(f"not a Top-K value: {piece!r}") from error
        if value < 1:
            raise ValueError(f"Top-K must be >= 1, got {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("no Top-K values given")
    return sorted(values)


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
        help="Primary white-box K for the Table 3 / Table 6 columns (default 3).",
    )
    parser.add_argument(
        "--white-box-k",
        type=str,
        default=None,
        help=(
            "Extra white-box windows to report, e.g. 2,3,5 (matches the upstream "
            "judge-once/derive-by-K scheme). Default: only the primary --top-k. "
            "The judge window is raised to the largest value automatically."
        ),
    )
    parser.add_argument(
        "--judge-top-k",
        type=int,
        default=None,
        help=(
            "Top-K window shown to the judge when it picks the support rank. "
            "Default: the largest white-box window (5 for --white-box-k 2,3,5), "
            "which is the upstream behaviour. Needs --stored-top-k to match, so "
            "the run must have persisted that many memories per question."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=runtime.default_config_path(),
        help="Retrival-Mem config supplying the judge model.",
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=1,
        help=(
            "Point 17: judge this many personas at once in separate processes. "
            "1 = the original in-process loop."
        ),
    )
    parser.add_argument(
        "--ollama-units",
        type=str,
        default=None,
        help=(
            "Point 16: comma-separated Ollama container base URLs, overriding "
            "OLLAMA_BASE_URLS from Experiment/.env (only matters when the judge "
            "model itself is an Ollama model)."
        ),
    )
    parser.add_argument("--method-name", type=str, default="EMIR²")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help=(
            "Do not render the live judge progress line on stderr. One persona "
            "is one step, because the judge reports at persona granularity."
        ),
    )
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


def count_short_memory_windows(personas: list[dict[str, Any]], judge_top_k: int) -> int:
    """Questions whose stored retrieval list is shorter than the judge window.

    ``run_experiment.py --stored-top-k`` decides how many memories survive into
    ``results.jsonl``. Judging with a larger window than that silently pads the
    white-box columns with misses, so the mismatch is reported instead of
    hidden.
    """
    short = 0
    for persona in personas:
        for session in persona.get("Sessions") or []:
            for question in session.get("Questions") or []:
                stored = question.get("Retrieved_Memories") or []
                if len(stored) < judge_top_k:
                    short += 1
    return short


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
                    "answer_error": raw.get("Answer_Error") or None,
                }
            )
        session_out = dict(session)
        session_out["Questions"] = questions_out
        sessions_out.append(session_out)

    scored_persona = dict(persona)
    scored_persona["Sessions"] = sessions_out
    return scored_persona, flat


def judge_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Judge one persona: in this process, or in a worker process (point 17).

    The Ollama unit of point 16 is applied before the judge client is built, so
    an Ollama judge spreads over the containers exactly like the runner does.
    """
    unit = payload.get("unit")
    if unit:
        ollama_units.apply_unit(str(unit))
    else:
        ollama_units.clear_unit()
    top_k = int(payload["top_k"])
    config = runtime.load_memory_config(Path(payload["config_path"]))
    judge = MemConflictJudge(config, top_k=top_k)
    scored_persona, flat = score_persona(payload["persona"], judge=judge, top_k=top_k)
    return {
        "Persona_ID": scored_persona.get("Persona_ID"),
        "Ollama_Unit": unit or "",
        "Scored_Persona": scored_persona,
        "Questions": flat,
    }


def load_personas_from_results(path: Path) -> list[dict[str, Any]]:
    """Load persona records written by ``run_experiment.py``.

    ``--resume`` appends to ``results.jsonl``, so a persona can appear more
    than once (finished, then replayed and rewritten). Only the richest record
    per persona is kept, otherwise those questions would be counted twice.
    """
    by_persona: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            persona_id = str(row.get("Persona_ID") or "")
            if not persona_id:
                continue
            previous = by_persona.get(persona_id)
            if previous is None:
                by_persona[persona_id] = row
                order.append(persona_id)
                continue
            previous_score = int(previous.get("Answered_Question_Count") or 0) or len(
                previous.get("Sessions") or []
            )
            row_score = int(row.get("Answered_Question_Count") or 0) or len(
                row.get("Sessions") or []
            )
            if row_score >= previous_score:
                by_persona[persona_id] = row
    return [by_persona[persona_id] for persona_id in order]


def rebuild_personas_from_sessions(path: Path) -> list[dict[str, Any]]:
    """Reconstruct persona records from the per-session progress log.

    ``run_experiment.py`` appends one line per completed session, so a run that
    was killed before finishing a persona can still be scored instead of being
    thrown away.
    """
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            persona_id = str(row.get("Persona_ID") or "")
            persona = grouped.get(persona_id)
            if persona is None:
                persona = {
                    "Persona_ID": persona_id,
                    "Memory_System": row.get("Memory_System"),
                    "Sessions": [],
                    "Rebuilt_From_Sessions_JSONL": True,
                }
                grouped[persona_id] = persona
                order.append(persona_id)
            persona["Sessions"].append(
                {
                    "Session_ID": row.get("Session_ID"),
                    "Date": row.get("Date"),
                    "Session_Type": row.get("Session_Type"),
                    "Question_Trigger_Types": row.get("Question_Trigger_Types") or [],
                    "Event_Types": row.get("Event_Types") or [],
                    "Ingest": row.get("Ingest") or {},
                    "Questions": row.get("Questions") or [],
                }
            )
    for persona in grouped.values():
        persona["Answered_Question_Count"] = sum(
            len(session.get("Questions") or []) for session in persona["Sessions"]
        )
    return [grouped[persona_id] for persona_id in order]


def merge_personas_with_sessions(
    personas: list[dict[str, Any]],
    sessions_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Backfill every session that ``results.jsonl`` does not carry (P0-2).

    A run can end with some personas written to ``results.jsonl`` and others
    only present as per-session rows (killed mid-flight, or failed and reported
    in ``errors.jsonl``). Scoring used to fall back to the session log only when
    ``results.jsonl`` was completely empty, which silently dropped the finished
    sessions of exactly the personas that had trouble. Here every persona in
    the session log that has no ``results.jsonl`` record is rebuilt and merged,
    and the missing ids are reported back to the caller.

    A ``--resume`` makes the same problem smaller but not gone: the resumed
    persona record only contains the sessions that this invocation replayed
    (2026-09-20: sessions 6-10 of a persona whose 0-5 crashed the first
    attempt), while the earlier sessions live only in ``sessions.jsonl``. Those
    sessions - and their already answered questions - are merged in here,
    deduplicated by ``Session_ID`` and with the ``results.jsonl`` copy winning.
    """
    have = {str(row.get("Persona_ID") or "") for row in personas}
    if not sessions_path.is_file():
        return personas, []

    rebuilt = rebuild_personas_from_sessions(sessions_path)
    by_persona = {str(row.get("Persona_ID") or ""): row for row in rebuilt}
    merged = list(personas)
    recovered: list[str] = []

    def session_sort_key(session: dict[str, Any]) -> tuple[str, int]:
        try:
            session_id = int(session.get("Session_ID"))
        except (TypeError, ValueError):
            session_id = 10**9
        return (str(session.get("Date") or ""), session_id)

    for persona in merged:
        persona_id = str(persona.get("Persona_ID") or "")
        logged = by_persona.get(persona_id)
        if logged is None:
            continue
        known = {
            str(session.get("Session_ID"))
            for session in persona.get("Sessions") or []
        }
        missing = [
            session
            for session in logged.get("Sessions") or []
            if str(session.get("Session_ID")) not in known
        ]
        if not missing:
            continue
        sessions = list(persona.get("Sessions") or []) + missing
        sessions.sort(key=session_sort_key)
        persona["Sessions"] = sessions
        persona["Session_Count"] = len(sessions)
        persona["Answered_Question_Count"] = sum(
            len(session.get("Questions") or []) for session in sessions
        )
        persona["Merged_Sessions_From_JSONL"] = [
            session.get("Session_ID") for session in sorted(missing, key=session_sort_key)
        ]

    for persona in rebuilt:
        persona_id = str(persona.get("Persona_ID") or "")
        if not persona_id or persona_id in have:
            continue
        sessions = persona.get("Sessions") or []
        if not sessions:
            continue
        # A resumed run can legitimately contribute only part of a persona.
        partial = persona
        partial["Rebuilt_From_Sessions_JSONL"] = True
        if have:
            partial["Partial_Persona"] = True
        merged.append(partial)
        recovered.append(persona_id)
    return merged, recovered


def main(argv: list[str] | None = None) -> int:
    runtime.ensure_utf8_console()
    args = build_arg_parser().parse_args(argv)
    primary_k = int(args.top_k)
    try:
        white_box_ks = (
            parse_k_values(args.white_box_k) if args.white_box_k else [primary_k]
        )
    except ValueError as error:
        print(f"[error] --white-box-k: {error}", file=sys.stderr)
        return 2
    if primary_k not in white_box_ks:
        # Tables 3 and 6 always report the primary window.
        white_box_ks = sorted(white_box_ks + [primary_k])
    judge_top_k = int(args.judge_top_k) if args.judge_top_k else max(white_box_ks)
    if judge_top_k < max(white_box_ks):
        print(
            f"[error] --judge-top-k {judge_top_k} is smaller than the largest "
            f"white-box window {max(white_box_ks)}; the judge must see the "
            "whole window it ranks inside of",
            file=sys.stderr,
        )
        return 2
    if judge_top_k < 1:
        print(f"[error] --judge-top-k must be >= 1, got {judge_top_k}", file=sys.stderr)
        return 2
    run_dir = args.run_dir
    results_path = args.results if args.results else (run_dir / "results.jsonl")
    if not results_path.is_file():
        print(f"[error] results file not found: {results_path}", file=sys.stderr)
        return 2
    output_dir = args.output_dir or (run_dir if run_dir else results_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    personas = load_personas_from_results(results_path)
    sessions_path = results_path.parent / "sessions.jsonl"
    personas, recovered = merge_personas_with_sessions(personas, sessions_path)
    if recovered:
        print(
            f"[warn] {len(recovered)} persona(s) were missing from "
            f"{results_path.name} and were rebuilt from {sessions_path.name}: "
            + ", ".join(recovered)
        )
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
    print(f"[judge] window    : Top-{judge_top_k} shown to the judge")
    print(
        f"[white] windows   : "
        + ", ".join(f"Top-{value}" for value in white_box_ks)
        + f" (Tables 3/5/6 report Top-{primary_k})"
    )
    print(f"[input] results  : {results_path}")

    short_windows = count_short_memory_windows(personas, judge_top_k)
    if short_windows:
        print(
            f"[warn] {short_windows} question(s) carry fewer than {judge_top_k} "
            "stored memories, so their white-box rank cannot reach the whole "
            "window. Re-run the experiment with "
            f"--stored-top-k {judge_top_k} to fix it.",
            file=sys.stderr,
        )

    try:
        runtime.load_env_file()
    except Exception as error:  # the .env file is optional; credentials still checked below
        print(f"[warn] could not load the .env file: {type(error).__name__}: {error}", file=sys.stderr)
    if args.ollama_units:
        os.environ[ollama_units.UNITS_ENV] = str(args.ollama_units)
    units = ollama_units.configured_units()
    judge_workers = max(1, int(args.judge_workers))
    if units:
        print(f"[ollama] units   : {len(units)} ({', '.join(units)})")
    if judge_workers > 1:
        print(f"[judge] workers  : {judge_workers} process(es)")

    try:
        MemConflictJudge(config, top_k=judge_top_k)
    except Exception as error:
        print(f"[error] could not build the judge model: {type(error).__name__}: {error}", file=sys.stderr)
        print("        " + runtime.credential_hint(), file=sys.stderr)
        return 3

    # Point 16: one container per persona, round robin (only matters when the
    # judge itself is an Ollama model). Point 17: judge several personas at once.
    assignments = ollama_units.assign_units(units, len(personas))
    scored_rows: list[dict[str, Any] | None] = [None] * len(personas)
    flat_rows: list[list[dict[str, Any]]] = [[] for _ in personas]
    failed_personas: list[str] = []
    jobs = [
        {
            "persona": persona,
            "config_path": str(args.config),
            "top_k": judge_top_k,
            "unit": assignments[index],
        }
        for index, persona in enumerate(personas)
    ]

    # The judge reports per persona (that is the unit of parallelism), so the
    # progress line is persona-level: it still answers "how much is left"
    # without waiting for scores.jsonl to appear.
    total_questions = sum(
        len(session.get("Questions") or [])
        for persona in personas
        for session in (persona.get("Sessions") or [])
    )
    progress = ProgressReporter(
        len(personas),
        label="personas",
        counters={"questions": 0, "judge_failures": 0},
        totals={"questions": total_questions},
        stream=sys.stderr,
        enabled=not args.no_progress,
    )
    progress.start()

    def collect(index: int, payload: dict[str, Any]) -> None:
        if isinstance(payload, parallel.JobError):
            failed_personas.append(str(personas[index].get("Persona_ID") or index))
            progress.log(
                f"[fail] persona {index + 1}/{len(personas)} "
                f"{personas[index].get('Persona_ID')} was not judged: {payload.summary}"
            )
            progress.advance(1, judge_failures=1)
            return
        scored_persona = payload["Scored_Persona"]
        flat = payload["Questions"]
        scored_persona["Evaluated_Question_Count"] = len(flat)
        scored_rows[index] = scored_persona
        flat_rows[index] = flat
        progress.log(
            f"[persona {index + 1}/{len(personas)}] {payload.get('Persona_ID')} "
            f"judged {len(flat)} questions "
            f"(unit {payload.get('Ollama_Unit') or 'config default'})"
        )
        progress.advance(1, questions=len(flat))
        progress.note(
            last=str(payload.get("Persona_ID") or "")[:8]
        )

    try:
        parallel.run_jobs(
            judge_job, jobs, workers=judge_workers, on_result=collect, raise_errors=False
        )
    finally:
        progress.finish()
    scored_rows = [row for row in scored_rows if row is not None]
    all_questions: list[dict[str, Any]] = [
        question for flat in flat_rows for question in flat
    ]

    # One judge pass, three tables: the primary window drives Tables 3 / 5 / 6,
    # and every other white-box window is derived from the same support rank.
    metrics = aggregate(all_questions, white_box_k=primary_k)
    by_k = aggregate_by_k(all_questions, white_box_ks)
    scored_ids = [str(row.get("Persona_ID") or "") for row in scored_rows]
    expected_ids: list[str] = []
    run_meta: dict[str, Any] = {}
    meta_path = (run_dir or results_path.parent) / "run_meta.json"
    if meta_path.is_file():
        try:
            run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expected_ids = list((run_meta.get("Persona_Unit_Assignments") or {}).keys())
        except (OSError, json.JSONDecodeError):
            expected_ids = []
    missing_ids = [persona_id for persona_id in expected_ids if persona_id not in scored_ids]
    answer_error_count = sum(1 for row in all_questions if row.get("answer_error"))
    merged_sessions = {
        str(persona.get("Persona_ID")): [
            session_id for session_id in persona.get("Merged_Sessions_From_JSONL") or []
        ]
        for persona in personas
        if persona.get("Merged_Sessions_From_JSONL")
    }

    scores_path = output_dir / "scores.jsonl"
    with open(scores_path, "w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "Memory_System": personas[0].get("Memory_System"),
                "Method": args.method_name,
                "Judge_Model": str(getattr(config.judge_model, "model", "") or ""),
                "Judge_Provider": str(getattr(config.judge_model, "provider", "") or ""),
                "White_Box_Top_K": primary_k,
                "White_Box_Windows": white_box_ks,
                "Judge_Top_K": judge_top_k,
                "Scored_At": datetime.now().isoformat(timespec="seconds"),
                "Personas_Scored": len(scored_ids),
                "Personas_Expected": len(expected_ids) or None,
                "Dataset_Personas_Expected": run_meta.get(
                    "Personas_Expected_From_Dataset"
                ),
                "Partial_Merge": run_meta.get("Partial_Merge"),
                "Shards_Merged": run_meta.get("Shards_Merged"),
                "Persona_Ids_Missing": missing_ids,
                "Personas_Rebuilt_From_Sessions": recovered,
                "Personas_Merged_From_Sessions": merged_sessions,
                "Answer_Error_Count": answer_error_count,
                "Questions_With_Short_Memory_Window": short_windows,
                "Metrics": metrics.to_dict(),
                "White_Box_By_K": {
                    key: by_k[key].to_dict() for key in sorted(by_k, key=lambda v: int(v))
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    table3 = render_table3(metrics, args.method_name)
    table5 = render_table5(metrics, args.method_name)
    table6 = render_table6(metrics, args.method_name)
    by_k_table = render_white_box_by_k(by_k, args.method_name)
    detail = render_detail_table(metrics)
    (output_dir / "table3.md").write_text(
        "# Table 3: conflict-aware evaluation\n\n"
        f"{table3}\n\n"
        "## Detail behind the row\n\n"
        f"{detail}\n",
        encoding="utf-8",
    )
    (output_dir / "table5.md").write_text(
        "# Table 5: black-box performance by conflict type\n\n"
        f"{table5}\n\n"
        "> AA / UOCS / CRS come from the same judge call that produced Table 3; "
        "UOCS is the dynamic conflict-handling flag and CRS the static one.\n",
        encoding="utf-8",
    )
    (output_dir / "table6.md").write_text(
        "# Table 6: white-box retrieval and ranking by conflict type\n\n"
        f"{table6}\n\n"
        "## White-box by Top-K\n\n"
        f"{by_k_table}\n\n"
        "> SEH@K / SRS are derived from the single support rank the judge "
        "returned, so the columns here match Table 3 at the primary window.\n",
        encoding="utf-8",
    )
    (output_dir / "tables.md").write_text(
        render_all_tables(metrics, args.method_name, by_k),
        encoding="utf-8",
    )

    print()
    print(table3)
    print()
    print(table5)
    print()
    print(table6)
    print()
    print(detail)
    print()
    print(f"[done] scores  : {scores_path}")
    print(f"[done] metrics : {output_dir / 'metrics.json'}")
    print(f"[done] tables  : {output_dir / 'table3.md'}, {output_dir / 'table5.md'}, "
          f"{output_dir / 'table6.md'} ({output_dir / 'tables.md'})")
    if missing_ids:
        print(
            f"[warn] {len(missing_ids)}/{len(expected_ids)} persona(s) of this run "
            "have no sessions at all and are absent from the table: "
            + ", ".join(missing_ids),
            file=sys.stderr,
        )
    if failed_personas:
        print(
            f"[fail] {len(failed_personas)}/{len(personas)} persona(s) were not "
            "judged: " + ", ".join(failed_personas),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
