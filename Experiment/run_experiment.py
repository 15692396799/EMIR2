"""Points 7, 16, 17 runner: per-session QA, several GPUs, personas in parallel.

For every persona the runner:

1. creates an isolated memory store (personas must not see each other),
2. walks the session chain in chronological order,
3. ingests that session's dialogue into memory,
4. immediately retrieves and answers that session's questions,
5. moves on to the next session, so later sessions can see earlier ones but no
   question can ever see a later session.

Points 16 and 17 add wall-clock parallelism on top of that contract:

* the GPU server runs one Ollama container per GPU, and ``OLLAMA_BASE_URLS``
  lists their base URLs; each persona worker owns one container, so the
  embedding / controller / window-planner calls of different workers land on
  different GPUs (see ``memconflict_eval.ollama_units``);
* ``--persona-workers N`` runs N personas at once in separate processes, which
  is safe precisely because personas share nothing. Inside one persona the
  session chain stays serial, so the point-7 guarantee is untouched.

Usage::

    python Experiment/run_experiment.py --persona-limit 1
    python Experiment/run_experiment.py --persona-workers 4
    python Experiment/run_experiment.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python Experiment/run_experiment.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from memconflict_eval import runtime  # noqa: E402
from memconflict_eval import ollama_units, parallel  # noqa: E402
from memconflict_eval.answering import MemConflictAnswerer  # noqa: E402
from memconflict_eval.data import dataset_summary, load_personas  # noqa: E402
from memconflict_eval.memory import MemConflictMemory, store_dir_for  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Retrival-Mem memory system on the MemConflict benchmark."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=runtime.default_dataset_path(),
        help="MemConflict JSONL release (default: MemConflict/Data/Step4_4.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run directory. Default: Experiment/runs/<timestamp>.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=runtime.default_config_path(),
        help="Retrival-Mem config (default: Retrival-Mem/configs/default.yaml).",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument(
        "--persona-limit",
        type=int,
        default=None,
        help="Only run the first N personas (after --start-index). Smoke tests use 1.",
    )
    parser.add_argument(
        "--persona-workers",
        type=int,
        default=1,
        help=(
            "Point 17: run this many personas at once in separate processes. "
            "Personas share nothing, so this only trades CPU/GPU/API "
            "concurrency for wall clock; the session chain inside one persona "
            "stays serial. 1 = the original in-process loop."
        ),
    )
    parser.add_argument(
        "--ollama-units",
        type=str,
        default=None,
        help=(
            "Point 16: comma-separated Ollama container base URLs, overriding "
            "OLLAMA_BASE_URLS from Experiment/.env. Use one URL to reproduce "
            "the single-container behaviour, four to give every worker its own "
            "GPU. Example: http://172.26.94.12:41133,http://172.26.94.12:41134"
        ),
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help=(
            "Only replay the first N sessions of each persona. Intended for "
            "pipeline smoke tests: it truncates the session chain, so the later "
            "sessions and their questions are skipped."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Memories placed in the answer prompt.",
    )
    parser.add_argument(
        "--stored-top-k",
        type=int,
        default=5,
        help="Retrieved memories persisted per question (must cover the largest judged K).",
    )
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument(
        "--keep-memory",
        action="store_true",
        help="Reuse an existing per-persona store instead of rebuilding it.",
    )
    return parser


def resolve_output_dir(requested: Path | None) -> Path:
    if requested is not None:
        directory = Path(requested)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = Path(__file__).resolve().parent / "runs" / stamp
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def persona_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Entry point for one persona: in this process, or in a worker process.

    Point 17 hands this function to a process pool, so it must stay a
    module-level function with a picklable payload (the persona, the paths and
    plain scalars). The unit assignment of point 16 is applied here, before the
    memory system is built, which is what pins every Ollama call of this
    persona to one container.
    """
    unit = payload.get("unit")
    if unit:
        ollama_units.apply_unit(str(unit))
    else:
        ollama_units.clear_unit()

    persona = payload["persona"]
    emit = payload.get("progress")
    if emit is not None:
        emit.put(
            {
                "Event": "persona_start",
                "Persona_Index": int(payload.get("persona_index", 0)),
                "Persona_ID": persona.persona_id,
                "Session_Count": len(persona.sessions),
                "Question_Count": persona.question_count,
                "Ollama_Unit": unit or "",
            }
        )

    return run_persona(
        persona=persona,
        output_dir=Path(payload["output_dir"]),
        config_path=Path(payload["config_path"]),
        answerer=None,
        top_k=int(payload["top_k"]),
        stored_top_k=int(payload["stored_top_k"]),
        version=str(payload["version"]),
        keep_memory=bool(payload["keep_memory"]),
        max_sessions=payload.get("max_sessions"),
        on_session=(emit.put if emit is not None else None),
    )


def run_persona(
    *,
    persona,
    output_dir: Path,
    config_path: Path,
    answerer: MemConflictAnswerer | None = None,
    top_k: int,
    stored_top_k: int,
    version: str,
    keep_memory: bool,
    max_sessions: int | None = None,
    on_session: Any = None,
) -> dict[str, Any]:
    store_dir = store_dir_for(output_dir, persona, version)
    persona_start = time.perf_counter()
    started_at = time.time()
    unit = ollama_units.active_unit() or ""
    sessions_out: list[dict[str, Any]] = []
    answered_questions = 0
    sessions = persona.sessions
    if max_sessions is not None:
        sessions = sessions[: max(0, int(max_sessions))]

    with MemConflictMemory(
        store_dir=store_dir,
        persona=persona,
        version=version,
        config_path=config_path,
        reset=not keep_memory,
    ) as memory:
        # The memory store just re-applied this process's Ollama unit (point
        # 16), so build the answer client after it: the answer model must talk
        # to the same container as the memory that produced its context.
        if answerer is None:
            answerer = MemConflictAnswerer(memory.config)
        for session in sessions:
            ingest = memory.ingest_session(session)
            questions_out: list[dict[str, Any]] = []
            for question in session.questions:
                retrieval = memory.retrieve(question.question, keep=stored_top_k)
                answer = answerer.answer(
                    question,
                    retrieval.memories[:top_k],
                    namespace=memory.namespace,
                )
                questions_out.append(
                    {
                        "question_id": question.question_id,
                        "question": question.question,
                        "answer": question.answer,
                        "conflict_type": question.conflict_type,
                        "ability_target": question.ability_target,
                        "difficulty": question.difficulty,
                        "Memory_System": runtime.MEMORY_SYSTEM_NAME,
                        "Model_Answer": answer.text,
                        "Retrieved_Memories": [
                            item.to_dict() for item in retrieval.memories
                        ],
                        "Retrieved_Memory_Count": len(retrieval.memories),
                        "Retrieval_Round_Count": retrieval.round_count,
                        "Retrieval_Duration_ms": retrieval.duration_ms,
                        "Answer_Duration_ms": answer.duration_ms,
                        "Answer_Top_K": top_k,
                    }
                )
                answered_questions += 1
            sessions_out.append(
                {
                    "Session_ID": session.session_id,
                    "Date": session.date,
                    "Session_Type": session.session_type,
                    "Question_Trigger_Types": list(session.question_trigger_types),
                    "Event_Types": list(session.event_types),
                    "Ingest": ingest.to_dict(),
                    "Questions": questions_out,
                }
            )
            if on_session is not None:
                on_session(
                    {
                        "Persona_ID": persona.persona_id,
                        "Ollama_Unit": unit,
                        "Memory_System": runtime.MEMORY_SYSTEM_NAME,
                        "Session_ID": session.session_id,
                        "Date": session.date,
                        "Session_Type": session.session_type,
                        "Question_Trigger_Types": list(session.question_trigger_types),
                        "Event_Types": list(session.event_types),
                        "Ingest": ingest.to_dict(),
                        "Questions": questions_out,
                    }
                )

    return {
        "Persona_ID": persona.persona_id,
        "Ollama_Unit": unit,
        "Memory_System": runtime.MEMORY_SYSTEM_NAME,
        "Memory_Namespace": MemConflictMemory.build_namespace(persona, version),
        "Memory_Store": str(store_dir),
        "Answer_Top_K": top_k,
        "Stored_Top_K": stored_top_k,
        "Sessions_Replayed": len(sessions),
        "Sessions_Available": len(persona.sessions),
        "Truncated": max_sessions is not None,
        "Session_Count": len(sessions_out),
        "Answered_Question_Count": answered_questions,
        "Persona_Runtime_ms": (time.perf_counter() - persona_start) * 1000.0,
        "Started_At": started_at,
        "Finished_At": time.time(),
        "Sessions": sessions_out,
    }


def main(argv: list[str] | None = None) -> int:
    runtime.ensure_utf8_console()
    args = build_arg_parser().parse_args(argv)
    output_dir = resolve_output_dir(args.output_dir)

    if not Path(args.input).is_file():
        print(f"[error] dataset not found: {args.input}", file=sys.stderr)
        return 2

    end_index = args.end_index
    if args.persona_limit is not None:
        limit_end = args.start_index + args.persona_limit
        end_index = limit_end if end_index is None else min(end_index, limit_end)

    personas = load_personas(args.input, start_index=args.start_index, end_index=end_index)
    summary = dataset_summary(personas)
    print(f"[data] {json.dumps(summary, ensure_ascii=False)}")
    if not personas:
        print("[error] no personas selected", file=sys.stderr)
        return 2

    print(f"[memory] Retrival-Mem root : {runtime.retrival_mem_root()}")
    print(f"[memory] config            : {args.config}")
    print(f"[output] run directory     : {output_dir}")

    # Point 16 reads OLLAMA_BASE_URLS, which lives in Experiment/.env next to the
    # credentials, so the file has to be in os.environ before the unit list is
    # read (load_memory_config would otherwise be the first to see it, and by
    # then the persona-to-container assignment would already be empty).
    try:
        runtime.load_env_file()
    except Exception as error:  # the .env file is optional; credentials still checked below
        print(f"[warn] could not load the .env file: {type(error).__name__}: {error}", file=sys.stderr)

    if args.ollama_units:
        # Written back to the environment so the workers inherit it too.
        os.environ[ollama_units.UNITS_ENV] = str(args.ollama_units)

    units = ollama_units.configured_units()
    if units:
        print(f"[ollama] units             : {len(units)} ({', '.join(units)})")
    persona_workers = max(1, int(args.persona_workers))
    if persona_workers > 1:
        print(f"[persona] workers          : {persona_workers} process(es)")
        if len(units) < persona_workers:
            print(
                f"[warn] {persona_workers} persona workers share "
                f"{len(units) or 1} Ollama unit(s)"
            )

    # Point 16: round-robin the personas over the containers. A persona keeps
    # its container for its whole lifetime, so its sessions and questions stay
    # on one GPU (and one KV cache) while other personas use the others.
    assignments = ollama_units.assign_units(units, len(personas))
    if assignments and assignments[0]:
        ollama_units.apply_unit(str(assignments[0]))

    config = runtime.load_memory_config(args.config)
    missing = runtime.missing_env_names(config, runtime.RUNNER_ROLES)
    if missing:
        print("[error] missing credentials: " + ", ".join(missing), file=sys.stderr)
        print("        " + runtime.credential_hint(), file=sys.stderr)
        return 3

    # Preflight only: fail fast on credentials/provider wiring before the first
    # persona starts. Every persona builds its own answer client inside its
    # worker, after its Ollama unit has been applied.
    try:
        MemConflictAnswerer(config)
    except Exception as error:
        print(f"[error] could not build the answer model: {type(error).__name__}: {error}", file=sys.stderr)
        print("        " + runtime.credential_hint(), file=sys.stderr)
        return 3
    results_path = output_dir / "results.jsonl"
    sessions_path = output_dir / "sessions.jsonl"

    run_started = time.time()
    jobs = [
        {
            "persona_index": index,
            "persona": persona,
            "output_dir": str(output_dir),
            "config_path": str(args.config),
            "unit": assignments[index],
            "top_k": args.top_k,
            "stored_top_k": args.stored_top_k,
            "version": args.version,
            "keep_memory": args.keep_memory,
            "max_sessions": args.max_sessions,
        }
        for index, persona in enumerate(personas)
    ]

    # results.jsonl is only written once a persona finishes, so a long persona
    # that dies part-way would otherwise lose every answered question. Every
    # completed session is therefore also appended to sessions.jsonl, which
    # run_scoring.py can fall back to.
    errors_path = output_dir / "errors.jsonl"
    failed_personas: list[str] = []
    with open(results_path, "w", encoding="utf-8") as handle, open(
        sessions_path, "w", encoding="utf-8"
    ) as sessions_handle, open(errors_path, "w", encoding="utf-8") as errors_handle:
        finished: list[dict[str, Any] | None] = [None] * len(personas)
        next_to_write = 0

        def record_session(row: dict[str, Any]) -> None:
            if row.get("Event") == "persona_start":
                print(
                    f"[persona {int(row.get('Persona_Index', 0)) + 1}/{len(personas)}] "
                    f"{row.get('Persona_ID')} sessions={row.get('Session_Count')} "
                    f"questions={row.get('Question_Count')} "
                    f"unit={row.get('Ollama_Unit') or '(config default)'}"
                )
                return
            sessions_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            sessions_handle.flush()

        def record_result(index: int, record: Any) -> None:
            """Append finished personas in dataset order, as soon as possible."""
            nonlocal next_to_write
            if isinstance(record, parallel.JobError):
                # One persona failing must not throw away the others' work.
                failed_personas.append(personas[index].persona_id)
                last_line = (record.traceback_text or "").strip().splitlines()
                print(
                    f"  !! persona {index + 1} failed ({personas[index].persona_id}): "
                    f"{record.summary}"
                )
                if last_line:
                    print(f"     {last_line[-1]}")
                finished[index] = {
                    "Persona_ID": personas[index].persona_id,
                    "Ollama_Unit": assignments[index] or "",
                    "Error": record.summary,
                    "Error_Traceback": record.traceback_text,
                }
            else:
                finished[index] = record
            while next_to_write < len(finished) and finished[next_to_write] is not None:
                done = dict(finished[next_to_write] or {})
                done["Persona_Index"] = next_to_write
                done["Started_At_s"] = round(
                    float(done.pop("Started_At", run_started) or run_started) - run_started, 3
                )
                done["Finished_At_s"] = round(
                    float(done.pop("Finished_At", run_started) or run_started) - run_started, 3
                )
                if isinstance(done.get("Error"), str):
                    # A failed persona is reported, not scored: it is written to
                    # errors.jsonl so the other personas' work is kept.
                    errors_handle.write(json.dumps(done, ensure_ascii=False) + "\n")
                    errors_handle.flush()
                else:
                    handle.write(json.dumps(done, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        f"  -> persona {next_to_write + 1} answered "
                        f"{done.get('Answered_Question_Count')} questions in "
                        f"{float(done.get('Persona_Runtime_ms', 0.0)) / 1000.0:.1f}s "
                        f"(unit {done.get('Ollama_Unit') or 'config default'})"
                    )
                next_to_write += 1

        parallel.run_jobs(
            persona_job,
            jobs,
            workers=persona_workers,
            on_row=record_session,
            on_result=record_result,
            raise_errors=False,
        )

    meta = {
        "Memory_System": runtime.MEMORY_SYSTEM_NAME,
        "Dataset": str(Path(args.input).resolve()),
        "Retrival_Mem_Root": str(runtime.retrival_mem_root()),
        "Config": str(Path(args.config).resolve()),
        "Answer_Top_K": args.top_k,
        "Stored_Top_K": args.stored_top_k,
        "Max_Sessions": args.max_sessions,
        "Version": args.version,
        "Persona_Workers": persona_workers,
        "Ollama_Units": units,
        "Persona_Unit_Assignments": {
            persona.persona_id: assignments[index]
            for index, persona in enumerate(personas)
        },
        "Failed_Personas": failed_personas,
        "Errors_Path": str(errors_path),
        "Wall_Clock_s": round(time.time() - run_started, 3),
        "Dataset_Summary": summary,
        "Results_Path": str(results_path),
        "Sessions_Path": str(sessions_path),
        "Created_At": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] wrote {results_path}")
    if failed_personas:
        print(
            f"[fail] {len(failed_personas)}/{len(personas)} persona(s) failed: "
            + ", ".join(failed_personas),
            file=sys.stderr,
        )
        print(f"       see {errors_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
