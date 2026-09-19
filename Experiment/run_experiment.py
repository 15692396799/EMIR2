"""Point 7 runner: MemConflict data reading + memory building + per-session QA.

For every persona the runner:

1. creates an isolated memory store (personas must not see each other),
2. walks the session chain in chronological order,
3. ingests that session's dialogue into memory,
4. immediately retrieves and answers that session's questions,
5. moves on to the next session, so later sessions can see earlier ones but no
   question can ever see a later session.

Usage::

    python Experiment/run_experiment.py --persona-limit 1
    python Experiment/run_experiment.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python Experiment/run_experiment.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from memconflict_eval import runtime  # noqa: E402
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


def run_persona(
    *,
    persona,
    output_dir: Path,
    config_path: Path,
    answerer: MemConflictAnswerer,
    top_k: int,
    stored_top_k: int,
    version: str,
    keep_memory: bool,
) -> dict[str, Any]:
    store_dir = store_dir_for(output_dir, persona, version)
    persona_start = time.perf_counter()
    sessions_out: list[dict[str, Any]] = []
    answered_questions = 0

    with MemConflictMemory(
        store_dir=store_dir,
        persona=persona,
        version=version,
        config_path=config_path,
        reset=not keep_memory,
    ) as memory:
        for session in persona.sessions:
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

    return {
        "Persona_ID": persona.persona_id,
        "Memory_System": runtime.MEMORY_SYSTEM_NAME,
        "Memory_Namespace": MemConflictMemory.build_namespace(persona, version),
        "Memory_Store": str(store_dir),
        "Answer_Top_K": top_k,
        "Stored_Top_K": stored_top_k,
        "Session_Count": len(sessions_out),
        "Answered_Question_Count": answered_questions,
        "Persona_Runtime_ms": (time.perf_counter() - persona_start) * 1000.0,
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

    config = runtime.load_memory_config(args.config)
    missing = runtime.missing_env_names(config, runtime.RUNNER_ROLES)
    if missing:
        print("[error] missing credentials: " + ", ".join(missing), file=sys.stderr)
        print("        " + runtime.credential_hint(), file=sys.stderr)
        return 3

    try:
        answerer = MemConflictAnswerer(config)
    except Exception as error:
        print(f"[error] could not build the answer model: {type(error).__name__}: {error}", file=sys.stderr)
        print("        " + runtime.credential_hint(), file=sys.stderr)
        return 3
    results_path = output_dir / "results.jsonl"

    with open(results_path, "w", encoding="utf-8") as handle:
        for index, persona in enumerate(personas, start=1):
            print(
                f"[persona {index}/{len(personas)}] {persona.persona_id} "
                f"sessions={len(persona.sessions)} questions={persona.question_count}"
            )
            record = run_persona(
                persona=persona,
                output_dir=output_dir,
                config_path=args.config,
                answerer=answerer,
                top_k=args.top_k,
                stored_top_k=args.stored_top_k,
                version=args.version,
                keep_memory=args.keep_memory,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"  -> answered {record['Answered_Question_Count']} questions "
                f"in {record['Persona_Runtime_ms'] / 1000.0:.1f}s"
            )

    meta = {
        "Memory_System": runtime.MEMORY_SYSTEM_NAME,
        "Dataset": str(Path(args.input).resolve()),
        "Retrival_Mem_Root": str(runtime.retrival_mem_root()),
        "Config": str(Path(args.config).resolve()),
        "Answer_Top_K": args.top_k,
        "Stored_Top_K": args.stored_top_k,
        "Version": args.version,
        "Dataset_Summary": summary,
        "Results_Path": str(results_path),
        "Created_At": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
