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
* ``--answer-workers N`` answers one session's questions concurrently (point
  17b). The memory state is frozen between session ingest and that session's
  questions, so concurrency cannot change what a question can see; it only
  stops the questions from queueing behind each other. The V4 store is opened
  with ``check_same_thread=False`` behind a re-entrant lock, FAISS searches and
  the API-history log are locked too, and every ``retrieve`` builds its own
  controller, so this is safe by construction.

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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):  # allow `python Experiment/run_experiment.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from memconflict_eval import runtime  # noqa: E402
from memconflict_eval import ollama_units, parallel  # noqa: E402
from memconflict_eval.answering import MemConflictAnswerer  # noqa: E402
from memconflict_eval.data import dataset_summary, load_personas  # noqa: E402
from memconflict_eval.memory import MemConflictMemory, store_dir_for  # noqa: E402
from memconflict_eval.progress import ProgressReporter, expected_work  # noqa: E402


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
        "--persona-indices",
        type=str,
        default=None,
        help=(
            "Point 27: explicit dataset indices, e.g. '0,4,18,27,28' or '0-4,10'. "
            "Use it for random shards: the personas are replayed in dataset order "
            "and merge_shards.py combines the shards later. Cannot be combined "
            "with --start-index/--end-index/--persona-limit."
        ),
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
        "--answer-workers",
        type=int,
        default=4,
        help=(
            "Point 17b: answer one session's questions with this many threads. "
            "The questions of a session share an immutable memory state, so "
            "only the schedule changes. 1 = the original serial loop."
        ),
    )
    parser.add_argument(
        "--extraction-workers",
        type=int,
        default=None,
        help=(
            "Override memory.memory_extraction_workers per worker process. "
            "Use it to keep total concurrency sane when --persona-workers > 1 "
            "(N workers x 8 threads all hit the same endpoints)."
        ),
    )
    parser.add_argument(
        "--entity-judge-workers",
        type=int,
        default=None,
        help="Override memory.backends.v4.entity_judge_workers per worker process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted run in the same --output-dir: sessions "
            "already present in sessions.jsonl are skipped and their memory "
            "store is kept instead of reset."
        ),
    )
    parser.add_argument(
        "--allow-model-change",
        action="store_true",
        help=(
            "Allow --resume even though the config now names different models "
            "than the recorded run_meta.json. Off by default: a store built by "
            "one embedding/planner model cannot be continued by another without "
            "mixing incompatible vector spaces and memory content."
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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help=(
            "Do not render the live progress line on stderr. The per-session "
            "rows in sessions.jsonl are written either way."
        ),
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


def load_completed_sessions(path: Path) -> dict[str, set[int]]:
    """Completed session ids per persona, from a previous run's sessions.jsonl.

    ``run_experiment.py`` appends one row per *finished* session, and a session
    row is only written after all of its questions have been answered, so the
    row is a safe "this session is done" marker for ``--resume``.
    """
    completed: dict[str, set[int]] = {}
    if not Path(path).is_file():
        return completed
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            persona_id = str(row.get("Persona_ID") or "")
            session_id = row.get("Session_ID")
            if not persona_id or not isinstance(session_id, int):
                continue
            completed.setdefault(persona_id, set()).add(session_id)
    return completed


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
        answer_workers=int(payload.get("answer_workers", 1)),
        skip_session_ids=tuple(payload.get("skip_session_ids") or ()),
        max_sessions=payload.get("max_sessions"),
        on_session=(emit.put if emit is not None else None),
    )


def _answer_one_question(
    *,
    memory: MemConflictMemory,
    answerer: MemConflictAnswerer,
    question: Any,
    top_k: int,
    stored_top_k: int,
) -> dict[str, Any]:
    """Retrieve + answer one question, never raising (point 17b helper)."""
    row: dict[str, Any] = {
        "question_id": question.question_id,
        "question": question.question,
        "answer": question.answer,
        "conflict_type": question.conflict_type,
        "ability_target": question.ability_target,
        "difficulty": question.difficulty,
        "Memory_System": runtime.MEMORY_SYSTEM_NAME,
        "Model_Answer": "",
        "Retrieved_Memories": [],
        "Retrieved_Memory_Count": 0,
        "Retrieval_Round_Count": 0,
        "Retrieval_Duration_ms": 0.0,
        "Answer_Duration_ms": 0.0,
        "Answer_Top_K": top_k,
        "Answer_Error": "",
    }
    try:
        retrieval = memory.retrieve(question.question, keep=stored_top_k)
        answer = answerer.answer(
            question,
            retrieval.memories[:top_k],
            namespace=memory.namespace,
        )
    except Exception as error:  # one bad question must not kill the persona
        row["Answer_Error"] = f"{type(error).__name__}: {error}"
        return row

    row.update(
        {
            "Model_Answer": answer.text,
            "Retrieved_Memories": [item.to_dict() for item in retrieval.memories],
            "Retrieved_Memory_Count": len(retrieval.memories),
            "Retrieval_Round_Count": retrieval.round_count,
            "Retrieval_Duration_ms": retrieval.duration_ms,
            "Answer_Duration_ms": answer.duration_ms,
        }
    )
    return row


def answer_session_questions(
    *,
    memory: MemConflictMemory,
    answerer: MemConflictAnswerer,
    session: Any,
    top_k: int,
    stored_top_k: int,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Answer one session's questions, in parallel when asked (point 17b).

    The memory state cannot change between the session's ingest and its
    questions, so the questions are independent reads of the same state and the
    original order is restored afterwards. ``workers=1`` is the historical
    serial loop.
    """
    questions = list(session.questions)
    if not questions:
        return []
    if int(workers) <= 1 or len(questions) == 1:
        return [
            _answer_one_question(
                memory=memory,
                answerer=answerer,
                question=question,
                top_k=top_k,
                stored_top_k=stored_top_k,
            )
            for question in questions
        ]
    with ThreadPoolExecutor(max_workers=min(int(workers), len(questions))) as pool:
        futures = [
            pool.submit(
                _answer_one_question,
                memory=memory,
                answerer=answerer,
                question=question,
                top_k=top_k,
                stored_top_k=stored_top_k,
            )
            for question in questions
        ]
        return [future.result() for future in futures]


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
    answer_workers: int = 1,
    skip_session_ids: frozenset[int] | set[int] | tuple[int, ...] = (),
    max_sessions: int | None = None,
    on_session: Any = None,
) -> dict[str, Any]:
    store_dir = store_dir_for(output_dir, persona, version)
    persona_start = time.perf_counter()
    started_at = time.time()
    unit = ollama_units.active_unit() or ""
    sessions_out: list[dict[str, Any]] = []
    answered_questions = 0
    answer_errors = 0
    sessions = persona.sessions
    if max_sessions is not None:
        sessions = sessions[: max(0, int(max_sessions))]
    skipped_sessions = 0
    if skip_session_ids:
        wanted = {int(value) for value in skip_session_ids}
        kept = tuple(session for session in sessions if session.session_id not in wanted)
        skipped_sessions = len(sessions) - len(kept)
        sessions = kept

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
            # Point 17b: the session is fully ingested before any of its
            # questions run, so the questions can go side by side.
            questions_out = answer_session_questions(
                memory=memory,
                answerer=answerer,
                session=session,
                top_k=top_k,
                stored_top_k=stored_top_k,
                workers=answer_workers,
            )
            session_answer_errors = sum(
                1 for row in questions_out if row.get("Answer_Error")
            )
            answered_questions += len(questions_out)
            answer_errors += session_answer_errors
            sessions_out.append(
                {
                    "Session_ID": session.session_id,
                    "Date": session.date,
                    "Session_Type": session.session_type,
                    "Question_Trigger_Types": list(session.question_trigger_types),
                    "Event_Types": list(session.event_types),
                    "Ingest": ingest.to_dict(),
                    "Questions": questions_out,
                    "Answer_Error_Count": session_answer_errors,
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
                        "Answer_Error_Count": session_answer_errors,
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
        "Answer_Workers": int(answer_workers),
        "Sessions_Replayed": len(sessions),
        "Sessions_Skipped": skipped_sessions,
        "Sessions_Available": len(persona.sessions),
        "Truncated": max_sessions is not None,
        "Session_Count": len(sessions_out),
        "Answered_Question_Count": answered_questions,
        "Answer_Error_Count": answer_errors,
        "Persona_Runtime_ms": (time.perf_counter() - persona_start) * 1000.0,
        "Started_At": started_at,
        "Finished_At": time.time(),
        "Sessions": sessions_out,
    }


def parse_persona_indices(raw: str) -> list[int]:
    """``"0,4,18-20"`` -> ``[0, 4, 18, 19, 20]`` (order kept, duplicates dropped).

    Point 27 asks for *random* persona shards, which the contiguous
    ``--start-index/--end-index`` window cannot express.
    """
    indices: list[int] = []
    for chunk in str(raw or "").replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            start_text, _, end_text = chunk.partition("-")
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as error:
                raise ValueError(f"not a range: {chunk!r}") from error
            if end < start:
                raise ValueError(f"empty range: {chunk!r}")
            candidates: Iterable[int] = range(start, end + 1)
        else:
            try:
                candidates = [int(chunk)]
            except ValueError as error:
                raise ValueError(f"not an index: {chunk!r}") from error
        for value in candidates:
            if value < 0:
                raise ValueError(f"negative index: {value}")
            if value not in indices:
                indices.append(value)
    if not indices:
        raise ValueError("no persona indices given")
    return indices


def select_personas_by_index(
    personas: Sequence[Any], indices: Sequence[int]
) -> tuple[list[Any], list[int]]:
    """Return the selected personas plus their dataset indices, in dataset order.

    A shard is a *set* of personas: the run replays them in dataset order and
    ``merge_shards.py`` restores that order anyway, so ordering here only affects
    which worker starts first.
    """
    total = len(personas)
    wanted = set(int(index) for index in indices)
    for index in sorted(wanted):
        if index >= total:
            raise ValueError(
                f"persona index {index} is outside the dataset (valid: 0..{total - 1})"
            )
    selected = [
        (index, persona) for index, persona in enumerate(personas) if index in wanted
    ]
    return [persona for _index, persona in selected], [index for index, _p in selected]


def short_persona_id(persona_id: Any) -> str:
    """Compact persona label for the one-line progress report.

    The released ids are UUIDs, so the first dash-separated block is both
    unique enough and short; anything unusual falls back to 8 characters.
    """
    text = str(persona_id or "?")
    head = text.split("-", 1)[0]
    if 4 <= len(head) <= 12:
        return head
    return text[:8] if len(text) > 8 else text


#: Roles recorded in ``run_meta.json`` so a run states which channel served
#: which stage (OpenRouter vs Bailian vs Ollama, judge model, ...).
MODEL_ROLES = (
    "embedding",
    "memory_builder",
    "adjudication_model",
    "answer_model",
    "judge_model",
    "controller",
    "window_planner",
    "entity_judge",
    "semantic_reducer",
    "slm",
    "decomposition_gate",
)


def model_summary(config: Any) -> dict[str, str]:
    """``{"memory_builder": "dashscope_bailian/glm-5.1", ...}`` for run_meta."""
    summary: dict[str, str] = {}
    for role in MODEL_ROLES:
        model = getattr(config, role, None)
        if model is None:
            continue
        provider = str(getattr(model, "provider", "") or "")
        name = str(getattr(model, "model", "") or "")
        if provider or name:
            summary[role] = f"{provider}/{name}" if provider else name
    return summary


def load_model_fingerprint(path: Path) -> dict[str, str]:
    """The ``Models`` map recorded by an earlier run, if it has one."""
    if not Path(path).is_file():
        return {}
    try:
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    models = meta.get("Models")
    if not isinstance(models, dict):
        return {}
    return {str(role): str(value) for role, value in models.items()}


def changed_model_roles(
    previous: dict[str, str], current: dict[str, str]
) -> dict[str, tuple[str, str]]:
    """Roles whose model differs between the recorded run and this config."""
    return {
        role: (previous.get(role, ""), current.get(role, ""))
        for role in sorted(set(previous) | set(current))
        if previous.get(role) != current.get(role)
    }


def expected_run_work(
    personas: Sequence[Any],
    *,
    max_sessions: int | None,
    resume_state: dict[str, set[int]] | None = None,
) -> tuple[int, int]:
    """``(sessions, questions)`` this invocation is going to replay.

    ``--max-sessions`` truncates every chain and ``--resume`` skips the sessions
    that already have a row in ``sessions.jsonl``; both shrink the work the
    progress bar has to divide by.
    """
    sessions_total = 0
    questions_total = 0
    for persona in personas:
        skipped = (resume_state or {}).get(persona.persona_id, ())
        sessions, questions = expected_work(
            persona.sessions,
            max_sessions=max_sessions,
            skip_session_ids=skipped,
        )
        sessions_total += sessions
        questions_total += questions
    return sessions_total, questions_total


def main(argv: list[str] | None = None) -> int:
    runtime.ensure_utf8_console()
    args = build_arg_parser().parse_args(argv)
    output_dir = resolve_output_dir(args.output_dir)

    if not Path(args.input).is_file():
        print(f"[error] dataset not found: {args.input}", file=sys.stderr)
        return 2

    if args.persona_indices:
        if args.start_index or args.end_index is not None or args.persona_limit is not None:
            print(
                "[error] --persona-indices cannot be combined with "
                "--start-index/--end-index/--persona-limit",
                file=sys.stderr,
            )
            return 2
        try:
            wanted = parse_persona_indices(args.persona_indices)
            personas, persona_indices = select_personas_by_index(
                load_personas(args.input), wanted
            )
        except ValueError as error:
            print(f"[error] --persona-indices: {error}", file=sys.stderr)
            return 2
        if not personas:
            print("[error] --persona-indices selected no persona", file=sys.stderr)
            return 2
    else:
        end_index = args.end_index
        if args.persona_limit is not None:
            limit_end = args.start_index + args.persona_limit
            end_index = limit_end if end_index is None else min(end_index, limit_end)
        personas = load_personas(
            args.input, start_index=args.start_index, end_index=end_index
        )
        persona_indices = list(range(args.start_index, args.start_index + len(personas)))

    summary = dataset_summary(personas)
    print(f"[data] {json.dumps(summary, ensure_ascii=False)}")
    if not personas:
        print("[error] no personas selected", file=sys.stderr)
        return 2
    print(
        f"[select] personas   : {len(personas)} "
        f"(dataset indices {','.join(str(value) for value in persona_indices)})"
    )

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

    # P1-2: concurrency is multiplicative. Four persona workers each running
    # the configured eight extraction threads means 32 calls in flight against
    # the same endpoints, which is what produced the builder validation
    # failures in the first parallel attempt. The overrides travel through the
    # environment so every worker process sees them.
    if args.extraction_workers is not None:
        os.environ[runtime.EXTRACTION_WORKERS_ENV] = str(max(1, int(args.extraction_workers)))
    if args.entity_judge_workers is not None:
        os.environ[runtime.ENTITY_JUDGE_WORKERS_ENV] = str(max(1, int(args.entity_judge_workers)))

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
    errors_path = output_dir / "errors.jsonl"

    # P1: --resume replays an interrupted run in place. sessions.jsonl already
    # holds one row per *finished* session, so those sessions are skipped and
    # the persona's memory store is kept instead of being rebuilt.
    resume_state: dict[str, set[int]] = {}
    if args.resume:
        resume_state = load_completed_sessions(sessions_path)
        if not resume_state and not results_path.is_file():
            print(
                "[error] --resume found nothing to continue in "
                f"{output_dir} (no sessions.jsonl rows)",
                file=sys.stderr,
            )
            return 2
        print(
            f"[resume] skipping {sum(len(v) for v in resume_state.values())} "
            f"completed session(s) across {len(resume_state)} persona(s)"
        )
        # A store is only valid for the models that built it: continuing with a
        # different embedding model mixes vector spaces, and a different
        # builder / planner / adjudicator mixes incompatible memory content.
        previous_models = load_model_fingerprint(output_dir / "run_meta.json")
        if previous_models:
            current_models = model_summary(config)
            changed = changed_model_roles(previous_models, current_models)
            if changed and not args.allow_model_change:
                print(
                    "[error] --resume would continue a store built with different "
                    "models:",
                    file=sys.stderr,
                )
                for role, (before, after) in changed.items():
                    print(
                        f"        {role}: {before or '(unset)'} -> {after or '(unset)'}",
                        file=sys.stderr,
                    )
                print(
                    "        Run this shard into a fresh --output-dir instead, or "
                    "pass --allow-model-change (the result then mixes models).",
                    file=sys.stderr,
                )
                return 2
            if changed:
                print(
                    f"[warn] --allow-model-change: continuing with "
                    f"{len(changed)} changed model role(s); results mix models",
                    file=sys.stderr,
                )
        else:
            print(
                "[resume] run_meta.json has no model fingerprint (older run); "
                "cannot verify that the models are unchanged"
            )

    answer_workers = max(1, int(args.answer_workers))
    print(f"[answer] workers           : {answer_workers} thread(s) per session")

    run_started = time.time()
    jobs = [
        {
            "persona_index": index,
            "dataset_index": persona_indices[index],
            "persona": persona,
            "output_dir": str(output_dir),
            "config_path": str(args.config),
            "unit": assignments[index],
            "top_k": args.top_k,
            "stored_top_k": args.stored_top_k,
            "version": args.version,
            "keep_memory": bool(args.keep_memory or args.resume),
            "answer_workers": answer_workers,
            "skip_session_ids": sorted(resume_state.get(persona.persona_id, ())),
            "max_sessions": args.max_sessions,
        }
        for index, persona in enumerate(personas)
    ]

    # Every persona is written the moment it finishes (P0-1). The dataset order
    # is recovered from Persona_Index, so a run that dies while personas are
    # still in flight keeps whatever finished; holding records back until the
    # earlier slots were filled used to lose them.
    #
    # sessions.jsonl is the finer-grained progress log (one row per finished
    # session) that run_scoring.py merges from and --resume reads.
    failed_personas: list[str] = []
    run_sessions, run_questions = expected_run_work(
        personas, max_sessions=args.max_sessions, resume_state=resume_state
    )
    progress = ProgressReporter(
        run_sessions,
        label="sessions",
        counters={"questions": 0, "errors": 0, "personas": 0, "failed": 0},
        totals={"questions": run_questions, "personas": len(personas)},
        stream=sys.stderr,
        enabled=not args.no_progress,
    )
    running_personas: set[str] = set()
    progress.start()
    file_mode = "a" if args.resume else "w"
    with open(results_path, file_mode, encoding="utf-8") as handle, open(
        sessions_path, file_mode, encoding="utf-8"
    ) as sessions_handle, open(errors_path, file_mode, encoding="utf-8") as errors_handle:
        def record_session(row: dict[str, Any]) -> None:
            if row.get("Event") == "persona_start":
                running_personas.add(str(row.get("Persona_ID") or ""))
                progress.note(running=f"{len(running_personas)}/{len(personas)}")
                progress.log(
                    f"[persona {int(row.get('Persona_Index', 0)) + 1}/{len(personas)}] "
                    f"{row.get('Persona_ID')} sessions={row.get('Session_Count')} "
                    f"questions={row.get('Question_Count')} "
                    f"unit={row.get('Ollama_Unit') or '(config default)'}"
                )
                return
            sessions_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            sessions_handle.flush()
            # The row is what --resume reads and what the progress line counts,
            # so both stay consistent even if the run is interrupted here.
            progress.advance(
                1,
                questions=len(row.get("Questions") or []),
                errors=int(row.get("Answer_Error_Count") or 0),
            )
            progress.note(
                last=f"{short_persona_id(row.get('Persona_ID'))} s{row.get('Session_ID')}"
            )

        def persona_finished(persona_id: str, *, failed: bool) -> None:
            running_personas.discard(persona_id)
            progress.advance(0, personas=1, failed=1 if failed else 0)
            progress.note(running=f"{len(running_personas)}/{len(personas)}")

        def record_result(index: int, record: Any) -> None:
            """Write each finished persona immediately (completion order)."""
            if isinstance(record, parallel.JobError):
                # One persona failing must not throw away the others' work.
                failed_personas.append(personas[index].persona_id)
                last_line = (record.traceback_text or "").strip().splitlines()
                progress.log(
                    f"  !! persona {index + 1} failed ({personas[index].persona_id}): "
                    f"{record.summary}"
                )
                if last_line:
                    progress.log(f"     {last_line[-1]}")
                record = {
                    "Persona_ID": personas[index].persona_id,
                    "Ollama_Unit": assignments[index] or "",
                    "Error": record.summary,
                    "Error_Traceback": record.traceback_text,
                }
                persona_finished(personas[index].persona_id, failed=True)
            done = dict(record)
            done["Persona_Index"] = index
            done["Dataset_Index"] = int(jobs[index].get("dataset_index", index))
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
                return
            handle.write(json.dumps(done, ensure_ascii=False) + "\n")
            handle.flush()
            progress.log(
                f"  -> persona {index + 1} answered "
                f"{done.get('Answered_Question_Count')} questions in "
                f"{float(done.get('Persona_Runtime_ms', 0.0)) / 1000.0:.1f}s "
                f"(unit {done.get('Ollama_Unit') or 'config default'})"
            )
            persona_finished(personas[index].persona_id, failed=False)

        try:
            parallel.run_jobs(
                persona_job,
                jobs,
                workers=persona_workers,
                on_row=record_session,
                on_result=record_result,
                raise_errors=False,
            )
        finally:
            progress.finish()

    meta = {
        "Memory_System": runtime.MEMORY_SYSTEM_NAME,
        "Dataset": str(Path(args.input).resolve()),
        "Retrival_Mem_Root": str(runtime.retrival_mem_root()),
        "Config": str(Path(args.config).resolve()),
        "Models": model_summary(config),
        "Persona_Indices": list(persona_indices),
        "Answer_Top_K": args.top_k,
        "Stored_Top_K": args.stored_top_k,
        "Max_Sessions": args.max_sessions,
        "Version": args.version,
        "Persona_Workers": persona_workers,
        "Answer_Workers": answer_workers,
        "Extraction_Workers": args.extraction_workers,
        "Entity_Judge_Workers": args.entity_judge_workers,
        "Resume": bool(args.resume),
        "Resumed_Session_Skips": {
            persona_id: sorted(session_ids)
            for persona_id, session_ids in resume_state.items()
        },
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
