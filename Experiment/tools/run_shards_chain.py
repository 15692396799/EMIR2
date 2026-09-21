"""Point 27 watchdog: run the persona shards one after another, unattended.

Typical use while ``shard_1`` is already running in your own terminal::

    python Experiment\\tools\\run_shards_chain.py

It waits for the shard that is currently running (no ``run_meta.json`` yet),
then for every later shard it runs

1. ``run_experiment.py`` with the plan's persona indices,
2. ``run_scoring.py`` for that shard (so each shard has its own table),
3. ``merge_shards.py --allow-partial`` over every finished shard plus a scoring
   pass for the merged directory (so the cumulative tables refresh by
   themselves).

A shard whose ``run_meta.json`` lists ``Failed_Personas`` is retried with
``--resume`` (bounded by ``--max-resume-attempts``); the checkpoint guard makes
that safe against the upstream stale-cache crash.

Everything is written to ``Experiment/runs/<shard>.chain.log`` and echoed to the
console, and the chain keeps its state in ``Experiment/shards/chain_state.json``
so an interrupted chain can be restarted with the same command.

Usage::

    python Experiment/tools/run_shards_chain.py --dry-run          # print, do nothing
    python Experiment/tools/run_shards_chain.py                    # wait for shard_1, then 2..6
    python Experiment/tools/run_shards_chain.py --shards shard_1,shard_2
    python Experiment/tools/run_shards_chain.py --start-if-idle    # start shard_1 too

Safety: it refuses to start a shard while its ``run_meta.json`` is missing
unless ``--start-if-idle`` is given (that is the "already running" state), and a
lock file stops two chains from running at once.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from memconflict_eval import runtime  # noqa: E402

DEFAULT_PLAN = EXPERIMENT_DIR / "shards" / "shard_plan.json"
DEFAULT_STATE = EXPERIMENT_DIR / "shards" / "chain_state.json"
DEFAULT_LOCK = EXPERIMENT_DIR / "shards" / "chain.lock"
DEFAULT_RUNS_ROOT = EXPERIMENT_DIR / "runs"
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "eval_large_bailian.yaml"

DONE = "done"
FAILED = "failed"
WAITING = "waiting"
IDLE = "idle"


def shard_status(run_dir: Path) -> str:
    """``done`` / ``failed`` / ``waiting`` / ``idle`` for one shard directory.

    ``run_meta.json`` is written only when the run returns, so its absence means
    the shard is still in flight (``waiting``) or has not started at all
    (``idle``) — the caller decides how to treat those two.

    A *resume* of a failed shard appends to ``sessions.jsonl`` after the last
    ``run_meta.json`` was written; that is detected here as ``waiting`` so the
    chain never starts a second process against the same memory store.
    """
    run_dir = Path(run_dir)
    meta_path = Path(run_dir) / "run_meta.json"
    sessions_path = Path(run_dir) / "sessions.jsonl"
    if meta_path.is_file():
        try:
            if (
                sessions_path.is_file()
                and sessions_path.stat().st_mtime
                > meta_path.stat().st_mtime + 1.0
            ):
                return WAITING
        except OSError:
            pass
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return FAILED
        return DONE if not (meta.get("Failed_Personas") or []) else FAILED
    if sessions_path.is_file() or (Path(run_dir) / "results.jsonl").is_file():
        return WAITING
    return IDLE


class Chain:
    """Drive the shards in order; every external step goes through ``run_step``."""

    def __init__(
        self,
        *,
        plan: dict[str, Any],
        runs_root: Path,
        config: Path,
        persona_workers: int = 5,
        answer_workers: int = 4,
        extraction_workers: int = 2,
        entity_judge_workers: int = 1,
        ollama_units: str | None = None,
        white_box_k: str = "2,3,5",
        max_resume_attempts: int = 2,
        poll_seconds: float = 120.0,
        wait_timeout_hours: float = 12.0,
        start_if_idle: bool = False,
        resume_stalled_after_minutes: float | None = None,
        score_shards: bool = True,
        merge_partial: bool = True,
        parallel: int = 1,
        units_map: dict[str, str] | None = None,
        units_pool: Sequence[str] | None = None,
        dry_run: bool = False,
        step_runner: Callable[[list[str], Path], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        printer: Callable[[str], None] = print,
    ) -> None:
        self.plan = plan
        self.runs_root = Path(runs_root)
        self.config = Path(config)
        self.persona_workers = persona_workers
        self.answer_workers = answer_workers
        self.extraction_workers = extraction_workers
        self.entity_judge_workers = entity_judge_workers
        self.ollama_units = ollama_units
        self.white_box_k = white_box_k
        self.max_resume_attempts = max_resume_attempts
        self.poll_seconds = poll_seconds
        self.wait_timeout_hours = wait_timeout_hours
        self.start_if_idle = start_if_idle
        self.resume_stalled_after_minutes = resume_stalled_after_minutes
        self.score_shards = score_shards
        self.merge_partial = merge_partial
        self.parallel = max(1, int(parallel))
        self.units_map = dict(units_map or {})
        self.units_pool = [str(unit) for unit in (units_pool or []) if str(unit).strip()]
        self.dry_run = dry_run
        self._step_runner = step_runner or self._subprocess_step
        self._clock = clock
        self._sleep = sleep
        self._print = printer
        self.state: dict[str, Any] = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "shards": {},
        }

    # -- commands ----------------------------------------------------------

    def unit_for(self, shard: dict[str, Any], index: int = 0) -> str | None:
        """The Ollama container this shard should use (point 27: spread shards)."""
        name = str(shard.get("name") or "")
        if name in self.units_map:
            return self.units_map[name]
        if self.units_pool:
            return self.units_pool[index % len(self.units_pool)]
        return self.ollama_units

    def run_command(self, shard: dict[str, Any], *, resume: bool = False) -> list[str]:
        indices = ",".join(str(value) for value in shard["personas"])
        command = [
            sys.executable,
            "-u",
            str(EXPERIMENT_DIR / "run_experiment.py"),
            "--config",
            str(self.config),
            "--persona-indices",
            indices,
            "--persona-workers",
            str(self.persona_workers),
            "--answer-workers",
            str(self.answer_workers),
            "--extraction-workers",
            str(self.extraction_workers),
            "--entity-judge-workers",
            str(self.entity_judge_workers),
            "--output-dir",
            str(self.runs_root / shard["name"]),
        ]
        unit = self.unit_for(shard, int(shard.get("_order", 0) or 0))
        if unit:
            command += ["--ollama-units", unit]
        if resume:
            command.append("--resume")
        return command

    def score_command(self, run_dir: Path) -> list[str]:
        return [
            sys.executable,
            "-u",
            str(EXPERIMENT_DIR / "run_scoring.py"),
            "--run-dir",
            str(run_dir),
            "--white-box-k",
            self.white_box_k,
        ]

    def merge_command(self, finished: Sequence[dict[str, Any]]) -> list[str]:
        command = [
            sys.executable,
            "-u",
            str(EXPERIMENT_DIR / "tools" / "merge_shards.py"),
            "--allow-partial",
            "--out",
            str(self.runs_root / "merged"),
            "--shards",
        ]
        command += [str(self.runs_root / shard["name"]) for shard in finished]
        return command

    # -- steps -------------------------------------------------------------

    def run_step(self, command: list[str], log_path: Path) -> int:
        self._print(f"[chain] $ {' '.join(command)}")
        if self.dry_run:
            return 0
        return int(self._step_runner(command, log_path))

    @staticmethod
    def _subprocess_step(command: list[str], log_path: Path) -> int:
        """Run a step, echoing its output to the console and a log file."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} "
                      f"{' '.join(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            return process.wait()

    # -- waiting -----------------------------------------------------------

    def wait_for_shard(self, shard: dict[str, Any]) -> str:
        """Wait until the shard directory reports done/failed (or start it)."""
        run_dir = self.runs_root / shard["name"]
        started = self._clock()
        last_size = -1
        last_change = self._clock()
        announced = False
        while True:
            status = shard_status(run_dir)
            if status in (DONE, FAILED):
                return status
            sessions_path = run_dir / "sessions.jsonl"
            size = sessions_path.stat().st_size if sessions_path.is_file() else 0
            if size != last_size:
                last_size, last_change = size, self._clock()
            if status == IDLE and self.start_if_idle:
                self._print(f"[chain] {shard['name']} has not started; starting it now")
                return "start"
            if not announced:
                self._print(
                    f"[chain] waiting for {shard['name']} (already running in "
                    "another terminal)"
                )
                announced = True
            stalled = (
                self.resume_stalled_after_minutes is not None
                and (self._clock() - last_change) / 60.0
                >= self.resume_stalled_after_minutes
            )
            if stalled:
                self._print(
                    f"[chain] {shard['name']} produced no new session row for "
                    f"{self.resume_stalled_after_minutes:.0f} min; resuming it"
                )
                return "stalled"
            if (self._clock() - started) / 3600.0 >= self.wait_timeout_hours:
                self._print(
                    f"[chain] gave up waiting for {shard['name']} after "
                    f"{self.wait_timeout_hours:.1f} h"
                )
                return "timeout"
            self._sleep(self.poll_seconds)

    # -- main loop ---------------------------------------------------------

    def run(self, shards: Sequence[dict[str, Any]]) -> int:
        for order, shard in enumerate(shards):
            shard["_order"] = order
        if self.parallel > 1:
            return self._run_parallel(shards)
        return self._run_sequential(shards)

    def _run_sequential(self, shards: Sequence[dict[str, Any]]) -> int:
        finished: list[dict[str, Any]] = []
        for position, shard in enumerate(shards):
            name = shard["name"]
            run_dir = self.runs_root / shard["name"]
            log_path = self.runs_root / f"{name}.chain.log"
            status = shard_status(run_dir)
            if self.dry_run:
                # Print exactly what a real chain would execute, without waiting
                # and without touching the shard directories.
                self._print(f"[chain] {name}: status={status} (dry run)")
                self.run_step(self.run_command(shard, resume=status == FAILED), log_path)
                if self.score_shards:
                    self.run_step(self.score_command(run_dir), log_path)
                finished.append(shard)
                if self.merge_partial:
                    self.run_step(self.merge_command(finished), log_path)
                    if self.score_shards:
                        self.run_step(self.score_command(self.runs_root / "merged"), log_path)
                continue
            if status == WAITING or (status == IDLE and not self.start_if_idle):
                waited = self.wait_for_shard(shard)
                if waited == "timeout":
                    return 1
                status = shard_status(run_dir)
                if waited in ("start", "stalled"):
                    status = FAILED if waited == "stalled" else IDLE
            attempts = 0
            while status != DONE:
                attempts += 1
                if attempts > self.max_resume_attempts + 1:
                    self._print(
                        f"[chain] {name} still failing after "
                        f"{self.max_resume_attempts} resume attempt(s); stopping"
                    )
                    return 1
                resume = status == FAILED
                self._print(
                    f"[chain] running {name}"
                    + (" (--resume)" if resume else "")
                )
                code = self.run_step(self.run_command(shard, resume=resume), log_path)
                self.state["shards"][name] = {
                    "exit_code": code,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "resumed": resume,
                }
                status = shard_status(run_dir)
                if code != 0 and status != DONE:
                    self._print(
                        f"[chain] {name} exited with {code}; see "
                        f"{self.runs_root / 'errors.jsonl'} and {log_path}"
                    )
            self.state["shards"].setdefault(name, {})["status"] = DONE
            finished.append(shard)
            if self.score_shards:
                self.run_step(self.score_command(run_dir), log_path)
            if self.merge_partial and len(finished) >= 1:
                merged_dir = self.runs_root / "merged"
                self.run_step(self.merge_command(finished), log_path)
                if self.score_shards:
                    self.run_step(self.score_command(merged_dir), log_path)
        self._print(
            f"[chain] all {len(finished)} shard(s) done: "
            + ", ".join(shard["name"] for shard in finished)
        )
        return 0

    def _run_one_shard(
        self, shard: dict[str, Any], run_dir: Path, log_path: Path
    ) -> int:
        """Wait if needed, run (resume on failure), then score one shard."""
        name = shard["name"]
        unit = self.unit_for(shard, int(shard.get("_order", 0) or 0))
        status = shard_status(run_dir)
        if self.dry_run:
            self._print(f"[chain] {name}: status={status} unit={unit or '(config default)'} (dry run)")
            self.run_step(self.run_command(shard, resume=status == FAILED), log_path)
            if self.score_shards:
                self.run_step(self.score_command(run_dir), log_path)
            return 0
        if status == WAITING or (status == IDLE and not self.start_if_idle):
            waited = self.wait_for_shard(shard)
            if waited == "timeout":
                return 1
            status = shard_status(run_dir)
            if waited in ("start", "stalled"):
                status = FAILED if waited == "stalled" else IDLE
        attempts = 0
        while status != DONE:
            attempts += 1
            if attempts > self.max_resume_attempts + 1:
                self._print(
                    f"[chain] {name} still failing after "
                    f"{self.max_resume_attempts} resume attempt(s); giving up"
                )
                return 1
            resume = status == FAILED
            self._print(
                f"[chain] running {name} unit={unit or '(config default)'}"
                + (" (--resume)" if resume else "")
            )
            code = self.run_step(self.run_command(shard, resume=resume), log_path)
            self.state["shards"].setdefault(name, {}).update(
                {
                    "exit_code": code,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "resumed": resume,
                    "ollama_unit": unit,
                }
            )
            status = shard_status(run_dir)
            if code != 0 and status != DONE:
                self._print(
                    f"[chain] {name} exited with {code}; see "
                    f"{self.runs_root / 'errors.jsonl'} and {log_path}"
                )
        self.state["shards"].setdefault(name, {})["status"] = DONE
        if self.score_shards:
            score_code = self.run_step(self.score_command(run_dir), log_path)
            if score_code != 0:
                self._print(f"[chain] scoring {name} failed with {score_code}")
                return score_code
        return 0

    def _run_parallel(self, shards: Sequence[dict[str, Any]]) -> int:
        """Run up to ``self.parallel`` shards at once, one container each."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        finished: list[dict[str, Any]] = []
        state_lock = threading.Lock()
        merge_lock = threading.Lock()
        codes: dict[str, int] = {}

        def handle(shard: dict[str, Any]) -> int:
            name = shard["name"]
            run_dir = self.runs_root / shard["name"]
            log_path = self.runs_root / f"{name}.chain.log"
            code = self._run_one_shard(shard, run_dir, log_path)
            with state_lock:
                codes[name] = code
                if code == 0:
                    finished.append(shard)
                snapshot = list(finished) if len(finished) >= 1 else []
            if snapshot and self.merge_partial:
                # One merge at a time: two writers must never share runs/merged.
                with merge_lock:
                    merged_dir = self.runs_root / "merged"
                    self.run_step(self.merge_command(snapshot), log_path)
                    if self.score_shards:
                        self.run_step(self.score_command(merged_dir), log_path)
            return code

        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            futures = {pool.submit(handle, shard): shard for shard in shards}
            for future in as_completed(futures):
                shard = futures[future]
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001 - reported, keep going
                    self._print(
                        f"[chain] {shard['name']} raised {type(error).__name__}: {error}"
                    )
                    codes.setdefault(shard["name"], 1)

        failed = sorted(name for name, code in codes.items() if code != 0)
        self._print(
            f"[chain] {len(finished)}/{len(shards)} shard(s) done"
            + (f"; failed: {', '.join(failed)}" if failed else "")
        )
        return 0 if not failed else 1


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if not plan.get("plan"):
        raise SystemExit(f"[error] {path} has no 'plan' list; run tools/shard_plan.py first")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--shards",
        type=str,
        default=None,
        help="Comma-separated shard names to chain (default: every shard in the plan).",
    )
    parser.add_argument("--config", type=Path, default=None, help="Default: the plan's config.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--persona-workers", type=int, default=5)
    parser.add_argument("--answer-workers", type=int, default=4)
    parser.add_argument("--extraction-workers", type=int, default=2)
    parser.add_argument("--entity-judge-workers", type=int, default=1)
    parser.add_argument("--ollama-units", type=str, default=None)
    parser.add_argument(
        "--units-map",
        type=str,
        default=None,
        help=(
            "Per-shard Ollama container, e.g. "
            "'shard_1=http://host:41135;shard_2=http://host:41133;shard_3=http://host:41136'. "
            "Shards without an entry fall back to --ollama-units (round robin)."
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "Run up to N shards at once (default 1). Give each one its own "
            "container with --units-map; the merged tables are refreshed one "
            "shard at a time under a lock."
        ),
    )
    parser.add_argument("--white-box-k", type=str, default="2,3,5")
    parser.add_argument("--max-resume-attempts", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--wait-timeout-hours", type=float, default=12.0)
    parser.add_argument(
        "--start-if-idle",
        action="store_true",
        help="Start a shard that has not been launched yet instead of waiting for it.",
    )
    parser.add_argument(
        "--resume-stalled-after-minutes",
        type=float,
        default=None,
        help=(
            "Resume a shard that stopped producing session rows for this long. "
            "Only use it when you are sure the original process is dead: two "
            "processes must never write one memory store."
        ),
    )
    parser.add_argument("--no-score", action="store_true", help="Run shards only.")
    parser.add_argument("--no-merge", action="store_true", help="Skip the merged tables.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    runtime.ensure_utf8_console()

    if args.lock.is_file() and not args.dry_run:
        print(
            f"[error] {args.lock} exists: another chain may be running. Delete it "
            "if you are sure it is stale.",
            file=sys.stderr,
        )
        return 2

    plan = load_plan(args.plan)
    shards = list(plan["plan"])
    if args.shards:
        wanted = [name.strip() for name in args.shards.split(",") if name.strip()]
        by_name = {shard["name"]: shard for shard in shards}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            print(f"[error] shard(s) not in the plan: {', '.join(missing)}", file=sys.stderr)
            return 2
        shards = [by_name[name] for name in wanted]

    config = args.config or Path(plan.get("config") or DEFAULT_CONFIG)
    units_map: dict[str, str] = {}
    for chunk in str(args.units_map or "").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            print(
                f"[error] --units-map entries must look like shard_1=http://host:port "
                f"(got {chunk!r})",
                file=sys.stderr,
            )
            return 2
        name, _, unit = chunk.partition("=")
        units_map[name.strip()] = unit.strip()
    units_pool = [
        unit.strip()
        for unit in str(args.ollama_units or "").split(",")
        if unit.strip()
    ]
    chain = Chain(
        plan=plan,
        runs_root=args.runs_root,
        config=config,
        persona_workers=args.persona_workers,
        answer_workers=args.answer_workers,
        extraction_workers=args.extraction_workers,
        entity_judge_workers=args.entity_judge_workers,
        ollama_units=args.ollama_units,
        white_box_k=args.white_box_k,
        max_resume_attempts=args.max_resume_attempts,
        poll_seconds=args.poll_seconds,
        wait_timeout_hours=args.wait_timeout_hours,
        start_if_idle=args.start_if_idle,
        resume_stalled_after_minutes=args.resume_stalled_after_minutes,
        score_shards=not args.no_score,
        merge_partial=not args.no_merge,
        parallel=args.parallel,
        units_map=units_map,
        units_pool=units_pool,
        dry_run=args.dry_run,
    )
    print(f"[chain] plan    : {args.plan} ({len(shards)} shard(s))")
    print(f"[chain] config  : {config}")
    print(f"[chain] runs    : {args.runs_root}")
    for shard in shards:
        shard["_order"] = shards.index(shard)
        print(
            f"[chain]   {shard['name']}: personas {shard['personas']} "
            f"-> {shard_status(args.runs_root / shard['name'])} "
            f"| unit {chain.unit_for(shard, shard['_order']) or '(config default)'}"
        )
    if args.parallel > 1:
        print(f"[chain] parallel : {args.parallel} shard(s) at once")

    if not args.dry_run:
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        code = chain.run(shards)
    finally:
        if not args.dry_run:
            args.lock.unlink(missing_ok=True)
        try:
            args.state.parent.mkdir(parents=True, exist_ok=True)
            args.state.write_text(
                json.dumps(chain.state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[chain] state   : {args.state}")
        except OSError:
            pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
