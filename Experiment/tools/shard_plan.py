"""Point 27: build the random persona shard plan and print the exact commands.

The personas are shuffled with a fixed seed and cut into equal shards, so the
split is *random, not contiguous* — each shard is an unbiased sample of the
benchmark, which is what makes the incremental tables meaningful. The same seed
always produces the same plan, and the manifest it writes is the contract used
later by ``merge_shards.py`` (disjoint persona sets, one shard per machine).

Usage::

    python Experiment/tools/shard_plan.py                       # 6 x 5, seed 27
    python Experiment/tools/shard_plan.py --only 1              # just shard 1
    python Experiment/tools/shard_plan.py --shards 3 --seed 7   # a different plan
    python Experiment/tools/shard_plan.py --config Experiment/configs/eval_large_bailian.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from memconflict_eval import runtime  # noqa: E402
from memconflict_eval.data import load_personas  # noqa: E402

DEFAULT_RUNS_ROOT = EXPERIMENT_DIR / "runs"
DEFAULT_MANIFEST = EXPERIMENT_DIR / "shards" / "shard_plan.json"


def build_plan(
    personas: Sequence[Any], *, shards: int, per_shard: int, seed: int
) -> list[dict[str, Any]]:
    """Shuffle the dataset indices and cut them into ``shards`` groups."""
    if shards < 1 or per_shard < 1:
        raise ValueError("shards and per-shard must be >= 1")
    total = len(personas)
    if shards * per_shard != total:
        raise ValueError(
            f"{shards} shards x {per_shard} personas = {shards * per_shard}, "
            f"but the dataset has {total} personas"
        )
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    plan: list[dict[str, Any]] = []
    for number in range(1, shards + 1):
        chunk = sorted(indices[(number - 1) * per_shard : number * per_shard])
        sessions = sum(len(personas[index].sessions) for index in chunk)
        questions = sum(personas[index].question_count for index in chunk)
        plan.append(
            {
                "name": f"shard_{number}",
                "personas": chunk,
                "sessions": sessions,
                "questions": questions,
            }
        )
    return plan


def run_command(
    shard: dict[str, Any],
    *,
    config: Path,
    runs_root: Path,
    workers: int,
    answer_workers: int,
    extraction_workers: int,
    entity_judge_workers: int,
    max_sessions: int | None,
    ollama_units: str | None,
) -> str:
    indices = ",".join(str(value) for value in shard["personas"])
    output_dir = _relative(runs_root / shard["name"])
    parts = [
        "python -u Experiment\\run_experiment.py",
        f"--config {_relative(config)}",
        f"--persona-indices {indices}",
        f"--persona-workers {workers}",
        f"--answer-workers {answer_workers}",
        f"--extraction-workers {extraction_workers}",
        f"--entity-judge-workers {entity_judge_workers}",
    ]
    if max_sessions is not None:
        parts.append(f"--max-sessions {max_sessions}")
    if ollama_units:
        parts.append(f"--ollama-units {ollama_units}")
    parts.append(f"--output-dir {output_dir}")
    # One long line on purpose: it works verbatim in cmd.exe and PowerShell.
    return " ".join(parts)


def _relative(path: Path) -> str:
    """Prefer a repo-relative path so the commands can be pasted on any host."""
    try:
        return str(Path(path).resolve().relative_to(EXPERIMENT_DIR.parent))
    except ValueError:
        return str(path)


def dataset_digest(path: Path) -> str | None:
    """SHA-256 of the dataset file: same hash -> same plan on every host."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=6, help="number of shards (default 6)")
    parser.add_argument("--per-shard", type=int, default=5, help="personas per shard (default 5)")
    parser.add_argument("--seed", type=int, default=27, help="shuffle seed (default 27)")
    parser.add_argument("--dataset", type=Path, default=runtime.default_dataset_path())
    parser.add_argument("--config", type=Path, default=EXPERIMENT_DIR / "configs" / "eval_large_bailian.yaml")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--persona-workers",
        type=int,
        default=10,
        help=(
            "Personas processed at once (default 10: two per Ollama lane with "
            "the six lanes of point 16). Capped by --per-shard, because a "
            "shard never runs more workers than it has personas."
        ),
    )
    parser.add_argument("--answer-workers", type=int, default=4)
    parser.add_argument("--extraction-workers", type=int, default=2)
    parser.add_argument("--entity-judge-workers", type=int, default=1)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--ollama-units", type=str, default=None)
    parser.add_argument("--only", type=int, default=None, help="print the commands for one shard only")
    args = parser.parse_args(argv)
    runtime.ensure_utf8_console()

    personas = load_personas(args.dataset)
    try:
        plan = build_plan(
            personas, shards=args.shards, per_shard=args.per_shard, seed=args.seed
        )
    except ValueError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 2

    digest = dataset_digest(Path(args.dataset))
    print(f"dataset        : {Path(args.dataset).resolve()}")
    print(f"dataset sha256 : {digest or '(unreadable)'}")
    print(f"personas       : {len(personas)}")
    print(f"plan           : {args.shards} shards x {args.per_shard} personas, seed {args.seed}")
    print(f"config         : {args.config}")
    print(
        "                 same dataset sha256 + same seed => the same plan on "
        "every host (no need to copy the file, but copying it is safer)"
    )
    print()
    print("| shard | personas | sessions | questions |")
    print("| --- | --- | ---: | ---: |")
    for shard in plan:
        print(
            "| {name} | {personas} | {sessions} | {questions} |".format(
                name=shard["name"],
                personas=",".join(str(value) for value in shard["personas"]),
                sessions=shard["sessions"],
                questions=shard["questions"],
            )
        )

    selected = [
        shard for shard in plan if args.only is None or shard["name"] == f"shard_{args.only}"
    ]
    if args.only is not None and not selected:
        print(f"[error] no such shard: {args.only}", file=sys.stderr)
        return 2

    for shard in selected:
        out_dir = args.runs_root / shard["name"]
        print()
        print("=" * 78)
        print(f"{shard['name']}: personas {shard['personas']} "
              f"({shard['sessions']} sessions, {shard['questions']} questions)")
        print("=" * 78)
        print(run_command(
            shard,
            config=args.config,
            runs_root=args.runs_root,
            workers=args.persona_workers,
            answer_workers=args.answer_workers,
            extraction_workers=args.extraction_workers,
            entity_judge_workers=args.entity_judge_workers,
            max_sessions=args.max_sessions,
            ollama_units=args.ollama_units,
        ))
        print()
        print("# score this shard on its own (same one judge pass):")
        print(
            f"python Experiment\\run_scoring.py --run-dir {_relative(out_dir)} "
            "--white-box-k 2,3,5"
        )

    print()
    print("=" * 78)
    print("merge + score (run after each shard finishes; partial merges are allowed)")
    print("=" * 78)
    finished = [
        shard
        for shard in plan
        if (args.runs_root / shard["name"] / "results.jsonl").is_file()
    ]
    if finished:
        shard_dirs = " ".join(
            _relative(args.runs_root / shard["name"]) for shard in finished
        )
        merged_dir = _relative(args.runs_root / "merged")
        print(
            f"python Experiment\\tools\\merge_shards.py --allow-partial "
            f"--out {merged_dir} --shards {shard_dirs}"
        )
        print(
            f"python Experiment\\run_scoring.py --run-dir {merged_dir} --white-box-k 2,3,5"
        )
        print()
        print(f"# {len(finished)}/{len(plan)} shard(s) have results.jsonl so far")
    else:
        print("# no shard has finished yet - the merge command appears here once one has")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_sha256": digest,
        "seed": args.seed,
        "shards": args.shards,
        "per_shard": args.per_shard,
        "persona_count": len(personas),
        "config": str(args.config),
        "runs_root": str(args.runs_root),
        "plan": plan,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"[plan] wrote {args.out}")
    print("[plan] keep this file: it is the record of which shard owns which persona.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
