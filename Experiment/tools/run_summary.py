"""Print the short summary of a finished run from its ``run_meta.json``.

Kept as a tool instead of an inline ``python -c`` inside the batch file: a
line like ``print('%.1f' % value)`` contains bare ``%`` characters, and cmd.exe
expands those before Python ever sees them, which turned the summary into a
``SyntaxError`` (seen on 2026-09-20 with ``run_scale1h.bat``).

Usage::

    python Experiment/tools/run_summary.py --run-dir Experiment/runs/scale1h
    python Experiment/tools/run_summary.py --meta Experiment/runs/<run>/run_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from memconflict_eval import runtime  # noqa: E402


def summarize(meta: Mapping[str, Any]) -> list[str]:
    """The summary lines, without printing anything (easy to unit-test)."""
    wall_clock = float(meta.get("Wall_Clock_s") or 0.0)
    workers = "%s persona / %s answer" % (
        meta.get("Persona_Workers", "?"),
        meta.get("Answer_Workers", "?"),
    )
    failed = meta.get("Failed_Personas") or []
    return [
        "   wall clock      : %.1f s (%.1f min)" % (wall_clock, wall_clock / 60.0),
        "   workers         : " + workers,
        "   failed personas : " + (", ".join(str(item) for item in failed) or "none"),
    ]


def load_meta(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Run directory holding run_meta.json.")
    source.add_argument("--meta", type=Path, help="Explicit run_meta.json path.")
    args = parser.parse_args(argv)
    runtime.ensure_utf8_console()

    meta_path = args.meta if args.meta is not None else Path(args.run_dir) / "run_meta.json"
    if not Path(meta_path).is_file():
        print(
            "   run_meta.json missing: the run did not finish, re-run with:"
            "  run_scale1h.bat resume",
            file=sys.stderr,
        )
        return 1
    try:
        meta = load_meta(meta_path)
    except (OSError, json.JSONDecodeError) as error:
        print(f"   could not read {meta_path}: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    for line in summarize(meta):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
