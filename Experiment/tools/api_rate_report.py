"""How hard is this run hitting the API platform, and how much headroom is left?

Reads the per-persona API logs (``Memory/*/v4_*.jsonl``) written by
``run_experiment.py`` and reports, per provider, the measured request rate and
token rate. Compare that with the platform quota to decide whether another
terminal/machine (another shard) fits::

    python Experiment\\tools\\api_rate_report.py --run-dir Experiment\\runs\\shard_1
    python Experiment\\tools\\api_rate_report.py --run-dir Experiment\\runs\\shard_1 --tpm 1000000 --rpm 500
    python Experiment\\tools\\api_rate_report.py --run-dir Experiment\\runs --provider dashscope_bailian --machines 2

The rates are averaged over the window the logs cover, which is what a quota
check needs. A quota is per account (and usually per model), so the projection
simply multiplies the measured rate by ``--machines``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from memconflict_eval import runtime  # noqa: E402

DEFAULT_RUNS_ROOT = EXPERIMENT_DIR / "runs"


@dataclass
class ProviderStats:
    provider: str
    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    first: datetime | None = None
    last: datetime | None = None
    modules: Counter = field(default_factory=Counter)

    def add(self, row: dict[str, Any]) -> None:
        timestamp = _parse_time(row.get("timestamp"))
        self.calls += 1
        if not row.get("success"):
            self.failures += 1
        usage = row.get("token_usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.modules[str(row.get("module") or "?")] += 1
        if timestamp is not None:
            self.first = timestamp if self.first is None else min(self.first, timestamp)
            self.last = timestamp if self.last is None else max(self.last, timestamp)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def minutes(self) -> float:
        if self.first is None or self.last is None:
            return 0.0
        return max(0.0, (self.last - self.first).total_seconds() / 60.0)

    def per_minute(self, value: int) -> float:
        return value / self.minutes if self.minutes > 0 else 0.0


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def collect(run_dirs: Iterable[Path]) -> dict[str, ProviderStats]:
    """Aggregate every ``v4_*.jsonl`` line under the given run directories."""
    stats: dict[str, ProviderStats] = {}
    for run_dir in run_dirs:
        for path in sorted(Path(run_dir).glob("Memory/*/v4_*.jsonl")):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                provider = str(row.get("provider") or "unknown")
                stats.setdefault(provider, ProviderStats(provider=provider)).add(row)
    return stats


def resolve_run_dirs(paths: Sequence[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if (Path(path) / "Memory").is_dir():
            resolved.append(Path(path))
            continue
        resolved.extend(sorted(p for p in Path(path).iterdir() if p.is_dir()))
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        default=[DEFAULT_RUNS_ROOT],
        help="Run directory (or a directory of run directories). Default: Experiment/runs.",
    )
    parser.add_argument("--provider", type=str, default="dashscope_bailian")
    parser.add_argument("--rpm", type=float, default=None, help="Platform requests/min quota.")
    parser.add_argument("--tpm", type=float, default=None, help="Platform tokens/min quota.")
    parser.add_argument(
        "--machines",
        type=float,
        default=1.0,
        help="Project the measured rate onto this many equally loaded machines.",
    )
    args = parser.parse_args(argv)
    runtime.ensure_utf8_console()

    run_dirs = resolve_run_dirs(args.run_dir)
    if not run_dirs:
        print("[error] no run directories with a Memory/ directory found", file=sys.stderr)
        return 2
    stats = collect(run_dirs)
    if not stats:
        print("[error] no API logs found", file=sys.stderr)
        return 2

    print(f"runs: {', '.join(str(path) for path in run_dirs)}")
    print()
    print("| provider | calls | failed | minutes | req/min | tokens/min | in/out tokens |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for provider in sorted(stats, key=lambda name: -stats[name].total_tokens):
        row = stats[provider]
        print(
            "| {p} | {calls} | {fail} | {minutes:.1f} | {rpm:.1f} | {tpm:,.0f} | {pin:,} / {pout:,} |".format(
                p=provider,
                calls=row.calls,
                fail=row.failures,
                minutes=row.minutes,
                rpm=row.per_minute(row.calls),
                tpm=row.per_minute(row.total_tokens),
                pin=row.prompt_tokens,
                pout=row.completion_tokens,
            )
        )

    target = stats.get(args.provider)
    if target is None:
        print()
        print(f"[warn] no calls to provider {args.provider!r} in these runs")
        return 0

    print()
    print(f"provider under test: {args.provider}")
    print(f"  measured          : {target.per_minute(target.total_tokens):,.0f} tokens/min, "
          f"{target.per_minute(target.calls):.1f} requests/min")
    print(f"  x{args.machines:g} machines     : {target.per_minute(target.total_tokens) * args.machines:,.0f} tokens/min, "
          f"{target.per_minute(target.calls) * args.machines:.1f} requests/min")
    if args.tpm:
        share = target.per_minute(target.total_tokens) * args.machines / args.tpm * 100.0
        verdict = "fits" if share < 100 else "EXCEEDS the quota"
        print(f"  vs TPM {args.tpm:,.0f}      : {share:.1f}% - {verdict}")
    if args.rpm:
        share = target.per_minute(target.calls) * args.machines / args.rpm * 100.0
        verdict = "fits" if share < 100 else "EXCEEDS the quota"
        print(f"  vs RPM {args.rpm:,.0f}      : {share:.1f}% - {verdict}")
    print()
    print("note: quotas are per account (usually per model) and are shared with any")
    print("      other application using the same key; the rate above is what this")
    print("      run measured, averaged over the window the logs cover.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
