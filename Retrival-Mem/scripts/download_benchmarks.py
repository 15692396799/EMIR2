#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory.config import load_config

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def prepare_proactive_membench(benchmark_dir: Path) -> None:
    zip_path = benchmark_dir / "ProactiveMemBench-76E1.zip"
    target = benchmark_dir / "proactive_membench"
    if (target / "data").exists():
        print(f"ProactiveMemBench already prepared at {target}")
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)
    print(f"Extracted ProactiveMemBench to {target}")


def prepare_locomo(benchmark_dir: Path) -> None:
    target_dir = benchmark_dir / "locomo"
    target = target_dir / "locomo10.json"
    if target.exists():
        print(f"LoCoMo already prepared at {target}")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LOCOMO_URL, target)
    print(f"Downloaded LoCoMo to {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--benchmarks", nargs="*", help="Subset to prepare.")
    args = parser.parse_args()
    config = load_config(args.config)
    benchmark_dir = Path(config.evaluation.benchmark_dir)
    benchmarks = args.benchmarks or config.evaluation.benchmarks
    unknown = sorted(set(benchmarks) - {"proactive_membench", "locomo"})
    if unknown:
        raise ValueError(f"Unsupported benchmarks: {', '.join(unknown)}")
    if "proactive_membench" in benchmarks:
        prepare_proactive_membench(benchmark_dir)
    if "locomo" in benchmarks:
        prepare_locomo(benchmark_dir)


if __name__ == "__main__":
    main()
