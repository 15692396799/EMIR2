#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory.config import load_config
from evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--memory-backend", help="Registered memory backend identifier.")
    parser.add_argument("--retrieval-strategy", help="Registered retrieval strategy identifier.")
    parser.add_argument("--benchmarks", nargs="*", help="Benchmarks to run, e.g. proactive_membench locomo.")
    parser.add_argument("--resume-run-dir", help="Resume an existing run directory instead of creating a new one.")
    parser.add_argument("--ollama-units-config", help="Optional YAML defining LoCoMo inference units; does not invalidate checkpoints.")
    parser.add_argument("--fork-run-from", help="Create a derived run from an existing run directory.")
    parser.add_argument(
        "--invalidate-stages", nargs="*",
        choices=("memory", "retrieval", "answer", "temporal_correction", "scoring"),
        default=(),
        help="Invalidate these stages and all downstream stages in a resumed or forked run.",
    )
    parser.add_argument("--locomo-reuse-memory-from")
    parser.add_argument("--locomo-max-questions", type=int)
    parser.add_argument("--locomo-example-indices", nargs="+", type=int,
                        help="Select zero-based LoCoMo example indices, preserving original IDs.")
    parser.add_argument("--locomo-categories", nargs="+", choices=("1", "2", "3", "4"))
    parser.add_argument("--locomo-reuse-predictions-from")
    parser.add_argument(
        "--locomo-rejudge-from",
        help="Reuse LoCoMo answers/traces and run only scoring with the current judge.",
    )
    parser.add_argument("--locomo-regenerate-answers", action="store_true")
    parser.add_argument(
        "--locomo-answer-only-frozen-retrieval",
        action="store_true",
        help="Reuse existing retrieval checkpoints without rerunning retrieval; use with --fork-run-from or --resume-run-dir.",
    )
    parser.add_argument("--locomo-force-rejudge", action="store_true")
    parser.add_argument("--proactive-reuse-memory-from")
    parser.add_argument("--proactive-reuse-predictions-from")
    parser.add_argument("--proactive-regenerate-answers", action="store_true")
    parser.add_argument("--proactive-force-rejudge", action="store_true")
    args = parser.parse_args()
    for conflict in _resume_arg_conflicts(args):
        parser.error(conflict)
    config = load_config(args.config)
    if args.ollama_units_config:
        from evaluation.ollama_units import load_ollama_units
        config.evaluation.locomo["ollama_units"] = load_ollama_units(args.ollama_units_config)["units"]
    if args.memory_backend:
        config.memory.backend = args.memory_backend
    if args.retrieval_strategy:
        config.retrieval["strategy"] = args.retrieval_strategy
    _apply_benchmark_overrides(config.evaluation.locomo, "locomo", args)
    _apply_benchmark_overrides(config.evaluation.proactive_membench, "proactive", args)
    run_evaluation(
        config,
        selected_benchmarks=args.benchmarks,
        resume_run_dir=args.resume_run_dir,
        fork_run_from=args.fork_run_from,
        invalidate_stages=args.invalidate_stages,
        config_input_path=args.config,
    )


def _resume_arg_conflicts(args: argparse.Namespace) -> list[str]:
    if args.resume_run_dir and getattr(args, "fork_run_from", None):
        return ["--resume-run-dir cannot be used with --fork-run-from"]
    return []


def _apply_benchmark_overrides(options: dict, prefix: str, args: argparse.Namespace) -> None:
    reuse_memory_from = getattr(args, f"{prefix}_reuse_memory_from")
    max_questions = getattr(args, f"{prefix}_max_questions", None)
    reuse_predictions_from = getattr(args, f"{prefix}_reuse_predictions_from")
    rejudge_from = getattr(args, f"{prefix}_rejudge_from", None)
    regenerate_answers = getattr(args, f"{prefix}_regenerate_answers")
    force_rejudge = getattr(args, f"{prefix}_force_rejudge")
    if reuse_memory_from:
        options["reuse_memory_from"] = reuse_memory_from
    example_indices = getattr(args, f"{prefix}_example_indices", None)
    if example_indices is not None:
        options["example_indices"] = example_indices
    categories = getattr(args, f"{prefix}_categories", None)
    if categories is not None:
        options["categories"] = categories
    if max_questions is not None:
        options["max_questions"] = max_questions
    if reuse_predictions_from:
        options["reuse_predictions_from"] = reuse_predictions_from
    if rejudge_from:
        options["rejudge_from"] = rejudge_from
    if regenerate_answers:
        options["regenerate_answers"] = True
    if getattr(args, f"{prefix}_answer_only_frozen_retrieval", False):
        options["answer_only_frozen_retrieval"] = True
    if force_rejudge:
        options["force_rejudge"] = True


if __name__ == "__main__":
    main()
