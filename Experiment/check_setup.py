"""Point 11 preflight: prove the configured models and endpoints really work.

Before spending anything on a run, this checks

1. the dataset is readable,
2. every credential the config needs is present,
3. every configured chat model answers a one-token request,
4. the configured embedding model returns a vector.

Each probe is deliberately tiny (a few tokens), so a full check costs well
under a cent.

Usage::

    python Experiment/check_setup.py --config Experiment/configs/smoke_small.yaml
    python Experiment/check_setup.py --config Experiment/configs/eval_large.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from memconflict_eval import runtime  # noqa: E402
from memconflict_eval.data import dataset_summary, load_personas  # noqa: E402


CHAT_ROLES = (
    "memory_builder",
    "answer_model",
    "judge_model",
    "controller",
    "window_planner",
    "entity_judge",
    "adjudication_model",
    "decomposition_gate",
    "slm",
)


@dataclass
class Probe:
    role: str
    provider: str
    model: str
    ok: bool
    detail: str
    seconds: float


def probe_chat(make_chat_client: Any, role: str, model_config: Any) -> Probe:
    started = time.perf_counter()
    try:
        client = make_chat_client(model_config)
        text = client.chat(
            [{"role": "user", "content": "Answer with the single word: pong"}]
        )
        seconds = time.perf_counter() - started
        reply = str(text).strip().replace("\n", " ")[:40]
        return Probe(role, model_config.provider, model_config.model, True, repr(reply), seconds)
    except Exception as error:
        seconds = time.perf_counter() - started
        return Probe(
            role,
            getattr(model_config, "provider", "?"),
            getattr(model_config, "model", "?"),
            False,
            f"{type(error).__name__}: {str(error)[:120]}",
            seconds,
        )


def probe_embedding(make_client: Any, model_config: Any) -> Probe:
    started = time.perf_counter()
    try:
        client = make_client(model_config, resilient=False)
        vector = client.embed_texts(["hello world"])
        seconds = time.perf_counter() - started
        row = vector[0] if isinstance(vector, list) and vector else []
        return Probe(
            "embedding",
            model_config.provider,
            model_config.model,
            True,
            f"dim={len(row)}",
            seconds,
        )
    except Exception as error:
        seconds = time.perf_counter() - started
        return Probe(
            "embedding",
            getattr(model_config, "provider", "?"),
            getattr(model_config, "model", "?"),
            False,
            f"{type(error).__name__}: {str(error)[:120]}",
            seconds,
        )


def main(argv: list[str] | None = None) -> int:
    runtime.ensure_utf8_console()
    parser = argparse.ArgumentParser(description="Verify the experiment configuration.")
    parser.add_argument("--config", type=Path, default=runtime.default_config_path())
    parser.add_argument(
        "--input", type=Path, default=runtime.default_dataset_path(), help="Dataset to check."
    )
    parser.add_argument(
        "--personas",
        type=int,
        default=1,
        help="How many dataset records to read while checking.",
    )
    parser.add_argument("--skip-models", action="store_true", help="Only check config and data.")
    args = parser.parse_args(argv)

    failures: list[str] = []

    print("=" * 78)
    print("environment")
    print("=" * 78)
    print("Retrival-Mem root :", runtime.retrival_mem_root())
    print("MemConflict root  :", runtime.memconflict_root())
    print("config            :", Path(args.config).resolve())
    print("env file          :", runtime.default_env_path())

    client = runtime.import_retrival_mem()
    config = runtime.load_memory_config(args.config)
    print("memory backend    :", config.memory.backend)

    print()
    print("credentials")
    print("-" * 78)
    required = runtime.required_env_names(config, runtime.RUNNER_ROLES)
    missing = runtime.missing_env_names(config, runtime.RUNNER_ROLES)
    for name in required:
        mark = "MISSING" if name in missing else "ok"
        print(f"  {name:<42} {mark}")
    if missing:
        failures.append("missing credentials: " + ", ".join(missing))

    print()
    print("dataset")
    print("-" * 78)
    if not Path(args.input).is_file():
        failures.append(f"dataset not found: {args.input}")
        print("  MISSING:", args.input)
    else:
        personas = load_personas(args.input, end_index=args.personas)
        summary = dataset_summary(personas)
        print(" ", json.dumps(summary, ensure_ascii=False))

    if args.skip_models:
        return 0 if not failures else 1

    print()
    print("model probes")
    print("-" * 78)
    probes: list[Probe] = [probe_embedding(client.make_embedding_client, config.embedding)]

    seen: set[tuple[str, str, float]] = set()
    for role in CHAT_ROLES:
        model_config = getattr(config, role, None)
        if model_config is None:
            continue
        key = (str(model_config.provider), str(model_config.model), float(model_config.temperature))
        if key in seen:
            continue
        seen.add(key)
        probes.append(probe_chat(client.make_chat_client, role, model_config))

    print(f"  {'role':<18} {'provider':<18} {'model':<26} {'result':<10} {'s':>5}")
    for probe in probes:
        status = "ok" if probe.ok else "FAILED"
        print(
            f"  {probe.role:<18} {probe.provider:<18} {probe.model:<26} {status:<10} {probe.seconds:5.1f}"
        )
        if not probe.ok:
            failures.append(f"{probe.role}: {probe.detail}")
        else:
            print(f"     -> {probe.detail}")

    print()
    print("=" * 78)
    if failures:
        print("FAILED")
        for item in failures:
            print("  -", item)
        return 1
    print("ALL CHECKS PASSED - the pipeline is ready to run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
