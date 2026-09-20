"""Point 16 helper: is every Ollama container in ``OLLAMA_BASE_URLS`` on a GPU?

``nvidia-smi`` is not a valid test: Ollama unloads a model after five idle
minutes, so an empty GPU proves nothing. This tool sends one small request per
container and then reads ``/api/ps``, where ``size_vram > 0`` and a sane
``eval_count / eval_duration`` are the real evidence (the 2026-09-19 archive
notes the failure this catches: a container that had lost its device ran at
0.8 tok/s with ``size_vram`` at zero).

Usage::

    python Experiment/tools/check_ollama_units.py
    python Experiment/tools/check_ollama_units.py --model qwen3.5:latest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memconflict_eval import runtime  # noqa: E402
from memconflict_eval.ollama_units import configured_units  # noqa: E402


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def probe(base_url: str, model: str, timeout: float) -> dict:
    started = time.perf_counter()
    payload = {
        "model": model,
        "prompt": "Reply with the single word ok.",
        "stream": False,
        "options": {"num_ctx": 512, "num_predict": 8},
    }
    generated = _post(base_url + "/api/generate", payload, timeout)
    wall = time.perf_counter() - started
    running = _post(base_url + "/api/ps", {}, timeout)
    loaded = next(
        (
            entry
            for entry in running.get("models", [])
            if str(entry.get("name", "")).split(":")[0] == model.split(":")[0]
        ),
        {},
    )
    eval_count = float(generated.get("eval_count") or 0)
    eval_seconds = float(generated.get("eval_duration") or 0) / 1e9
    return {
        "base_url": base_url,
        "wall_s": round(wall, 2),
        "eval_count": int(eval_count),
        "tokens_per_second": round(eval_count / eval_seconds, 1) if eval_seconds else 0.0,
        "size_vram_gb": round(float(loaded.get("size_vram") or 0) / 1e9, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.5:latest")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    runtime.ensure_utf8_console()
    runtime.load_env_file()

    units = configured_units()
    if not units:
        print("[error] no Ollama units configured (set OLLAMA_BASE_URLS)", file=sys.stderr)
        return 2

    failures = 0
    for base_url in units:
        try:
            row = probe(base_url, args.model, args.timeout)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            failures += 1
            print(f"[fail] {base_url}: {type(error).__name__}: {error}")
            continue
        verdict = "gpu" if row["size_vram_gb"] > 0 else "CPU? (size_vram=0)"
        if row["size_vram_gb"] <= 0:
            failures += 1
        print(
            f"[{'ok  ' if row['size_vram_gb'] > 0 else 'warn'}] {row['base_url']} "
            f"vram={row['size_vram_gb']}GB {row['tokens_per_second']}tok/s "
            f"wall={row['wall_s']}s -> {verdict}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
