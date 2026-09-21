"""Check that the model roles of a config really answer (proxy / geo sanity).

OpenRouter's behaviour on this machine is host-routing dependent: openrouter.ai
has to go through the local proxy (OpenAI's models answer 403 "This model is not
available in your region" when dialled from a CN IP), while the GPU server and
DeepSeek are dialled directly. Run this before a long job instead of discovering
it inside `errors.jsonl` hours later.

Usage::

    python Experiment/tools/check_channels.py
    python Experiment/tools/check_channels.py --roles memory_builder,answer_model,judge_model
    python Experiment/tools/check_channels.py --config Experiment/configs/eval_large_dsjudge.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memconflict_eval import runtime  # noqa: E402

DEFAULT_ROLES = ("memory_builder", "answer_model", "judge_model")


def _check_chat(role: str, model_config, prompt: str) -> tuple[bool, str]:
    client = runtime.build_chat_client(model_config)
    started = time.perf_counter()
    text = client.chat([{"role": "user", "content": prompt}])
    return True, "%.1fs -> %s" % (time.perf_counter() - started, str(text).strip()[:50])


def _check_embedding(model_config) -> tuple[bool, str]:
    client = runtime.import_retrival_mem().make_embedding_client(model_config)
    started = time.perf_counter()
    vectors = client.embed_texts(["channel check"])
    return True, "%.1fs -> %d vector(s), dim %d" % (
        time.perf_counter() - started,
        len(vectors),
        len(vectors[0]) if vectors else 0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None, help="Config to read the roles from."
    )
    parser.add_argument(
        "--roles",
        default=",".join(DEFAULT_ROLES),
        help="Comma separated model roles (default: %s)." % ",".join(DEFAULT_ROLES),
    )
    parser.add_argument("--prompt", default="reply with the single word ok")
    args = parser.parse_args(argv)
    runtime.ensure_utf8_console()

    runtime.load_env_file()
    config = runtime.load_memory_config(args.config)
    roles = [role.strip() for role in str(args.roles).split(",") if role.strip()]

    failures = 0
    for role in roles:
        model_config = getattr(config, role, None)
        if model_config is None:
            print("[skip] %-18s (role not in this config)" % role)
            continue
        provider = str(getattr(model_config, "provider", "") or "?")
        model = str(getattr(model_config, "model", "") or "?")
        try:
            if role == "embedding":
                _ok, detail = _check_embedding(model_config)
            else:
                _ok, detail = _check_chat(role, model_config, args.prompt)
            print("[ok  ] %-18s %-24s %-26s %s" % (role, provider, model, detail))
        except Exception as error:  # noqa: BLE001 - reported, not raised
            failures += 1
            print(
                "[fail] %-18s %-24s %-26s %s: %s"
                % (role, provider, model, type(error).__name__, str(error)[:140])
            )

    if failures:
        print()
        print("hint: openrouter.ai must stay OUT of NO_PROXY (the local proxy is what")
        print("      makes OpenAI's models reachable from this network), and the GPU")
        print("      server / DeepSeek should be dialled directly.")
        print("      If OpenRouter is unusable, score with")
        print("      --config Experiment/configs/eval_large_dsjudge.yaml (DeepSeek direct).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
