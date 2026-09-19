"""Wire the experiment to the unmodified Retrival-Mem and MemConflict checkouts.

Both upstream repositories are treated as read-only dependencies:

* ``Retrival-Mem/`` supplies the memory system (V4 backend) and its chat clients.
  We import it by putting ``Retrival-Mem/src`` on ``sys.path``; no file in that
  checkout is ever written to.
* ``MemConflict/`` supplies the benchmark data (``Data/Step4_4.jsonl``) and is
  read as data only. Its evaluation scripts are deliberately *not* imported, so
  the experiment stays free to define its own prompts and metrics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace


# Experiment/ -> EMIR2/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RETRIVAL_MEM_ROOT = PROJECT_ROOT / "Retrival-Mem"
DEFAULT_MEMCONFLICT_ROOT = PROJECT_ROOT / "MemConflict"

MEMORY_SYSTEM_NAME = "retrival_mem_v4"


class DependencyError(RuntimeError):
    """Raised when an upstream checkout or credential is missing."""


def retrival_mem_root() -> Path:
    override = os.getenv("RETRIVAL_MEM_ROOT")
    return Path(override).expanduser().resolve() if override else DEFAULT_RETRIVAL_MEM_ROOT


def memconflict_root() -> Path:
    override = os.getenv("MEMCONFLICT_ROOT")
    return Path(override).expanduser().resolve() if override else DEFAULT_MEMCONFLICT_ROOT


def default_dataset_path() -> Path:
    return memconflict_root() / "Data" / "Step4_4.jsonl"


def default_config_path() -> Path:
    return retrival_mem_root() / "configs" / "default.yaml"


def default_env_path() -> Path:
    """Where this experiment reads credentials from.

    Preference order keeps the experiment self-contained: ``Experiment/.env``
    first, then the memory checkout's own ``.env`` if the operator already had
    one. Neither location is inside the tracked source of the upstream repos.
    """
    local = Path(__file__).resolve().parents[1] / ".env"
    if local.is_file():
        return local
    return retrival_mem_root() / ".env"


def import_retrival_mem(root: Path | None = None):
    """Import the memory-system API from the Retrival-Mem checkout.

    Returns a namespace with the pieces the experiment needs. The checkout's
    ``src`` directory is prepended to ``sys.path`` so the import works without
    installing the package.
    """
    root = Path(root) if root is not None else retrival_mem_root()
    source_dir = root / "src"
    if not source_dir.is_dir():
        raise DependencyError(
            f"Retrival-Mem src directory not found at {source_dir}. "
            "Set RETRIVAL_MEM_ROOT to the checkout location."
        )
    source = str(source_dir)
    if source not in sys.path:
        sys.path.insert(0, source)

    from api_history import ApiHistoryLogger  # type: ignore[import-not-found]
    from memory import MemorySystem  # type: ignore[import-not-found]
    from memory.clients import (  # type: ignore[import-not-found]
        make_chat_client,
        make_embedding_client,
    )
    from memory.config import (  # type: ignore[import-not-found]
        configure_backend_output_paths,
        load_config,
    )

    return SimpleNamespace(
        MemorySystem=MemorySystem,
        make_chat_client=make_chat_client,
        make_embedding_client=make_embedding_client,
        load_config=load_config,
        configure_backend_output_paths=configure_backend_output_paths,
        ApiHistoryLogger=ApiHistoryLogger,
        root=root,
    )


def load_memory_config(config_path: Path | None = None):
    """Load the Retrival-Mem config, reading ``Retrival-Mem/.env`` for secrets."""
    runtime = import_retrival_mem()
    path = Path(config_path) if config_path else default_config_path()
    env_path = default_env_path()
    return runtime.load_config(path, env_path=env_path)


# ---------------------------------------------------------------------------
# credential preflight
# ---------------------------------------------------------------------------

_PROVIDER_PREFIX = {
    "openai": "OPENAI",
    "dashscope_bailian": "DASHSCOPE",
    "openrouter": "OPENROUTER",
    "greatrouter": "GREATROUTER",
}

_EXTRA_MODEL_ROLES = (
    "controller",
    "window_planner",
    "entity_judge",
    "adjudication_model",
    "slm",
    "decomposition_gate",
)

#: Roles that the runner needs (memory building, retrieval and answering).
RUNNER_ROLES = ("embedding", "memory_builder", "answer_model") + _EXTRA_MODEL_ROLES

#: Roles that scoring needs (the judge only).
SCORING_ROLES = ("judge_model",)


def required_env_names(config, roles: tuple[str, ...] = RUNNER_ROLES) -> list[str]:
    """Environment variables the configured models need, derived from the config."""
    required: set[str] = set()
    for role in roles:
        model = getattr(config, role, None)
        if model is None:
            continue
        provider = str(getattr(model, "provider", "") or "").lower()
        if not provider:
            continue
        if provider == "ollama":
            if role == "embedding":
                required.update(
                    {"OLLAMA_EMBED_ENDPOINT", "OLLAMA_LEGACY_EMBEDDINGS_ENDPOINT"}
                )
            else:
                required.add("OLLAMA_CHAT_ENDPOINT")
            continue
        prefix = _PROVIDER_PREFIX.get(provider)
        if prefix:
            required.add(f"{prefix}_API_KEY")
            if role == "embedding":
                required.add(f"{prefix}_EMBEDDINGS_ENDPOINT")
            else:
                required.add(f"{prefix}_CHAT_COMPLETIONS_ENDPOINT")
    return sorted(required)


def missing_env_names(config, roles: tuple[str, ...] = RUNNER_ROLES) -> list[str]:
    """Subset of :func:`required_env_names` that is unset or empty."""
    return [name for name in required_env_names(config, roles) if not os.getenv(name)]


def credential_hint() -> str:
    return (
        "Copy Experiment/.env.example to Experiment/.env and fill in the missing "
        "values. The memory checkout is left untouched; the experiment loads "
        "Experiment/.env first and only falls back to Retrival-Mem/.env."
    )


def ensure_utf8_console() -> None:
    """Keep the CLIs from dying on a non-UTF-8 console (e.g. Windows GBK).

    The report files are always written as UTF-8; this only affects stdout, and
    degrades to replacement characters instead of raising UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
