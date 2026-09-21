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

#: P1-2 concurrency overrides. One persona worker multiplies whatever the
#: config asks for, so the runner exposes these instead of editing YAML per
#: worker count.
EXTRACTION_WORKERS_ENV = "MEMCONFLICT_EXTRACTION_WORKERS"
ENTITY_JUDGE_WORKERS_ENV = "MEMCONFLICT_ENTITY_JUDGE_WORKERS"


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


def build_chat_client(model_config: Any):
    """Build the chat client for a role, honouring the ``azure`` provider.

    Everything else goes through the unmodified Retrival-Mem factory; Azure
    needs its own client because it authenticates with ``api-key`` and puts the
    deployment in the path (see ``azure_client``).

    The result is wrapped in the retrying client: OpenRouter's upstreams flip
    between reachable and geo-filtered (and the local proxy drops connections),
    and both are two-second problems that should not cost a persona's judging
    pass. ``MEMCONFLICT_CHAT_RETRIES=1`` restores the raw client.
    """
    from .retrying import wrap_retrying

    provider = str(getattr(model_config, "provider", "") or "").lower()
    if provider == "azure":
        from .azure_client import build_azure_chat_client

        client = build_azure_chat_client(model_config)
    else:
        client = import_retrival_mem().make_chat_client(model_config)
    return wrap_retrying(client, prefix="MEMCONFLICT_CHAT")


def load_memory_config(config_path: Path | None = None):
    """Load the Retrival-Mem config, reading ``Retrival-Mem/.env`` for secrets.

    Point 16: ``load_config`` loads the ``.env`` file with ``override=True``, so
    a persona worker's Ollama container assignment (``OLLAMA_BASE_URLS`` /
    ``MEMCONFLICT_OLLAMA_UNIT``) has to be re-applied afterwards -- otherwise
    every worker would silently fall back to the one endpoint written in
    ``.env`` no matter how many containers are running.
    """
    from . import ollama_units

    runtime = import_retrival_mem()
    path = Path(config_path) if config_path else default_config_path()
    env_path = default_env_path()
    config = runtime.load_config(path, env_path=env_path)
    ollama_units.apply_active_unit()
    apply_concurrency_overrides(config)
    return config


def apply_concurrency_overrides(config) -> dict[str, int]:
    """Apply the per-worker concurrency limits (P1-2) to a loaded config.

    ``--persona-workers N`` multiplies every internal thread pool by N, so the
    runner passes these limits through the environment instead of requiring a
    hand-edited YAML per worker count. Only the values that are actually set
    are touched.
    """
    applied: dict[str, int] = {}
    memory = getattr(config, "memory", None)
    if memory is None:
        return applied

    def positive(name: str) -> int | None:
        raw = os.getenv(name)
        if raw is None or not str(raw).strip():
            return None
        try:
            value = int(str(raw).strip())
        except ValueError:
            return None
        return value if value > 0 else None

    extraction = positive(EXTRACTION_WORKERS_ENV)
    if extraction is not None:
        memory.memory_extraction_workers = extraction
        applied["memory_extraction_workers"] = extraction

    entity_judge = positive(ENTITY_JUDGE_WORKERS_ENV)
    if entity_judge is not None:
        backends = getattr(memory, "backends", None)
        if isinstance(backends, dict):
            backend = backends.setdefault("v4", {})
            if isinstance(backend, dict):
                backend["entity_judge_workers"] = entity_judge
                applied["entity_judge_workers"] = entity_judge
    return applied


def load_env_file() -> Path:
    """Load the experiment's ``.env`` into ``os.environ`` (same file as above).

    Used by the standalone tools, which need ``OLLAMA_BASE_URLS`` before any
    config exists. ``load_dotenv`` is the upstream helper, so ``.env`` parsing
    stays identical to a real run.
    """
    import_retrival_mem()  # puts Retrival-Mem/src on sys.path
    from memory.config import load_dotenv  # type: ignore[import-not-found]

    env_path = default_env_path()
    if env_path.is_file():
        load_dotenv(env_path, override=True)
    return env_path


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
            continue
        if provider == "azure":
            required.add("AZURE_API_KEY")
            required.add("AZURE_OPENAI_ENDPOINT")
            continue
        # Any other OpenAI-compatible provider (modelscope_openai_compatible,
        # vllm_openai_compatible, ...): trust the env names the config itself
        # declares instead of guessing a prefix, so alternative channels
        # (Qianfan, Zhipu, an internal gateway) preflight correctly.
        for attribute in (
            "api_key_env",
            "chat_completions_endpoint_env",
            "embeddings_endpoint_env",
            "base_url_env",
        ):
            name = getattr(model, attribute, None)
            if name:
                required.add(str(name))
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
