"""Point 16: spread the Ollama roles over several containers / GPUs.

The GPU server runs one Ollama container per GPU (``device4`` ... ``device7`` at
the time of writing), each publishing its own host port. A single experiment
process can only talk to one of them, so the experiment keeps two knobs:

* ``OLLAMA_BASE_URLS`` -- the comma-separated container list, e.g.
  ``http://172.26.94.12:41133,http://172.26.94.12:41134,...``;
* ``MEMCONFLICT_OLLAMA_UNIT`` -- the one container *this* process owns. The
  persona workers in ``run_experiment.py`` set it before they touch the memory
  system, which is what turns four containers into four lanes.

Why an environment variable instead of Retrival-Mem's ``ollama_routes`` context
variable: the V4 builder extracts windows inside a ``ThreadPoolExecutor``, and a
``ContextVar`` is *not* inherited by new threads, so a route set in the
submitting thread would silently fall back to the single ``.env`` endpoint for
every extraction call. ``make_chat_client`` / ``make_embedding_client`` read
these environment variables each time a client is built, and a child process
never shares them with its siblings, so this is the lever that reaches the
worker threads and stays local to one lane.

``Retrival-Mem``'s ``load_config(..., env_path=...)`` calls
``load_dotenv(override=True)``, which overwrites ``os.environ`` from
``Experiment/.env``. The unit assignment is therefore re-applied after every
config load by ``runtime.load_memory_config``.
"""

from __future__ import annotations

import json
import os
import urllib.request
from urllib.parse import urlsplit

#: Comma/space separated list of Ollama containers available to a run.
UNITS_ENV = "OLLAMA_BASE_URLS"

#: The container one process owns. Set by the persona workers, read by
#: ``apply_active_unit`` after each config load.
ACTIVE_UNIT_ENV = "MEMCONFLICT_OLLAMA_UNIT"

#: Endpoint variables the unmodified Retrival-Mem Ollama clients read.
ENDPOINT_ENV_KEYS = (
    "OLLAMA_BASE_URL",
    "OLLAMA_CHAT_ENDPOINT",
    "OLLAMA_EMBED_ENDPOINT",
    "OLLAMA_LEGACY_EMBEDDINGS_ENDPOINT",
)


def normalize_base_url(raw: object) -> str:
    """Validate one container base URL and strip the trailing slash."""
    base = str(raw or "").strip().rstrip("/")
    if not base:
        raise ValueError("Ollama unit base URL must not be empty")
    parts = urlsplit(base)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.path
        or parts.query
        or parts.fragment
        or parts.username
        or parts.password
    ):
        raise ValueError(
            f"Ollama unit {base!r} must be an HTTP(S) base URL without "
            "credentials or /api paths"
        )
    return base


def parse_base_urls(raw: object) -> list[str]:
    """Split a unit list, keeping the order and dropping duplicates."""
    units: list[str] = []
    seen: set[str] = set()
    for chunk in str(raw or "").replace(",", " ").replace(";", " ").split():
        base = normalize_base_url(chunk)
        if base not in seen:
            seen.add(base)
            units.append(base)
    return units


def configured_units() -> list[str]:
    """Every container this run may use.

    ``OLLAMA_BASE_URLS`` is the multi-container knob (point 16); a single
    ``OLLAMA_BASE_URL`` keeps the original one-container behaviour, and an empty
    result means "leave the endpoints exactly as the config/.env set them".
    """
    raw = os.getenv(UNITS_ENV)
    if raw and raw.strip():
        return parse_base_urls(raw)
    single = os.getenv("OLLAMA_BASE_URL", "")
    return [normalize_base_url(single)] if single.strip() else []


def active_unit() -> str | None:
    """The container this process owns, if any."""
    raw = os.getenv(ACTIVE_UNIT_ENV)
    return normalize_base_url(raw) if raw and raw.strip() else None


def routes_for(base_url: str) -> dict[str, str]:
    """The three Ollama routes Retrival-Mem asks for."""
    base = normalize_base_url(base_url)
    return {
        "chat": base + "/api/chat",
        "embed": base + "/api/embed",
        "legacy_embed": base + "/api/embeddings",
    }


def apply_unit(base_url: str) -> dict[str, str]:
    """Point every Ollama endpoint variable at ``base_url``."""
    base = normalize_base_url(base_url)
    routes = routes_for(base)
    values = {
        "OLLAMA_BASE_URL": base,
        "OLLAMA_CHAT_ENDPOINT": routes["chat"],
        "OLLAMA_EMBED_ENDPOINT": routes["embed"],
        "OLLAMA_LEGACY_EMBEDDINGS_ENDPOINT": routes["legacy_embed"],
    }
    for key, value in values.items():
        os.environ[key] = value
    os.environ[ACTIVE_UNIT_ENV] = base
    return values


def clear_unit() -> None:
    """Undo :func:`apply_unit` (used when a process owns no unit)."""
    os.environ.pop(ACTIVE_UNIT_ENV, None)


def apply_active_unit() -> dict[str, str] | None:
    """Re-apply this process's unit after ``load_dotenv(override=True)``."""
    unit = active_unit()
    return apply_unit(unit) if unit else None


def assign_units(units: list[str], count: int) -> list[str | None]:
    """Round-robin ``count`` personas over ``units``; ``None`` means default."""
    if not units:
        return [None] * count
    return [units[index % len(units)] for index in range(count)]


def probe_unit(base_url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Ask one container for its model list; ``GET /api/tags`` is the cheap check.

    Deliberately not ``nvidia-smi`` and not ``/api/ps``: a healthy but idle
    container unloads its model, and a container that lost its GPU still
    answers -- this probe only answers "is this lane up at all".
    """
    base = normalize_base_url(base_url)
    request = urllib.request.Request(base + "/api/tags", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=max(0.5, float(timeout))) as response:
            payload = json.load(response)
    except Exception as error:  # noqa: BLE001 - reported to the caller
        return False, f"{type(error).__name__}: {error}"
    models = payload.get("models")
    return True, f"{len(models) if isinstance(models, list) else 0} model(s)"


def split_reachable(
    units: list[str], timeout: float = 5.0
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split a lane list into ``(reachable, [(unreachable, reason), ...])``.

    Point 16 asks for many containers; a container that is down (or still
    starting) takes its personas with it, so the caller checks the lanes before
    the run instead of discovering it hours later.
    """
    reachable: list[str] = []
    unreachable: list[tuple[str, str]] = []
    for base in units:
        try:
            ok, detail = probe_unit(base, timeout)
        except ValueError as error:  # malformed URL, reported like a dead lane
            ok, detail = False, str(error)
        if ok:
            reachable.append(base)
        else:
            unreachable.append((base, detail))
    return reachable, unreachable


__all__ = [
    "ACTIVE_UNIT_ENV",
    "ENDPOINT_ENV_KEYS",
    "UNITS_ENV",
    "active_unit",
    "apply_active_unit",
    "apply_unit",
    "assign_units",
    "clear_unit",
    "configured_units",
    "normalize_base_url",
    "parse_base_urls",
    "probe_unit",
    "routes_for",
    "split_reachable",
]
