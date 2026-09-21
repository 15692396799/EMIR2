"""Harness-side repair for upstream's "never think" shape on OpenRouter.

``Retrival-Mem/src/memory/clients.py::disable_request_thinking`` rewrites every
OpenRouter request to

    "reasoning": {"enabled": false, "effort": "none"}

Point 33 moved the answering and judging roles to ``openai/gpt-5-mini``, and
that endpoint refuses to run without reasoning:

    400 {"error": {"message": "Reasoning is mandatory for this endpoint and
         cannot be disabled."}}

Measured against the live lane on 2026-09-21 with the exact body the client
builds: the upstream shape is a 400, while dropping the field or asking for
``effort=minimal`` / ``effort=low`` all answer 200. A 400 is not transient, so
neither ``retry_v4_call`` nor ``wrap_retrying`` retries it, and every question of
both shard runs recorded ``Answer_Error`` instead of an answer (9/9 and 12/12);
the judge would fail the same way.

The patch keeps upstream's intent -- do not spend the output budget on thinking
-- in a shape the endpoint accepts: for a reasoning-mandatory model on
OpenRouter it sends ``reasoning: {"effort": <effort>}`` with the effort from
``MEMCONFLICT_REASONING_EFFORT`` (default ``minimal``; ``off`` drops the field
and takes the provider default). Every other provider and model keeps the
upstream behaviour exactly; ``MEMCONFLICT_REASONING_GUARD=0`` switches the patch
off.
"""

from __future__ import annotations

import os
import sys
from typing import Any

GUARD_ENV = "MEMCONFLICT_REASONING_GUARD"
EFFORT_ENV = "MEMCONFLICT_REASONING_EFFORT"
DEFAULT_EFFORT = "minimal"
_FALSE_VALUES = {"0", "false", "no", "off"}
_DROP_VALUES = {"", "default", "drop", "off", "none"}
#: Model families whose endpoint rejects ``reasoning.enabled = false``.
_MANDATORY_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_INSTALLED_FLAG = "_memconflict_reasoning_guard"


def guard_enabled() -> bool:
    """The patch is on unless ``MEMCONFLICT_REASONING_GUARD=0``."""

    raw = os.getenv(GUARD_ENV)
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in _FALSE_VALUES


def reasoning_effort() -> str:
    """Effort written for a reasoning-mandatory model; "" drops the field."""

    raw = os.getenv(EFFORT_ENV)
    value = str(raw).strip().lower() if raw is not None and str(raw).strip() else DEFAULT_EFFORT
    return "" if value in _DROP_VALUES else value


def requires_reasoning(model: Any) -> bool:
    """True for the model families that refuse ``reasoning.enabled = false``."""

    name = str(model or "").strip().lower().split("/")[-1]
    return name.startswith(_MANDATORY_PREFIXES)


def rewrite_reasoning(payload: dict[str, Any], provider: Any) -> bool:
    """Replace upstream's illegal disable-reasoning shape; True when rewritten."""

    if str(provider or "").strip().lower() != "openrouter":
        return False
    if not requires_reasoning(payload.get("model")):
        return False
    payload.pop("reasoning", None)
    effort = reasoning_effort()
    if effort:
        payload["reasoning"] = {"effort": effort}
    return True


def install_reasoning_guard() -> bool:
    """Wrap ``memory.clients.disable_request_thinking`` once per process."""

    if not guard_enabled():
        return False

    from memory import clients  # type: ignore[import-not-found]

    if getattr(clients, _INSTALLED_FLAG, False):
        return True

    original = clients.disable_request_thinking

    def disable_request_thinking(payload: dict[str, Any], provider: str) -> None:
        original(payload, provider)
        if guard_enabled():
            rewrite_reasoning(payload, provider)

    clients.disable_request_thinking = disable_request_thinking
    setattr(clients, _INSTALLED_FLAG, True)
    print(
        "[guard] OpenRouter reasoning-mandatory models ask for reasoning.effort="
        f"{reasoning_effort() or '(provider default)'}"
        f" ({GUARD_ENV}=0 restores upstream behaviour)",
        file=sys.stderr,
    )
    return True
