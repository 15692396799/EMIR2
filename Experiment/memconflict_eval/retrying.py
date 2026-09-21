"""Retry wrapper for the chat clients this experiment builds itself.

OpenRouter geo-gates its OpenAI endpoint from this machine in bursts: the same
request returns 200 and then 403 within seconds, and the 403 arrives *before*
any model runs, so it is a routing artefact rather than a bad request. The
upstream client (kept read-only) treats 403 as permanent and aborts, which
would fail a whole persona over a two-second flap.

The judge is cheap and idempotent, so it is wrapped here: transient HTTP
statuses *and* 403 are retried with a small backoff. The wrapper only ever
sees the judge and answer clients, both of which are constructed by this
experiment rather than inside the memory system.
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping

#: Retryable statuses: the usual transient set plus 403, because here a 403 is
#: how OpenRouter reports "this upstream is geo-filtered right now".
RETRYABLE_STATUS = {403, 408, 409, 425, 429, 500, 502, 503, 504}

DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 2.0


def _status_of(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(error, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def is_retryable(error: BaseException) -> bool:
    try:
        import requests
    except ImportError:  # pragma: no cover - requests is a hard dependency
        return False
    status = _status_of(error)
    if status is not None:
        return status in RETRYABLE_STATUS
    return isinstance(error, requests.exceptions.RequestException)


class RetryingChatClient:
    """Wrap a chat client so transient failures are retried."""

    def __init__(
        self,
        inner: Any,
        *,
        attempts: int = DEFAULT_ATTEMPTS,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        sleeper=time.sleep,
    ) -> None:
        self._inner = inner
        self.attempts = max(1, int(attempts))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self._sleep = sleeper
        self.retry_count = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_schema: Mapping[str, Any] | None = None,
    ) -> str:
        last_error: BaseException | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                return self._inner.chat(
                    messages, json_mode=json_mode, json_schema=json_schema
                )
            except BaseException as error:  # noqa: BLE001 - re-raised below
                last_error = error
                if attempt >= self.attempts or not is_retryable(error):
                    raise
                self.retry_count += 1
                self._sleep(self.backoff_seconds * attempt)
        raise last_error  # pragma: no cover - the loop returns or raises

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_retrying(client: Any, *, prefix: str = "MEMCONFLICT") -> Any:
    """Apply :class:`RetryingChatClient` with env-tunable attempts/backoff."""
    attempts = os.getenv(f"{prefix}_CHAT_RETRIES", "")
    backoff = os.getenv(f"{prefix}_CHAT_RETRY_BACKOFF", "")
    try:
        parsed_attempts = int(attempts) if str(attempts).strip() else DEFAULT_ATTEMPTS
    except ValueError:
        parsed_attempts = DEFAULT_ATTEMPTS
    try:
        parsed_backoff = float(backoff) if str(backoff).strip() else DEFAULT_BACKOFF_SECONDS
    except ValueError:
        parsed_backoff = DEFAULT_BACKOFF_SECONDS
    if parsed_attempts <= 1:
        return client
    return RetryingChatClient(client, attempts=parsed_attempts, backoff_seconds=parsed_backoff)


__all__ = [
    "RETRYABLE_STATUS",
    "RetryingChatClient",
    "is_retryable",
    "wrap_retrying",
]
