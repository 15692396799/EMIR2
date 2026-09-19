from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class V4OperationContext:
    stage: str
    unit_id: str
    provider: str | None = None
    model: str | None = None
    checkpoint_key: str | None = None


class V4StageError(RuntimeError):
    """Base class for exhausted V4 result-affecting operations."""

    def __init__(
        self,
        context: V4OperationContext,
        attempts: int,
        cause: Exception,
    ) -> None:
        self.stage = context.stage
        self.unit_id = context.unit_id
        self.attempts = int(attempts)
        self.provider = context.provider
        self.model = context.model
        self.checkpoint_key = context.checkpoint_key
        self.cause = cause
        details = [
            f"stage={self.stage}",
            f"unit_id={self.unit_id}",
            f"attempts={self.attempts}",
        ]
        if self.provider:
            details.append(f"provider={self.provider}")
        if self.model:
            details.append(f"model={self.model}")
        if self.checkpoint_key:
            details.append(f"checkpoint_key={self.checkpoint_key}")
        details.append(f"cause={type(cause).__name__}: {cause}")
        super().__init__("V4 operation failed (" + ", ".join(details) + ")")


class V4BuildStageError(V4StageError):
    pass


class V4RetrievalError(V4StageError):
    pass


class V4AnswerError(V4StageError):
    pass


class V4TemporalCorrectionError(V4StageError):
    pass


def is_retryable_external_error(error: Exception) -> bool:
    """Return whether an exception represents a transient external failure."""
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    name = type(error).__name__.casefold()
    module = type(error).__module__.casefold()
    transient_names = (
        "apiconnection",
        "apitimeout",
        "ratelimit",
        "serviceunavailable",
        "temporarilyunavailable",
        "toomanyrequests",
        "httpstatus",
        "httperror",
        "requestexception",
    )
    if any(value in name for value in transient_names):
        status = _status_code(error)
        return status is None or status == 408 or status == 409 or status == 429 or status >= 500
    if "urllib" in module or "requests" in module or "httpx" in module or "openai" in module:
        status = _status_code(error)
        return status is None or status == 408 or status == 409 or status == 429 or status >= 500
    return False


def retry_v4_call(
    operation: Callable[[], T],
    *,
    policy: Any,
    context: V4OperationContext,
    error_type: type[V4StageError],
    sleeper: Callable[[float], None] | None = None,
    on_failure: Callable[[int, Exception], None] | None = None,
    validation_errors: tuple[type[Exception], ...] = (),
    validation_retries: int = 0,
) -> T:
    """Retry external failures and optionally retry invalid output within the budget."""
    max_attempts = max(1, int(policy.max_attempts))
    delay = max(0.0, float(policy.retry_initial_delay_seconds))
    multiplier = max(1.0, float(policy.retry_backoff_multiplier))
    sleep = sleeper or time.sleep
    last_error: Exception | None = None
    validation_failures = 0
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            last_error = error
            if on_failure is not None:
                on_failure(attempt, error)
            if isinstance(error, validation_errors):
                validation_failures += 1
                if validation_failures > validation_retries:
                    raise error_type(context, attempt, error) from error
                retryable = True
            else:
                retryable = is_retryable_external_error(error)
            if not retryable:
                raise
            if attempt == max_attempts:
                break
            sleep(delay)
            delay *= multiplier
    assert last_error is not None
    raise error_type(context, max_attempts, last_error) from last_error


def _status_code(error: Exception) -> int | None:
    for attribute in ("status_code", "status", "code"):
        value = getattr(error, attribute, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    response = getattr(error, "response", None)
    try:
        value = getattr(response, "status_code", None)
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
