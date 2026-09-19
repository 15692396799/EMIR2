"""Embedding-client adapter for cloud providers (point 12 substitution).

The upstream default runs embeddings on a local Ollama server, which accepts a
whole batch of texts in one request. Cloud providers cap that batch:

* Bailian / DashScope ``text-embedding-v3`` and ``v4`` reject more than **10**
  inputs per request with HTTP 400
  (``batch size is invalid, it should not be larger than 10``).

The V4 builder embeds entity aliases in one call, which routinely exceeds 10, so
substituting a Bailian embedding model for the Ollama one breaks the build. The
fix lives here: wrap the provider client and split large batches, instead of
touching the memory-system code.

``MemorySystem`` accepts an ``embedding_client`` argument, so this adapter is
injected rather than patched in.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterable, Sequence


#: Verified per-request input limits. ``None`` means "send the batch as-is".
PROVIDER_BATCH_LIMITS: dict[str, int | None] = {
    "dashscope_bailian": 10,
    "openai": None,
    "vllm_openai_compatible": None,
    "modelscope_openai_compatible": 10,
    "ollama": None,
    "hash": None,
}

#: HTTP statuses worth retrying (rate limits, gateway hiccups, server errors).
TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_SECONDS = 1.5


def _is_transient(error: BaseException) -> bool:
    """True for network/rate-limit failures, False for real request errors.

    A 400 (bad parameter) must never be retried: it will fail identically and
    would only burn time. Chunked-encoding, connection and timeout errors are
    common against cloud endpoints and are worth retrying.
    """
    try:
        import requests
    except ImportError:  # pragma: no cover - requests is a hard dependency
        return False
    if isinstance(error, requests.exceptions.HTTPError):
        status = getattr(getattr(error, "response", None), "status_code", None)
        return status in TRANSIENT_STATUS
    return isinstance(error, requests.exceptions.RequestException)


def embed_batch_limit(provider: str) -> int | None:
    """Batch limit for a provider, overridable with MEMCONFLICT_EMBED_BATCH."""
    override = os.getenv("MEMCONFLICT_EMBED_BATCH")
    if override:
        try:
            value = int(override)
            return value if value > 0 else None
        except ValueError:
            pass
    return PROVIDER_BATCH_LIMITS.get(str(provider or "").lower())


class BatchedEmbeddingClient:
    """Split ``embed_texts`` calls so each request stays under the provider cap.

    Vector order is preserved, so callers cannot tell the batches apart.
    """

    def __init__(
        self,
        inner: Any,
        max_batch: int,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        self._inner = inner
        self.max_batch = max(1, int(max_batch))
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_seconds = float(backoff_seconds)
        self.batch_count = 0
        self.retry_count = 0

    def _embed_chunk(self, chunk: list[str]) -> list[list[float]]:
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                self.batch_count += 1
                return list(self._inner.embed_texts(chunk))
            except BaseException as error:  # noqa: BLE001 - re-raised below
                last_error = error
                if attempt >= self.max_attempts or not _is_transient(error):
                    raise
                self.retry_count += 1
                time.sleep(self.backoff_seconds * attempt)
        raise last_error  # pragma: no cover - loop either returns or raises

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        rows = list(texts)
        if not rows:
            return []
        if len(rows) <= self.max_batch:
            return self._embed_chunk(rows)

        vectors: list[list[float]] = []
        for start in range(0, len(rows), self.max_batch):
            chunk = rows[start : start + self.max_batch]
            if not chunk:
                continue
            vectors.extend(self._embed_chunk(chunk))
        return vectors

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_embedding_client(client: Any, provider: str) -> Any:
    """Apply the provider batch limit to a freshly built embedding client."""
    limit = embed_batch_limit(provider)
    if limit is None or isinstance(client, BatchedEmbeddingClient):
        return client
    return BatchedEmbeddingClient(client, limit)


__all__ = [
    "BatchedEmbeddingClient",
    "PROVIDER_BATCH_LIMITS",
    "embed_batch_limit",
    "wrap_embedding_client",
]
