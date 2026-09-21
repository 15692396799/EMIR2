"""Memory adapter for the unmodified Retrival-Mem V4 backend (point 7).

LoCoMo usage in ``Retrival-Mem/src/evaluation/runner.py`` ingests a whole
example once and then answers every question. MemConflict needs the opposite
schedule: ingest session 0, answer session 0's questions, ingest session 1,
answer session 1's questions, and so on, so a question can never see a future
session.

The V4 builder merges each artifact with the scope's existing nodes, so
ingesting one session at a time accumulates properly. Personas stay isolated by
giving each one its own database/FAISS directory and namespace.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import runtime
from .data import Persona, Session
from .embedding import wrap_embedding_client


#: V4 keeps a durable build cache in ``v4_build_checkpoints`` and replays a
#: ``status='succeeded'`` row whenever the same unit (``checkpoint_key``) is
#: built again. The loader matches on the key and the status only, so an output
#: that was validated against an older semantic state is replayed as-is; a
#: ``reinforce <fact_key>`` operation then references a fact key that no longer
#: exists (``Memory/BUILD`` -> ``builder.py:2023-2031`` -> ``semantic.py:422``)
#: and the persona dies. The switches below let the harness drop those entries
#: instead of replaying them, and re-sample the answer that failed.
CHECKPOINT_GUARD_ENV = "MEMCONFLICT_CHECKPOINT_GUARD"
DEFAULT_CHECKPOINT_STAGE = "semantic_update"
#: How many times an ingest that fails with one of the messages below is retried
#: after invalidating this namespace's checkpoints. One retry was not enough for
#: the 2026-09-21 runs: the semantic reducer and the cross-window adjudicator
#: both reject a *sampled* answer (a trigger ref outside ``local_event_refs``, a
#: candidate pair answered twice or left out), so a retry is a fresh draw and
#: often the only thing between a flaky answer and a dead persona. Every retry
#: recomputes the session's stages, so the budget is deliberately small and
#: configurable: ``MEMCONFLICT_CHECKPOINT_RETRIES=1`` restores the old single
#: retry, ``=0`` turns retrying off (the guard then only invalidates).
CHECKPOINT_RETRIES_ENV = "MEMCONFLICT_CHECKPOINT_RETRIES"
DEFAULT_CHECKPOINT_RETRIES = 3
MAX_CHECKPOINT_RETRIES = 10
_STALE_CHECKPOINT_MARKERS = (
    "references unknown fact key",
    "references evidence outside supplied local event refs",
    "resolves to an existing fact key",
)
_STAGE_PATTERN = re.compile(r"stage=([A-Za-z_]+)")


def configured_checkpoint_retries() -> int:
    """Retries after a stale-cache validation failure (default 3, clamped 0..10)."""

    raw = os.getenv(CHECKPOINT_RETRIES_ENV)
    if raw is None or not str(raw).strip():
        return DEFAULT_CHECKPOINT_RETRIES
    try:
        value = int(str(raw).strip())
    except ValueError:
        return DEFAULT_CHECKPOINT_RETRIES
    return max(0, min(value, MAX_CHECKPOINT_RETRIES))


@dataclass(frozen=True)
class RetrievedMemory:
    """One retrieved memory, in the shape the prompts and metrics expect."""

    rank: int
    memory: str
    created_at: str
    score: float | None
    node_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "memory": self.memory,
            "created_at": self.created_at,
            "score": self.score,
            "node_type": self.node_type,
        }


@dataclass
class IngestReport:
    session_id: int
    message_count: int
    duration_ms: float
    namespace_ready: bool | None = None
    skipped: bool = False
    stale_checkpoints_dropped: int = 0
    retried_after_validation_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "Session_ID": self.session_id,
            "Message_Count": self.message_count,
            "Duration_ms": self.duration_ms,
            "Namespace_Ready": self.namespace_ready,
            "Skipped": self.skipped,
            "Stale_Checkpoints_Dropped": self.stale_checkpoints_dropped,
            "Retried_After_Validation_Error": self.retried_after_validation_error,
        }


@dataclass
class RetrievalReport:
    memories: list[RetrievedMemory] = field(default_factory=list)
    duration_ms: float = 0.0
    round_count: int = 0

    def top(self, k: int) -> list[RetrievedMemory]:
        return self.memories[:k]


def _memory_text(item: Any) -> str:
    title = str(getattr(item, "title", "") or "").strip()
    body = str(getattr(item, "text", "") or getattr(item, "summary", "") or "").strip()
    parts: list[str] = []
    if title and title != body:
        parts.append(title)
    if body:
        parts.append(body)

    semantic_memories = getattr(item, "semantic_memories", None) or []
    semantic = "; ".join(
        " ".join(
            str(value).strip()
            for value in (
                getattr(memory, "subject", ""),
                getattr(memory, "predicate", ""),
                getattr(memory, "object", ""),
            )
            if str(value or "").strip()
        )
        for memory in semantic_memories
    )
    if semantic:
        parts.append(f"semantic: {semantic}")
    return " | ".join(parts) or "(empty memory)"


def _is_stale_checkpoint_error(error: BaseException) -> bool:
    """True for the validation failures a stale build cache can produce.

    V4 raises ``SemanticValidationError`` (sometimes wrapped in
    ``V4BuildStageError``) when a reducer output does not fit the current
    semantic state; those are exactly the cases where recomputing the unit
    instead of replaying the cached output fixes the run.
    """
    name = type(error).__name__
    if name not in {"SemanticValidationError", "V4BuildStageError", "V4StageError"}:
        return False
    message = str(error)
    return any(marker in message for marker in _STALE_CHECKPOINT_MARKERS)


def _stage_from_error(error: BaseException) -> str | None:
    """The ``stage=...`` named in a V4 error message, if it carries one."""
    match = _STAGE_PATTERN.search(str(error))
    return match.group(1) if match else None


def _memory_created_at(item: Any) -> str:
    metadata = getattr(item, "metadata", None) or {}
    for candidate in (
        metadata.get("absolute_time_start"),
        metadata.get("observed_at"),
        getattr(item, "timestamp_start", None),
    ):
        if candidate not in (None, ""):
            return str(candidate)
    return "Unknown Time"


def persona_slug(persona_id: str) -> str:
    """Filesystem- and namespace-safe identity for one persona.

    The full identifier is kept rather than a suffix: two personas colliding on
    a short suffix would silently share one memory store, which breaks the
    requirement that personas stay independent.
    """
    cleaned = "".join(
        char if (char.isalnum() or char in "-_") else "_" for char in str(persona_id or "")
    ).strip("_")
    return cleaned[:64] or "unknown_persona"


class MemConflictMemory:
    """One persona's memory: an isolated V4 store driven session by session."""

    def __init__(
        self,
        *,
        store_dir: str | Path,
        persona: Persona,
        version: str = "v1",
        config_path: str | Path | None = None,
        reset: bool = True,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.persona = persona
        self.version = version
        self.namespace = self.build_namespace(persona, version)
        self._closed = False

        if reset and self.store_dir.exists():
            shutil.rmtree(self.store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        client = runtime.import_retrival_mem()
        config = runtime.load_memory_config(config_path)
        client.configure_backend_output_paths(
            config,
            str(self.store_dir / "memory.sqlite3"),
            str(self.store_dir / "faiss"),
        )
        self.config = config
        embedding_client = wrap_embedding_client(
            client.make_embedding_client(config.embedding, resilient=False),
            config.embedding.provider,
        )
        self.system = client.MemorySystem(
            config,
            embedding_client=embedding_client,
            api_history_logger=client.ApiHistoryLogger(str(self.store_dir)),
        )
        self.embedding_client = embedding_client

    @staticmethod
    def build_namespace(persona: Persona, version: str) -> str:
        return f"memconflict:{persona_slug(persona.persona_id)}:{version}"

    # -- point 7: build memory one session at a time -----------------------

    def ingest_session(self, session: Session) -> IngestReport:
        if not session.dialogue:
            return IngestReport(session.session_id, 0, 0.0, None, skipped=True)

        dropped = self.drop_stale_checkpoints()
        # One attempt plus MEMCONFLICT_CHECKPOINT_RETRIES retries. The failing
        # answer is sampled again on every retry, which is the only repair these
        # validation errors have; the cache is invalidated first so the retry
        # recomputes the stage instead of replaying the output that just failed.
        attempts = 1
        budget = configured_checkpoint_retries() + 1
        while True:
            try:
                return self._ingest_once(
                    session, dropped=dropped, retried=attempts > 1
                )
            except Exception as error:  # noqa: BLE001 - re-raised unless the guard applies
                if not self.checkpoint_guard_enabled() or not _is_stale_checkpoint_error(error):
                    raise
                invalidated = self.invalidate_checkpoints(
                    stage=_stage_from_error(error) or DEFAULT_CHECKPOINT_STAGE,
                    reason=f"{type(error).__name__}: {error}",
                )
                if invalidated <= 0:
                    # The failing stage carried no cache entry; fall back to
                    # every succeeded checkpoint of this namespace, so a resumed
                    # run recomputes instead of replaying the stale output.
                    invalidated = self.invalidate_checkpoints(
                        stage=None, reason=f"{type(error).__name__}: {error}"
                    )
                dropped += invalidated
                if attempts >= budget:
                    raise
                attempts += 1
                print(
                    f"[warn] {type(error).__name__} while ingesting session "
                    f"{session.session_id} of {self.namespace}: invalidated "
                    f"{invalidated} cached checkpoint(s) and retrying "
                    f"(attempt {attempts}/{budget})",
                    file=sys.stderr,
                )

    def _ingest_once(
        self,
        session: Session,
        *,
        dropped: int = 0,
        retried: bool = False,
    ) -> IngestReport:
        start = time.perf_counter()
        self.system.ingest_conversation(
            self.namespace,
            [session.to_memory_session()],
            {
                "dataset": "memconflict",
                "persona_id": self.persona.persona_id,
                "session_id": session.session_id,
                "session_date": session.date,
                "session_type": session.session_type,
            },
        )
        duration_ms = (time.perf_counter() - start) * 1000.0
        try:
            ready: bool | None = bool(self.system.is_namespace_ready(self.namespace))
        except Exception:  # readiness is informational only
            ready = None
        return IngestReport(
            session_id=session.session_id,
            message_count=len(session.dialogue),
            duration_ms=duration_ms,
            namespace_ready=ready,
            stale_checkpoints_dropped=dropped,
            retried_after_validation_error=retried,
        )

    # -- stale build-cache guard ------------------------------------------

    @staticmethod
    def checkpoint_guard_enabled() -> bool:
        """The guard is on unless ``MEMCONFLICT_CHECKPOINT_GUARD=0``."""
        raw = os.getenv(CHECKPOINT_GUARD_ENV)
        if raw is None or not str(raw).strip():
            return True
        return str(raw).strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def checkpoint_retries() -> int:
        """Retries after a stale-cache failure (``MEMCONFLICT_CHECKPOINT_RETRIES``)."""
        return configured_checkpoint_retries()

    def drop_stale_checkpoints(self, *, reason: str = "stale_scope_revision") -> int:
        """Invalidate cached outputs recorded against an older scope revision.

        Called before every session ingest. Within one build the scope revision
        does not move, so a legitimate checkpoint (written by this session's own
        earlier attempt) survives; entries from earlier sessions or from a
        previous run cannot be replayed against a state that has since changed.
        """
        if not self.checkpoint_guard_enabled():
            return 0
        return self._invalidate_checkpoints(
            stale_only=True, stage=None, reason=reason
        )

    def invalidate_checkpoints(self, *, stage: str | None, reason: str) -> int:
        """Mark this namespace's succeeded build checkpoints as failed."""
        return self._invalidate_checkpoints(
            stale_only=False, stage=stage, reason=reason
        )

    def _invalidate_checkpoints(
        self, *, stale_only: bool, stage: str | None, reason: str
    ) -> int:
        db_path = self.store_dir / "memory.sqlite3"
        if not db_path.is_file():
            return 0
        payload = json.dumps(
            {"type": "harness_invalidation", "message": str(reason)},
            ensure_ascii=False,
        )
        sql = [
            "UPDATE v4_build_checkpoints",
            "SET status='failed', error_json=?",
            "WHERE namespace=? AND status='succeeded'",
        ]
        params: list[Any] = [payload, self.namespace]
        if stage:
            sql.append("AND stage=?")
            params.append(str(stage))
        if stale_only:
            sql.append(
                "AND EXISTS (SELECT 1 FROM v4_participant_scopes s"
                " WHERE s.id = v4_build_checkpoints.scope_id"
                " AND v4_build_checkpoints.scope_revision < s.revision)"
            )
        try:
            connection = sqlite3.connect(str(db_path), timeout=10.0)
        except sqlite3.Error:
            return 0
        try:
            connection.execute("PRAGMA busy_timeout=10000")
            cursor = connection.execute(" ".join(sql), params)
            count = int(cursor.rowcount) if cursor.rowcount and cursor.rowcount > 0 else 0
            connection.commit()
            return count
        except sqlite3.Error:
            connection.rollback()
            return 0
        finally:
            connection.close()

    # -- retrieval ---------------------------------------------------------

    def retrieve(self, question: str, *, keep: int | None = None) -> RetrievalReport:
        start = time.perf_counter()
        result = self.system.retrieve(question, self.namespace)
        duration_ms = (time.perf_counter() - start) * 1000.0

        items = list(getattr(result, "episodic_memories", []) or [])
        memories = [
            RetrievedMemory(
                rank=index,
                memory=_memory_text(item),
                created_at=_memory_created_at(item),
                score=getattr(item, "score", None),
                node_type=str(getattr(item, "node_type", "") or ""),
            )
            for index, item in enumerate(items, start=1)
        ]
        if keep is not None:
            memories = memories[:keep]
        trace = list(getattr(result, "trace", []) or [])
        return RetrievalReport(memories=memories, duration_ms=duration_ms, round_count=len(trace))

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.system.close()
        except Exception:
            pass

    def __enter__(self) -> "MemConflictMemory":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def store_dir_for(output_dir: str | Path, persona: Persona, version: str) -> Path:
    slug = persona_slug(persona.persona_id)
    return Path(output_dir) / "Memory" / f"{runtime.MEMORY_SYSTEM_NAME}_{slug}_{version}"


def first_turns(session: Session, limit: int = 2) -> Sequence[Any]:
    """Small helper used by tests and debug output."""
    return session.dialogue[:limit]
