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

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import runtime
from .data import Persona, Session


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "Session_ID": self.session_id,
            "Message_Count": self.message_count,
            "Duration_ms": self.duration_ms,
            "Namespace_Ready": self.namespace_ready,
            "Skipped": self.skipped,
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
        self.system = client.MemorySystem(
            config,
            api_history_logger=client.ApiHistoryLogger(str(self.store_dir)),
        )

    @staticmethod
    def build_namespace(persona: Persona, version: str) -> str:
        return f"memconflict:{persona_slug(persona.persona_id)}:{version}"

    # -- point 7: build memory one session at a time -----------------------

    def ingest_session(self, session: Session) -> IngestReport:
        if not session.dialogue:
            return IngestReport(session.session_id, 0, 0.0, None, skipped=True)

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
        )

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
