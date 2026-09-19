from __future__ import annotations

import json
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

from memory.v4.cross_scope_visibility import is_cross_scope_visible
from memory.v4.schemas import (
    SCHEMA_VERSION,
    EXACT_MENTION_MAX_LENGTH,
    MemoryEdge,
    MemoryNode,
    Participant,
    ParticipantScope,
    SemanticFact,
    TopicChain,
    participant_scope_id,
    stable_id,
)


_SCHEMA_REBUILD_MESSAGE = (
    "database is not a fresh compatible V4 store; rebuild fresh V4 memory"
)
_SQLITE_BUSY_TIMEOUT_SECONDS = 60.0
_SQLITE_BUSY_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _facts_text(facts: list[dict[str, Any]]) -> str:
    """Plain-text projection of semantic facts for FTS indexing."""
    parts: list[str] = []
    for fact in facts:
        for key in ("subject", "dimension", "aspect", "predicate", "value"):
            value = fact.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                parts.append(text)
    return " ".join(parts)


def _node_fts_content(node: MemoryNode) -> str:
    """FTS content for a V4 node.

    Aggregation nodes (chain_head / semantic_state) are indexed by their
    reducer-generated summary, facts, and entities only. Their ``text`` field
    must not carry raw turn wording; the original evidence lives in
    ``v4_evidence_records`` and is fetched by reference at answer time.
    Short exact_mentions are indexed for every node type so verbatim strings
    (titles, slogans, names, numbers, dates) stay lexically retrievable even
    when the summary paraphrases them.
    """
    parts = [node.title, node.summary]
    if node.node_type not in {"chain_head", "semantic_state"}:
        parts.append(node.text)
    parts.append(" ".join(node.entities))
    exact_mentions = [
        str(item).strip()
        for item in (node.metadata.get("exact_mentions") or ())
        if str(item).strip() and len(str(item).strip()) <= EXACT_MENTION_MAX_LENGTH
    ]
    if exact_mentions:
        parts.append(" ".join(exact_mentions))
    if node.node_type in {"chain_head", "semantic_state"}:
        parts.append(_facts_text(node.facts))
    return "\n".join(part for part in parts if part)


def _safe_fts_terms(values: Sequence[str], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = " ".join(
            str(value).replace('"', " ").replace("*", " ").split()
        ).strip()
        marker = term.casefold()
        if not term or marker in seen:
            continue
        seen.add(marker)
        output.append(term[:120])
        if len(output) == limit:
            break
    return output


def _load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _stable_unique_values(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def _is_sqlite_busy(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message


def _reliable_time_condition(prefix: str = "") -> str:
    """SQL predicate allowing only reliable time statuses into hard time filters.

    Vague or unresolved times remain reachable through FTS, embeddings, and
    raw_time_expression, but must not participate in before/after/nearest/
    overlap constraints.
    """
    return (
        " AND COALESCE(json_extract("
        f"{prefix}metadata_json, '$.normalization_status'), 'absolute') "
        "IN ('absolute','resolved_relative')"
    )


class RevisionConflictError(RuntimeError):
    pass


class V4SQLiteStore:
    """Authoritative Version-4 topic graph store.

    Data is isolated by participant scope. A scope publication uses optimistic
    revision checking and commits graph rows and index-generation metadata in
    the same SQLite transaction.
    """

    def __init__(self, db_path: str, read_only: bool = False):
        self.db_path = db_path
        self.read_only = read_only
        self._lock = threading.RLock()
        if read_only:
            escaped = quote(Path(db_path).resolve().as_posix(), safe="/")
            self.conn = sqlite3.connect(
                f"file:{escaped}?mode=ro", uri=True, check_same_thread=False,
                timeout=_SQLITE_BUSY_TIMEOUT_SECONDS,
            )
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(
                db_path, check_same_thread=False, timeout=_SQLITE_BUSY_TIMEOUT_SECONDS
            )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(f"PRAGMA busy_timeout={int(_SQLITE_BUSY_TIMEOUT_SECONDS * 1000)}")
        if read_only:
            self._validate_schema()
        else:
            self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def _begin_immediate(self) -> None:
        for delay in (*_SQLITE_BUSY_RETRY_DELAYS, None):
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as exc:
                if delay is None or not _is_sqlite_busy(exc):
                    raise
                time.sleep(delay)

    def _validate_schema(self) -> None:
        try:
            row = self.conn.execute(
                "SELECT value FROM v4_schema_meta WHERE key='schema_version'"
            ).fetchone()
            backend_row = self.conn.execute(
                "SELECT value FROM v4_schema_meta WHERE key='backend_id'"
            ).fetchone()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(_SCHEMA_REBUILD_MESSAGE) from exc
        if (
            row is None
            or int(row["value"]) != SCHEMA_VERSION
            or backend_row is None
            or backend_row["value"] != "v4"
        ):
            raise RuntimeError(_SCHEMA_REBUILD_MESSAGE)
        self._validate_required_schema_objects()

    def init_schema(self) -> None:
        with self._lock:
            version_table = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='v4_schema_meta'"
            ).fetchone()
            existing_objects = self.conn.execute(
                """SELECT name FROM sqlite_master
                   WHERE type IN ('table','view','trigger')
                     AND name NOT LIKE 'sqlite_%'"""
            ).fetchall()
            if existing_objects and not version_table:
                raise RuntimeError(_SCHEMA_REBUILD_MESSAGE)
            if version_table:
                row = self.conn.execute(
                    "SELECT value FROM v4_schema_meta WHERE key='schema_version'"
                ).fetchone()
                backend_row = self.conn.execute(
                    "SELECT value FROM v4_schema_meta WHERE key='backend_id'"
                ).fetchone()
                if (
                    row is None
                    or int(row["value"]) != SCHEMA_VERSION
                    or backend_row is None
                    or backend_row["value"] != "v4"
                ):
                    raise RuntimeError(_SCHEMA_REBUILD_MESSAGE)
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS v4_schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS v4_participant_scopes (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    visibility_policy TEXT NOT NULL DEFAULT 'scope_only',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    revision INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(namespace, id)
                );
                CREATE INDEX IF NOT EXISTS idx_v4_scopes_namespace
                    ON v4_participant_scopes(namespace);
                CREATE TABLE IF NOT EXISTS v4_scope_participants (
                    scope_id TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    PRIMARY KEY(scope_id, participant_id, role),
                    FOREIGN KEY(scope_id) REFERENCES v4_participant_scopes(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_v4_scope_participants_participant
                    ON v4_scope_participants(participant_id);
                CREATE TABLE IF NOT EXISTS v4_topic_chains (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    rolling_summary TEXT NOT NULL DEFAULT '',
                    representative_entities_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(scope_id, topic_id),
                    FOREIGN KEY(scope_id) REFERENCES v4_participant_scopes(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_v4_chains_scope ON v4_topic_chains(scope_id, topic_id);
                CREATE TABLE IF NOT EXISTS v4_topic_aliases (
                    scope_id TEXT NOT NULL,
                    chain_id TEXT NOT NULL,
                    alias TEXT NOT NULL COLLATE NOCASE,
                    PRIMARY KEY(scope_id, alias),
                    FOREIGN KEY(chain_id) REFERENCES v4_topic_chains(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS v4_memory_nodes (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    chain_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    node_type TEXT NOT NULL CHECK(node_type IN ('chain_head','event','semantic_state')),
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    text TEXT NOT NULL DEFAULT '',
                    actors_json TEXT NOT NULL DEFAULT '[]',
                    action TEXT NOT NULL DEFAULT '',
                    objects_json TEXT NOT NULL DEFAULT '[]',
                    location TEXT,
                    event_time_start TEXT,
                    event_time_end TEXT,
                    time_precision TEXT,
                    observed_at TEXT,
                    valid_from TEXT,
                    valid_to TEXT,
                    facts_json TEXT NOT NULL DEFAULT '[]',
                    changed_keys_json TEXT NOT NULL DEFAULT '[]',
                    previous_state_id TEXT,
                    trigger_event_ids_json TEXT NOT NULL DEFAULT '[]',
                    entities_json TEXT NOT NULL DEFAULT '[]',
                    entity_ids_json TEXT NOT NULL DEFAULT '[]',
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    evidence_refs_json TEXT NOT NULL DEFAULT '[]',
                    embedding_json TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(scope_id) REFERENCES v4_participant_scopes(id) ON DELETE CASCADE,
                    FOREIGN KEY(chain_id) REFERENCES v4_topic_chains(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_v4_nodes_scope_type ON v4_memory_nodes(scope_id, node_type);
                CREATE INDEX IF NOT EXISTS idx_v4_nodes_chain_time ON v4_memory_nodes(chain_id, event_time_start, id);
                CREATE INDEX IF NOT EXISTS idx_v4_nodes_valid_time ON v4_memory_nodes(scope_id, valid_from, valid_to);
                CREATE INDEX IF NOT EXISTS idx_v4_nodes_namespace_event_time
                    ON v4_memory_nodes(namespace, node_type, event_time_start, event_time_end, id);
                CREATE VIRTUAL TABLE IF NOT EXISTS v4_memory_fts USING fts5(
                    node_id UNINDEXED, scope_id UNINDEXED, content, tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS v4_semantic_facts (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    chain_id TEXT NOT NULL,
                    state_node_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    dimension TEXT NOT NULL,
                    aspect TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    evidence_refs_json TEXT NOT NULL DEFAULT '[]',
                    valid_from TEXT,
                    valid_to TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    history_origin TEXT,
                    status TEXT NOT NULL DEFAULT 'confirmed',
                    FOREIGN KEY(state_node_id) REFERENCES v4_memory_nodes(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_v4_facts_scope_key ON v4_semantic_facts(scope_id, fact_key, valid_from);
                CREATE TABLE IF NOT EXISTS v4_pending_conflicts (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    event_node_id TEXT NOT NULL,
                    chain_id TEXT NOT NULL,
                    boundary_time TEXT,
                    reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(event_node_id) REFERENCES v4_memory_nodes(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_v4_pending_scope_status
                    ON v4_pending_conflicts(scope_id, status);
                CREATE TABLE IF NOT EXISTS v4_semantic_reconciliation_failures (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    event_node_id TEXT NOT NULL,
                    error_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(event_node_id) REFERENCES v4_memory_nodes(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS v4_memory_edges (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    direction TEXT NOT NULL DEFAULT 'directed',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    trust TEXT NOT NULL CHECK(trust IN ('explicit','inferred','derived')),
                    evidence_refs_json TEXT NOT NULL DEFAULT '[]',
                    created_method TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    explanation TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(source_id) REFERENCES v4_memory_nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES v4_memory_nodes(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_v4_edges_source ON v4_memory_edges(scope_id, source_id, edge_type);
                CREATE INDEX IF NOT EXISTS idx_v4_edges_target ON v4_memory_edges(scope_id, target_id, edge_type);
                CREATE TABLE IF NOT EXISTS v4_cross_scope_edges (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    source_scope_id TEXT NOT NULL,
                    target_scope_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    direction TEXT NOT NULL DEFAULT 'directed',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    trust TEXT NOT NULL CHECK(trust IN ('explicit','inferred','derived')),
                    evidence_refs_json TEXT NOT NULL DEFAULT '[]',
                    explanation TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_v4_cross_scope_source
                    ON v4_cross_scope_edges(namespace, source_scope_id, source_id, edge_type);
                CREATE INDEX IF NOT EXISTS idx_v4_cross_scope_target
                    ON v4_cross_scope_edges(namespace, target_scope_id, target_id, edge_type);
                CREATE TABLE IF NOT EXISTS v4_node_entities (
                    scope_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    entity TEXT NOT NULL COLLATE NOCASE,
                    entity_id TEXT NOT NULL DEFAULT '',
                    significance REAL NOT NULL DEFAULT 1.0,
                    PRIMARY KEY(node_id, entity),
                    FOREIGN KEY(node_id) REFERENCES v4_memory_nodes(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_v4_entities_scope_entity ON v4_node_entities(scope_id, entity);
                CREATE INDEX IF NOT EXISTS idx_v4_entities_scope_entity_id
                    ON v4_node_entities(scope_id, entity_id);
                CREATE INDEX IF NOT EXISTS idx_v4_entities_entity_id_node
                    ON v4_node_entities(entity_id, node_id);
                CREATE TABLE IF NOT EXISTS v4_entity_registry (
                    namespace TEXT NOT NULL,
                    canonical_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    representative_alias TEXT NOT NULL,
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    embedding_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(namespace, canonical_id)
                );
                CREATE INDEX IF NOT EXISTS idx_v4_entity_registry_namespace_type
                    ON v4_entity_registry(namespace, entity_type);
                CREATE TABLE IF NOT EXISTS v4_entity_aliases (
                    namespace TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    alias TEXT NOT NULL COLLATE NOCASE,
                    canonical_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(namespace, entity_type, alias),
                    FOREIGN KEY(namespace, canonical_id)
                        REFERENCES v4_entity_registry(namespace, canonical_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_v4_entity_aliases_canonical
                    ON v4_entity_aliases(namespace, canonical_id);
                CREATE TABLE IF NOT EXISTS v4_evidence_records (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    session_id TEXT,
                    turn_id TEXT,
                    role TEXT,
                    participant_id TEXT,
                    observed_at TEXT,
                    content TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(node_id) REFERENCES v4_memory_nodes(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_v4_evidence_node ON v4_evidence_records(scope_id, node_id);
                CREATE INDEX IF NOT EXISTS idx_v4_evidence_turn ON v4_evidence_records(scope_id, session_id, turn_id);
                CREATE TABLE IF NOT EXISTS v4_scope_index_state (
                    scope_id TEXT PRIMARY KEY,
                    generation TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    node_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(scope_id) REFERENCES v4_participant_scopes(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS v4_build_checkpoints (
                    checkpoint_key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    scope_revision INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('failed','succeeded')),
                    cumulative_attempts INTEGER NOT NULL DEFAULT 0,
                    provider TEXT,
                    model TEXT,
                    prompt_version TEXT NOT NULL DEFAULT '',
                    backend_schema INTEGER NOT NULL,
                    output_json TEXT,
                    output_blob BLOB,
                    output_blob_dimension INTEGER,
                    error_json TEXT,
                    attempt_history_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_v4_build_checkpoints_scope_stage
                    ON v4_build_checkpoints(scope_id, stage, status);
                CREATE TABLE IF NOT EXISTS v4_scope_stage_state (
                    scope_id TEXT NOT NULL,
                    generation TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('failed','succeeded')),
                    error_json TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(scope_id, generation, stage)
                );
                """
            )
            # Remove obsolete identity columns when opening an existing native V4 store.
            self._begin_immediate()
            for table, obsolete in {
                "v4_build_checkpoints": ("fingerprint", "config_digest", "input_hash", "snapshot_identity"),
                "v4_scope_stage_state": ("fingerprint",),
                "v4_scope_index_state": ("mapping_checksum",),
            }.items():
                columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
                for column in obsolete:
                    if column in columns:
                        self.conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
            self._validate_required_schema_objects()
            self.conn.execute(
                "INSERT OR REPLACE INTO v4_schema_meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO v4_schema_meta(key,value) VALUES('backend_id','v4')"
            )
            self.conn.commit()

    def _validate_required_schema_objects(self) -> None:
        missing: list[str] = []
        if not self._table_exists("v4_cross_scope_edges"):
            missing.append("v4_cross_scope_edges table")
        evidence_columns = self._table_columns("v4_evidence_records")
        if "participant_id" not in evidence_columns:
            missing.append("v4_evidence_records.participant_id column")
        node_columns = self._table_columns("v4_memory_nodes")
        if "entity_ids_json" not in node_columns:
            missing.append("v4_memory_nodes.entity_ids_json column")
        node_entity_columns = self._table_columns("v4_node_entities")
        if "entity_id" not in node_entity_columns:
            missing.append("v4_node_entities.entity_id column")
        if not self._table_exists("v4_entity_registry"):
            missing.append("v4_entity_registry table")
        if not self._table_exists("v4_entity_aliases"):
            missing.append("v4_entity_aliases table")
        if not self._table_exists("v4_build_checkpoints"):
            missing.append("v4_build_checkpoints table")
        if not self._table_exists("v4_scope_stage_state"):
            missing.append("v4_scope_stage_state table")
        if missing:
            raise RuntimeError(f"{_SCHEMA_REBUILD_MESSAGE}: missing {', '.join(missing)}")

    def _table_exists(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None

    def _table_columns(self, name: str) -> set[str]:
        try:
            rows = self.conn.execute(f"PRAGMA table_info({name})").fetchall()
        except sqlite3.OperationalError:
            return set()
        return {str(row["name"]) for row in rows}

    def list_scopes(self, namespace: str) -> list[ParticipantScope]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM v4_participant_scopes WHERE namespace=? ORDER BY id", (namespace,)
            ).fetchall()
            return [self._scope(row) for row in rows]

    def get_scope(self, scope_id: str) -> ParticipantScope | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM v4_participant_scopes WHERE id=?", (scope_id,)).fetchone()
            return self._scope(row) if row else None

    def resolve_scope(self, namespace: str, participants: Iterable[Any] | None = None) -> ParticipantScope:
        if participants is not None:
            scope_id = participant_scope_id(namespace, participants)
            scope = self.get_scope(scope_id)
            if scope is None:
                raise KeyError(f"participant scope {scope_id!r} does not exist in namespace {namespace!r}")
            return scope
        scopes = self.list_scopes(namespace)
        if not scopes:
            raise KeyError(f"namespace {namespace!r} has no participant scope")
        if len(scopes) > 1:
            raise ValueError(
                f"namespace {namespace!r} contains multiple participant scopes; participants are required"
            )
        return scopes[0]

    def scope_revision(self, scope_id: str) -> int:
        scope = self.get_scope(scope_id)
        return scope.revision if scope else 0

    def get_scope_index_state(self, scope_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT generation, revision, node_count FROM v4_scope_index_state WHERE scope_id=?",
                (scope_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "generation": str(row["generation"]), "revision": int(row["revision"]),
            "node_count": int(row["node_count"]),
        }

    def record_build_checkpoint(
        self,
        *,
        checkpoint_key: str,
        namespace: str,
        scope_id: str,
        scope_revision: int,
        stage: str,
        unit_id: str,
        status: str,
        attempt: int,
        provider: str | None,
        model: str | None,
        prompt_version: str,
        output: Any = None,
        embedding: Iterable[float] | None = None,
        error: Any = None,
    ) -> None:
        if self.read_only:
            raise RuntimeError("cannot checkpoint through a read-only V4 store")
        if status not in {"failed", "succeeded"}:
            raise ValueError("build checkpoint status must be failed or succeeded")
        increment = max(1, int(attempt))
        vector = list(embedding) if embedding is not None else None
        blob = (
            struct.pack(f"<{len(vector)}f", *[float(value) for value in vector])
            if vector is not None
            else None
        )
        history_item = {
            "status": status,
            "attempt": increment,
            "error": error,
        }
        with self._lock:
            self._begin_immediate()
            try:
                previous = self.conn.execute(
                    "SELECT cumulative_attempts,attempt_history_json "
                    "FROM v4_build_checkpoints WHERE checkpoint_key=?",
                    (checkpoint_key,),
                ).fetchone()
                cumulative = (
                    int(previous["cumulative_attempts"]) if previous is not None else 0
                ) + increment
                history = (
                    _load(previous["attempt_history_json"], [])
                    if previous is not None
                    else []
                )
                history.append(history_item)
                self.conn.execute(
                    """INSERT INTO v4_build_checkpoints(
                           checkpoint_key,namespace,scope_id,scope_revision,stage,unit_id,
                           status,cumulative_attempts,provider,model,prompt_version,
                           backend_schema,output_json,
                           output_blob,output_blob_dimension,error_json,attempt_history_json)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(checkpoint_key) DO UPDATE SET
                         namespace=excluded.namespace,scope_id=excluded.scope_id,
                         scope_revision=excluded.scope_revision,stage=excluded.stage,
                         unit_id=excluded.unit_id,
                         status=excluded.status,cumulative_attempts=excluded.cumulative_attempts,
                         provider=excluded.provider,model=excluded.model,
                         prompt_version=excluded.prompt_version,
                         backend_schema=excluded.backend_schema,
                         output_json=excluded.output_json,output_blob=excluded.output_blob,
                         output_blob_dimension=excluded.output_blob_dimension,
                         error_json=excluded.error_json,
                         attempt_history_json=excluded.attempt_history_json,
                         updated_at=CURRENT_TIMESTAMP""",
                    (
                        checkpoint_key, namespace, scope_id, int(scope_revision), stage,
                        unit_id, status, cumulative, provider, model,
                        prompt_version, SCHEMA_VERSION, _dump(output) if output is not None else None,
                        blob, len(vector) if vector is not None else None,
                        _dump(error) if error is not None else None, _dump(history),
                    ),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def load_build_checkpoint(
        self, checkpoint_key: str
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM v4_build_checkpoints "
                "WHERE checkpoint_key=? AND status='succeeded'",
                (checkpoint_key,),
            ).fetchone()
        if row is None:
            return None
        dimension = row["output_blob_dimension"]
        blob = row["output_blob"]
        embedding = None
        if blob is not None and dimension is not None:
            embedding = list(struct.unpack(f"<{int(dimension)}f", bytes(blob)))
        return {
            "checkpoint_key": str(row["checkpoint_key"]),
            "stage": str(row["stage"]),
            "unit_id": str(row["unit_id"]),
            "status": str(row["status"]),
            "cumulative_attempts": int(row["cumulative_attempts"]),
            "output": _load(row["output_json"], None),
            "embedding": embedding,
            "attempt_history": _load(row["attempt_history_json"], []),
        }

    def record_scope_stage(
        self,
        scope_id: str,
        generation: str,
        stage: str,
        status: str,
        *,
        error: Any = None,
    ) -> None:
        if self.read_only:
            raise RuntimeError("cannot checkpoint through a read-only V4 store")
        if status not in {"failed", "succeeded"}:
            raise ValueError("scope stage status must be failed or succeeded")
        with self._lock:
            self.conn.execute(
                """INSERT INTO v4_scope_stage_state(
                       scope_id,generation,stage,status,error_json)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(scope_id,generation,stage) DO UPDATE SET
                     status=excluded.status,
                     error_json=excluded.error_json,updated_at=CURRENT_TIMESTAMP""",
                (
                    scope_id, generation, stage, status,
                    _dump(error) if error is not None else None,
                ),
            )
            self.conn.commit()

    def is_scope_stage_complete(
        self, scope_id: str, generation: str, stage: str
    ) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT status FROM v4_scope_stage_state "
                "WHERE scope_id=? AND generation=? AND stage=?",
                (scope_id, generation, stage),
            ).fetchone()
        return row is not None and str(row["status"]) == "succeeded"

    def get_scope_stage(
        self, scope_id: str, generation: str, stage: str
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT status,error_json,updated_at "
                "FROM v4_scope_stage_state "
                "WHERE scope_id=? AND generation=? AND stage=?",
                (scope_id, generation, stage),
            ).fetchone()
        if row is None:
            return None
        return {
            "status": str(row["status"]),
            "error": _load(row["error_json"], None),
            "updated_at": str(row["updated_at"]),
        }

    def publish_scope(
        self,
        scope: ParticipantScope,
        chains: Iterable[TopicChain],
        nodes: Iterable[MemoryNode],
        edges: Iterable[MemoryEdge],
        facts: Iterable[SemanticFact],
        *,
        generation: str,
        node_count: int,
        expected_revision: int,
    ) -> int:
        if self.read_only:
            raise RuntimeError("cannot publish through a read-only V4 store")
        chains, nodes, edges, facts = list(chains), list(nodes), list(edges), list(facts)
        if node_count != len(nodes):
            raise ValueError("scope index count does not match the published node view")
        self._validate_scope_rows(scope, chains, nodes, edges, facts)
        new_revision = expected_revision + 1
        with self._lock:
            self._begin_immediate()
            try:
                row = self.conn.execute(
                    "SELECT revision FROM v4_participant_scopes WHERE id=?", (scope.id,)
                ).fetchone()
                actual = int(row["revision"]) if row else 0
                if actual != expected_revision:
                    raise RevisionConflictError(
                        f"scope revision changed from {expected_revision} to {actual}"
                    )
                self.conn.execute(
                    """INSERT INTO v4_participant_scopes(id,namespace,visibility_policy,metadata_json,revision)
                       VALUES(?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET metadata_json=excluded.metadata_json,
                         visibility_policy=excluded.visibility_policy, revision=excluded.revision,
                         updated_at=CURRENT_TIMESTAMP""",
                    (scope.id, scope.namespace, scope.visibility_policy, _dump(scope.metadata), new_revision),
                )
                for participant in scope.participants:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO v4_scope_participants(scope_id,participant_id,role) VALUES(?,?,?)",
                        (scope.id, participant.participant_id, participant.role),
                    )
                for chain in chains:
                    self._upsert_chain(chain)
                for node in nodes:
                    self._upsert_node(node)
                # Incremental edge publish: the artifact carries the complete edge
                # view, but we only delete rows whose id is no longer present and
                # upsert the rest. Unchanged edges (same stable id) keep their row
                # identity — no delete/insert churn, downstream consumers see
                # stable ids across ingests. EVENT/STATE edges are preserved in the
                # artifact by the builder, so they survive unless their endpoints
                # were removed.
                new_edge_ids = {edge.id for edge in edges}
                stale_rows = self.conn.execute(
                    "SELECT id FROM v4_memory_edges WHERE scope_id=?", (scope.id,)
                ).fetchall()
                stale_ids = [str(row["id"]) for row in stale_rows if str(row["id"]) not in new_edge_ids]
                if stale_ids:
                    self.conn.executemany(
                        "DELETE FROM v4_memory_edges WHERE id=?",
                        [(edge_id,) for edge_id in stale_ids],
                    )
                for edge in edges:
                    self._upsert_edge(edge)
                # The artifact is the complete fact view. Re-materialize it in
                # the same transaction so in-place extend/retract cannot leave
                # stale rows behind.
                self.conn.execute(
                    "DELETE FROM v4_semantic_facts WHERE scope_id=?", (scope.id,)
                )
                for fact in facts:
                    self._upsert_fact(fact)
                self.conn.execute(
                    """INSERT INTO v4_scope_index_state(scope_id,generation,revision,node_count)
                       VALUES(?,?,?,?) ON CONFLICT(scope_id) DO UPDATE SET
                         generation=excluded.generation, revision=excluded.revision,
                         node_count=excluded.node_count,
                         updated_at=CURRENT_TIMESTAMP""",
                    (scope.id, generation, new_revision, node_count),
                )
                self.conn.commit()
                return new_revision
            except BaseException:
                self.conn.rollback()
                raise

    def _validate_scope_rows(self, scope, chains, nodes, edges, facts) -> None:
        chain_ids = {chain.id for chain in chains}
        node_ids = {node.id for node in nodes}
        for row in [*chains, *nodes, *edges, *facts]:
            if row.namespace != scope.namespace or row.scope_id != scope.id:
                raise ValueError("cross-scope V4 publication is forbidden")
        if any(node.chain_id not in chain_ids for node in nodes):
            raise ValueError("all nodes must belong to a published topic chain")
        if any(edge.source_id not in node_ids or edge.target_id not in node_ids for edge in edges):
            raise ValueError("memory edges cannot cross participant scopes")

    def _upsert_chain(self, chain: TopicChain) -> None:
        self.conn.execute(
            """INSERT INTO v4_topic_chains(id,namespace,scope_id,topic_id,name,description,rolling_summary,representative_entities_json,metadata_json)
               VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,
                 description=excluded.description, rolling_summary=excluded.rolling_summary,
                 representative_entities_json=excluded.representative_entities_json,
                 metadata_json=excluded.metadata_json, updated_at=CURRENT_TIMESTAMP""",
            (chain.id, chain.namespace, chain.scope_id, chain.topic_id, chain.name, chain.description,
             chain.rolling_summary, _dump(chain.representative_entities), _dump(chain.metadata)),
        )
        for alias in dict.fromkeys([chain.topic_id, chain.name, *chain.aliases]):
            if str(alias).strip():
                self.conn.execute(
                    "INSERT OR IGNORE INTO v4_topic_aliases(scope_id,chain_id,alias) VALUES(?,?,?)",
                    (chain.scope_id, chain.id, str(alias).strip()),
                )

    def _upsert_node(self, node: MemoryNode) -> None:
        values = (
            node.id, node.namespace, node.scope_id, node.chain_id, node.topic_id, node.node_type,
            node.title, node.summary, node.text, _dump(node.actors), node.action, _dump(node.objects),
            node.location, node.event_time_start, node.event_time_end, node.time_precision, node.observed_at,
            node.valid_from, node.valid_to, _dump(node.facts), _dump(node.changed_keys), node.previous_state_id,
            _dump(node.trigger_event_ids), _dump(node.entities), _dump(node.entity_ids), node.importance, node.confidence,
            _dump(node.evidence_refs), _dump(node.embedding) if node.embedding is not None else None,
            _dump(node.metadata),
        )
        self.conn.execute(
            """INSERT INTO v4_memory_nodes(
                 id,namespace,scope_id,chain_id,topic_id,node_type,title,summary,text,actors_json,action,
                 objects_json,location,event_time_start,event_time_end,time_precision,observed_at,valid_from,
                 valid_to,facts_json,changed_keys_json,previous_state_id,trigger_event_ids_json,entities_json,
                 entity_ids_json,importance,confidence,evidence_refs_json,embedding_json,metadata_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET title=excluded.title, summary=excluded.summary, text=excluded.text,
                 actors_json=excluded.actors_json, action=excluded.action, objects_json=excluded.objects_json,
                 location=excluded.location, event_time_start=excluded.event_time_start,
                 event_time_end=excluded.event_time_end, time_precision=excluded.time_precision,
                 observed_at=excluded.observed_at, valid_from=excluded.valid_from, valid_to=excluded.valid_to,
                 facts_json=excluded.facts_json, changed_keys_json=excluded.changed_keys_json,
                 previous_state_id=excluded.previous_state_id, trigger_event_ids_json=excluded.trigger_event_ids_json,
                 entities_json=excluded.entities_json, entity_ids_json=excluded.entity_ids_json,
                 importance=excluded.importance,
                 confidence=excluded.confidence, evidence_refs_json=excluded.evidence_refs_json,
                 embedding_json=excluded.embedding_json, metadata_json=excluded.metadata_json,
                 updated_at=CURRENT_TIMESTAMP""", values,
        )
        self.conn.execute("DELETE FROM v4_memory_fts WHERE node_id=?", (node.id,))
        self.conn.execute(
            "INSERT INTO v4_memory_fts(node_id,scope_id,content) VALUES(?,?,?)",
            (node.id, node.scope_id, _node_fts_content(node)),
        )
        self.conn.execute("DELETE FROM v4_node_entities WHERE node_id=?", (node.id,))
        entity_ids = list(node.entity_ids or node.entities)
        for index, entity in enumerate(dict.fromkeys(node.entities)):
            entity_id = entity_ids[index] if index < len(entity_ids) else entity
            self.conn.execute(
                "INSERT INTO v4_node_entities(scope_id,node_id,entity,entity_id,significance) VALUES(?,?,?,?,?)",
                (node.scope_id, node.id, entity, entity_id, 1.0),
            )
        for ref in node.evidence_refs:
            evidence_id = str(ref.get("evidence_id") or "")
            if evidence_id:
                link_id = stable_id("evidence_record", evidence_id, node.id)
                self.conn.execute(
                    """INSERT INTO v4_evidence_records(id,namespace,scope_id,node_id,session_id,turn_id,role,participant_id,observed_at,content,metadata_json)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET node_id=excluded.node_id,
                         observed_at=excluded.observed_at, content=excluded.content, metadata_json=excluded.metadata_json,
                         participant_id=excluded.participant_id""",
                    (link_id, node.namespace, node.scope_id, node.id, ref.get("session_id"),
                     ref.get("turn_id"), ref.get("role"), ref.get("participant_id"),
                     ref.get("observed_at"), ref.get("content"),
                     _dump(ref.get("metadata") or {})),
                )
        self.conn.execute(
            "DELETE FROM v4_pending_conflicts WHERE event_node_id=?", (node.id,)
        )
        for index, conflict in enumerate(node.metadata.get("pending_semantic_conflicts") or []):
            conflict_id = stable_id("pending_conflict", node.id, index, conflict)
            self.conn.execute(
                """INSERT INTO v4_pending_conflicts(
                     id,namespace,scope_id,event_node_id,chain_id,boundary_time,reason,payload_json,status
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    conflict_id, node.namespace, node.scope_id, node.id, node.chain_id,
                    conflict.get("boundary_time"), str(conflict.get("reason") or ""),
                    _dump(conflict), "pending",
                ),
            )
        self.conn.execute(
            "DELETE FROM v4_semantic_reconciliation_failures WHERE event_node_id=?", (node.id,)
        )
        failure = node.metadata.get("semantic_reconciliation_error")
        if failure:
            self.conn.execute(
                """INSERT INTO v4_semantic_reconciliation_failures(
                     id,namespace,scope_id,event_node_id,error_json
                   ) VALUES(?,?,?,?,?)""",
                (
                    stable_id("semantic_failure", node.id, failure), node.namespace,
                    node.scope_id, node.id, _dump(failure),
                ),
            )

    def _upsert_edge(self, edge: MemoryEdge) -> None:
        self.conn.execute(
            """INSERT INTO v4_memory_edges(id,namespace,scope_id,source_id,target_id,edge_type,direction,
                 confidence,trust,evidence_refs_json,created_method,valid_from,valid_to,explanation,metadata_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                 confidence=excluded.confidence, trust=excluded.trust,
                 evidence_refs_json=excluded.evidence_refs_json, valid_from=excluded.valid_from,
                 valid_to=excluded.valid_to, explanation=excluded.explanation, metadata_json=excluded.metadata_json""",
            (edge.id, edge.namespace, edge.scope_id, edge.source_id, edge.target_id, edge.edge_type,
             edge.direction, edge.confidence, edge.trust, _dump(edge.evidence_refs), edge.created_method,
             edge.valid_from, edge.valid_to, edge.explanation, _dump(edge.metadata)),
        )

    def _upsert_fact(self, fact: SemanticFact) -> None:
        self.conn.execute(
            """INSERT INTO v4_semantic_facts(id,namespace,scope_id,chain_id,state_node_id,fact_key,subject,
                 predicate,dimension,aspect,value_json,confidence,evidence_refs_json,valid_from,valid_to,
                 created_at,updated_at,history_origin,status)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET value_json=excluded.value_json,
                 confidence=excluded.confidence, evidence_refs_json=excluded.evidence_refs_json,
                 valid_from=excluded.valid_from, valid_to=excluded.valid_to,
                 updated_at=excluded.updated_at, history_origin=excluded.history_origin,
                 status=excluded.status""",
            (fact.id, fact.namespace, fact.scope_id, fact.chain_id, fact.state_node_id, fact.key,
             fact.subject, fact.predicate, fact.dimension, fact.aspect, _dump(fact.value),
             fact.confidence, _dump(fact.evidence_refs), fact.valid_from, fact.valid_to,
             fact.created_at, fact.updated_at, fact.history_origin, fact.status),
        )

    def list_chains(self, scope_id: str) -> list[TopicChain]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM v4_topic_chains WHERE scope_id=? ORDER BY topic_id", (scope_id,)
            ).fetchall()
            aliases = self.conn.execute(
                "SELECT chain_id,alias FROM v4_topic_aliases WHERE scope_id=? ORDER BY alias", (scope_id,)
            ).fetchall()
        by_chain: dict[str, list[str]] = {}
        for row in aliases:
            by_chain.setdefault(str(row["chain_id"]), []).append(str(row["alias"]))
        return [self._chain(row, by_chain.get(str(row["id"]), [])) for row in rows]

    def list_nodes(self, scope_id: str, node_types: Iterable[str] | None = None) -> list[MemoryNode]:
        params: list[Any] = [scope_id]
        clause = ""
        types = list(node_types or [])
        if types:
            clause = f" AND node_type IN ({','.join('?' for _ in types)})"
            params.extend(types)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM v4_memory_nodes WHERE scope_id=?{clause} ORDER BY chain_id,event_time_start,valid_from,id",
                params,
            ).fetchall()
        return [self._node(row) for row in rows]

    def get_node(self, node_id: str, scope_id: str | None = None) -> MemoryNode | None:
        query, params = "SELECT * FROM v4_memory_nodes WHERE id=?", [node_id]
        if scope_id:
            query += " AND scope_id=?"
            params.append(scope_id)
        with self._lock:
            row = self.conn.execute(query, params).fetchone()
        return self._node(row) if row else None

    def get_nodes(self, node_ids: Iterable[str], scope_id: str | None = None) -> list[MemoryNode]:
        ids = list(dict.fromkeys(str(value) for value in node_ids))
        if not ids:
            return []
        query = f"SELECT * FROM v4_memory_nodes WHERE id IN ({','.join('?' for _ in ids)})"
        params: list[Any] = list(ids)
        if scope_id:
            query += " AND scope_id=?"
            params.append(scope_id)
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        by_id = {str(row["id"]): self._node(row) for row in rows}
        return [by_id[node_id] for node_id in ids if node_id in by_id]

    def list_edges(self, scope_id: str) -> list[MemoryEdge]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM v4_memory_edges WHERE scope_id=? ORDER BY edge_type,id", (scope_id,)
            ).fetchall()
        return [self._edge(row) for row in rows]

    def adjacent(
        self, scope_id: str, node_ids: Iterable[str], edge_types: Iterable[str] | None = None,
        direction: str = "both",
    ) -> list[tuple[MemoryEdge, str]]:
        ids = list(dict.fromkeys(str(value) for value in node_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        if direction == "out":
            where, params = f"source_id IN ({placeholders})", ids
        elif direction == "in":
            where, params = f"target_id IN ({placeholders})", ids
        else:
            where, params = f"(source_id IN ({placeholders}) OR target_id IN ({placeholders}))", [*ids, *ids]
        types = [str(value).upper() for value in (edge_types or [])]
        type_clause = ""
        if types:
            type_clause = f" AND edge_type IN ({','.join('?' for _ in types)})"
            params.extend(types)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM v4_memory_edges WHERE scope_id=? AND {where}{type_clause} ORDER BY id",
                [scope_id, *params],
            ).fetchall()
        requested = set(ids)
        output = []
        for row in rows:
            edge = self._edge(row)
            neighbor = edge.target_id if edge.source_id in requested else edge.source_id
            output.append((edge, neighbor))
        return output

    def list_facts(self, scope_id: str, state_node_ids: Iterable[str] | None = None) -> list[SemanticFact]:
        ids = list(state_node_ids or [])
        clause, params = "", [scope_id]
        if ids:
            clause = f" AND state_node_id IN ({','.join('?' for _ in ids)})"
            params.extend(ids)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM v4_semantic_facts WHERE scope_id=?{clause} ORDER BY valid_from,id", params
            ).fetchall()
        return [self._fact(row) for row in rows]

    # ----- cross-scope participant index + navigation queries (#7) -----

    def scopes_by_participant(self, namespace: str, participant_id: str) -> list[ParticipantScope]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT DISTINCT s.* FROM v4_participant_scopes s
                   JOIN v4_scope_participants p ON p.scope_id=s.id
                   WHERE s.namespace=? AND p.participant_id=?
                   ORDER BY s.id""",
                (namespace, participant_id),
            ).fetchall()
            return [self._scope(row) for row in rows]

    def scope_participants(self, scope_id: str) -> list[Participant]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT participant_id,role FROM v4_scope_participants WHERE scope_id=? ORDER BY role,participant_id",
                (scope_id,),
            ).fetchall()
        return [Participant(str(row["participant_id"]), str(row["role"])) for row in rows]

    def node_evidence_participants(self, node_id: str) -> set[str]:
        """Return the set of participant_ids that contributed evidence to a node.

        Used by the cross-scope visibility filter (target scope participants must
        be a superset of this set). Missing participant_id rows are treated as
        "no consent" — the node cannot cross scope.
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT DISTINCT participant_id FROM v4_evidence_records WHERE node_id=? AND participant_id IS NOT NULL",
                (node_id,),
            ).fetchall()
        return {str(row["participant_id"]) for row in rows if row["participant_id"]}

    def evidence_records_by_node_ids(
        self, node_ids: Iterable[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Return complete source records grouped by node without N+1 queries."""
        ids = list(dict.fromkeys(str(value) for value in node_ids if str(value)))
        grouped: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in ids}
        if not ids:
            return grouped
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT id,node_id,session_id,turn_id,role,participant_id,
                           observed_at,content,metadata_json
                      FROM v4_evidence_records
                     WHERE node_id IN ({placeholders})
                     ORDER BY node_id,session_id,turn_id,id""",
                ids,
            ).fetchall()
        for row in rows:
            node_id = str(row["node_id"])
            grouped[node_id].append({
                "id": str(row["id"]),
                "node_id": node_id,
                "session_id": row["session_id"],
                "turn_id": row["turn_id"],
                "role": row["role"],
                "participant_id": row["participant_id"],
                "observed_at": row["observed_at"],
                "content": str(row["content"]),
                "metadata": _load(row["metadata_json"], {}),
            })
        return grouped

    def get_entity_vectors(self, namespace: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT namespace,canonical_id,entity_type,representative_alias,aliases_json,
                          embedding_json,created_at,updated_at
                   FROM v4_entity_registry WHERE namespace=?
                   ORDER BY created_at,canonical_id""",
                (namespace,),
            ).fetchall()
        return [
            {
                "namespace": str(row["namespace"]),
                "canonical_id": str(row["canonical_id"]),
                "entity_type": str(row["entity_type"]),
                "representative_alias": str(row["representative_alias"]),
                "aliases": _load(row["aliases_json"], []),
                "embedding": _load(row["embedding_json"], None),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    def upsert_entities(self, namespace: str, entries: Iterable[dict[str, Any]]) -> None:
        if self.read_only:
            raise RuntimeError("cannot publish through a read-only V4 store")
        with self._lock:
            for raw in entries:
                canonical_id = str(raw.get("canonical_id") or "").strip()
                entity_type = str(raw.get("entity_type") or "entity").strip().lower()
                representative = str(raw.get("representative_alias") or "").strip()
                if not canonical_id or not representative:
                    continue
                existing = self.conn.execute(
                    """SELECT representative_alias,aliases_json,embedding_json
                       FROM v4_entity_registry WHERE namespace=? AND canonical_id=?""",
                    (namespace, canonical_id),
                ).fetchone()
                aliases = _stable_unique_values([
                    representative,
                    *list(raw.get("aliases") or []),
                    *(_load(existing["aliases_json"], []) if existing else []),
                ])
                representative = str(existing["representative_alias"]) if existing else representative
                embedding = raw.get("embedding")
                if embedding is None and existing:
                    embedding = _load(existing["embedding_json"], None)
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                self.conn.execute(
                    """INSERT INTO v4_entity_registry(
                           namespace,canonical_id,entity_type,representative_alias,aliases_json,embedding_json)
                       VALUES(?,?,?,?,?,?)
                       ON CONFLICT(namespace, canonical_id) DO UPDATE SET
                         entity_type=excluded.entity_type,
                         representative_alias=excluded.representative_alias,
                         aliases_json=excluded.aliases_json,
                         embedding_json=excluded.embedding_json,
                         updated_at=CURRENT_TIMESTAMP""",
                    (namespace, canonical_id, entity_type, representative, _dump(aliases),
                     _dump(embedding) if embedding is not None else None),
                )
                for alias in aliases:
                    alias_text = str(alias).strip()
                    if not alias_text:
                        continue
                    self.conn.execute(
                        """INSERT INTO v4_entity_aliases(namespace,entity_type,alias,canonical_id)
                           VALUES(?,?,?,?)
                           ON CONFLICT(namespace, entity_type, alias) DO UPDATE SET
                             canonical_id=excluded.canonical_id""",
                        (namespace, entity_type, alias_text, canonical_id),
                    )
            self.conn.commit()

    def nodes_by_entity_ids(
        self,
        namespace: str,
        entity_ids: Iterable[str],
        node_type: str | None = None,
        *,
        match_all: bool = False,
        limit: int | None = None,
    ) -> list[MemoryNode]:
        ids = list(dict.fromkeys(str(value) for value in entity_ids if str(value).strip()))
        if not ids:
            return []
        params: list[Any] = [namespace, *ids]
        type_clause = " AND n.node_type!='chain_head'"
        if node_type:
            type_clause = " AND n.node_type=?"
            params.append(node_type)
        having_clause = ""
        if match_all:
            having_clause = " HAVING COUNT(DISTINCT e.entity_id)=?"
            params.append(len(ids))
        limit_clause = ""
        if limit is not None:
            limit_clause = " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT n.* FROM v4_memory_nodes n
                   JOIN v4_node_entities e ON e.node_id=n.id
                   WHERE n.namespace=? AND e.entity_id IN ({','.join('?' for _ in ids)}){type_clause}
                   GROUP BY n.id{having_clause}
                   ORDER BY n.scope_id,n.event_time_start,n.id{limit_clause}""",
                params,
            ).fetchall()
        return [self._node(row) for row in rows]

    def merge_entity_ids(self, namespace: str, source_canonical_id: str, target_canonical_id: str) -> None:
        source = str(source_canonical_id).strip()
        target = str(target_canonical_id).strip()
        if not source or not target or source == target:
            return
        if self.read_only:
            raise RuntimeError("cannot publish through a read-only V4 store")
        with self._lock:
            self._begin_immediate()
            try:
                source_row = self.conn.execute(
                    "SELECT * FROM v4_entity_registry WHERE namespace=? AND canonical_id=?",
                    (namespace, source),
                ).fetchone()
                target_row = self.conn.execute(
                    "SELECT * FROM v4_entity_registry WHERE namespace=? AND canonical_id=?",
                    (namespace, target),
                ).fetchone()
                if source_row is None or target_row is None:
                    self.conn.commit()
                    return
                alias_rows = self.conn.execute(
                    """SELECT entity_type,alias FROM v4_entity_aliases
                       WHERE namespace=? AND canonical_id IN (?,?)
                       ORDER BY created_at,alias""",
                    (namespace, source, target),
                ).fetchall()
                target_type = str(target_row["entity_type"])
                aliases = _stable_unique_values([
                    str(target_row["representative_alias"]),
                    *_load(target_row["aliases_json"], []),
                    str(source_row["representative_alias"]),
                    *_load(source_row["aliases_json"], []),
                    *(str(row["alias"]) for row in alias_rows),
                ])
                embedding = _load(target_row["embedding_json"], None)
                self.conn.execute(
                    """UPDATE v4_entity_registry SET aliases_json=?, embedding_json=?,
                         updated_at=CURRENT_TIMESTAMP
                       WHERE namespace=? AND canonical_id=?""",
                    (_dump(aliases), _dump(embedding) if embedding is not None else None, namespace, target),
                )
                self.conn.execute(
                    "DELETE FROM v4_entity_aliases WHERE namespace=? AND canonical_id IN (?,?)",
                    (namespace, source, target),
                )
                for alias in aliases:
                    alias_text = str(alias).strip()
                    if alias_text:
                        self.conn.execute(
                            """INSERT OR REPLACE INTO v4_entity_aliases(
                                   namespace,entity_type,alias,canonical_id)
                               VALUES(?,?,?,?)""",
                            (namespace, target_type, alias_text, target),
                        )
                self.conn.execute(
                    "DELETE FROM v4_entity_registry WHERE namespace=? AND canonical_id=?",
                    (namespace, source),
                )
                self.conn.execute(
                    """UPDATE v4_node_entities SET entity_id=?
                       WHERE node_id IN (SELECT id FROM v4_memory_nodes WHERE namespace=?)
                         AND entity_id=?""",
                    (target, namespace, source),
                )
                rows = self.conn.execute(
                    "SELECT id,entity_ids_json FROM v4_memory_nodes WHERE namespace=?",
                    (namespace,),
                ).fetchall()
                for row in rows:
                    entity_ids = _load(row["entity_ids_json"], [])
                    if source not in entity_ids:
                        continue
                    updated = [target if value == source else value for value in entity_ids]
                    self.conn.execute(
                        "UPDATE v4_memory_nodes SET entity_ids_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                        (_dump(updated), row["id"]),
                    )
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise

    def nodes_overlapping_time(
        self,
        namespace: str,
        start: str | None,
        end: str | None,
        node_type: str | None = None,
        *,
        limit: int | None = None,
    ) -> list[MemoryNode]:
        if not start:
            return []
        end_value = end or start
        params: list[Any] = [namespace, end_value, start]
        type_clause = ""
        if node_type:
            type_clause = " AND node_type=?"
            params.append(node_type)
        limit_clause = ""
        if limit is not None:
            limit_clause = " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT * FROM v4_memory_nodes
                   WHERE namespace=? AND node_type!='chain_head'
                     AND event_time_start IS NOT NULL
                     AND event_time_start <= ? AND COALESCE(event_time_end, event_time_start) >= ?
                     {_reliable_time_condition()}
                     {type_clause}
                   ORDER BY scope_id,event_time_start,id{limit_clause}""",
                params,
            ).fetchall()
        return [self._node(row) for row in rows]

    def nodes_before_time(
        self,
        namespace: str,
        reference: str | None,
        node_type: str | None = None,
        *,
        max_distance_days: int | None = None,
        limit: int | None = None,
    ) -> list[MemoryNode]:
        """Return events ending before ``reference``, nearest first."""
        if not reference:
            return []
        params: list[Any] = [namespace, reference]
        type_clause = ""
        if node_type:
            type_clause = " AND node_type=?"
            params.append(node_type)
        distance_clause = ""
        if max_distance_days is not None:
            distance_clause = (
                " AND julianday(?) - "
                "julianday(COALESCE(event_time_end,event_time_start)) <= ?"
            )
            params.extend((reference, max(0, int(max_distance_days))))
        limit_clause = ""
        if limit is not None:
            limit_clause = " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT * FROM v4_memory_nodes
                   WHERE namespace=? AND node_type!='chain_head'
                     AND event_time_start IS NOT NULL
                     AND COALESCE(event_time_end,event_time_start) < ?
                     {_reliable_time_condition()}
                     {type_clause}{distance_clause}
                   ORDER BY COALESCE(event_time_end,event_time_start) DESC,id
                   {limit_clause}""",
                params,
            ).fetchall()
        return [self._node(row) for row in rows]

    def nodes_after_time(
        self,
        namespace: str,
        reference: str | None,
        node_type: str | None = None,
        *,
        max_distance_days: int | None = None,
        limit: int | None = None,
    ) -> list[MemoryNode]:
        """Return events starting after ``reference``, nearest first."""
        if not reference:
            return []
        params: list[Any] = [namespace, reference]
        type_clause = ""
        if node_type:
            type_clause = " AND node_type=?"
            params.append(node_type)
        distance_clause = ""
        if max_distance_days is not None:
            distance_clause = " AND julianday(event_time_start) - julianday(?) <= ?"
            params.extend((reference, max(0, int(max_distance_days))))
        limit_clause = ""
        if limit is not None:
            limit_clause = " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT * FROM v4_memory_nodes
                   WHERE namespace=? AND node_type!='chain_head'
                     AND event_time_start IS NOT NULL
                     AND event_time_start > ?
                     {_reliable_time_condition()}
                     {type_clause}{distance_clause}
                   ORDER BY event_time_start ASC,id{limit_clause}""",
                params,
            ).fetchall()
        return [self._node(row) for row in rows]

    def nodes_nearest_time(
        self,
        namespace: str,
        reference: str | None,
        node_type: str | None = None,
        *,
        max_distance_days: int | None = None,
        limit: int | None = None,
    ) -> list[MemoryNode]:
        """Return events closest to ``reference`` by interval distance."""
        if not reference:
            return []
        distance_sql = """CASE
            WHEN event_time_start <= ?
             AND COALESCE(event_time_end,event_time_start) >= ? THEN 0.0
            WHEN COALESCE(event_time_end,event_time_start) < ?
              THEN julianday(?) - julianday(COALESCE(event_time_end,event_time_start))
            ELSE julianday(event_time_start) - julianday(?) END"""
        params: list[Any] = [
            reference, reference, reference, reference, reference, namespace,
        ]
        type_clause = ""
        if node_type:
            type_clause = " AND node_type=?"
            params.append(node_type)
        distance_clause = ""
        if max_distance_days is not None:
            distance_clause = " AND time_distance <= ?"
            params.append(max(0, int(max_distance_days)))
        limit_clause = ""
        if limit is not None:
            limit_clause = " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT * FROM (
                       SELECT n.*, {distance_sql} AS time_distance
                       FROM v4_memory_nodes n
                       WHERE namespace=? AND node_type!='chain_head'
                         AND event_time_start IS NOT NULL
                         {_reliable_time_condition('n.')}
                         {type_clause}
                   ) WHERE 1=1{distance_clause}
                   ORDER BY time_distance ASC,event_time_start ASC,id{limit_clause}""",
                params,
            ).fetchall()
        return [self._node(row) for row in rows]

    # ----- cross-scope EVENT edges (persisted, #7) -----

    def _validate_cross_scope_edge(
        self, edge: MemoryEdge, target_scope_id: str
    ) -> None:
        if edge.scope_id == target_scope_id:
            raise ValueError("cross-scope edge endpoints must use distinct scopes")
        source_node = self.get_node(edge.source_id, edge.scope_id)
        target_node = self.get_node(edge.target_id, target_scope_id)
        if source_node is None or target_node is None:
            raise ValueError("cross-scope edge endpoints must exist in their scopes")
        source_scope_participants = {
            item.participant_id for item in self.scope_participants(edge.scope_id)
        }
        target_scope_participants = {
            item.participant_id for item in self.scope_participants(target_scope_id)
        }
        if not is_cross_scope_visible(
            source_scope_participants,
            self.node_evidence_participants(edge.source_id),
            target_scope_participants,
            self.node_evidence_participants(edge.target_id),
        ):
            raise ValueError("cross-scope edge violates the visibility contract")

    def _upsert_cross_scope_edge_row(
        self, edge: MemoryEdge, target_scope_id: str
    ) -> None:
        metadata = dict(edge.metadata)
        metadata.setdefault("target_scope_id", target_scope_id)
        self.conn.execute(
            """INSERT INTO v4_cross_scope_edges(
                   id,namespace,source_scope_id,target_scope_id,source_id,target_id,edge_type,
                   direction,confidence,trust,evidence_refs_json,explanation,metadata_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET confidence=excluded.confidence, trust=excluded.trust,
                 evidence_refs_json=excluded.evidence_refs_json, explanation=excluded.explanation,
                 metadata_json=excluded.metadata_json""",
            (
                edge.id,
                edge.namespace,
                edge.scope_id,
                target_scope_id,
                edge.source_id,
                edge.target_id,
                edge.edge_type,
                edge.direction,
                edge.confidence,
                edge.trust,
                _dump(edge.evidence_refs),
                edge.explanation,
                _dump(metadata),
            ),
        )

    def replace_cross_scope_edges_for_scope(
        self,
        namespace: str,
        scope_id: str,
        edges: Iterable[tuple[MemoryEdge, str]],
    ) -> None:
        """Validate and replace all edges touching one scope in one transaction."""
        if self.read_only:
            raise RuntimeError("cannot publish through a read-only V4 store")
        replacements = [(edge, str(target_scope_id)) for edge, target_scope_id in edges]
        with self._lock:
            self._begin_immediate()
            try:
                seen_ids: set[str] = set()
                for edge, target_scope_id in replacements:
                    if edge.id in seen_ids:
                        raise ValueError(f"duplicate cross-scope edge id: {edge.id}")
                    seen_ids.add(edge.id)
                    if edge.namespace != namespace:
                        raise ValueError(
                            "cross-scope edge namespace does not match replacement"
                        )
                    if scope_id not in {edge.scope_id, target_scope_id}:
                        raise ValueError(
                            "replacement edge does not touch the requested scope"
                        )
                    self._validate_cross_scope_edge(edge, target_scope_id)
                self.conn.execute(
                    """DELETE FROM v4_cross_scope_edges
                       WHERE namespace=?
                         AND (source_scope_id=? OR target_scope_id=?)""",
                    (namespace, scope_id, scope_id),
                )
                for edge, target_scope_id in replacements:
                    self._upsert_cross_scope_edge_row(edge, target_scope_id)
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise

    def cross_scope_adjacent(
        self, namespace: str, scope_id: str, node_ids: Iterable[str],
        edge_types: Iterable[str] | None = None, direction: str = "both",
    ) -> list[tuple[MemoryEdge, str, str]]:
        """Return (edge, neighbor_id, neighbor_scope_id) for cross-scope edges.

        Visibility is enforced at atomic write time, so no read-time filtering here.
        """
        ids = list(dict.fromkeys(str(value) for value in node_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        clauses = ["namespace=?", "(source_scope_id=? OR target_scope_id=?)"]
        params: list[Any] = [namespace, scope_id, scope_id]
        if direction == "out":
            clauses.append(f"source_id IN ({placeholders})")
            params.extend(ids)
        elif direction == "in":
            clauses.append(f"target_id IN ({placeholders})")
            params.extend(ids)
        else:
            clauses.append(f"(source_id IN ({placeholders}) OR target_id IN ({placeholders}))")
            params.extend(ids)
            params.extend(ids)
        types = [str(value).upper() for value in (edge_types or [])]
        if types:
            clauses.append(f"edge_type IN ({','.join('?' for _ in types)})")
            params.extend(types)
        where = " AND ".join(clauses)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM v4_cross_scope_edges WHERE {where} ORDER BY id", params
            ).fetchall()
        requested = set(ids)
        output: list[tuple[MemoryEdge, str, str]] = []
        for row in rows:
            edge = self._cross_scope_edge(row)
            if edge.source_id in requested:
                neighbor_id = edge.target_id
                neighbor_scope = str(row["target_scope_id"])
            else:
                neighbor_id = edge.source_id
                neighbor_scope = edge.scope_id
            output.append((edge, neighbor_id, neighbor_scope))
        return output

    @staticmethod
    def _cross_scope_edge(row: sqlite3.Row) -> MemoryEdge:
        metadata = _load(row["metadata_json"], {})
        return MemoryEdge(
            id=str(row["id"]), namespace=str(row["namespace"]), scope_id=str(row["source_scope_id"]),
            source_id=str(row["source_id"]), target_id=str(row["target_id"]), edge_type=str(row["edge_type"]),
            direction=str(row["direction"]), confidence=float(row["confidence"]), trust=str(row["trust"]),
            evidence_refs=_load(row["evidence_refs_json"], []), created_method="cross_scope_builder",
            explanation=str(row["explanation"]), metadata=metadata,
        )

    def search_fts(self, scope_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        terms = [token for token in query.replace('"', " ").split() if token]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms[:16])
        return self._search_fts_expression(scope_id, expression, limit)

    def search_fts_structured(
        self,
        scope_id: str,
        must_terms: Sequence[str],
        should_terms: Sequence[str],
        limit: int,
    ) -> list[tuple[str, float]]:
        """Search strict and broad lexical probes without accepting raw FTS syntax."""
        must = _safe_fts_terms(must_terms, 4)
        should = _safe_fts_terms(should_terms, 8)
        strict = self._search_fts_expression(
            scope_id,
            " AND ".join(f'"{term}"' for term in must),
            limit,
        ) if must else []
        broad_terms = list(dict.fromkeys([*must, *should]))
        broad = self._search_fts_expression(
            scope_id,
            " OR ".join(f'"{term}"' for term in broad_terms),
            limit,
        ) if broad_terms else []
        if not strict:
            return broad
        strict_scores = dict(strict)
        broad_scores = dict(broad)
        node_ids = list(dict.fromkeys([*strict_scores, *broad_scores]))
        combined = [
            (
                node_id,
                0.7 * strict_scores.get(node_id, 0.0)
                + 0.3 * broad_scores.get(node_id, 0.0),
            )
            for node_id in node_ids
        ]
        return sorted(combined, key=lambda item: (-item[1], item[0]))[: int(limit)]

    def _search_fts_expression(
        self, scope_id: str, expression: str, limit: int
    ) -> list[tuple[str, float]]:
        if not expression:
            return []
        try:
            with self._lock:
                rows = self.conn.execute(
                    "SELECT node_id,bm25(v4_memory_fts) AS rank FROM v4_memory_fts WHERE scope_id=? AND v4_memory_fts MATCH ? ORDER BY rank LIMIT ?",
                    (scope_id, expression, int(limit)),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        # FTS5 bm25() returns lower (normally negative) values for better matches.
        # Convert to positive relevance and normalize within this query so fusion
        # preserves BM25 differences without depending on corpus-dependent scale.
        relevance = [max(0.0, -float(row["rank"])) for row in rows]
        scale = max(relevance, default=0.0)
        if scale <= 0.0:
            return [(str(row["node_id"]), 0.0) for row in rows]
        return [
            (str(row["node_id"]), score / scale)
            for row, score in zip(rows, relevance)
        ]

    def search_entities(self, scope_id: str, entities: Iterable[str], limit: int) -> list[tuple[str, float]]:
        values = list(dict.fromkeys(str(value).casefold() for value in entities if str(value).strip()))
        if not values:
            return []
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT node_id,COUNT(*) AS matches FROM v4_node_entities
                    WHERE scope_id=? AND lower(entity) IN ({','.join('?' for _ in values)})
                    GROUP BY node_id ORDER BY matches DESC,node_id LIMIT ?""",
                [scope_id, *values, int(limit)],
            ).fetchall()
        denominator = max(1, len(values))
        return [(str(row["node_id"]), float(row["matches"]) / denominator) for row in rows]

    def _scope(self, row: sqlite3.Row) -> ParticipantScope:
        participants = self.conn.execute(
            "SELECT participant_id,role FROM v4_scope_participants WHERE scope_id=? ORDER BY role,participant_id",
            (row["id"],),
        ).fetchall()
        return ParticipantScope(
            id=str(row["id"]), namespace=str(row["namespace"]),
            participants=[Participant(str(value["participant_id"]), str(value["role"])) for value in participants],
            visibility_policy=str(row["visibility_policy"]), metadata=_load(row["metadata_json"], {}),
            revision=int(row["revision"]),
        )

    @staticmethod
    def _chain(row: sqlite3.Row, aliases: list[str]) -> TopicChain:
        return TopicChain(
            id=str(row["id"]), namespace=str(row["namespace"]), scope_id=str(row["scope_id"]),
            topic_id=str(row["topic_id"]), name=str(row["name"]), description=str(row["description"]),
            aliases=aliases, rolling_summary=str(row["rolling_summary"]),
            representative_entities=_load(row["representative_entities_json"], []),
            metadata=_load(row["metadata_json"], {}),
        )

    @staticmethod
    def _node(row: sqlite3.Row) -> MemoryNode:
        return MemoryNode(
            id=str(row["id"]), namespace=str(row["namespace"]), scope_id=str(row["scope_id"]),
            chain_id=str(row["chain_id"]), topic_id=str(row["topic_id"]), node_type=str(row["node_type"]),
            title=str(row["title"]), summary=str(row["summary"]), text=str(row["text"]),
            actors=_load(row["actors_json"], []), action=str(row["action"]),
            objects=_load(row["objects_json"], []), location=row["location"],
            event_time_start=row["event_time_start"], event_time_end=row["event_time_end"],
            time_precision=row["time_precision"], observed_at=row["observed_at"], valid_from=row["valid_from"],
            valid_to=row["valid_to"], facts=_load(row["facts_json"], []),
            changed_keys=_load(row["changed_keys_json"], []), previous_state_id=row["previous_state_id"],
            trigger_event_ids=_load(row["trigger_event_ids_json"], []), entities=_load(row["entities_json"], []),
            entity_ids=_load(row["entity_ids_json"], []),
            importance=float(row["importance"]), confidence=float(row["confidence"]),
            evidence_refs=_load(row["evidence_refs_json"], []), embedding=_load(row["embedding_json"], None),
            metadata=_load(row["metadata_json"], {}),
        )

    @staticmethod
    def _edge(row: sqlite3.Row) -> MemoryEdge:
        return MemoryEdge(
            id=str(row["id"]), namespace=str(row["namespace"]), scope_id=str(row["scope_id"]),
            source_id=str(row["source_id"]), target_id=str(row["target_id"]), edge_type=str(row["edge_type"]),
            direction=str(row["direction"]), confidence=float(row["confidence"]), trust=str(row["trust"]),
            evidence_refs=_load(row["evidence_refs_json"], []), created_method=str(row["created_method"]),
            valid_from=row["valid_from"], valid_to=row["valid_to"], explanation=str(row["explanation"]),
            metadata=_load(row["metadata_json"], {}),
        )

    @staticmethod
    def _fact(row: sqlite3.Row) -> SemanticFact:
        return SemanticFact(
            id=str(row["id"]), namespace=str(row["namespace"]), scope_id=str(row["scope_id"]),
            chain_id=str(row["chain_id"]), state_node_id=str(row["state_node_id"]),
            key=str(row["fact_key"]), subject=str(row["subject"]), predicate=str(row["predicate"]),
            dimension=str(row["dimension"]), aspect=str(row["aspect"]),
            value=_load(row["value_json"], None), confidence=float(row["confidence"]),
            evidence_refs=_load(row["evidence_refs_json"], []), valid_from=row["valid_from"], valid_to=row["valid_to"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            history_origin=row["history_origin"], status=str(row["status"]),
        )
