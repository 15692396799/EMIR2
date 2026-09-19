from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class RerankCheckpointStore:
    """Transactional, resumable checkpoints for final rerank batches."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path, timeout=30.0, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout=30000")
        if self.path != ":memory:":
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rerank_batch_checkpoints (
                checkpoint_key TEXT NOT NULL,
                batch_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                output_json TEXT,
                error_json TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(checkpoint_key, batch_index)
            )
            """
        )
        self.connection.execute("BEGIN IMMEDIATE")
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(rerank_batch_checkpoints)")}
        if "fingerprint" in columns:
            self.connection.execute("ALTER TABLE rerank_batch_checkpoints DROP COLUMN fingerprint")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def invalidate(self, checkpoint_key: str) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM rerank_batch_checkpoints WHERE checkpoint_key=?",
                (checkpoint_key,),
            )

    def load_success(
        self, checkpoint_key: str, batch_index: int
    ) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT output_json FROM rerank_batch_checkpoints
            WHERE checkpoint_key=? AND batch_index=?
              AND status='succeeded'
            """,
            (checkpoint_key, int(batch_index)),
        ).fetchone()
        if row is None or row["output_json"] is None:
            return None
        value = json.loads(str(row["output_json"]))
        if not isinstance(value, dict):
            raise ValueError("Corrupt rerank batch checkpoint output")
        return value

    def record_success(
        self,
        checkpoint_key: str,
        batch_index: int,
        output: Mapping[str, Any],
    ) -> None:
        self._record(
            checkpoint_key,
            batch_index,
            "succeeded",
            output=dict(output),
            error=None,
        )

    def record_failure(
        self,
        checkpoint_key: str,
        batch_index: int,
        error: Exception,
    ) -> None:
        self._record(
            checkpoint_key,
            batch_index,
            "failed",
            output=None,
            error={"type": type(error).__name__, "message": str(error)},
        )

    def _record(
        self,
        checkpoint_key: str,
        batch_index: int,
        status: str,
        *,
        output: Mapping[str, Any] | None,
        error: Mapping[str, Any] | None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO rerank_batch_checkpoints(
                    checkpoint_key,batch_index,status,
                    output_json,error_json,updated_at
                ) VALUES(?,?,?,?,?,?)
                ON CONFLICT(checkpoint_key,batch_index) DO UPDATE SET
                    status=excluded.status,
                    output_json=excluded.output_json,
                    error_json=excluded.error_json,
                    updated_at=excluded.updated_at
                """,
                (
                    checkpoint_key,
                    int(batch_index),
                    status,
                    json.dumps(output, ensure_ascii=False, sort_keys=True)
                    if output is not None
                    else None,
                    json.dumps(error, ensure_ascii=False, sort_keys=True)
                    if error is not None
                    else None,
                    now,
                ),
            )


__all__ = ["RerankCheckpointStore"]
