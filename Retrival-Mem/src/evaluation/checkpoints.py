from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


QUESTION_STAGES = frozenset({"retrieval", "answer", "temporal_correction", "scoring"})


class QuestionStageLedger:
    """Transactional per-question stage state shared by evaluation workers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(self.path), timeout=30.0, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS question_stage_checkpoints (
                question_key TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                cumulative_attempts INTEGER NOT NULL DEFAULT 0,
                output_json TEXT,
                error_json TEXT,
                attempt_history_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL,
                PRIMARY KEY(question_key, stage)
            )
            """
        )
        self._connection.execute("BEGIN IMMEDIATE")
        columns = {row[1] for row in self._connection.execute("PRAGMA table_info(question_stage_checkpoints)")}
        if "fingerprint" in columns:
            self._connection.execute("ALTER TABLE question_stage_checkpoints DROP COLUMN fingerprint")
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "QuestionStageLedger":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def load_success(
        self, question_key: str, stage: str
    ) -> dict[str, Any] | None:
        self._validate_stage(stage)
        row = self._connection.execute(
            """
            SELECT output_json
            FROM question_stage_checkpoints
            WHERE question_key=? AND stage=? AND status='succeeded'
            """,
            (question_key, stage),
        ).fetchone()
        if row is None or row["output_json"] is None:
            return None
        value = json.loads(str(row["output_json"]))
        if not isinstance(value, dict):
            raise ValueError(
                f"Corrupt question checkpoint output for {question_key}:{stage}"
            )
        return value

    def record_success(
        self,
        question_key: str,
        stage: str,
        output: Mapping[str, Any],
        *,
        attempts: int = 1,
    ) -> None:
        self._record(
            question_key,
            stage,
            status="succeeded",
            output=dict(output),
            error=None,
            attempts=attempts,
        )

    def record_failure(
        self,
        question_key: str,
        stage: str,
        error: Exception,
        *,
        attempts: int = 1,
    ) -> None:
        self._record(
            question_key,
            stage,
            status="failed",
            output=None,
            error={
                "type": type(error).__name__,
                "message": str(error),
            },
            attempts=attempts,
        )

    def get_state(self, question_key: str, stage: str) -> dict[str, Any] | None:
        self._validate_stage(stage)
        row = self._connection.execute(
            """
            SELECT question_key,stage,status,cumulative_attempts,
                   output_json,error_json,attempt_history_json,updated_at
            FROM question_stage_checkpoints
            WHERE question_key=? AND stage=?
            """,
            (question_key, stage),
        ).fetchone()
        if row is None:
            return None
        value = dict(row)
        for key in ("output_json", "error_json", "attempt_history_json"):
            raw = value.pop(key)
            value[key.removesuffix("_json")] = json.loads(raw) if raw else None
        return value

    def invalidate_question(self, question_key: str, *, keep_retrieval: bool = False) -> None:
        with self._connection:
            self._connection.execute(
                "DELETE FROM question_stage_checkpoints WHERE question_key=?"
                + (" AND stage != 'retrieval'" if keep_retrieval else ""),
                (question_key,),
            )

    def invalidate_stages(self, stages: set[str] | frozenset[str]) -> int:
        values = sorted(set(stages))
        for stage in values:
            self._validate_stage(stage)
        if not values:
            return 0
        placeholders = ",".join("?" for _ in values)
        with self._connection:
            cursor = self._connection.execute(
                f"DELETE FROM question_stage_checkpoints WHERE stage IN ({placeholders})",
                values,
            )
        return int(cursor.rowcount)

    def _record(
        self,
        question_key: str,
        stage: str,
        *,
        status: str,
        output: Mapping[str, Any] | None,
        error: Mapping[str, Any] | None,
        attempts: int,
    ) -> None:
        self._validate_stage(stage)
        attempt_count = max(0, int(attempts))
        now = datetime.now(timezone.utc).isoformat()
        with self._connection:
            row = self._connection.execute(
                """
                SELECT cumulative_attempts,attempt_history_json
                FROM question_stage_checkpoints
                WHERE question_key=? AND stage=?
                """,
                (question_key, stage),
            ).fetchone()
            cumulative = int(row["cumulative_attempts"]) if row is not None else 0
            history = (
                json.loads(str(row["attempt_history_json"]))
                if row is not None
                else []
            )
            history.append(
                {
                    "status": status,
                    "attempts": attempt_count,
                    "error": dict(error) if error is not None else None,
                    "recorded_at": now,
                }
            )
            self._connection.execute(
                """
                INSERT INTO question_stage_checkpoints(
                    question_key,stage,status,cumulative_attempts,
                    output_json,error_json,attempt_history_json,updated_at
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(question_key,stage) DO UPDATE SET
                    status=excluded.status,
                    cumulative_attempts=excluded.cumulative_attempts,
                    output_json=excluded.output_json,
                    error_json=excluded.error_json,
                    attempt_history_json=excluded.attempt_history_json,
                    updated_at=excluded.updated_at
                """,
                (
                    question_key,
                    stage,
                    status,
                    cumulative + attempt_count,
                    json.dumps(output, ensure_ascii=False, sort_keys=True)
                    if output is not None
                    else None,
                    json.dumps(error, ensure_ascii=False, sort_keys=True)
                    if error is not None
                    else None,
                    json.dumps(history, ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if stage not in QUESTION_STAGES:
            raise ValueError(f"Unsupported question checkpoint stage: {stage}")
