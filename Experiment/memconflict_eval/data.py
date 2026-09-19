"""MemConflict dataset reading (point 7, input side).

Replaces the LoCoMo reader used by the Retrival-Mem harness. The differences
that matter for the experiment:

* LoCoMo is one flat conversation per example, answered once at the end.
  MemConflict is a *chain* of dated sessions per persona, and each session
  carries its own ``Session_Questions`` that must be answered once that session
  has been ingested.
* ``Session_Dialogue`` is a dict keyed ``dialogue_turn_1..N``, so the keys must
  be sorted numerically (lexicographic order puts ``turn_10`` before ``turn_2``).
* Sessions, conflict annotations and questions are all optional per session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


CONFLICT_TYPES = ("dynamic_conflict", "static_conflict", "conditional_conflict")


@dataclass(frozen=True)
class DialogueTurn:
    turn_id: int
    role: str
    content: str
    session_date: str

    def to_memory_turn(self) -> dict[str, Any]:
        """Turn payload for the V4 builder.

        ``turn_id`` must be a *string*. With a bare JSON number the builder
        model tends to invent a prefix (``"turn_33"``), which then fails the
        builder's evidence-id validation because ``turn_33`` is not a supplied
        id. A self-describing string is copied verbatim, which is also what the
        upstream LoCoMo path does (it uses ``dia_id`` strings such as ``D1:1``).
        """
        return {
            "turn_id": f"turn_{self.turn_id}",
            "role": self.role,
            "speaker": self.role,
            "content": self.content,
            "timestamp": self.session_date,
            "session_timestamp": self.session_date,
        }


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    answer: str
    conflict_type: str
    ability_target: str
    difficulty: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Question":
        return cls(
            question_id=str(raw.get("question_id") or ""),
            question=str(raw.get("question") or ""),
            answer=str(raw.get("answer") or ""),
            conflict_type=str(raw.get("conflict_type") or ""),
            ability_target=str(raw.get("ability_target") or ""),
            difficulty=str(raw.get("difficulty") or ""),
        )

    @property
    def key(self) -> str:
        """Stable per-question key.

        ``question_id`` alone is NOT unique: it restarts for every conflict
        family inside a persona (``Q_001`` appears once per family), so the
        conflict type must be part of the key.
        """
        return f"{self.conflict_type}:{self.question_id}"


@dataclass(frozen=True)
class Session:
    session_id: int
    date: str
    session_type: str
    dialogue: tuple[DialogueTurn, ...]
    questions: tuple[Question, ...]
    question_trigger_types: tuple[str, ...]
    event_types: tuple[str, ...]
    updated_attributes: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    static_conflict_information: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    conditional_conflict_information: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    others_dynamic_information: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    outline: str = ""

    def to_memory_session(self) -> dict[str, Any]:
        """Session payload accepted by the V4 builder's turn normalizer."""
        return {
            "session_id": str(self.session_id),
            "timestamp": self.date,
            "session_timestamp": self.date,
            "turns": [turn.to_memory_turn() for turn in self.dialogue],
        }


@dataclass(frozen=True)
class Persona:
    persona_id: str
    sessions: tuple[Session, ...]
    raw: dict[str, Any]

    @property
    def question_count(self) -> int:
        return sum(len(session.questions) for session in self.sessions)

    def questions_by_type(self) -> dict[str, int]:
        counts = {conflict_type: 0 for conflict_type in CONFLICT_TYPES}
        for session in self.sessions:
            for question in session.questions:
                counts[question.conflict_type] = counts.get(question.conflict_type, 0) + 1
        return counts


def _dialogue_turn_order(key: str) -> int:
    try:
        return int(str(key).rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return 10**9


def flatten_session_dialogue(session_dialogue: Any, session_date: str) -> tuple[DialogueTurn, ...]:
    """Flatten ``Session_Dialogue`` into chronological user/assistant turns."""
    if not isinstance(session_dialogue, dict):
        return ()

    turns: list[DialogueTurn] = []
    for turn_key in sorted(session_dialogue, key=_dialogue_turn_order):
        group = session_dialogue.get(turn_key)
        if not isinstance(group, list):
            continue
        for message in group:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role not in ("user", "assistant") or content in (None, ""):
                continue
            turns.append(
                DialogueTurn(
                    turn_id=len(turns) + 1,
                    role=str(role),
                    content=str(content),
                    session_date=session_date,
                )
            )
    return tuple(turns)


def _tuple_of_dicts(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def parse_session(raw: dict[str, Any], fallback_index: int) -> Session:
    date = str(raw.get("Date") or "")
    session_id = raw.get("Session_ID")
    if not isinstance(session_id, int):
        session_id = fallback_index
    questions = tuple(
        Question.from_dict(item)
        for item in (raw.get("Session_Questions") or [])
        if isinstance(item, dict)
    )
    return Session(
        session_id=session_id,
        date=date,
        session_type=str(raw.get("Session_Type") or ""),
        dialogue=flatten_session_dialogue(raw.get("Session_Dialogue"), date),
        questions=questions,
        question_trigger_types=tuple(str(x) for x in (raw.get("Question_Trigger_Types") or [])),
        event_types=tuple(str(x) for x in (raw.get("Event_Types") or [])),
        updated_attributes=_tuple_of_dicts(raw.get("Updated_Attributes")),
        static_conflict_information=_tuple_of_dicts(raw.get("Static_Conflict_Information")),
        conditional_conflict_information=_tuple_of_dicts(
            raw.get("Conditional_Conflict_Information")
        ),
        others_dynamic_information=_tuple_of_dicts(raw.get("Others_Dynamic_Information")),
        outline=str(raw.get("Session_Outline") or ""),
    )


def parse_persona(raw: dict[str, Any]) -> Persona:
    chain = raw.get("Full_Session_Chain")
    if not isinstance(chain, list):
        raise ValueError("persona record has no Full_Session_Chain list")
    sessions = [
        parse_session(item, index)
        for index, item in enumerate(chain)
        if isinstance(item, dict)
    ]
    # The release is already chronological, but the experiment must not depend
    # on that: the per-session QA order is what defines the benchmark.
    sessions.sort(key=lambda session: (session.date, session.session_id))
    return Persona(
        persona_id=str(raw.get("ID") or ""),
        sessions=tuple(sessions),
        raw=raw,
    )


def iter_personas(path: str | Path) -> Iterator[Persona]:
    """Yield personas from a MemConflict JSONL release."""
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            yield parse_persona(raw)


def load_personas(
    path: str | Path,
    *,
    start_index: int = 0,
    end_index: int | None = None,
) -> list[Persona]:
    personas: list[Persona] = []
    for index, persona in enumerate(iter_personas(path)):
        if index < start_index:
            continue
        if end_index is not None and index >= end_index:
            break
        personas.append(persona)
    return personas


def dataset_summary(personas: list[Persona]) -> dict[str, Any]:
    """Cheap sanity summary used by the CLI and by tests."""
    question_counts = {conflict_type: 0 for conflict_type in CONFLICT_TYPES}
    sessions = 0
    questions = 0
    triggered_sessions = 0
    for persona in personas:
        for session in persona.sessions:
            sessions += 1
            questions += len(session.questions)
            if session.questions:
                triggered_sessions += 1
            for question in session.questions:
                question_counts[question.conflict_type] = (
                    question_counts.get(question.conflict_type, 0) + 1
                )
    return {
        "Persona_Count": len(personas),
        "Session_Count": sessions,
        "Triggered_Session_Count": triggered_sessions,
        "Question_Count": questions,
        "Question_Count_By_Conflict_Type": question_counts,
    }
