from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from typing import Any, Mapping, Sequence

import tiktoken
from tiktoken.core import Encoding

from memory.prompt_safety import UNTRUSTED_DATA_INSTRUCTION
from memory.structured_output import (
    array_schema,
    nullable,
    object_schema,
    string_array,
)
from memory.v4.schemas import EDGE_TYPE_DESCRIPTIONS, EVENT_EDGE_TYPES


MEMORY_BUILDER_PROMPT_VERSION = "memory-builder-v3"

BUILDER_SYSTEM_MESSAGE = (
    "Return only valid JSON grounded in supplied evidence. "
    + UNTRUSTED_DATA_INSTRUCTION
)

EVENT_RELATION_SEMANTICS = {
    edge_type: EDGE_TYPE_DESCRIPTIONS[edge_type]
    for edge_type in sorted(EVENT_EDGE_TYPES)
}

_BUILDER_INSTRUCTIONS = [
    "Apply a high-recall question-answering evidence gate to this complete conversation window.",
    "Set write=true whenever any supplied turn contains concrete, grounded content that could support a future question, even when the detail is temporary or mentioned only once.",
    "Extract askable facts, events, experiences, feelings, opinions, reasons, advice, plans, activities, locations, possessions, media, and interpersonal interactions.",
    "Treat specific advice or recommendations given or received by any participant, including an assistant, as askable evidence and attribute them to the correct speaker.",
    "Treat explicitly supplied image captions or image descriptions as askable evidence, but do not infer visual details that are not stated in the supplied text.",
    "Copy every exact short string that a future question could quote verbatim into the event's exact_mentions field: named entities, book/work/media titles, slogans, short quotes, numbers, dates, and short image caption phrases. Keep each entry at most 120 characters; do not copy full-sentence quotes or whole image captions because their original wording is already preserved in the supplied evidence. Preserve the original wording character for character without paraphrasing or summarizing; include only strings that appear literally in the supplied turns.",
    "For every substantive turn, ensure at least one event or candidate_claims entry cites that turn in evidence_turn_ids; skip a turn only when it is purely a greeting, acknowledgement, content-free repetition, or has no reliably attributable information.",
    "Include all directly supporting, elaborating, or correcting supplied turn IDs in evidence_turn_ids instead of citing only the main turn; never cite unrelated turns.",
    "Preserve negation, tense, and modality. A plan or intention must remain a plan and must not be represented as an event that already happened.",
    "Resolve pronouns from the turn speaker and supplied participant information; if attribution is uncertain, do not guess the subject.",
    "Attribute quoted or reported content to its original subject, not automatically to the current speaker.",
    "Extract atomic events rather than one node per turn, and do not discard secondary details merely because another event is more important.",
    "Keep event_time distinct from observed_at and cite only supplied turn IDs.",
    "If the source explicitly states an absolute date, copy it to event_time_start using ISO 8601 and preserve its original precision.",
    "If the source uses a relative or approximate time expression, do not resolve or guess an absolute date. Set event_time_start and event_time_end to null, and copy the exact phrase to raw_time_expression.",
    "Never convert vague quantifiers such as \"few\", \"several\", \"some time\", \"recently\", or \"a while ago\" into fixed numeric offsets.",
    "Use the coarsest precision that fits an explicitly stated absolute date: prefer year/month/date over datetime. Do not include a time-of-day in event_time_start unless the utterance explicitly states one, and never copy the time component of observed_at into event_time.",
    "Always provide time_precision when event_time_start is provided.",
    "Represent temporary or one-off askable details as events; represent stable preferences, health, relationships, identity, and long-term plans as candidate_claims.",
    "Apply the same absolute-only time policy to candidate_claims.valid_from: copy an explicitly stated absolute date as ISO 8601 with its original precision; for a relative or approximate expression, leave valid_from null and copy the exact phrase to raw_time_expression.",
    "For each relation use exactly two distinct supplied event refs and one of CAUSES, CONTRIBUTES_TO, CONTEXT_FOR, ENABLES, PART_OF, FOLLOW_UP_OF, or SAME_EVENT in the documented direction.",
    "Return at most one relation per ordered source-target event pair; if several types seem applicable, keep only the highest-confidence relation.",
    "Do not decide whether a claim reinforces, extends, revises, or conflicts with history.",
    "Do not invent facts or evidence.",
    UNTRUSTED_DATA_INSTRUCTION,
]

MEMORY_BUILDER_JSON_SCHEMA = object_schema(
    {
        "write": {"type": "boolean"},
        "events": array_schema(
            object_schema(
                {
                    "ref": {"type": "string"},
                    "topic_id": {"type": "string"},
                    "topic_name": {"type": "string"},
                    "topic_description": {"type": "string"},
                    "topic_aliases": string_array(),
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "actors": string_array(),
                    "action": {"type": "string"},
                    "objects": string_array(),
                    "location": nullable({"type": "string"}),
                    "entities": string_array(),
                    "exact_mentions": string_array(),
                    "event_time_start": nullable({"type": "string"}),
                    "event_time_end": nullable({"type": "string"}),
                    "time_precision": nullable({"type": "string"}),
                    "raw_time_expression": nullable({"type": "string"}),
                    "observed_at": nullable({"type": "string"}),
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "evidence_turn_ids": string_array(min_items=1),
                },
                required=(
                    "ref",
                    "topic_id",
                    "topic_name",
                    "topic_description",
                    "topic_aliases",
                    "title",
                    "summary",
                    "actors",
                    "action",
                    "objects",
                    "location",
                    "entities",
                    "event_time_start",
                    "event_time_end",
                    "time_precision",
                    "raw_time_expression",
                    "observed_at",
                    "importance",
                    "confidence",
                    "evidence_turn_ids",
                ),
            )
        ),
        "candidate_claims": array_schema(
            object_schema(
                {
                    "topic_id": {"type": "string"},
                    "claims": array_schema(
                        object_schema(
                            {
                                "subject": {"type": "string"},
                                "dimension": {"type": "string"},
                                "aspect": {"type": "string"},
                                "value": {},
                            },
                            required=("subject", "dimension", "aspect", "value"),
                        ),
                        min_items=1,
                    ),
                    "valid_from": nullable({"type": "string"}),
                    "trigger_event_ref": nullable({"type": "string"}),
                    "time_precision": nullable(
                        {
                            "type": "string",
                            "enum": [
                                "datetime",
                                "date",
                                "month",
                                "year",
                                "range",
                                "season",
                                "unknown",
                            ],
                        }
                    ),
                    "raw_time_expression": nullable({"type": "string"}),
                    "evidence_turn_ids": string_array(min_items=1),
                },
                required=(
                    "topic_id",
                    "claims",
                    "valid_from",
                    "trigger_event_ref",
                    "time_precision",
                    "raw_time_expression",
                    "evidence_turn_ids",
                ),
            )
        ),
        "relations": array_schema(
            object_schema(
                {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "edge_type": {
                        "type": "string",
                        "enum": sorted(EVENT_EDGE_TYPES),
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                required=("source", "target", "edge_type", "confidence"),
            )
        ),
    },
    required=("write", "events", "candidate_claims", "relations"),
)


class TokenEncodingError(RuntimeError):
    """The configured exact token encoding could not be loaded."""


@lru_cache(maxsize=8)
def get_token_encoding(name: str) -> Encoding:
    encoding_name = str(name).strip()
    if not encoding_name:
        raise TokenEncodingError("token_encoding must not be empty")
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:
        raise TokenEncodingError(
            f"unable to load required tiktoken encoding {encoding_name!r}"
        ) from exc


def turn_observed_at(turn: Mapping[str, Any]) -> str | None:
    value = (
        turn.get("timestamp")
        or turn.get("datetime")
        or turn.get("date")
        or turn.get("session_timestamp")
    )
    text = "" if value is None else str(value).strip()
    return text or None


def build_memory_builder_payload(
    session_id: str,
    turns: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the one canonical V4 extraction payload without side effects."""
    public_turns = []
    for source_turn in turns:
        copied = dict(source_turn)
        copied["observed_at"] = turn_observed_at(source_turn)
        public_turns.append(copied)
    return {
        "task": "memory_window_extraction",
        "session_id": session_id,
        "metadata": dict(metadata or {}),
        "turns": public_turns,
        "instructions": list(_BUILDER_INSTRUCTIONS),
        "event_relation_semantics": dict(EVENT_RELATION_SEMANTICS),
        "output_schema": deepcopy(MEMORY_BUILDER_JSON_SCHEMA),
    }


def build_memory_builder_messages(
    session_id: str,
    turns: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    """Build the exact system/user messages used for planning and extraction."""
    payload = build_memory_builder_payload(session_id, turns, metadata)
    return [
        {"role": "system", "content": BUILDER_SYSTEM_MESSAGE},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def count_chat_tokens(
    messages: Sequence[Mapping[str, Any]],
    encoding_name: str,
) -> int:
    """Count message values plus conservative role/framing overhead."""
    encoding = get_token_encoding(encoding_name)
    total = 3  # Conservative assistant-reply priming.
    for message in messages:
        total += 6  # Per-message framing and separators.
        for key, value in message.items():
            total += len(encoding.encode(str(key)))
            total += len(encoding.encode(str(value)))
    return total


def count_builder_input_tokens(
    session_id: str,
    turns: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None,
    encoding_name: str,
) -> int:
    return count_chat_tokens(
        build_memory_builder_messages(session_id, turns, metadata),
        encoding_name,
    )


__all__ = [
    "BUILDER_SYSTEM_MESSAGE",
    "EVENT_RELATION_SEMANTICS",
    "MEMORY_BUILDER_JSON_SCHEMA",
    "MEMORY_BUILDER_PROMPT_VERSION",
    "TokenEncodingError",
    "build_memory_builder_messages",
    "build_memory_builder_payload",
    "count_builder_input_tokens",
    "count_chat_tokens",
    "get_token_encoding",
    "turn_observed_at",
]
