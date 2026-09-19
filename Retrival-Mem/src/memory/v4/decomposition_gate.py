"""Question-only binary gate for V4 initial subproblem decomposition."""

from __future__ import annotations

import json

from memory.clients import ChatClient
from memory.prompt_safety import UNTRUSTED_DATA_INSTRUCTION
from memory.structured_output import (
    JsonSchema,
    messages_with_json_schema,
    object_schema,
    parse_json_object_with_schema,
)


DECOMPOSITION_GATE_PRE_ANCHOR_REFINEMENT_SYSTEM_PROMPT = (
    "Decide conservatively whether answering the supplied question requires "
    "retrieving and combining at least two distinct facts through separate "
    "subproblems. Return true for comparisons across entities or events, "
    "elapsed-time calculations requiring two anchors, explicit multi-hop "
    "questions, or broad aggregation across multiple events. Also return true "
    "when a question asks for all events, activities, or ways involving one "
    "person across multiple times and the expected answer contains multiple "
    "independent items. A question may request a single conclusion and still "
    "require decomposition when a reliable inference depends on complementary "
    "independent evidence dimensions, such as behaviors, preferences, skills, "
    "constraints, long-term patterns, or competing hypotheses. Return false for "
    "a single fact about one subject or event and for an inference that needs only "
    "one direct fact plus ordinary common knowledge. Do not return true merely "
    "because a question is long, hypothetical, or contains several nouns. When "
    "uncertain, return false. Return only the schema fields. "
    + UNTRUSTED_DATA_INSTRUCTION
)


DECOMPOSITION_GATE_SYSTEM_PROMPT = (
    "Decide conservatively whether answering the supplied question requires "
    "retrieving and combining at least two distinct facts through separate "
    "subproblems. Return true for comparisons across entities or events, "
    "explicit multi-hop questions, or broad aggregation across multiple events. "
    "For elapsed-time calculations, return true only when both time anchors must "
    "be independently retrieved. Return false when one anchor is supplied by the "
    "question or conversation time and only ordinary arithmetic is needed. Also "
    "return true when a question asks for all events, activities, or ways involving "
    "one person across multiple times and the expected answer contains multiple "
    "independent items. A question may request a single conclusion and still "
    "require decomposition when a reliable inference depends on complementary "
    "independent evidence dimensions, such as behaviors, preferences, skills, "
    "constraints, long-term patterns, or competing hypotheses. Return false for "
    "a single fact about one subject or event and for an inference that needs only "
    "one direct fact plus ordinary common knowledge. Do not return true merely "
    "because a question is long, hypothetical, or contains several nouns. When "
    "uncertain, return false. Return only the schema fields. "
    + UNTRUSTED_DATA_INSTRUCTION
)

DECOMPOSITION_GATE_JSON_SCHEMA: JsonSchema = object_schema(
    {"needs_decomposition": {"type": "boolean"}},
    required=("needs_decomposition",),
)


class V4DecompositionGate:
    """Classify whether the initial Controller should decompose a question."""

    def __init__(self, client: ChatClient):
        self.client = client

    def needs_decomposition(self, question: str) -> bool:
        payload = {"question": str(question).strip()}
        raw = self.client.chat(
            messages_with_json_schema(
                [
                    {
                        "role": "system",
                        "content": DECOMPOSITION_GATE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                DECOMPOSITION_GATE_JSON_SCHEMA,
            ),
            json_mode=True,
            json_schema=DECOMPOSITION_GATE_JSON_SCHEMA,
        )
        value = parse_json_object_with_schema(
            raw,
            DECOMPOSITION_GATE_JSON_SCHEMA,
            source="V4 decomposition gate",
        )
        return value["needs_decomposition"]


__all__ = [
    "DECOMPOSITION_GATE_JSON_SCHEMA",
    "DECOMPOSITION_GATE_PRE_ANCHOR_REFINEMENT_SYSTEM_PROMPT",
    "DECOMPOSITION_GATE_SYSTEM_PROMPT",
    "V4DecompositionGate",
]
