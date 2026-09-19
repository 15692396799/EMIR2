"""Structured extraction and deterministic arithmetic for duration answers."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from memory.clients import ChatClient, parse_json_object
from agent.evidence_answer import (
    EVIDENCE_SELECTION_GUIDANCE,
    evidence_extraction_schema,
    split_extraction_evidence,
)
from memory.prompt_safety import UNTRUSTED_DATA_INSTRUCTION
from memory.structured_output import (
    JsonSchema,
    enum_string,
    messages_with_json_schema,
    nullable,
    object_schema,
    parse_json_object_with_schema,
)


_DURATION_QUESTION_RE = re.compile(
    r"\bhow\s+many\s+(?:days?|weeks?|months?|years?)\b"
    r"|\bafter\s+how\s+many\s+(?:days?|weeks?|months?|years?)\b"
    r"|\bhow\s+long\s+(?:did|does|has|have|was|were|is|are|will|would)\b"
    r"|\bfor\s+how\s+long\b",
    flags=re.IGNORECASE,
)
_HOW_LONG_AGO_RE = re.compile(r"\bhow\s+long\s+ago\b", flags=re.IGNORECASE)
_TEMPORAL_VALUE_QUESTION_RE = re.compile(
    r"^\s*when\b"
    r"|\bwhat\s+(?:date|day|month|year)\b"
    r"|\bwhich\s+(?:date|day|month|year)\b"
    r"|\bin\s+which\s+(?:month|year)\b",
    flags=re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"^(?P<year>\d{4})(?:-(?P<month>\d{2})(?:-(?P<day>\d{2}))?)?$")
_ISO_DATE_SCHEMA_PATTERN = r"^\d{4}(-\d{2}(-\d{2})?)?$"
_ANSWER_TYPE_LABELS = {
    "activity",
    "count",
    "date",
    "duration",
    "event",
    "person",
    "place",
    "time",
}


DURATION_EXTRACTION_SYSTEM_PROMPT_PRE_EVENT_BINDING = (
    "Extract the duration evidence needed to answer the question. Use only the "
    "retrieved memories. Choose explicit_duration when a matching memory directly "
    "states the elapsed duration. Choose date_difference only when the matching start "
    "and end events and both dates are supported. Choose unavailable otherwise. Dates "
    "must be ISO YYYY, YYYY-MM, or YYYY-MM-DD and must describe the events, not merely "
    "their message timestamps. Select the unit explicitly requested by the question; "
    "for an unspecified 'how long' question, select the coarsest unit that preserves a "
    "meaningful nonzero interval. Use a modifier only when the evidence or endpoint "
    "precision supports it. Return only the schema fields. "
    + UNTRUSTED_DATA_INSTRUCTION
)

DURATION_EXTRACTION_JSON_SCHEMA: JsonSchema = object_schema(
    {
        "mode": enum_string(("explicit_duration", "date_difference", "unavailable")),
        "explicit_duration": nullable({"type": "string", "minLength": 1}),
        "start_date": nullable(
            {"type": "string", "pattern": _ISO_DATE_SCHEMA_PATTERN}
        ),
        "end_date": nullable(
            {"type": "string", "pattern": _ISO_DATE_SCHEMA_PATTERN}
        ),
        "unit": nullable(enum_string(("days", "weeks", "months", "years"))),
        "modifier": enum_string(("none", "about", "nearly", "over")),
    },
    required=(
        "mode",
        "explicit_duration",
        "start_date",
        "end_date",
        "unit",
        "modifier",
    ),
)


TEMPORAL_VALUE_EXTRACTION_SYSTEM_PROMPT_PRE_EVENT_BINDING = (
    "Extract the concrete time requested by the question from the retrieved memories. "
    "First match the exact person, event occurrence, ordinal or qualifier, and requested "
    "time window. Bind the chosen time to that event's own original evidence; do not use "
    "a nearby event's date or a summary timestamp. Copy a short supporting quote verbatim "
    "into evidence_quote. Resolve a relative expression exactly once from that quote's "
    "observed_at timestamp. Prefer an explicit event date or a matching absolute-time or "
    "Time Evidence value when it describes the same occurrence. Preserve source precision "
    "and return a range only when the source denotes a range. Choose resolved_time only "
    "when the event, evidence quote, and concrete value are supported; otherwise choose "
    "unavailable. Return only the schema fields. "
    + UNTRUSTED_DATA_INSTRUCTION
)

# JSON-shaped user data alone does not satisfy the JSON-mode message contract.
# Keep historical prompts intact; append only the missing output-format instruction.
_JSON_OUTPUT_INSTRUCTION = (
    " Return a single valid JSON object matching the supplied output_schema."
)
DURATION_EXTRACTION_SYSTEM_PROMPT = (
    DURATION_EXTRACTION_SYSTEM_PROMPT_PRE_EVENT_BINDING + _JSON_OUTPUT_INSTRUCTION
)
TEMPORAL_VALUE_EXTRACTION_SYSTEM_PROMPT = (
    TEMPORAL_VALUE_EXTRACTION_SYSTEM_PROMPT_PRE_EVENT_BINDING + _JSON_OUTPUT_INSTRUCTION
)

TEMPORAL_VALUE_EXTRACTION_JSON_SCHEMA: JsonSchema = object_schema(
    {
        "mode": enum_string(("resolved_time", "unavailable")),
        "event": nullable({"type": "string", "minLength": 1}),
        "evidence_quote": nullable({"type": "string", "minLength": 1}),
        "observed_at": nullable({"type": "string", "minLength": 1}),
        "time_expression": nullable({"type": "string", "minLength": 1}),
        "resolved_value": nullable({"type": "string", "minLength": 1}),
        "precision": nullable(
            enum_string(("day", "week", "month", "year", "range", "unknown"))
        ),
    },
    required=(
        "mode",
        "event",
        "evidence_quote",
        "observed_at",
        "time_expression",
        "resolved_value",
        "precision",
    ),
)


@dataclass(frozen=True)
class _DatePoint:
    year: int
    month: int | None
    day: int | None

    @property
    def precision(self) -> str:
        if self.day is not None:
            return "day"
        if self.month is not None:
            return "month"
        return "year"


def is_structured_duration_question(question: str) -> bool:
    """Return whether the question asks for an elapsed duration."""
    normalized = str(question).strip()
    return bool(
        normalized
        and not _HOW_LONG_AGO_RE.search(normalized)
        and _DURATION_QUESTION_RE.search(normalized)
    )


def is_structured_temporal_value_question(question: str) -> bool:
    """Return whether the question explicitly asks for a date or time value."""
    normalized = str(question).strip()
    return bool(normalized and _TEMPORAL_VALUE_QUESTION_RE.search(normalized))


def is_compound_temporal_question(question: str) -> bool:
    """Conservatively keep a temporal sub-answer from replacing other facets."""
    text = str(question).strip()
    # Multiple interrogative clauses, including an embedded duration request.
    if re.search(r"(?:\band\b|\balso\b|[;,?])\s*(?:for\s+)?"
                 r"(?:what|which|where|when|why|who|how|list|describe|name|explain|identify|"
                 r"did|does|do|is|was|were|are|has|have|will|would)\b", text, re.I):
        return True
    match = _DURATION_QUESTION_RE.search(text)
    return bool(match and re.search(r"\b(?:what|which|where|why|who)\b",
                                   text[:match.start()], re.I))


def _recover_extraction_fields(
    raw: str, schema: JsonSchema, diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Retain independently valid fields, never coercing invalid values."""
    candidate = parse_json_object_with_schema(
        raw, {"type": "object"}, source="Temporal extraction")
    valid: dict[str, Any] = {}
    errors = []
    for field, value in candidate.items():
        if field not in schema["properties"]:
            continue
        try:
            parse_json_object_with_schema(
                json.dumps({field: value}),
                object_schema({field: schema["properties"][field]}),
                source="Temporal extraction")
        except ValueError as error:
            errors.append(str(error))
        else:
            valid[field] = value
    diagnostics["valid_fields"] = dict(valid)
    if errors:
        diagnostics["field_errors"] = errors
    return valid


def calculate_duration_answer(
    *,
    start_date: str,
    end_date: str,
    unit: str,
) -> str:
    """Calculate a non-negative elapsed duration from validated ISO endpoints."""
    start = _parse_date_point(start_date)
    end = _parse_date_point(end_date)
    normalized_unit = str(unit).strip().lower()
    if normalized_unit in {"days", "weeks"}:
        if start.precision != "day" or end.precision != "day":
            raise ValueError(f"{normalized_unit} require day precision endpoints")
        start_value = date(start.year, start.month or 1, start.day or 1)
        end_value = date(end.year, end.month or 1, end.day or 1)
        elapsed_days = (end_value - start_value).days
        if elapsed_days < 0:
            raise ValueError("duration end date precedes start date")
        amount = (
            elapsed_days
            if normalized_unit == "days"
            else math.floor(elapsed_days / 7 + 0.5)
        )
    elif normalized_unit == "months":
        if start.month is None or end.month is None:
            raise ValueError("months require month precision endpoints")
        amount = (end.year - start.year) * 12 + end.month - start.month
        if amount < 0:
            raise ValueError("duration end date precedes start date")
    elif normalized_unit == "years":
        amount = end.year - start.year
        if amount < 0:
            raise ValueError("duration end date precedes start date")
    else:
        raise ValueError(f"unsupported duration unit: {unit}")
    label = normalized_unit[:-1] if amount == 1 else normalized_unit
    return f"{amount} {label}"


def resolve_duration_extraction(value: Mapping[str, Any]) -> str | None:
    """Resolve validated extraction fields into a concrete answer when possible."""
    mode = str(value.get("mode") or "")
    if mode == "unavailable":
        return None
    if mode == "explicit_duration":
        duration = str(value.get("explicit_duration") or "").strip()
        if not duration:
            raise ValueError("explicit_duration mode requires a duration")
        if duration.casefold() in _ANSWER_TYPE_LABELS:
            raise ValueError("explicit_duration must contain a concrete value")
        return duration
    if mode != "date_difference":
        raise ValueError(f"unsupported duration extraction mode: {mode}")
    start_date = value.get("start_date")
    end_date = value.get("end_date")
    unit = value.get("unit")
    if not all(isinstance(item, str) and item.strip() for item in (start_date, end_date, unit)):
        raise ValueError("date_difference mode requires start_date, end_date, and unit")
    answer = calculate_duration_answer(
        start_date=start_date,
        end_date=end_date,
        unit=unit,
    )
    modifier = str(value.get("modifier") or "none")
    return answer if modifier == "none" else f"{modifier} {answer}"


CAT2_SOURCE_TIME_GUIDANCE = (
    " Use your judgment to reconcile the original passage, summaries, and Time Evidence; "
    "stored dates are useful candidates, not facts to automatically reject or obey. "
    "Match the requested person and occurrence, including first/third, named object, "
    "and planned versus completed status, before choosing a time. Resolve relative "
    "expressions from their own passage's timestamp; a later recollection can describe "
    "an earlier event. Prefer the best-supported interpretation across records, and "
    "retain the supported precision or approximation instead of inventing an exact day. "
    "A missing explicit absolute date does not by itself make an answer unavailable."
)
CAT2_ANSWER_TYPE_GUIDANCE = (
    " Answer the actual question type: a location, activity, count, or ordering question "
    "needs that answer, not a date merely because time is mentioned. Use common knowledge "
    "for entity-to-attribute mappings where needed (for example city to country). "
    "Check that the final answer has the requested specificity and any necessary units."
)
CAT2_DURATION_GUIDANCE = (
    " Briefly record in selection_basis why the evidence matches the requested interval. "
    "For date_difference, fill start_support and end_support with each endpoint event, "
    "its supporting passage and its own time anchor. One passage may establish both "
    "endpoints, but a completion mention alone does not establish the start. Reconsider "
    "identical endpoints before returning zero; do not force vague dates into exact days. "
    "For explicit_duration include the unit in the answer. Keep support fields null "
    "when they do not apply; use concise evidence, not an extended reasoning transcript."
)


def prepare_cat2_time_context(memory_context: str) -> str:
    """Keep the complete evidence, including the stored temporal index."""
    return str(memory_context)


def _cat2_selection_schema(base: JsonSchema, *, duration: bool) -> JsonSchema:
    """Add concise model-selected evidence without changing other categories."""
    extra = {"selection_basis": nullable({"type": "string", "minLength": 1})}
    if duration:
        extra.update({name: nullable({"type": "string", "minLength": 1})
                      for name in ("start_support", "end_support")})
    return object_schema({**base["properties"], **extra},
                         required=(*base["required"], *extra))


def extract_structured_duration(
    client: ChatClient,
    *,
    question: str,
    memory_context: str,
    source_time_binding: bool = False,
    json_mode: bool = True,
    diagnostics: dict[str, Any] | None = None,
    selected_evidence: list[dict[str, Any]] | None = None,
    execution_work: dict[str, Any] | None = None,
    contract_v2: bool = False,
) -> str | None:
    """Ask the answer model for grounded duration fields and compute the result."""
    payload = {
        "question": str(question).strip(),
        "retrieved_memories": str(memory_context),
    }
    schema = (_cat2_selection_schema(DURATION_EXTRACTION_JSON_SCHEMA, duration=True)
              if source_time_binding or diagnostics is not None or contract_v2 else DURATION_EXTRACTION_JSON_SCHEMA)
    request_schema = evidence_extraction_schema(schema) if selected_evidence is not None else schema
    execution_guidance = ""
    if execution_work is not None:
        from agent.category_execution import TIME_GUIDANCE, time_binding_schema
        request_schema = time_binding_schema(request_schema, contract_v2=contract_v2)
        execution_guidance = TIME_GUIDANCE
        if contract_v2:
            from agent.execution_contract import TIME_IDENTITY_GUIDANCE, temporal_output_examples
            execution_guidance += TIME_IDENTITY_GUIDANCE
            payload["format_examples"] = temporal_output_examples(duration=True)
            payload["format_note"] = "Synthetic format examples only; never use example facts as evidence. Answer the actual question from retrieved_memories."
    raw = client.chat(
        messages_with_json_schema(
            [
                {"role": "system", "content": DURATION_EXTRACTION_SYSTEM_PROMPT
                 + (CAT2_SOURCE_TIME_GUIDANCE + CAT2_DURATION_GUIDANCE if source_time_binding else "")
                 + (" Record each endpoint event and its supporting passage in start_support "
                    "and end_support when applicable; omit inactive fields if unnecessary."
                    if diagnostics is not None else "")
                 + (EVIDENCE_SELECTION_GUIDANCE if selected_evidence is not None else "")
                 + execution_guidance},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            request_schema,
        ),
        json_mode=json_mode,
        json_schema=request_schema,
    )
    if selected_evidence is not None:
        raw = split_extraction_evidence(raw, selected_evidence)
    if execution_work is not None:
        from agent.category_execution import apply_time_bindings
        value = apply_time_bindings(parse_json_object(raw), memory_context, execution_work, contract_v2=contract_v2)
        if contract_v2:
            from agent.execution_contract import resolve_interval_duration
            interval_answer = resolve_interval_duration(value, execution_work.get("bindings", []))
            if interval_answer is not None:
                return interval_answer
        raw = json.dumps(value)
    if diagnostics is not None:
        value = _recover_extraction_fields(raw, schema, diagnostics)
        return resolve_duration_extraction(value)
    if source_time_binding:
        # Accept omitted inactive fields, but retain the full schema's type and
        # active-field checks. Keep the model-facing contract unchanged.
        candidate = parse_json_object_with_schema(
            raw, {**schema, "required": ["mode"]},
            source="Duration answer mode",
        )
        inactive_fields = {
            "date_difference": ("explicit_duration",),
            "explicit_duration": (
                "start_date", "end_date", "unit", "start_support", "end_support",
            ),
            "unavailable": (
                "explicit_duration", "start_date", "end_date", "unit",
                "selection_basis", "start_support", "end_support",
            ),
        }.get(candidate["mode"], ())
        for field in inactive_fields:
            candidate.setdefault(field, None)
        if candidate["mode"] in {"explicit_duration", "unavailable"}:
            candidate.setdefault("modifier", "none")
        raw = json.dumps(candidate, ensure_ascii=False)
    value = parse_json_object_with_schema(
        raw,
        schema,
        source="Duration answer extractor",
    )
    return resolve_duration_extraction(value)


def resolve_temporal_value_extraction(
    value: Mapping[str, Any],
    *,
    memory_context: str,
) -> str | None:
    """Return a grounded extracted time value, or fall back when grounding fails."""
    mode = str(value.get("mode") or "")
    if mode == "unavailable":
        return None
    if mode != "resolved_time":
        raise ValueError(f"unsupported temporal extraction mode: {mode}")

    event = str(value.get("event") or "").strip()
    quote = str(value.get("evidence_quote") or "").strip()
    resolved = str(value.get("resolved_value") or "").strip()
    if not event or not quote or not resolved:
        raise ValueError("resolved_time requires event, evidence_quote, and resolved_value")
    if resolved.casefold() in _ANSWER_TYPE_LABELS:
        raise ValueError("resolved_value must contain a concrete value")
    if quote.casefold() not in str(memory_context).casefold():
        raise ValueError("temporal evidence_quote is not present in retrieved memories")
    return resolved


def extract_structured_temporal_value(
    client: ChatClient,
    *,
    question: str,
    memory_context: str,
    source_time_binding: bool = False,
    json_mode: bool = True,
    diagnostics: dict[str, Any] | None = None,
    selected_evidence: list[dict[str, Any]] | None = None,
    execution_work: dict[str, Any] | None = None,
    contract_v2: bool = False,
) -> str | None:
    """Extract a grounded event-bound date or time value with one model call."""
    payload = {
        "question": str(question).strip(),
        "retrieved_memories": str(memory_context),
    }
    schema = (_cat2_selection_schema(TEMPORAL_VALUE_EXTRACTION_JSON_SCHEMA, duration=False)
              if source_time_binding else TEMPORAL_VALUE_EXTRACTION_JSON_SCHEMA)
    request_schema = evidence_extraction_schema(schema) if selected_evidence is not None else schema
    execution_guidance = ""
    if execution_work is not None:
        from agent.category_execution import TIME_GUIDANCE, time_binding_schema
        request_schema = time_binding_schema(request_schema, contract_v2=contract_v2)
        execution_guidance = TIME_GUIDANCE
        if contract_v2:
            from agent.execution_contract import TIME_IDENTITY_GUIDANCE, temporal_output_examples
            execution_guidance += TIME_IDENTITY_GUIDANCE
            payload["format_examples"] = temporal_output_examples(duration=False)
            payload["format_note"] = "Synthetic format examples only; never use example facts as evidence. Answer the actual question from retrieved_memories."
    raw = client.chat(
        messages_with_json_schema(
            [
                {"role": "system", "content": TEMPORAL_VALUE_EXTRACTION_SYSTEM_PROMPT
                 + (CAT2_SOURCE_TIME_GUIDANCE + " In selection_basis briefly state why the chosen "
                    "evidence matches this occurrence rather than competing mentions."
                    if source_time_binding else "")
                 + (EVIDENCE_SELECTION_GUIDANCE if selected_evidence is not None else "")
                 + execution_guidance},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            request_schema,
        ),
        json_mode=json_mode,
        json_schema=request_schema,
    )
    if selected_evidence is not None:
        raw = split_extraction_evidence(raw, selected_evidence)
    if execution_work is not None:
        from agent.category_execution import apply_time_bindings
        raw = json.dumps(apply_time_bindings(parse_json_object(raw), memory_context, execution_work, contract_v2=contract_v2))
    if diagnostics is not None:
        value = _recover_extraction_fields(raw, schema, diagnostics)
        return resolve_temporal_value_extraction(value, memory_context=memory_context)
    value = parse_json_object_with_schema(
        raw,
        schema,
        source="Temporal value answer extractor",
    )
    return resolve_temporal_value_extraction(value, memory_context=memory_context)


def _parse_date_point(value: str) -> _DatePoint:
    matched = _ISO_DATE_RE.fullmatch(str(value).strip())
    if matched is None:
        raise ValueError(f"invalid ISO duration endpoint: {value}")
    year = int(matched.group("year"))
    month_text = matched.group("month")
    day_text = matched.group("day")
    month = int(month_text) if month_text is not None else None
    day = int(day_text) if day_text is not None else None
    if month is not None:
        date(year, month, day or 1)
    return _DatePoint(year=year, month=month, day=day)


__all__ = [
    "DURATION_EXTRACTION_JSON_SCHEMA",
    "DURATION_EXTRACTION_SYSTEM_PROMPT",
    "TEMPORAL_VALUE_EXTRACTION_JSON_SCHEMA",
    "TEMPORAL_VALUE_EXTRACTION_SYSTEM_PROMPT",
    "calculate_duration_answer",
    "extract_structured_duration",
    "extract_structured_temporal_value",
    "is_structured_duration_question",
    "is_structured_temporal_value_question",
    "resolve_duration_extraction",
    "resolve_temporal_value_extraction",
]
