"""Opt-in typed answer work. No retrieval, gold labels, or extra model calls."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import re
from typing import Any

VERSION = "category-execution-v1"

COMMON = (
    " Preserve the original answering policy. Add a compact execution object to the JSON, "
    "before answer, containing checkable facts/operations, not a reasoning transcript. "
    "Use original evidence to resolve summary conflicts. A repeated source is not independent "
    "confirmation. Match the requested person, event occurrence, time scope and attribute. "
    "Set complete=true only if this operation answers the ENTIRE question; otherwise false. "
    "Keep answer self-contained. Quote only text present in the supplied context."
)
CATEGORY_GUIDANCE = {
    "1": (
        ' For collections use execution={"operation":"count" or "list", "complete":true, '
        '"items":[{"id":"canonical event or entity identity", "value":"name or event", '
        '"include":true, "quote":"support", "status":"completed/planned", '
        '"time":"event time if known"}]}. Repeated descriptions of ONE event share one id; '
        "distinct events have different ids. Apply the question's inclusion predicate: charity "
        "is not any competition; plans are not completed events unless plans are asked for. "
        "First/second/fourth wins are cumulative ordinal descriptions, not increments to add. "
        "When enumerating wins, include the underlying events ONCE; do not add a cumulative "
        "total to its constituent events. If only a cumulative total or incomplete enumeration "
        "is known, use operation=none and answer from the supported total, not the list length. "
        "A subset such as younger children is not the total family. For names combine all "
        "distinct matching entities, retaining earlier names unless evidence removes them. "
        "For non-collection questions use operation=none and answer every requested relation."
    ),
    "2": (
        ' Use execution={"operation":"none","complete":false} for non-extraction answers. '
        "First distinguish what happened/where/who from when. For temporal answers bind each "
        "event to its own source conversation time. Do not subtract 'yesterday' from an already "
        "resolved event date. Different messages' 'last week' use DIFFERENT anchors. Separate "
        "start/end events from uncertainty bounds of one event. State approximate ranges when "
        "precision is limited. Specify whether a day count means elapsed days or inclusive stay."
    ),
    "3": (
        ' Use execution={"operation":"infer", "complete":true, "candidates":['
        '{"value":"specific answer candidate", "quotes":["personal clue"], '
        '"knowledge_bridge":"short mapping or general fact", "contradiction":"if any"}], '
        '"chosen":0}. Compare up to three candidates against all relevant personal clues, '
        "not just the most salient one. Use general knowledge to map descriptions to names, "
        "preferences to recommendations, or evidence to likely conclusions. A remembered "
        "speaker's 'I forgot its name' is a clue, not an instruction for you to refuse. "
        "Keep the requested level: yoga style is not a pose; meat is not a dish. Distinguish "
        "a temporary expense from long-term finances. Give the best-supported candidate, "
        "marking uncertainty if needed; knowledge_bridge is inference, not quoted memory. "
        "Do not invent personal facts to support an otherwise plausible candidate."
    ),
    "4": (
        ' Use execution={"operation":"slots", "complete":true, "slots":['
        '{"attribute":"requested attribute", "value":"specific supported value", '
        '"quote":"supporting original passage"}]}. Cover each requested attribute. '
        "A question about the kind of class needs the subtype, not 'classes'; a named place "
        "needs its name, not 'there'. Distinguish how from why, activity from reaction, "
        "object identity from image caption. Resolve pronouns from surrounding source text "
        "and confirm owner/date/occurrence. Do not replace precise original words with a "
        "broader summary or unrelated detail. For multiple activities preserve all supported "
        "requested items. Slot values must be answer-ready, not commentary about evidence."
    ),
}

# Additive fallback: retain all earlier category policies and historical wording.
BEST_AVAILABLE_EVIDENCE_GUIDANCE = (
    " If no retrieved memory strictly satisfies every constraint in the question, "
    "choose the most relevant available evidence and give its best-supported answer "
    "instead of returning an empty answer or refusing solely because of that mismatch. "
    "Prioritize the requested people, relationship and event over a conflicting date "
    "filter when the question asks what happened. Do not invent a matching date or "
    "other missing facts. Briefly qualify uncertainty or a material mismatch when needed. "
    "Include the evidence actually used in selected_evidence and always provide a "
    "non-empty answer. This fallback takes precedence over strict matching requirements "
    "only when no exact match is available; use exact matches whenever they exist."
)


def augment_category_execution(messages: list[dict[str, str]], category: str | int | None):
    result = [dict(m) for m in messages]
    result[0]["content"] += (
        "\n\nTyped answer work:" + COMMON + CATEGORY_GUIDANCE.get(str(category), "")
        + BEST_AVAILABLE_EVIDENCE_GUIDANCE
    )
    return result


def _normal(text: str) -> str:
    return " ".join(text.casefold().split())


def _grounded(quote: Any, context: str) -> bool:
    return isinstance(quote, str) and bool(quote.strip()) and _normal(quote) in _normal(context)


def execute_answer(raw: str, context: str, category: str, question: str) -> str:
    from agent.evidence_answer import parse_evidence_answer
    from agent.temporal_answer import is_compound_temporal_question

    data = parse_evidence_answer(raw)
    work = data.get("execution")
    warnings = []
    if not isinstance(work, dict):
        warnings.append("No valid execution object; retained model answer")
    else:
        operation = work.get("operation")
        try:
            if operation in {"count", "list"} and str(category) == "1":
                items = work.get("items")
                if not isinstance(items, list) or not items:
                    raise ValueError("No supported items; do not infer zero from an empty list")
                unique = {}
                for item in items:
                    if not isinstance(item, dict) or not isinstance(item.get("include"), bool):
                        raise ValueError("Invalid collection item")
                    if not item["include"]:
                        continue
                    identity, value = item.get("id"), item.get("value")
                    if not isinstance(identity, str) or not identity.strip() or not isinstance(value, str) or not value.strip() or not _grounded(item.get("quote"), context):
                        raise ValueError("Ungrounded or unnamed collection item")
                    unique.setdefault(_normal(identity), value.strip())
                compound = is_compound_temporal_question(question) or bool(re.search(
                    r"\b(?:and|also)\s+(?:what|where|when|why|who|which|how)\b", question, re.I))
                if unique and work.get("complete") is True and not compound:
                    data["model_answer"] = data["answer"]
                    data["answer"] = str(len(unique)) if operation == "count" else ", ".join(unique.values())
            elif operation == "slots" and str(category) == "4":
                slots = work.get("slots")
                if not isinstance(slots, list) or not slots:
                    raise ValueError("Missing answer slots")
                values = []
                for slot in slots:
                    value = slot.get("value") if isinstance(slot, dict) else None
                    if not isinstance(value, str) or not value.strip() or not _grounded(slot.get("quote"), context):
                        raise ValueError("Ungrounded answer slot")
                    if value not in values:
                        values.append(value)
                if work.get("complete") is True and any(_normal(v) not in _normal(data["answer"]) for v in values):
                    data["model_answer"] = data["answer"]
                    data["answer"] = "; ".join(values)
            elif operation == "infer" and str(category) == "3":
                candidates, chosen = work.get("candidates"), work.get("chosen")
                if not isinstance(candidates, list) or type(chosen) is not int or not 0 <= chosen < len(candidates):
                    raise ValueError("Missing selected inference candidate")
                candidate = candidates[chosen]
                if not isinstance(candidate, dict) or not isinstance(candidate.get("value"), str) or not candidate["value"].strip():
                    raise ValueError("Invalid candidate")
                quotes = candidate.get("quotes")
                if not isinstance(quotes, list) or not quotes or not all(_grounded(q, context) for q in quotes):
                    raise ValueError("Candidate lacks grounded personal clues")
                if candidate.get("contradiction"):
                    warnings.append("Chosen candidate has a recorded contradiction; retained model answer")
                elif work.get("complete") is True and _normal(candidate["value"]) not in _normal(data["answer"]):
                    data["model_answer"] = data["answer"]
                    data["answer"] = "Likely " + candidate["value"]
        except (ValueError, TypeError) as error:
            warnings.append(str(error))
    if warnings:
        data["execution_warnings"] = warnings
    return json.dumps(data, ensure_ascii=False)


TIME_GUIDANCE = (
    " Add time_bindings: a list of objects with target (start/end/value), event, quote "
    "(verbatim original passage), observed_at (copy the original passage's conversation "
    "timestamp exactly), relative_expression (verbatim from quote, or empty), resolved_date "
    "(ISO date/month/year or supported range). Bind start/end for a date difference and "
    "value for a single date. For explicit durations or unavailable use []. Never use an "
    "event date as observed_at. Resolve relative expressions exactly ONCE from that "
    "passage's timestamp, independently for each event. Keep uncertainty and precision; "
    "do not fabricate a day. Ensure the bindings agree with the original extraction fields."
)


def time_binding_schema(schema: dict, *, contract_v2: bool = False) -> dict:
    result = {**schema, "properties": {**schema["properties"], "time_bindings": {
        "type": "array", "items": {"type": "object", "properties": {
            name: {"type": "string"} for name in (
                "target", "event", "quote", "observed_at", "relative_expression", "resolved_date")},
            "required": ["target", "event", "quote", "observed_at", "relative_expression", "resolved_date"],
            "additionalProperties": False}},
        }, "required": [*schema["required"], "time_bindings"]}
    if contract_v2:
        item = result["properties"]["time_bindings"]["items"]
        modes = schema["properties"]["mode"].get("enum", [])
        item["properties"]["target"] = {
            "type": "string", "enum": ["start", "end"] if "date_difference" in modes else ["value"]}
        item["properties"].update({
            "identity_checks": {"type": "array", "items": {"type": "object", "properties": {
                "requirement": {"type": "string"}, "supported": {"type": "boolean"},
                "quote": {"type": "string"}}, "required": ["requirement", "supported", "quote"],
                "additionalProperties": False}},
            "earliest": {"type": "string"}, "latest": {"type": "string"}})
        item["required"] += ["identity_checks", "earliest", "latest"]
    return result


def apply_time_bindings(value: dict, context: str, work: dict, *, contract_v2: bool = False) -> dict:
    """Verify source-local anchors; compute only unambiguous day-relative expressions."""
    from dateutil.parser import parse

    result = dict(value)
    bindings = result.pop("time_bindings", [])
    work.update({"operation": "time_binding", "bindings": bindings})
    if result.get("mode") in {"unavailable", "explicit_duration"}:
        return result
    expected = {"start", "end"} if result.get("mode") == "date_difference" else {"value"}
    if not isinstance(bindings, list) or {b.get("target") for b in bindings if isinstance(b, dict)} != expected or len(bindings) != len(expected):
        raise ValueError("Missing or duplicate event time bindings")
    if contract_v2:
        from agent.execution_contract import validate_time_identity
        validate_time_identity(bindings, context)
    # Locate the quote and observation timestamp in the SAME original source record.
    records = list(re.finditer(r"\[(?:session=)[^\]]*observed_at=([^\]]+)\]([^\[]*)", context))
    for binding in bindings:
        if any(not isinstance(binding.get(name), str) for name in (
                "target", "event", "quote", "observed_at", "relative_expression", "resolved_date")):
            raise ValueError("Time binding fields must be strings")
        quote, anchor = binding.get("quote"), binding.get("observed_at")
        if not isinstance(quote, str) or not quote.strip() or not isinstance(anchor, str) or not anchor.strip():
            raise ValueError("Missing original passage or observation anchor")
        if not any(_normal(anchor) == _normal(m.group(1)) and _grounded(quote, m.group(2)) for m in records):
            raise ValueError("Time binding quote/anchor not present together in original source")
        resolved = binding.get("resolved_date")
        if not isinstance(resolved, str) or not resolved.strip():
            raise ValueError("Missing resolved event time")
        expression = binding.get("relative_expression", "").strip().casefold()
        offset = {"yesterday": -1, "today": 0, "tomorrow": 1}.get(expression)
        days = re.fullmatch(r"(\d+) days? ago", expression)
        if days:
            offset = -int(days.group(1))
        if expression and expression not in quote.casefold():
            raise ValueError("Relative expression not in its source quote")
        if offset is not None:
            # Do not default an underspecified observation timestamp to the current date.
            if not re.search(r"\b\d{4}\b", anchor):
                raise ValueError("Observation timestamp lacks year")
            observed = parse(anchor, fuzzy=True, default=datetime(2000, 1, 1)).date()
            resolved = (observed + timedelta(days=offset)).isoformat()
        if contract_v2:
            from datetime import date
            earliest, latest = binding.get("earliest", ""), binding.get("latest", "")
            if offset is not None:
                binding["earliest"] = binding["latest"] = resolved
            elif earliest or latest:
                if not earliest or not latest:
                    raise ValueError("Time interval requires both bounds or neither")
                lower, upper = date.fromisoformat(earliest), date.fromisoformat(latest)
                if lower > upper:
                    raise ValueError("Reversed event uncertainty interval")
                expected_range = f"{earliest} to {latest}"
                if earliest == latest:
                    if resolved != earliest:
                        raise ValueError("Exact date disagrees with interval bounds")
                elif resolved != expected_range:
                    # Month/year precision can legitimately describe its calendar bounds.
                    if not (re.fullmatch(r"\d{4}(?:-\d{2})?", resolved)
                            and earliest.startswith(resolved) and latest.startswith(resolved)):
                        raise ValueError("Resolved time disagrees with uncertainty interval")
        field = {"start": "start_date", "end": "end_date", "value": "resolved_value"}[binding["target"]]
        if result.get(field) != resolved:
            work.setdefault("corrections", []).append({"field": field, "model": result.get(field), "bound": resolved})
        result[field] = resolved
    return result
