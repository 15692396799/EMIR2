"""Evidence-first answers with an optional, minimal duration calculation."""
from __future__ import annotations

import json
import re
from datetime import date

from agent.category_execution import _grounded
from agent.evidence_answer import augment_evidence_answer_messages, parse_evidence_answer
from agent.execution_contract import _format_only, resolve_interval_duration
from agent.temporal_answer import calculate_duration_answer, is_compound_temporal_question


COMPACT_DURATION_GUIDANCE = (
    ' For a duration requiring subtraction, add calculation before answer: '
    '{"start":"YYYY-MM-DD", "end":"YYYY-MM-DD", "unit":"days|weeks|months|years"}. '
    'These are dates of the two requested events, not their first/last mentions. '
    'In selected_evidence supports, briefly identify each endpoint and its source-local '
    'time anchor. Resolve relative dates once from the matching message timestamp. '
    'Dates may use YYYY or YYYY-MM precision, or "YYYY-MM-DD to YYYY-MM-DD" for '
    'an explicit uncertainty range of ONE endpoint. Never invent missing months/days. '
    'Omit calculation when the duration is directly stated or endpoints are unknown. '
    'The program computes the difference. Still give a complete provisional answer. '
    'For a compound question, include the provisional duration as {{duration:three months}} '
    '(replace three months with your provisional value) within the full answer; the '
    'program replaces only that marker, preserving the other requested facts. '
    'No other JSON fields are needed.'
)


TEMPORAL_EVIDENCE_GUIDANCE = (
    '\n\nOutput contract (original task rules still apply; this replaces '
    'answer-only formatting): Return one JSON object, with selected_evidence first '
    'and answer last. Quote short verbatim passages from the supplied memories. '
    'In each supports, briefly identify the event and its source-local time anchor '
    'or explicit duration. Match the same event occurrence asked about, including '
    'first/another, planned/completed, and the requested activity or attribute. '
    'Do not use the date of a similar event or a later mention as the event date. '
    'Resolve relative time from that passage\'s own conversation timestamp, once. '
    'Treat derived dates as candidates to check against the original text. Preserve '
    'uncertainty and date precision rather than inventing a day or month. '
    'A range of possible dates for one event is not an elapsed duration. '
    'Give a concise, self-contained answer to every part of the question. '
    'Do not add audit tables or text outside JSON.'
)

TEMPORAL_CALCULATION_GUIDANCE = (
    ' If the answer requires subtracting two supported event dates, you MUST include '
    'calculation between selected_evidence and answer. This is how you invoke the '
    'program calculator; giving only a duration in answer does not invoke it. '
    'Complete format example (illustrative dates, never evidence for this question): '
    '{"selected_evidence": [{"quote": "I started on April 2, 2020.", '
    '"supports": "start of the requested activity: 2020-04-02"}, '
    '{"quote": "I finished on April 16, 2020.", '
    '"supports": "end of the same activity: 2020-04-16"}], '
    '"calculation": {"start": "2020-04-02", "end": "2020-04-16", "unit": "days"}, '
    '"answer": "14 days"}. '
    'Use the requested unit: days, weeks, months, or years. Dates may be YYYY, '
    'YYYY-MM, YYYY-MM-DD, or "YYYY-MM-DD to YYYY-MM-DD" for uncertainty in ONE '
    'endpoint. Do not use an announcement date in place of departure, arrival, '
    'start, or completion. For an ongoing duration, use the relevant question or '
    'conversation reference date, not today. '
    'If the requested duration is explicitly stated, use it directly and omit '
    'calculation; likewise omit it if an endpoint cannot be grounded. In that case '
    'return only selected_evidence and answer, with the best-supported duration or '
    'approximation, without manufacturing operands. Always provide a provisional '
    'answer. For compound questions put only the duration in a marker such as '
    '{{duration:two weeks}} within the complete answer; code replaces that span '
    'and preserves the other facts.'
)


SINGLE_HOP_PRECISION_GUIDANCE = (
    '\n\nAnswer precision: Identify the requested attribute before choosing '
    'evidence: a name, subtype, subject, action, purpose, or depicted content. '
    'In supports, briefly map that requested attribute to the concrete value '
    'provided by the quote. Match the person, object and event in the question; '
    'a related background description is not a substitute for the requested value. '
    'Preserve specific names, subtypes and relevant details supplied by the evidence '
    'instead of replacing them with a broader category or vague paraphrase. For '
    'how/what-to-practice questions, give the concrete actions or practice points, '
    'not merely an attitude, difficulty or instruction to practice. Retain all '
    'directly relevant purposes or actions when several jointly answer the question. '
    'If an image caption is present, consider its content together with the '
    'associated conversation when relevant, especially for questions about a shared '
    'photo or depicted object, scene or activity. Use only visual details actually '
    'described; do not invent unseen details or treat an image search query as a '
    'verified caption. Before finishing, ensure answer actually states the specific '
    'values supported by selected_evidence and answers the requested attribute '
    'rather than repeating the question or its background. Be concise without '
    'dropping those values; do not invent specificity missing from the evidence. '
    'Keep the existing JSON fields; no additional checklist is needed.'
)


def compact_messages(messages: list[dict[str, str]], *, duration: bool,
                     temporal: bool = False, single_hop: bool = False) -> list[dict[str, str]]:
    # Normalize template instructions only; preserve question and memory bytes.
    messages = [dict(message) for message in messages]
    messages[0]["content"] = _format_only(messages[0]["content"])
    for message in messages[1:]:
        if message["role"] != "user":
            continue
        prefix, marker, suffix = message["content"].rpartition("\n\nAnswer requirements:\n")
        if marker:
            suffix = _format_only(suffix)
            if suffix.endswith("\n\nShort answer:"):
                suffix = suffix[:-len("\n\nShort answer:")] + "\n\nJSON response:"
            message["content"] = prefix + marker + suffix
    if temporal:
        result = [dict(message) for message in messages]
        result[0]["content"] += TEMPORAL_EVIDENCE_GUIDANCE
        result[0]["content"] += (
            TEMPORAL_CALCULATION_GUIDANCE if duration else
            ' Return exactly two top-level fields: '
            '{"selected_evidence": [{"quote": "verbatim event passage", '
            '"supports": "event identity and its date anchor"}], '
            '"answer": "the requested date, period, or attribute"}.'
        )
        return result
    # Keep the existing non-Cat2 policy and evidence-selection guidance.
    result = augment_evidence_answer_messages(messages)
    result[0]["content"] += (
        ' Output exactly selected_evidence and answer, except for the optional duration '
        'calculation described below. First quote the memories you use and briefly '
        'explain their relevance in supports, then give the complete answer. '
        'If no exact match exists, use the most relevant evidence and the original '
        'category inference policy; do not fabricate quotations.'
        if duration else
        ' Output exactly two top-level fields: selected_evidence and answer. First '
        'quote the memories you use and briefly explain their relevance in supports, '
        'then give the complete answer. If no exact match exists, use the most relevant '
        'evidence and the original category inference policy; do not fabricate quotations.'
    )
    if duration:
        result[0]["content"] += COMPACT_DURATION_GUIDANCE
    if single_hop:
        result[0]["content"] += SINGLE_HOP_PRECISION_GUIDANCE
    return result


def _duration(calculation: dict) -> str:
    start, end, unit = (calculation[key] for key in ("start", "end", "unit"))
    if not all(isinstance(value, str) for value in (start, end, unit)):
        raise ValueError("Duration parameters must be strings")
    bindings = []
    for target, value in (("start", start), ("end", end)):
        bounds = value.split(" to ")
        if len(bounds) == 1:
            bounds *= 2
        if len(bounds) != 2:
            raise ValueError("Invalid endpoint range")
        bindings.append({"target": target, "earliest": bounds[0], "latest": bounds[1]})
    if " to " in start or " to " in end:
        # Validate chronology independently, including month/year interval arithmetic.
        for binding in bindings:
            lo, hi = (date.fromisoformat(binding[key]) for key in ("earliest", "latest"))
            if lo > hi:
                raise ValueError("Reversed endpoint range")
        first, last = bindings
        if last["earliest"] < first["latest"]:
            raise ValueError("Endpoint ranges overlap or are reversed")
        answer = resolve_interval_duration({"mode": "date_difference", "unit": unit}, bindings)
        if answer is not None:
            return answer
        low = calculate_duration_answer(start_date=first["latest"], end_date=last["earliest"], unit=unit)
        high = calculate_duration_answer(start_date=first["earliest"], end_date=last["latest"], unit=unit)
        return low if low == high else f"about {low} to {high}"
    # Reject reversed day endpoints even for calendar-month/year computations.
    if len(start) == len(end) and end < start:
        raise ValueError("Duration end precedes start")
    answer = calculate_duration_answer(start_date=start, end_date=end, unit=unit)
    return f"about {answer}" if len(start) != len(end) else answer


def _warn(output: dict, message: str) -> None:
    output.setdefault("execution_warnings", []).append(message)


def finish_compact_answer(raw: str, context: str, question: str) -> str:
    """Ground quotes and compute dates without asking the model for audit tables."""
    data = parse_evidence_answer(raw)
    # Do not pass legacy model-generated execution tables downstream.
    output = {"selected_evidence": data["selected_evidence"], "answer": data["answer"]}
    ungrounded = [index for index, item in enumerate(data["selected_evidence"], 1)
                  if not _grounded(item["quote"], context)]
    if ungrounded:
        # A paraphrased quote is a diagnostic; the answer itself stays usable.
        _warn(output, f"Selected evidence not found verbatim in the memories: items {ungrounded}")
    calculation = data.get("calculation")
    if calculation is not None:
        output["calculation"] = calculation
        try:
            if not isinstance(calculation, dict) or not output["selected_evidence"]:
                raise ValueError("Calculation requires selected evidence and an object")
            computed = _duration(calculation)
        except (KeyError, TypeError, ValueError) as error:
            # Diagnostics are code-generated, never another model output requirement.
            _warn(output, f"Calculation not applied: {error}")
        else:
            if not is_compound_temporal_question(question):
                output["answer"] = computed
            elif re.search(r"\{\{duration(?::[^{}]*)?\}\}", output["answer"]):
                output["answer"] = re.sub(r"\{\{duration(?::[^{}]*)?\}\}", lambda _: computed, output["answer"])
            else:
                # A unique duration phrase can be replaced safely without touching names.
                pattern = r"\b(?:(?:about|nearly|over)\s+)?(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:days?|weeks?|months?|years?)\b"
                matches = list(re.finditer(pattern, output["answer"], re.I))
                if len(matches) == 1:
                    match = matches[0]
                    output["answer"] = output["answer"][:match.start()] + computed + output["answer"][match.end():]
                else:
                    _warn(output, "Compound duration span is ambiguous; model answer retained")
    output["answer"] = re.sub(r"\{\{duration:([^{}]*)\}\}", r"\1", output["answer"])
    return json.dumps(output, ensure_ascii=False)
