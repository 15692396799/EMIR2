"""Opt-in coherent output contract; legacy prompts and execution remain available."""
from __future__ import annotations

from datetime import date
import json
import re

from agent.category_execution import (
    BEST_AVAILABLE_EVIDENCE_GUIDANCE, CATEGORY_GUIDANCE, _grounded, _normal,
    execute_answer,
)

VERSION = "execution-contract-v2"

# Keep legacy semantic guidance, but omit its separate partial JSON examples.
_POLICY_START = {"1": "Repeated descriptions", "2": "First distinguish",
                 "3": "Compare up to", "4": "Cover each"}
CATEGORY_POLICY_V2 = {
    key: start + CATEGORY_GUIDANCE[key].partition(start)[2]
    for key, start in _POLICY_START.items()
}

STEPS = (
    " Work in this order within ONE response: (1) identify the requested attributes and "
    "event constraints; (2) select passages that establish those constraints, including "
    "nearby references and contrary evidence; (3) fill the operation and check each "
    "required part against its supporting quote; (4) give the complete final answer. "
    "Record only compact, checkable evidence mappings, not a private reasoning transcript. "
    "Do not stop at a passage that merely repeats the question's subject. Find the value "
    "of the requested property. Distinguish event existence from event identity: person, "
    "object, occurrence/ordinal, planned versus completed, and time scope must match. "
    "A source timestamp does not prove an event is the first/third occurrence. "
    "For each constraint put requirement, supported (boolean), and quote in execution.checks. "
    "Use multiple checks for a multi-part question. complete=true means every required "
    "part is supported, not merely that a related quote exists. If support is partial, "
    "set complete=false but still provide your best-supported answer with any needed qualification. "
    "For attributes, extract the specific value, not the enclosing category; quote the "
    "passage that actually contains that value. Join adjacent utterances to resolve pronouns. "
    "Use image captions when they contain the requested fact; an image search query is "
    "only a clue, not proof of what an image depicts. "
)


def _format_only(text: str) -> str:
    # Exact known template phrases only, never broad edits to retrieved memories.
    for old in (
        "Return only a concise answer phrase, without explanation.",
        "Return only the complete answer, without explanation.",
        "Return only the answer, without explanation.",
    ):
        text = text.replace(old, "Keep the final answer field concise and complete.")
    return text


def build_contract_messages(messages: list[dict[str, str]], category: str) -> list[dict[str, str]]:
    """Called on base prompts, BEFORE the legacy two-field format is appended."""
    result = [dict(m) for m in messages]
    result[0]["content"] = _format_only(result[0]["content"])
    result[0]["content"] += (
        "\n\nOutput contract: Return one valid JSON object with selected_evidence, execution, "
        "and answer, in that order. All three fields are required. No Markdown or text "
        "outside JSON. Conciseness instructions apply to answer only. Preserve the "
        "original category answering policy. selected_evidence contains verbatim quote "
        "and short supports mappings; only answer is judged."
        + STEPS + CATEGORY_POLICY_V2.get(str(category), "")
        + BEST_AVAILABLE_EVIDENCE_GUIDANCE
    )
    # Only the known prompt suffix is formatting; leave context bytes untouched.
    for message in result[1:]:
        prefix, marker, suffix = message["content"].rpartition("\n\nAnswer requirements:\n")
        if marker:
            suffix = _format_only(suffix)
            if suffix.endswith("\n\nShort answer:"):
                suffix = suffix[:-len("\n\nShort answer:")] + "\n\nJSON response:"
            message["content"] = prefix + marker + suffix
    work = {
        "operation": "none", "complete": False,
        "checks": [{"requirement": "requested constraint", "supported": True, "quote": "verbatim support"}],
    }
    if str(category) == "1":
        work.update(operation="list", items=[{
            "id": "distinct identity", "value": "specific item", "include": True,
            "quote": "verbatim support", "status": "completed", "time": "if known",
        }])
    elif str(category) == "3":
        work.update(operation="infer", candidates=[{
            "value": "specific candidate", "quotes": ["verbatim personal clue"],
            "knowledge_bridge": "brief mapping", "contradiction": "",
        }], chosen=0)
    elif str(category) == "4":
        work.update(operation="slots", slots=[{
            "attribute": "requested property", "value": "specific value",
            "quote": "original passage containing this value",
        }])
    result.append({"role": "user", "content": json.dumps({
        "output_example": {"selected_evidence": [{"quote": "verbatim support", "supports": "question part"}],
                           "execution": work, "answer": "complete final answer"},
        "format_note": "Replace placeholders; choose the applicable operation. Always include execution and checks. "
                       "For counts use count, collections use list, other questions use none. "
                       "No applicable operation: use none. Empty support: checks=[] and complete=false, but give a best-supported answer.",
    }, ensure_ascii=False)})
    return result


def validate_checks(checks: object, context: str) -> None:
    """Structural/quotation validation, NOT an automatic semantic entailment judge."""
    if not isinstance(checks, list) or not checks:
        raise ValueError("Missing evidence-to-requirement checks")
    for check in checks:
        if (not isinstance(check, dict) or not isinstance(check.get("requirement"), str)
                or not check["requirement"].strip() or check.get("supported") is not True
                or not _grounded(check.get("quote"), context)):
            raise ValueError("Unsupported or ungrounded event/attribute requirement")


def audit_execution(raw: str, context: str, category: str, question: str) -> str:
    from agent.evidence_answer import parse_evidence_answer

    data = parse_evidence_answer(raw)
    try:
        work = data.get("execution")
        if not isinstance(work, dict):
            raise ValueError("Missing execution object")
        allowed = {"1": {"none", "count", "list"}, "2": {"none"},
                   "3": {"infer", "none"}, "4": {"slots", "none"}}
        if work.get("operation") not in allowed.get(str(category), {"none"}):
            raise ValueError("Unsupported operation for category")
        if not isinstance(work.get("complete"), bool):
            raise ValueError("Missing boolean completeness flag")
        validate_checks(work.get("checks"), context)
        if work.get("complete") is not True:
            raise ValueError("Model reports incomplete support; retained answer without execution")
        if work.get("operation") == "slots":
            slots = work.get("slots")
            if not isinstance(slots, list) or not slots:
                raise ValueError("Missing slots")
            for slot in slots:
                if (not isinstance(slot, dict) or not str(slot.get("attribute") or "").strip()
                        or not isinstance(slot.get("value"), str) or not slot["value"].strip()
                        or not _grounded(slot.get("quote"), context)
                        or _normal(slot["value"]) not in _normal(slot["quote"])):
                    raise ValueError("Slot lacks attribute or extractive value support; retained model answer")
        data = json.loads(execute_answer(raw, context, category, question))
        data["execution_status"] = "degraded" if data.get("execution_warnings") else "validated_structure"
    except (ValueError, TypeError) as error:
        data["execution_status"] = "degraded"
        data["execution_warnings"] = [*data.get("execution_warnings", []), str(error)]
    return json.dumps(data, ensure_ascii=False)


TIME_IDENTITY_GUIDANCE_PRE_TARGET_FIX = (
    " Before resolving dates, identify each endpoint's person, action, object, ordinal "
    "and state. In each time binding add identity_checks: [{requirement, supported, quote}]. "
    "Quotes must establish the question's actual occurrence, not merely a similar event. "
    "Use adjacent passages to identify first/second/third events and completed versus planned. "
    "For each binding add earliest and latest (ISO day bounds, or empty strings when unknown). "
    "A precise day has equal bounds. A calendar week has a range, not an arbitrary first day. "
    "Do not invent finite bounds for recently/some time ago. Preserve unknown precision. "
    "resolved_date may be YYYY-MM, YYYY, or an ISO start to end range. "
    "For duration ranges calculate earliest end minus latest start through latest end minus "
    "earliest start. Never confuse two bounds of ONE event with TWO endpoint events. "
    "If identity cannot be established, use unavailable and retain provisional bindings."
)

TIME_IDENTITY_GUIDANCE = TIME_IDENTITY_GUIDANCE_PRE_TARGET_FIX.replace(
    "If identity cannot be established, use unavailable and retain provisional bindings.",
    "If identity cannot be established, set mode=unavailable and time_bindings=[]. "
    "Retain useful quoted clues in selected_evidence and briefly state the unresolved "
    "constraint in selection_basis; do NOT retain provisional time_bindings in unavailable mode. "
    "target is a role label, NOT a field name: a single event date uses exactly value; "
    "a duration uses start and end. Never put resolved_date or resolved_value in target."
)


def temporal_output_examples(*, duration: bool) -> list[dict]:
    """Complete synthetic examples; no benchmark-specific facts or answers."""
    quote = "I started today." if duration else "I left yesterday."
    binding = {"target": "start" if duration else "value", "event": "project start" if duration else "departure",
               "quote": quote, "observed_at": "20 January, 2023",
               "relative_expression": "today" if duration else "yesterday",
               "resolved_date": "2023-01-20" if duration else "2023-01-19",
               "identity_checks": [{"requirement": "requested event", "supported": True, "quote": quote}],
               "earliest": "2023-01-20" if duration else "2023-01-19",
               "latest": "2023-01-20" if duration else "2023-01-19"}
    common = {"selected_evidence": [{"quote": quote, "supports": "requested event"}],
              "selection_basis": "The passage describes the requested event.", "time_bindings": [binding]}
    if duration:
        end = {**binding, "target": "end", "event": "project completion", "quote": "I finished today.",
               "observed_at": "23 January, 2023", "resolved_date": "2023-01-23",
               "earliest": "2023-01-23", "latest": "2023-01-23",
               "identity_checks": [{"requirement": "same project completed", "supported": True,
                                    "quote": "I finished today."}]}
        common["time_bindings"].append(end)
        common["selected_evidence"].append({"quote": "I finished today.", "supports": "completion"})
        resolved = {**common, "mode": "date_difference", "explicit_duration": None,
                    "start_date": "2023-01-20", "end_date": "2023-01-23", "unit": "days", "modifier": "none",
                    "start_support": quote, "end_support": "I finished today."}
        inactive = {"explicit_duration": None, "start_date": None, "end_date": None,
                    "unit": None, "modifier": "none", "start_support": None, "end_support": None}
    else:
        resolved = {**common, "mode": "resolved_time", "event": "departure", "evidence_quote": quote,
                    "observed_at": "20 January, 2023", "time_expression": "yesterday",
                    "resolved_value": "2023-01-19", "precision": "day"}
        inactive = {name: None for name in ("event", "evidence_quote", "observed_at", "time_expression", "resolved_value", "precision")}
    unavailable = {**inactive, "mode": "unavailable", "selected_evidence": [],
                   "selection_basis": "The requested occurrence could not be identified.", "time_bindings": []}
    return [resolved, unavailable]


def validate_time_identity(bindings: list[dict], context: str) -> None:
    for binding in bindings:
        validate_checks(binding.get("identity_checks"), context)
        expression = binding.get("relative_expression", "").casefold()
        if re.search(r"\b(recently|lately|few|last week|this week|next week)\b", expression):
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", binding.get("resolved_date", "")):
                raise ValueError("Vague/week expression collapsed to an exact day; retain an interval")


def resolve_interval_duration(value: dict, bindings: list[dict]) -> str | None:
    if value.get("mode") != "date_difference" or len(bindings) != 2:
        return None
    by_target = {b["target"]: b for b in bindings}
    if set(by_target) != {"start", "end"}:
        return None
    endpoints = []
    for target in ("start", "end"):
        binding = by_target[target]
        if not binding.get("earliest") or not binding.get("latest"):
            return None
        lo, hi = (date.fromisoformat(binding[k]) for k in ("earliest", "latest"))
        if lo > hi:
            raise ValueError("Reversed event uncertainty interval")
        endpoints.append((lo, hi))
    (start_lo, start_hi), (end_lo, end_hi) = endpoints
    if start_lo == start_hi and end_lo == end_hi:
        return None  # Original exact-date calculator remains authoritative.
    if end_hi < start_lo:
        raise ValueError("Duration end precedes start")
    low, high = max(0, (end_lo - start_hi).days), (end_hi - start_lo).days
    unit = value.get("unit")
    if unit == "days":
        return f"{low}–{high} days"
    if unit == "weeks":
        return f"about {low / 7:.2g}–{high / 7:.2g} weeks"
    # Calendar month/year semantics use the existing resolver or its evidence-preserving fallback.
    return None
