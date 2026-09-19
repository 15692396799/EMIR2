"""MemConflict prompts (point 8).

The Retrival-Mem harness ships LoCoMo prompts, which are wrong for this
benchmark in two ways:

``ANSWER``
    ``agent/prompts.py`` routes by LoCoMo *category* (multi-hop / temporal /
    open-ended / single-hop) and talks about answer types, dates, durations and
    list shaping. MemConflict answers are about handling a *conflict*: the
    updated state, the true fact, or the condition attached to a preference.
    So we route by ``conflict_type`` and ``ability_target`` instead.

``JUDGE``
    ``evaluation/prompts.py`` uses ``MEM0_ACCURACY_PROMPT``: a binary
    CORRECT/WRONG label, explicitly "generous" grading, and no memory-level
    judgement. MemConflict needs a graded answer score plus a white-box rank
    (SEH@K / SRS) and the UOCS/CRS diagnostics.

Routed mapping (LoCoMo -> MemConflict):

======================  ==========================  ==================
LoCoMo category         MemConflict conflict_type  ability_target
======================  ==========================  ==================
1 multi-hop             dynamic_conflict            track_state_over_time
2 temporal              static_conflict             recover_truth
3 open-ended            conditional_conflict        bind_condition
4 single-hop            -                           -
======================  ==========================  ==================
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


# --------------------------------------------------------------------------
# Answer prompts
# --------------------------------------------------------------------------

_ANSWER_COMMON_RULES = """Shared rules for every question:
1. Answer only from the retrieved memories. Never invent facts.
2. The memories are a dated conversation history. Prefer the memory whose time
   period matches what the question asks about.
3. Some memories are false, outdated, or describe a different person. Do not
   copy a value into the answer merely because it is mentioned by the user.
4. If the retrieved memories do not support an answer, say that you cannot
   confirm it.
5. Answer in one or two short sentences. No bullets, no headings, no meta
   commentary, and never mention retrieval, memories, sources, or scoring."""

_DYNAMIC_ANSWER_PROMPT = """You answer a question about how a user's state CHANGED over time.

Extra rules for this question type:
- Give the CURRENT (latest) state, and when the question asks what changed, name
  both the previous value and the new value in the correct direction.
- For a yes/no question about whether something changed, answer Yes or No and
  briefly name the change (or the value that stayed the same).
- For a question that asks whether something stayed the same, answer Yes or No
  and give the unchanged value.
- An outdated earlier value must never become the answer of a current-state
  question.""" + "\n\n" + _ANSWER_COMMON_RULES

_STATIC_ANSWER_PROMPT = """You answer a question about a STABLE user fact that may have been contradicted.

Extra rules for this question type:
- State the true value of the fact, and briefly note that another statement
  contradicts it when the retrieved memories really contain both.
- Never adopt the contradicting value as the answer, and never present the
  contradiction as the user's own later correction.
- A value that belongs to a different person is not the user's fact.""" + "\n\n" + _ANSWER_COMMON_RULES

_CONDITIONAL_ANSWER_PROMPT = """You answer a question about WHEN a preference applies.

Extra rules for this question type:
- Associate the preference with its condition: state the circumstance, activity,
  time, or purpose under which the user holds it.
- Give the condition itself, not just the preference item, and keep the pairing
  that the evidence supports.
- Do not attach a condition that belongs to a different preference item or to a
  different person.""" + "\n\n" + _ANSWER_COMMON_RULES


ANSWER_SYSTEM_PROMPTS: dict[str, str] = {
    "dynamic_conflict": _DYNAMIC_ANSWER_PROMPT,
    "static_conflict": _STATIC_ANSWER_PROMPT,
    "conditional_conflict": _CONDITIONAL_ANSWER_PROMPT,
}


def answer_system_prompt(conflict_type: str) -> str:
    try:
        return ANSWER_SYSTEM_PROMPTS[conflict_type]
    except KeyError as error:
        raise ValueError(f"unsupported conflict_type: {conflict_type!r}") from error


def format_memory_context(memories: Sequence[Mapping[str, Any]], namespace: str = "") -> str:
    """Render retrieved memories exactly as the answer prompt sees them."""
    header = f"Memories for user {namespace}:" if namespace else "Retrieved memories:"
    lines = [header]
    if not memories:
        return "\n".join(lines + ["No relevant memories were retrieved."])
    for index, memory in enumerate(memories, start=1):
        created_at = memory.get("created_at") or "Unknown Time"
        text = str(memory.get("memory") or "").strip()
        score = memory.get("score")
        suffix = f" (score={score})" if score is not None else ""
        lines.append(f"{index}. [{created_at}] {text}{suffix}")
    return "\n".join(lines)


def build_answer_messages(
    question: str,
    memory_context: str,
    conflict_type: str,
) -> list[dict[str, str]]:
    """MemConflict counterpart of ``agent.prompts.build_answer_messages``."""
    user_prompt = (
        "Retrieved Memory Context:\n"
        f"{memory_context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": answer_system_prompt(conflict_type)},
        {"role": "user", "content": user_prompt},
    ]


# --------------------------------------------------------------------------
# Judge prompt
# --------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a memory-conflict benchmark.
You score one question at a time using the reference answer as the gold standard.
Use the model answer for the answer-level metrics and the retrieved memories for the
memory-level metric only. Do not reward confident but unsupported answers.
Return a single valid JSON object and nothing else."""


_JUDGE_ANSWER_RULES: dict[str, str] = {
    "dynamic_conflict": """- answer_accuracy:
    1.0  the answer gives the core updated fact and the old -> new direction correctly;
    0.5  it names a correct entity, state, or direction but misses important
         old/new detail or is too incomplete for full credit;
    0.0  it is wrong, contradicts the reference, states the outdated value as the
         current one, or is too uncertain to be usable.
- conflict_handling (UOCS): 1 if the answer clearly shows that an update happened and
  keeps the correct old -> new direction; otherwise 0. A value-only answer that never
  signals the change scores 0.""",
    "static_conflict": """- answer_accuracy:
    1.0  the answer gives the correct stable fact (and flags the contradiction when the
         reference does);
    0.5  it contains a correct key fact but misses the contradiction or an important
         detail, or is vague about which value is true;
    0.0  it is wrong, adopts the false value, or refuses to answer.
- conflict_handling (CRS): 1 if the answer recognises that the sources disagree or that
  the user's statement is not reliable; otherwise 0.""",
    "conditional_conflict": """- answer_accuracy:
    1.0  the answer states the condition that the reference answer gives, even with
         different wording;
    0.0  the condition is wrong, absent, attached to the wrong preference, or the answer
         only names the preference item without its condition.
  This type has no partial credit, so only 0.0 or 1.0 are valid.""",
}

_JUDGE_RANK_RULE = """- support_rank: an integer from 0 to {top_k}. Look only at the
  retrieved memories listed below. Set it to the 1-based rank of the FIRST memory that
  contains the evidence supporting the reference answer. Set it to 0 when no listed
  memory supports it. A memory that only mentions the topic, or that states the value
  the reference answer rejects, does not count."""


def judge_json_schema(top_k: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer_accuracy": {"type": "number", "enum": [0.0, 0.5, 1.0]},
            "conflict_handling": {"type": "integer", "enum": [0, 1]},
            "support_rank": {"type": "integer", "minimum": 0, "maximum": int(top_k)},
            "reasoning": {"type": "string"},
        },
        "required": ["answer_accuracy", "conflict_handling", "support_rank", "reasoning"],
        "additionalProperties": False,
    }


def build_judge_messages(
    *,
    question: str,
    gold_answer: str,
    model_answer: str,
    conflict_type: str,
    memories: Sequence[Mapping[str, Any]],
    top_k: int,
) -> list[dict[str, str]]:
    """MemConflict counterpart of ``build_locomo_judge_messages``."""
    if conflict_type not in _JUDGE_ANSWER_RULES:
        raise ValueError(f"unsupported conflict_type: {conflict_type!r}")
    retrieved = format_memory_context(memories)
    user_prompt = f"""Evaluate one {conflict_type} question.

Question:
{question}

Reference Answer:
{gold_answer}

Model Answer:
{model_answer}

Top-{top_k} Retrieved Memories:
{retrieved}

Metrics to score:
{_JUDGE_ANSWER_RULES[conflict_type]}
{_JUDGE_RANK_RULE.format(top_k=top_k)}

Return JSON in exactly this shape:
{{
  "answer_accuracy": 0.0,
  "conflict_handling": 0,
  "support_rank": 0,
  "reasoning": "short explanation"
}}
"""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------


def extract_json_object(text: str) -> dict[str, Any]:
    """Tolerantly pull one JSON object out of a model response."""
    if text is None:
        raise ValueError("empty judge response")
    normalized = str(text).strip().lstrip("\ufeff").strip()
    if not normalized:
        raise ValueError("empty judge response")

    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()

    try:
        value = json.loads(normalized)
    except json.JSONDecodeError:
        value = _scan_for_object(normalized)
    if not isinstance(value, dict):
        raise ValueError("judge response is not a JSON object")
    return value


def _scan_for_object(text: str) -> Any:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for position in range(start, len(text)):
            char = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : position + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in judge response")
