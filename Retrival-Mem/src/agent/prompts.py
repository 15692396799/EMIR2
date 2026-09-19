from __future__ import annotations

import re


ANSWER_SYSTEM_PROMPT = (
    "Answer the question using the retrieved memories as evidence. "
    "First identify the answer type requested by the question (a time, duration, "
    "person, place, event, count, list, yes/no, or other fact) and return exactly "
    "that type; a date in the question is usually only a constraint, not the answer. "
    "Resolve each relative time expression once, from its own utterance timestamp, "
    "and do not apply the same offset twice. "
    "Return only a concise answer phrase, without explanation. "
    "Use wording from the retrieved memories whenever possible. "
    "Preserve required proper nouns, titles, names, and every requested parallel item; "
    "do not merge, drop, or paraphrase them into generic descriptions. "
    "For count or list questions, first form the list of distinct matching events or "
    "items, then answer with the count or the complete list. "
    "When original evidence is provided alongside a summary, prefer details "
    "and exact wording from the original evidence over the summary. "
    "Do not introduce factual details that are unrelated to or contradicted "
    "by the retrieved memories. Evidence-grounded inference is allowed when "
    "required by the answer instructions. "
    "When time is relevant, use the conversation dates and temporal evidence. "
    "Use an absolute date when it can be derived reliably, but preserve the "
    "precision of the evidence and do not invent missing date components. "
    "When memories conflict, resolve the conflict with respect to the time "
    "asked about in the question. For current-state questions, prefer the "
    "latest applicable update; for historical questions, use the fact valid "
    "at the queried time. "
    "Do not mention retrieval traces, tool calls, node IDs, or scores."
)


LOCOMO_TEMPORAL_ANSWER_SYSTEM_PROMPT_PRE_EVENT_BINDING = (
    "Solve the temporal question from the retrieved memories. Before answering, silently "
    "identify the requested value and the exact event occurrence, extract the relevant "
    "time anchors, normalize each relative expression exactly once from its own source "
    "timestamp, and verify event order, arithmetic, and units. For elapsed-time questions, "
    "calculate the difference between the supported endpoints rather than estimating it. "
    "Treat dates mentioned in the question as filters unless a date is the requested value. "
    "Prefer time evidence attached to the matching event and preserve its precision. Return "
    "only the final concise value requested, with no reasoning, answer-type label, retrieval "
    "metadata, or unsupported date components."
)


LOCOMO_TEMPORAL_ANSWER_SYSTEM_PROMPT = (
    LOCOMO_TEMPORAL_ANSWER_SYSTEM_PROMPT_PRE_EVENT_BINDING.replace(
        "Solve the temporal question", "Answer the question"
    )
)

LOCOMO_OPEN_ENDED_ANSWER_SYSTEM_PROMPT_PRE_REQUIRED_ANSWER = (
    "Answer the open-ended question as accurately and specifically as possible. Use both "
    "the retrieved memories and your full general, commonsense, geographic, cultural, "
    "scientific, and real-world knowledge. The memories are evidence about the people and "
    "events, not a restriction on knowledge you may use to interpret them. Perform any "
    "needed multi-step, causal, abductive, or entity-mapping inference, and combine all "
    "relevant clues before choosing the answer. Do not refuse merely because the answer is "
    "not stated verbatim: when several interpretations are possible, return the one best "
    "supported by the memories and world knowledge. Use Cannot be determined only when no "
    "meaningful inference or plausible choice can be made. Match the requested answer type "
    "and granularity, include every supported item for plural questions, and do not "
    "contradict explicit memories. Return a concise direct answer; when useful, include one "
    "brief reason. Do not mention retrieval traces, tool calls, node IDs, or scores."
)


LOCOMO_OPEN_ENDED_ANSWER_REQUIREMENTS_V2 = (
    "Treat questions containing likely, might, could, would, potentially, suspected, "
    "or based on as requests for the best evidence-grounded inference, not only for a "
    "verbatim fact. Before answering, silently combine all relevant memories, including "
    "preferences, repeated behavior, plans, outcomes, relationships, and contrary "
    "evidence; do not answer from one superficially similar memory. The conclusion does "
    "not need to appear verbatim, and the absence of an explicit label is not by itself "
    "evidence for No or cannot be determined. Use Cannot be determined only when the "
    "retrieved evidence supports no reasonable choice after this inference. "
    "Stable, widely known general knowledge may bridge evidence to the requested answer, "
    "for example city to country, description to a well-known named place, work to "
    "creator, or product to company; never invent a personal fact. When the question "
    "describes an unnamed real-world entity, return its canonical specific name rather "
    "than repeating the description. For future, hypothetical, preference, or "
    "recommendation questions, weigh the person's demonstrated preferences and "
    "constraints together with the latest relevant outcome. For questions asking for "
    "another or additional option, do not merely repeat an activity already stated in "
    "the question or evidence. For explicit alternatives, name the selected alternative. "
    "For yes/no questions, begin with Yes, No, Likely yes, or Likely no. Always give the "
    "specific conclusion followed by one brief evidence-grounded reason; answer in one "
    "or two concise sentences."
)


# V4 is preserved above for reproducing the candidate-check experiment.
# V5 keeps V2 inference guidance and makes answering mandatory for this experiment.
LOCOMO_OPEN_ENDED_ANSWER_SYSTEM_PROMPT = (
    LOCOMO_OPEN_ENDED_ANSWER_SYSTEM_PROMPT_PRE_REQUIRED_ANSWER.replace(
        "Use Cannot be determined only when no meaningful inference or plausible choice can be made.",
        "Always give a concrete answer. When evidence is incomplete, choose the most "
        "likely candidate using the available clues and world knowledge. Never abstain "
        "or answer Cannot be determined, unknown, or insufficient information. "
        "Express uncertainty with Likely when appropriate, while still naming an answer.",
    ).replace("Answer the open-ended question", "Answer the question")
)

LOCOMO_OPEN_ENDED_ANSWER_REQUIREMENTS_V5 = (
    LOCOMO_OPEN_ENDED_ANSWER_REQUIREMENTS_V2.replace(
        "Use Cannot be determined only when the retrieved evidence supports no reasonable "
        "choice after this inference. ",
        "You must choose and state the most likely answer even when the evidence is "
        "incomplete; do not abstain. ",
    )
)
LOCOMO_OPEN_ENDED_ANSWER_REQUIREMENTS = LOCOMO_OPEN_ENDED_ANSWER_REQUIREMENTS_V5


LOCOMO_TEMPORAL_ANSWER_REQUIREMENTS_V7 = (
    "Silently determine the concrete value requested by the question. This is only a "
    "planning step: return the concrete value itself. Never return an answer-type label "
    "such as time, duration, count, person, place, event, or activity. A date or interval "
    "in the question is often only a filter; in that case return the requested fact, not "
    "the date. Inspect all matching memories and select the same entity, event occurrence, "
    "and requested time window before answering. Qualifiers such as first, last, during "
    "summer, in a named month, before, and after must distinguish similar occurrences. "
    "For a requested time, prefer a supported `absolute time:` or matching `Time Evidence` "
    "for the same event. Otherwise use the matching original evidence's explicit date, or "
    "resolve a deterministic relative expression once from that utterance's `observed_at`. "
    "Never return vague relative wording when a supported absolute time can be recovered. "
    "A summary timestamp is not necessarily the event date. Preserve source precision: do "
    "not expand a year or month into artificial first-day/last-day boundaries. For a "
    "duration, return an explicitly stated duration or the calculated interval, never the "
    "word duration itself. Do not claim information is missing until every retrieved "
    "memory matching the entity and event has been checked. Return only one concise "
    "answer of the requested form, without explanation."
)


LOCOMO_TEMPORAL_ANSWER_REQUIREMENTS = LOCOMO_TEMPORAL_ANSWER_REQUIREMENTS_V7


# Preserved verbatim for replaying the original multi-hop answer prompt.
LOCOMO_MULTI_HOP_ANSWER_REQUIREMENTS_V1 = (
    "Use all relevant retrieved memories before answering. First identify "
    "each distinct supported item or event requested and list it, preserving "
    "concrete names, places, titles, dates, and numbers over vague "
    "descriptions. If multiple items or events are requested, combine every "
    "distinct supported item and keep all requested parallel items. For "
    "counting questions, count all distinct matching events from that list "
    "and answer with a numeral. Do not collapse distinct occurrences or "
    "merge similar names. Return only the complete answer, without "
    "explanation."
)


LOCOMO_MULTI_HOP_ANSWER_REQUIREMENTS = LOCOMO_MULTI_HOP_ANSWER_REQUIREMENTS_V1


LOCOMO_MULTI_HOP_COUNT_ANSWER_REQUIREMENTS_V1 = (
    "The question asks for a quantity. Silently collect every matching event or item from "
    "all retrieved memories, then deduplicate events rather than memory entries or mentions. "
    "A summary, its source quotation, and a later reference to the same event count once. "
    "Use participants, activity, time, and explicit references to identify duplicates; "
    "similar activities on different occasions remain distinct. Count plans only when plans "
    "are requested, and do not count a planned event again when its completion is described. "
    "Check for omissions and duplicates, then return only the final numeral with no list, "
    "dates, units, or explanation. Do not infer zero solely from missing evidence."
)


_MULTI_HOP_COUNT_QUESTION_RE = re.compile(
    r"^\s*(?:how\s+many|what\s+(?:is\s+)?the\s+number\s+of)\b",
    flags=re.IGNORECASE,
)
_MULTI_HOP_DURATION_QUESTION_RE = re.compile(
    r"\bhow\s+many\s+(?:days?|weeks?|months?|years?)\b"
    r"|\bafter\s+how\s+many\s+(?:days?|weeks?|months?|years?)\b"
    r"|\bhow\s+long\s+(?:did|does|has|have|was|were|is|are|will|would)\b"
    r"|\bfor\s+how\s+long\b",
    flags=re.IGNORECASE,
)


LOCOMO_SINGLE_HOP_ANSWER_REQUIREMENTS_PRE_EXACT = (
    "Use the most directly relevant retrieved memory. Prefer concrete "
    "details and wording from the original evidence, and preserve required "
    "proper nouns, titles, and parallel items exactly as supported. Return "
    "only the specific fact requested, without additional context or "
    "explanation."
)

LOCOMO_SINGLE_HOP_ANSWER_REQUIREMENTS = LOCOMO_SINGLE_HOP_ANSWER_REQUIREMENTS_PRE_EXACT


def get_locomo_answer_requirements(
    category: str | None,
    question: str,
) -> str | None:
    """Select narrow answer instructions from the question's requested output form."""
    if category == "1":
        if _MULTI_HOP_DURATION_QUESTION_RE.search(question):
            return LOCOMO_TEMPORAL_ANSWER_REQUIREMENTS
        if _MULTI_HOP_COUNT_QUESTION_RE.search(question):
            return LOCOMO_MULTI_HOP_COUNT_ANSWER_REQUIREMENTS_V1
        return LOCOMO_MULTI_HOP_ANSWER_REQUIREMENTS_V1
    return LOCOMO_ANSWER_REQUIREMENTS.get(category or "")


LOCOMO_ANSWER_REQUIREMENTS = {
    "1": LOCOMO_MULTI_HOP_ANSWER_REQUIREMENTS,
    "2": LOCOMO_TEMPORAL_ANSWER_REQUIREMENTS,
    "3": LOCOMO_OPEN_ENDED_ANSWER_REQUIREMENTS,
    "4": LOCOMO_SINGLE_HOP_ANSWER_REQUIREMENTS,
}


def build_answer_messages(
    memory_context: str,
    question: str,
    *,
    category: str | int | None = None,
) -> list[dict[str, str]]:
    normalized_category = None if category is None else str(category).strip()
    requirements = get_locomo_answer_requirements(normalized_category, question)
    if normalized_category == "2" or (
        normalized_category == "1"
        and _MULTI_HOP_DURATION_QUESTION_RE.search(question)
    ):
        system_prompt = LOCOMO_TEMPORAL_ANSWER_SYSTEM_PROMPT
    elif normalized_category == "3":
        system_prompt = LOCOMO_OPEN_ENDED_ANSWER_SYSTEM_PROMPT
    else:
        system_prompt = ANSWER_SYSTEM_PROMPT
    requirements_block = (
        f"\n\nAnswer requirements:\n{requirements}" if requirements else ""
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Retrieved memories:\n{memory_context}\n\n"
                f"Question:\n{question}"
                f"{requirements_block}\n\n"
                "Short answer:"
            ),
        },
    ]
