"""Validation helpers for model-generated V4 retrieval probes."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_TERM_META = re.compile(r'["():*^{}\[\]]+')
_SPACE = re.compile(r"\s+")
_QUOTED = re.compile(r'["“”]([^"“”]{1,80})["“”]')
_DATE_OR_NUMBER = re.compile(r"\b(?:\d{4}(?:-\d{1,2}(?:-\d{1,2})?)?|\d+)\b")
_CAPITALIZED = re.compile(r"\b[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*){0,3}\b")
_QUESTION_WORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "why",
    "how", "did", "does", "do", "is", "are", "was", "were", "tell",
    "list", "compare",
})
_LEXICAL_STOP_WORDS = _QUESTION_WORDS | frozenset({
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "from",
    "by", "about", "as", "than", "then", "there", "their", "they", "them",
    "he", "she", "it", "its", "we", "you", "i", "me", "my", "your", "our",
    "his", "her", "him", "be", "been", "being", "am", "are", "is", "was",
    "were", "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "and", "or", "not", "also",
})
_CJK = re.compile(r"[\u4e00-\u9fff]")
_BOOLEAN_TERMS = frozenset({"and", "or", "not", "near"})
_TIME_KEYS = frozenset({
    "operator", "start", "end", "value", "at", "precision", "hard",
})
_RETRIEVAL_FIELDS = frozenset({
    "query", "must_terms", "should_terms", "entities", "time_constraint", "facets",
})
_FACETS = frozenset({
    "actor", "action", "object", "time", "location", "cause", "outcome",
    "state", "conflict", "count",
})
_FACET_ALIASES = {
    "event": "action",
    "purpose": "cause",
    "reason": "cause",
    "result": "outcome",
    "participant": "actor",
    "who": "actor",
    "subject": "actor",
    "place": "location",
    "where": "location",
    "when": "time",
    "what": "object",
    "content": "object",
    "topic": "object",
    "number": "count",
    "effect": "outcome",
    "goal": "outcome",
}
_RETIRED_FIELDS = frozenset({
    "semantic_query", "question", "kind", "lane", "attempted", "exhausted",
    "evidence_refs",
})
_TIME_OPERATOR_ALIASES = {
    "eq": "at",
    "equals": "at",
    "exact": "at",
    "in": "at",
    "during": "at",
    "gte": "after",
    ">=": "after",
    "since": "after",
    "lte": "before",
    "<=": "before",
    "<": "before",
    "until": "before",
    "overlap": "between",
    "range": "between",
    "within": "between",
    "none": "any",
}
_TIME_RANGE = re.compile(r"\s+(?:to|through|until)\s+", flags=re.IGNORECASE)


def normalize_subproblem(
    value: Mapping[str, Any],
    *,
    original_question: str,
    include_required: bool = True,
) -> dict[str, Any]:
    """Return a strict, safe structured subproblem for runtime execution."""
    allowed_fields = _RETRIEVAL_FIELDS | _RETIRED_FIELDS | {"required"}
    if set(value) - allowed_fields or not (
        _RETRIEVAL_FIELDS - {"query", "facets"}
    ).issubset(value):
        raise ValueError("V4 subproblem uses invalid protocol fields")
    query_source = value.get("query") if "query" in value else value.get(
        "semantic_query"
    )
    query = _plain_text(query_source)
    if not query:
        raise ValueError("V4 subproblem requires query")

    must_terms = _terms(value.get("must_terms"), limit=4)
    should_terms = _terms(value.get("should_terms"), limit=8)
    must_markers = {term.casefold() for term in must_terms}
    should_terms = [
        term for term in should_terms if term.casefold() not in must_markers
    ]
    entities = _terms(value.get("entities"), limit=4)
    facets = value.get("facets", [])
    if not isinstance(facets, list):
        facets = []
    mapped: list[str] = []
    for item in facets:
        if not isinstance(item, str):
            continue
        key = item.strip().casefold()
        if key in _FACETS:
            mapped.append(key)
        elif key in _FACET_ALIASES:
            mapped.append(_FACET_ALIASES[key])
    facets = list(dict.fromkeys(mapped))
    time_constraint = _time_constraint(
        value.get("time_constraint"), original_question=original_question
    )
    normalized = {
        "query": query,
        "must_terms": must_terms,
        "should_terms": should_terms,
        "entities": entities,
        "facets": list(dict.fromkeys(facets)),
        "time_constraint": time_constraint,
    }
    if include_required:
        required_value = value.get("required", True)
        if not isinstance(required_value, bool):
            raise ValueError("V4 subproblem required must be boolean")
        normalized["required"] = required_value
    return normalized


def normalize_subproblems(
    values: Any,
    *,
    original_question: str,
    max_subproblems: int = 4,
) -> list[dict[str, Any]]:
    if max_subproblems < 1:
        raise ValueError("max_subproblems must be positive")
    if (
        not isinstance(values, list)
        or len(values) > max_subproblems
        or any(not isinstance(item, Mapping) for item in values)
    ):
        raise ValueError(
            "V4 initial Controller requires zero to "
            f"{max_subproblems} subproblems"
        )
    if not values:
        return []
    normalized = [
        normalize_subproblem(item, original_question=original_question)
        for item in values
    ]
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for item in normalized:
        marker = item["query"].casefold()
        if marker in seen:
            continue
        seen.add(marker)
        output.append(item)
    if not output:
        raise ValueError("V4 initial Controller produced no distinct subproblems")
    return output


def _plain_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return _SPACE.sub(" ", value).strip()


def _terms(value: Any, *, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("V4 structured probe terms must be string lists")
    output: list[str] = []
    seen: set[str] = set()
    for raw in value:
        tokens = [
            token
            for token in _SPACE.sub(" ", _TERM_META.sub(" ", raw)).split()
            if token.casefold() not in _BOOLEAN_TERMS
        ]
        term = " ".join(tokens).strip()
        marker = term.casefold()
        if not term or marker in seen:
            continue
        seen.add(marker)
        output.append(term[:120])
        if len(output) == limit:
            break
    return output


def _grounded_anchors(question: str) -> list[str]:
    candidates = [
        *(_plain_text(item) for item in _QUOTED.findall(question)),
        *_DATE_OR_NUMBER.findall(question),
        *_CAPITALIZED.findall(question),
    ]
    output: list[str] = []
    for candidate in candidates:
        if candidate.casefold() in _QUESTION_WORDS:
            continue
        if candidate and not _contains_casefold(output, candidate):
            output.append(candidate)
    return output


def _contains_casefold(values: Sequence[str], expected: str) -> bool:
    marker = expected.casefold()
    return any(value.casefold() == marker for value in values)


def _time_constraint(value: Any, *, original_question: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = _plain_text(value)
        if not text:
            return None
        bounds = _TIME_RANGE.split(text, maxsplit=1)
        operator = "between" if len(bounds) == 2 else "at"
        return {
            "operator": operator,
            "start": bounds[0],
            "end": bounds[1] if len(bounds) == 2 else None,
            "precision": None,
            "hard": False,
        }
    if not isinstance(value, Mapping) or set(value) - _TIME_KEYS:
        raise ValueError("V4 time_constraint is invalid")
    if "hard" in value and not isinstance(value["hard"], bool):
        raise ValueError("V4 time_constraint hard must be boolean")
    operator = str(value.get("operator") or "any").strip().lower()
    operator = _TIME_OPERATOR_ALIASES.get(operator, operator)
    if operator not in {"any", "at", "before", "after", "between"}:
        operator = "any"
    start = _optional_text(
        value.get("start") or value.get("value") or value.get("at")
    )
    end = _optional_text(value.get("end"))
    precision = _optional_text(value.get("precision"))
    hard = bool(value.get("hard", False))
    grounded = all(
        not item or item in original_question
        for item in (start, end)
    )
    complete = bool(start) and (operator != "between" or bool(end))
    return {
        "operator": operator,
        "start": start,
        "end": end,
        "precision": precision,
        "hard": hard and grounded and complete and operator != "any",
    }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = _plain_text(value)
    return text or None


def derive_probe_hints(query: str) -> dict[str, list[str]]:
    """Return conservative lexical hints for a shortened followup_probe.

    The Controller may now return only ``{query, time_constraint}`` for a
    follow-up probe.  This helper regenerates the deterministic parts that
    used to come from the model: capitalized/quoted/date anchors become
    entities and must-terms, remaining non-stopword tokens fill the rest of
    the must-terms, and should-terms stay empty because synonym generation
    is not deterministic.
    """
    normalized = _plain_text(query)
    grounded = [
        item for item in _grounded_anchors(normalized)
        if item.casefold() not in _LEXICAL_STOP_WORDS
    ][:4]
    entities = list(dict.fromkeys(grounded))
    entity_markers = {item.casefold() for item in entities}
    extra: list[str] = []
    if not _CJK.search(normalized):
        for token in _SPACE.sub(" ", _TERM_META.sub(" ", normalized)).split():
            token = token.strip("?!.,;:")
            if not token:
                continue
            terms = _terms([token], limit=1)
            if not terms:
                continue
            term = terms[0]
            marker = term.casefold()
            if marker in _LEXICAL_STOP_WORDS or marker in entity_markers:
                continue
            extra.append(term)
            if len(entities) + len(extra) == 4:
                break
    return {
        "must_terms": [*entities, *extra],
        "should_terms": [],
        "entities": entities,
    }


__all__ = ["normalize_subproblem", "normalize_subproblems", "derive_probe_hints"]
