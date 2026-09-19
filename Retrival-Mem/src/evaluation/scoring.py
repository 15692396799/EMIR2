from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence


def score_references(prediction: str, references: Sequence[str]) -> dict[str, float]:
    aliases = [str(reference) for reference in references if str(reference).strip()]
    if not aliases:
        return {"exact_match": 0.0, "token_f1": 0.0}
    return {
        "exact_match": max(float(_normalise(prediction) == _normalise(reference)) for reference in aliases),
        "token_f1": max(_token_f1(prediction, reference) for reference in aliases),
    }


def _normalise(value: str) -> str:
    value = str(value).lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _token_f1(prediction: str, reference: str) -> float:
    predicted = _normalise(prediction).split()
    expected = _normalise(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = sum((Counter(predicted) & Counter(expected)).values())
    if not common:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2.0 * precision * recall / (precision + recall)
