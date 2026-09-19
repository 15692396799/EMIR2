from __future__ import annotations

import json
from typing import Any

from memory.structured_output import array_schema, nullable, object_schema


PROACTIVE_RECALL_SYSTEM_PROMPT = (
    "You are an evaluation expert. Determine whether each ground-truth memory unit "
    "has a semantically identical or highly similar match in the model retrieval results. "
    "Output JSON only."
)

PROACTIVE_PRECISION_SYSTEM_PROMPT = (
    "You are a memory recall evaluation expert. Given the user message, raw retrieved "
    "content, and each retrieved memory unit, score each unit from 0 to 5. "
    "0 means fabricated or directly copied from the query; 5 means precise, grounded, "
    "and highly valuable. Output JSON only."
)

MEM0_ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


PROACTIVE_RECALL_JSON_SCHEMA = object_schema(
    {
        "matches": array_schema(
            object_schema(
                {
                    "candidate_unit": {"type": "string"},
                    "matched": {"type": "boolean"},
                    "matched_retrieved_unit": nullable({"type": "string"}),
                },
                required=(
                    "candidate_unit",
                    "matched",
                    "matched_retrieved_unit",
                ),
            )
        )
    },
    required=("matches",),
)

PROACTIVE_PRECISION_JSON_SCHEMA = object_schema(
    {
        "judgments": array_schema(
            object_schema(
                {
                    "memory_unit": {"type": "string"},
                    "score": {"type": "number", "minimum": 0, "maximum": 5},
                    "reason": {"type": "string"},
                },
                required=("memory_unit", "score", "reason"),
            ),
            min_items=1,
        ),
        "overall_comment": {"type": "string"},
    },
    required=("judgments", "overall_comment"),
)

LOCOMO_JUDGE_JSON_SCHEMA = object_schema(
    {"label": {"type": "string", "enum": ["CORRECT", "WRONG"]}},
    required=("label",),
)

def build_proactive_recall_messages(
    question: str,
    candidate_units: list[str],
    retrieved_units: list[str],
) -> list[dict[str, str]]:
    user = {
        "user_message": question,
        "ground_truth_candidate_set": candidate_units,
        "model_retrieved_units": retrieved_units,
    }
    return [
        {"role": "system", "content": PROACTIVE_RECALL_SYSTEM_PROMPT},
        {"role": "user", "content": _dumps(user)},
    ]

def build_proactive_precision_messages(
    trigger_type: str,
    question: str,
    context_str: str,
    retrieved_units: list[str],
) -> list[dict[str, str]]:
    user = {
        "trigger_type": trigger_type,
        "user_message": question,
        "raw_retrieved_content": context_str,
        "retrieved_units": retrieved_units,
    }
    return [
        {"role": "system", "content": PROACTIVE_PRECISION_SYSTEM_PROMPT},
        {"role": "user", "content": _dumps(user)},
    ]

def build_locomo_judge_messages(question: str, gold_answer: str, generated_answer: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": MEM0_ACCURACY_PROMPT.format(
                question=question,
                gold_answer=gold_answer,
                generated_answer=generated_answer,
            ),
        }
    ]

def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


__all__ = [
    "LOCOMO_JUDGE_JSON_SCHEMA",
    "MEM0_ACCURACY_PROMPT",
    "PROACTIVE_PRECISION_JSON_SCHEMA",
    "PROACTIVE_PRECISION_SYSTEM_PROMPT",
    "PROACTIVE_RECALL_JSON_SCHEMA",
    "PROACTIVE_RECALL_SYSTEM_PROMPT",
    "build_locomo_judge_messages",
    "build_proactive_precision_messages",
    "build_proactive_recall_messages",
]
