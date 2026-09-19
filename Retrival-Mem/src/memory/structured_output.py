"""Provider-neutral helpers for native structured chat outputs."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator


JsonSchema = dict[str, Any]
_SCHEMA_SYSTEM_PREFIX = "Authoritative output JSON Schema: "
_JSON_CODE_FENCE = re.compile(
    r"```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```",
    flags=re.IGNORECASE | re.DOTALL,
)


def object_schema(
    properties: Mapping[str, Any],
    *,
    required: Iterable[str] | None = None,
    additional_properties: bool = False,
) -> JsonSchema:
    """Build a JSON object schema with explicit property handling."""
    schema: JsonSchema = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": additional_properties,
    }
    required_fields = list(required or ())
    if required_fields:
        schema["required"] = required_fields
    return schema


def array_schema(
    items: Mapping[str, Any],
    *,
    min_items: int | None = None,
    max_items: int | None = None,
    unique_items: bool = False,
) -> JsonSchema:
    """Build a JSON array schema with optional cardinality constraints."""
    schema: JsonSchema = {"type": "array", "items": dict(items)}
    if min_items is not None:
        schema["minItems"] = int(min_items)
    if max_items is not None:
        schema["maxItems"] = int(max_items)
    if unique_items:
        schema["uniqueItems"] = True
    return schema


def nullable(schema: Mapping[str, Any]) -> JsonSchema:
    """Allow either the supplied schema or the JSON null literal."""
    return {"anyOf": [dict(schema), {"type": "null"}]}


def enum_string(values: Iterable[str]) -> JsonSchema:
    """Build a string schema constrained to values when any are available."""
    choices = list(dict.fromkeys(str(value) for value in values))
    schema: JsonSchema = {"type": "string"}
    if choices:
        schema["enum"] = choices
    return schema


def string_array(
    *,
    enum: Iterable[str] = (),
    min_items: int | None = None,
    max_items: int | None = None,
    unique_items: bool = True,
) -> JsonSchema:
    """Build an array of strings, optionally limited to supplied values."""
    return array_schema(
        enum_string(enum),
        min_items=min_items,
        max_items=max_items,
        unique_items=unique_items,
    )


def messages_with_json_schema(
    messages: Iterable[Mapping[str, str]],
    schema: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Return copied messages whose prompt exposes the exact native schema.

    JSON-object user payloads receive an ``output_schema`` field. Free-form
    prompts receive a leading system message. In both cases the schema object is
    the same contract passed to the provider's native structured-output setting.
    """
    output = [dict(message) for message in messages]
    for index in range(len(output) - 1, -1, -1):
        message = output[index]
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(message.get("content", ""))
        except json.JSONDecodeError:
            break
        if not isinstance(payload, dict):
            break
        payload["output_schema"] = deepcopy(dict(schema))
        message["content"] = json.dumps(payload, ensure_ascii=False)
        return output
    output.insert(
        0,
        {
            "role": "system",
            "content": _SCHEMA_SYSTEM_PREFIX
            + json.dumps(schema, ensure_ascii=False, sort_keys=True),
        },
    )
    return output


def parse_json_object_with_schema(
    text: str,
    schema: Mapping[str, Any],
    *,
    source: str = "Structured model",
) -> dict[str, Any]:
    """Parse a JSON object and locally enforce the provider-facing schema.

    Some providers wrap otherwise valid structured output in one Markdown JSON
    code fence. Accept that bounded representation, but continue to reject
    surrounding prose, multiple fences, and trailing content.
    """
    normalized = str(text).strip().removeprefix("\ufeff").strip()
    if not normalized:
        raise ValueError(f"{source} returned an empty JSON response")
    fenced = _JSON_CODE_FENCE.fullmatch(normalized)
    if fenced is not None:
        normalized = fenced.group("body").strip()
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source} returned invalid or truncated JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{source} returned JSON that is not an object")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise ValueError(
            f"{source} violated JSON Schema at {path}: {error.message}"
        )
    return value


__all__ = [
    "JsonSchema",
    "array_schema",
    "enum_string",
    "nullable",
    "messages_with_json_schema",
    "object_schema",
    "parse_json_object_with_schema",
    "string_array",
]
