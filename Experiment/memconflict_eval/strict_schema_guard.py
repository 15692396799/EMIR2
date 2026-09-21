"""Harness-side strict structured output for the semantic reducer.

Why: the reducer may only cite events extracted in *this* window. Upstream puts
that rule in the prompt (`local_event_refs` + an instruction) and validates the
answer with

    unknown_local_refs = [ref for ref in cited refs if ref not in event_refs]
    if unknown_local_refs:
        raise SemanticValidationError(
            "semantic operation references evidence outside supplied local event refs")

Measured on 2026-09-21 over the API logs of every finished store: the request
never advertises an unresolvable ref (0 of 2,873 calls), but the *answer* cites
one in ~0.1% of the calls (1/1,391 and 2/1,285 in the glm-5.1 arms, 0/104 in the
gpt-4o-mini arm). At `temperature: 0.0` the feedback retry reproduces the same
answer, so the checkpoint guard has to invalidate the namespace and re-run the
whole session -- minutes of work per event, and a dead persona when the retry
reproduces it again.

The fix, instead of repairing the answer after the fact: send the reducer's
answer schema as an OpenAI *structured output* request
(``response_format: {"type": "json_schema", "json_schema": {"strict": true ...}}``)
with ``operations[].evidence_event_refs.items`` constrained to an ``enum`` of the
refs the request advertises. The provider then cannot emit a foreign ref at all.

Only the lanes that implement OpenAI strict structured outputs are patched --
``openrouter`` (OpenAI/Azure upstreams), ``openai`` and ``azure``; Bailian and
Ollama keep the upstream ``json_object`` request byte for byte, and the harness
side reducer guard stays in place as their safety net.

``MEMCONFLICT_STRICT_SCHEMA_GUARD=0`` switches the patch off.
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from typing import Any, Mapping

GUARD_ENV = "MEMCONFLICT_STRICT_SCHEMA_GUARD"
_FALSE_VALUES = {"0", "false", "no", "off"}
_INSTALLED_FLAG = "_memconflict_strict_schema_guard"

#: Providers that accept `response_format: {"type": "json_schema", ...}`.
_STRICT_PROVIDERS = {"openrouter", "openai", "azure", "azure_openai"}

#: JSON Schema keywords OpenAI's strict mode rejects (and that the reducer's
#: schema uses: `value: {}`, minLength, minimum/maximum, uniqueItems, minItems).
_UNSUPPORTED_KEYS = (
    "minItems",
    "maxItems",
    "uniqueItems",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "pattern",
    "format",
    "default",
    "examples",
)

#: Replacement for the reducer schema's one unconstrained slot
#: (`operations[].value`), which strict mode cannot express as `{}`.
_ANY_VALUE: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "boolean"},
        {"type": "null"},
        {"type": "array", "items": {"type": "string"}},
    ]
}

#: The ref list every reducer operation cites its evidence from.
_REF_FIELD = "evidence_event_refs"


def guard_enabled() -> bool:
    """The patch is on unless ``MEMCONFLICT_STRICT_SCHEMA_GUARD=0``."""

    raw = os.getenv(GUARD_ENV)
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in _FALSE_VALUES


def reducer_payload(messages: Any) -> dict[str, Any] | None:
    """The reducer request payload inside ``messages``, if this is one."""

    for message in reversed(list(messages or [])):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(data, dict)
            and isinstance(data.get("event"), dict)
            and isinstance(data.get("local_event_refs"), list)
        ):
            return data
    return None


def allowed_refs(payload: Mapping[str, Any] | None) -> list[str]:
    """The refs the reducer request advertises, in order, de-duplicated."""

    refs: list[str] = []
    for value in (payload or {}).get("local_event_refs") or ():
        text = str(value).strip()
        if text and text not in refs:
            refs.append(text)
    return refs


def _strictify(node: Any) -> Any:
    """Rewrite one schema node into the subset OpenAI's strict mode accepts."""

    if isinstance(node, list):
        return [_strictify(item) for item in node]
    if not isinstance(node, dict):
        return node
    if not node:
        return deepcopy(_ANY_VALUE)
    output: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_KEYS:
            continue
        if key in {"oneOf", "anyOf", "allOf"}:
            # `oneOf` is not accepted by strict mode; `anyOf` carries the same
            # meaning for generation, which is all this request needs.
            output["anyOf"] = [_strictify(item) for item in value]
            continue
        if key == "items":
            output["items"] = _strictify(value)
            continue
        if key == "properties":
            output["properties"] = {name: _strictify(sub) for name, sub in value.items()}
            continue
        output[key] = deepcopy(value)
    properties = output.get("properties")
    if output.get("type") == "object" and isinstance(properties, dict):
        output["additionalProperties"] = False
        output["required"] = sorted(properties)
    return output


def _inject_ref_enum(node: Any, refs: list[str]) -> None:
    """Constrain every ``evidence_event_refs`` list to the window's refs."""

    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict) and _REF_FIELD in properties:
            properties[_REF_FIELD] = {
                "type": "array",
                "items": {"type": "string", "enum": list(refs)},
            }
        for value in node.values():
            _inject_ref_enum(value, refs)
    elif isinstance(node, list):
        for value in node:
            _inject_ref_enum(value, refs)


def to_strict_schema(schema: Mapping[str, Any], refs: list[str]) -> dict[str, Any]:
    """Return the strict-mode copy of ``schema`` with the refs enum injected."""

    strict = _strictify(deepcopy(dict(schema)))
    _inject_ref_enum(strict, list(refs))
    return strict


def strict_response_format(
    messages: Any, provider: Any
) -> dict[str, Any] | None:
    """The ``response_format`` for a reducer request, or None to leave it alone."""

    if not guard_enabled():
        return None
    if str(provider or "").strip().lower() not in _STRICT_PROVIDERS:
        return None
    payload = reducer_payload(messages)
    if payload is None:
        return None
    refs = allowed_refs(payload)
    if not refs:
        return None
    schema = payload.get("output_schema")
    if not isinstance(schema, dict) or not schema:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_reducer",
            "strict": True,
            "schema": to_strict_schema(schema, refs),
        },
    }


def install_strict_schema_guard() -> bool:
    """Wrap ``OpenAICompatibleChatClient`` once per process."""

    if not guard_enabled():
        return False

    from memory import clients  # type: ignore[import-not-found]

    if getattr(clients, _INSTALLED_FLAG, False):
        return True

    client_class = clients.OpenAICompatibleChatClient
    original = client_class.chat_with_metadata

    def chat_with_metadata(
        self: Any,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_schema: Mapping[str, Any] | None = None,
    ) -> Any:
        response_format = strict_response_format(messages, getattr(self, "provider", ""))
        if response_format is None:
            return original(self, messages, json_mode=json_mode, json_schema=json_schema)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        payload.update(self.extra_body or {})
        payload["response_format"] = response_format
        clients.disable_request_thinking(payload, self.provider)
        response = clients.requests.post(
            self.endpoint_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        raw_usage = clients._raw_usage_from_response(data)
        return clients.ChatCallResult(
            content=data["choices"][0]["message"]["content"],
            token_usage=clients._standard_token_usage(raw_usage),
            raw_usage=raw_usage,
        )

    client_class.chat_with_metadata = chat_with_metadata
    setattr(clients, _INSTALLED_FLAG, True)
    print(
        "[guard] semantic reducer answers use a strict JSON schema whose "
        f"evidence refs are an enum of the window's refs ({GUARD_ENV}=0 restores "
        "upstream behaviour)",
        file=sys.stderr,
    )
    return True


__all__ = [
    "GUARD_ENV",
    "allowed_refs",
    "guard_enabled",
    "install_strict_schema_guard",
    "reducer_payload",
    "strict_response_format",
    "to_strict_schema",
]
