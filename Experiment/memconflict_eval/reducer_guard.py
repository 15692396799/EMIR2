"""Harness-side repair for the semantic reducer's trigger-ref mismatch.

Every semantic update that reaches the reducer carries two ref lists:

* ``event.trigger_event_ref`` -- the event that triggered the claim, copied from
  the extraction answer;
* ``local_event_refs`` -- the events extracted in *this* window.

Upstream then validates the answer against the second list only::

    unknown_local_refs = [
        ref for ref in local_refs
        if f"{ref_prefix}:{ref}" not in event_refs
    ]
    if unknown_local_refs:
        raise SemanticValidationError(
            "semantic operation references evidence outside supplied local event refs"
        )

``event_refs`` holds exactly the window's extracted events, so a claim whose
trigger was extracted in another window (or was never extracted as an event)
advertises a ref the answer may not cite. A model that cites the ref the prompt
itself handed it is rejected, and because the reducer role runs at
``temperature: 0.0`` the retry produces the same answer:
``retry_v4_call`` re-asks once, the checkpoint guard re-runs the session (the
``[warn] V4BuildStageError ... retrying`` lines), and the persona dies once the
budget is spent. Session 0 of a fresh run shows it immediately -- nothing stale
is involved, which is what makes the warning look wrong there.

The unresolvable ref cannot contribute anything to memory anyway:
``builder.py::_apply_reducer_update`` resolves every cited ref through
``event_refs[f"{ref_prefix}:{ref}"]`` and silently drops the ones that miss.
So this guard drops exactly the refs that cannot resolve, and nothing else:

* the request stops advertising a trigger ref its own window cannot resolve;
* the answer has unresolvable ``evidence_event_refs`` (and an unresolvable
  operation-level ``trigger_event_ref``) removed, so the unit passes instead of
  aborting the session with a validation error that no retry can fix.

Payloads without ``local_event_refs`` -- the extraction, adjudication, planner
and entity-judge calls -- are never touched, and an answer whose refs are all
legal is returned unchanged, byte for byte. ``MEMCONFLICT_REDUCER_GUARD=0``
restores upstream behaviour.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

GUARD_ENV = "MEMCONFLICT_REDUCER_GUARD"
_FALSE_VALUES = {"0", "false", "no", "off"}
_INSTALLED_FLAG = "_memconflict_reducer_guard"
_REF_KEY = "local_event_refs"


def guard_enabled() -> bool:
    """The guard is on unless ``MEMCONFLICT_REDUCER_GUARD=0``."""

    raw = os.getenv(GUARD_ENV)
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in _FALSE_VALUES


def guard_request(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], set[str] | None]:
    """Return the messages to send and the refs the answer may cite.

    ``None`` means "this is not a semantic reducer call": the messages are
    returned untouched and the answer is passed through unexamined.
    """

    for index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        if _REF_KEY not in content:
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        refs = payload.get(_REF_KEY)
        if not isinstance(refs, list) or not refs:
            continue
        allowed = {str(ref) for ref in refs}
        event = payload.get("event")
        trigger = event.get("trigger_event_ref") if isinstance(event, dict) else None
        if trigger is None or str(trigger) in allowed:
            return list(messages), allowed
        guarded = {**payload, "event": {
            key: value for key, value in event.items() if key != "trigger_event_ref"
        }}
        rewritten = list(messages)
        rewritten[index] = {
            **message,
            "content": json.dumps(guarded, ensure_ascii=False),
        }
        return rewritten, allowed
    return list(messages), None


def guard_answer(raw: Any, allowed: set[str] | None) -> Any:
    """Drop refs the builder could not resolve; return ``raw`` when nothing drops."""

    if not allowed or not isinstance(raw, str):
        return raw
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(payload, dict):
        return raw
    operations = payload.get("operations")
    if not isinstance(operations, list):
        return raw
    changed = False
    guarded: list[Any] = []
    for operation in operations:
        if not isinstance(operation, dict):
            guarded.append(operation)
            continue
        cleaned = dict(operation)
        evidence = operation.get("evidence_event_refs")
        if isinstance(evidence, list):
            kept = [ref for ref in evidence if str(ref) in allowed]
            if len(kept) != len(evidence):
                cleaned["evidence_event_refs"] = kept
                changed = True
        trigger = operation.get("trigger_event_ref")
        if trigger is not None and str(trigger) not in allowed:
            cleaned.pop("trigger_event_ref", None)
            changed = True
        guarded.append(cleaned)
    if not changed:
        return raw
    return json.dumps({**payload, "operations": guarded}, ensure_ascii=False)


class ReducerGuardedChatClient:
    """Chat client that keeps the reducer's refs inside its own window."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def chat(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_schema: Any | None = None,
    ) -> Any:
        if not guard_enabled():
            return self._client.chat(
                messages, json_mode=json_mode, json_schema=json_schema
            )
        guarded, allowed = guard_request(messages)
        answer = self._client.chat(
            guarded, json_mode=json_mode, json_schema=json_schema
        )
        return guard_answer(answer, allowed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def install_reducer_guard() -> bool:
    """Wrap ``make_chat_client`` so every V4 client gets the guard once."""

    if not guard_enabled():
        return False

    from memory import clients  # type: ignore[import-not-found]
    from memory.v4 import memory_system  # type: ignore[import-not-found]

    if getattr(clients, _INSTALLED_FLAG, False):
        return True

    original = clients.make_chat_client

    def make_chat_client(config: Any) -> Any:
        client = original(config)
        if not guard_enabled() or isinstance(client, clients.NoopChatClient):
            return client
        return ReducerGuardedChatClient(client)

    clients.make_chat_client = make_chat_client
    # ``memory_system`` imported the factory by name, so patch its binding too.
    memory_system.make_chat_client = make_chat_client
    setattr(clients, _INSTALLED_FLAG, True)
    print(
        "[guard] semantic reducer refs are limited to the window's "
        f"local_event_refs ({GUARD_ENV}=0 restores upstream behaviour)",
        file=sys.stderr,
    )
    return True
