"""Strict structured outputs for the semantic reducer (point 37).

The guard turns the reducer request into an OpenAI structured-output request
whose `evidence_event_refs` are an enum of the refs the request advertises, so
the lane cannot answer with a ref from another window -- the failure that made
the checkpoint guard re-run whole sessions and eventually killed personas.

Run with::

    python -m unittest discover -s Experiment/tests -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RETRIVAL_MEM_SRC = EXPERIMENT_DIR.parent / "Retrival-Mem" / "src"
for _path in (str(EXPERIMENT_DIR), str(RETRIVAL_MEM_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from memconflict_eval import strict_schema_guard as guard  # noqa: E402

from memory import clients  # noqa: E402
from memory.v4.semantic import SEMANTIC_REDUCER_JSON_SCHEMA  # noqa: E402


REFS = ["ev_1", "ev_2", "ev_7"]


def reducer_messages(refs: list[str] | None = None) -> list[dict[str, str]]:
    """A reducer request shaped like the one Retrival-Mem builds."""

    refs = list(REFS if refs is None else refs)
    payload = {
        "chain": {"topic": "t", "name": "topic"},
        "active_state": None,
        "fact_snapshot": [],
        "event": {
            "valid_from": "2022-01-01",
            "trigger_event_ref": refs[0] if refs else None,
            "evidence": ["x"],
        },
        "candidate_claims": [{"subject": "s", "dimension": "d", "aspect": "a", "value": "v"}],
        "known_fact_keys": [],
        "local_event_refs": refs,
        "output_schema": json.loads(json.dumps(SEMANTIC_REDUCER_JSON_SCHEMA)),
    }
    return [
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


class StrictSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.pop(guard.GUARD_ENV, None)

    def tearDown(self) -> None:
        if self._saved is not None:
            os.environ[guard.GUARD_ENV] = self._saved

    def test_strict_schema_is_inside_the_supported_subset(self) -> None:
        strict = guard.to_strict_schema(SEMANTIC_REDUCER_JSON_SCHEMA, REFS)
        nodes = list(walk(strict))
        for node in nodes:
            for key in ("oneOf", "minItems", "maxItems", "uniqueItems", "minLength",
                        "maxLength", "minimum", "maximum", "pattern", "format"):
                self.assertNotIn(key, node, key)
        for node in nodes:
            if isinstance(node.get("properties"), dict):
                self.assertEqual(node.get("type"), "object")
                self.assertIs(node.get("additionalProperties"), False)
                self.assertEqual(node.get("required"), sorted(node["properties"]))
        # The one unconstrained slot has to be typed for strict mode.
        value_schemas = [
            node["properties"]["value"]
            for node in nodes
            if isinstance(node.get("properties"), dict) and "value" in node["properties"]
        ]
        self.assertTrue(value_schemas)
        for value_schema in value_schemas:
            self.assertIn("anyOf", value_schema)

    def test_every_evidence_ref_list_is_an_enum_of_the_advertised_refs(self) -> None:
        strict = guard.to_strict_schema(SEMANTIC_REDUCER_JSON_SCHEMA, REFS)
        lists = [
            node["properties"][guard._REF_FIELD]
            for node in walk(strict)
            if isinstance(node.get("properties"), dict) and guard._REF_FIELD in node["properties"]
        ]
        self.assertGreaterEqual(len(lists), 3)  # add / replace / reinforce+retract
        for schema in lists:
            self.assertEqual(schema["items"]["enum"], REFS)

    def test_only_reducer_requests_on_strict_lanes_are_rewritten(self) -> None:
        messages = reducer_messages()
        for provider in ("openrouter", "openai", "azure"):
            response_format = guard.strict_response_format(messages, provider)
            self.assertIsNotNone(response_format, provider)
            self.assertEqual(response_format["type"], "json_schema")
            self.assertTrue(response_format["json_schema"]["strict"])
            schema = json.dumps(response_format["json_schema"]["schema"])
            self.assertIn('"enum": ["ev_1", "ev_2", "ev_7"]', schema)
        for provider in ("dashscope_bailian", "ollama", "modelscope_openai_compatible"):
            self.assertIsNone(guard.strict_response_format(messages, provider), provider)

    def test_non_reducer_payloads_are_left_alone(self) -> None:
        builder_messages = [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": json.dumps({"turns": [{"turn_id": "turn_1"}]})},
        ]
        self.assertIsNone(guard.strict_response_format(builder_messages, "openrouter"))
        # A reducer payload without refs (empty window) keeps upstream behaviour.
        self.assertIsNone(guard.strict_response_format(reducer_messages([]), "openrouter"))

    def test_guard_can_be_switched_off(self) -> None:
        os.environ[guard.GUARD_ENV] = "0"
        try:
            self.assertIsNone(guard.strict_response_format(reducer_messages(), "openrouter"))
            self.assertFalse(guard.guard_enabled())
        finally:
            os.environ.pop(guard.GUARD_ENV, None)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class StrictSchemaClientTests(unittest.TestCase):
    """The patched client must send the strict schema and nothing else new."""

    def setUp(self) -> None:
        os.environ.pop(guard.GUARD_ENV, None)
        self.sent: list[dict] = []
        self.original_post = clients.requests.post

        def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
            self.sent.append(dict(json))
            return _FakeResponse({
                "choices": [{"message": {"content": '{"status": "uncertain", "operations": [], "full_summary": "", "reason": "r"}'}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            })

        clients.requests.post = fake_post
        self.original_method = clients.OpenAICompatibleChatClient.chat_with_metadata
        # The installation flag is process-wide and survives a restore of the
        # method, so reset it before installing again.
        self.flag_saved = getattr(clients, guard._INSTALLED_FLAG, None)
        setattr(clients, guard._INSTALLED_FLAG, False)
        guard.install_strict_schema_guard()

    def tearDown(self) -> None:
        clients.requests.post = self.original_post
        clients.OpenAICompatibleChatClient.chat_with_metadata = self.original_method
        if self.flag_saved is None:
            try:
                delattr(clients, guard._INSTALLED_FLAG)
            except AttributeError:
                pass
        else:
            setattr(clients, guard._INSTALLED_FLAG, self.flag_saved)
        os.environ.pop(guard.GUARD_ENV, None)

    def test_reducer_call_uses_json_schema(self) -> None:
        client = clients.OpenAICompatibleChatClient(
            endpoint_url="http://gateway/v1/chat/completions",
            model="openai/gpt-4o-mini",
            api_key="k",
            provider="openrouter",
        )
        client.chat(reducer_messages(), json_mode=True, json_schema=SEMANTIC_REDUCER_JSON_SCHEMA)
        response_format = self.sent[-1]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(
            response_format["json_schema"]["schema"]["properties"]["status"]["enum"],
            ["apply", "uncertain"],
        )

    def test_other_calls_keep_the_json_object_shape(self) -> None:
        client = clients.OpenAICompatibleChatClient(
            endpoint_url="http://gateway/v1/chat/completions",
            model="openai/gpt-4o-mini",
            api_key="k",
            provider="openrouter",
        )
        client.chat(
            [{"role": "user", "content": json.dumps({"turns": [{"turn_id": "turn_1"}]})}],
            json_mode=True,
            json_schema={"type": "object"},
        )
        self.assertEqual(self.sent[-1]["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
