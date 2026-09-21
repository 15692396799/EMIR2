"""Tests for the harness-side semantic reducer guard.

The fixtures below are the recorded payload and answer of the call that killed
session 0 of ``a7850e51`` in the 2026-09-21 shard_1 run: the prompt advertises
``event.trigger_event_ref = "turn_17"`` while ``local_event_refs`` lists
``turn_1, turn_11, turn_13, turn_18, turn_3, turn_40, turn_5, turn_7, turn_80,
turn_9`` -- the trigger was not extracted as an event in that window. The model
cited the ref it was shown, upstream rejected it
(``semantic operation references evidence outside supplied local event refs``),
the checkpoint guard re-ran the session and, at temperature 0.0, the identical
answer came back on every retry (the run logged the same eight payloads four
times).

``memconflict_eval/reducer_guard.py`` drops exactly the refs the builder could
not have resolved anyway. These tests pin that, and that nothing else changes.

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

from memconflict_eval import runtime  # noqa: E402
from memconflict_eval.reducer_guard import (  # noqa: E402
    GUARD_ENV,
    ReducerGuardedChatClient,
    guard_answer,
    guard_request,
    install_reducer_guard,
)

# Puts Retrival-Mem/src on sys.path and installs the guards, like a run does.
runtime.import_retrival_mem()

LOCAL_REFS = [
    "turn_1",
    "turn_11",
    "turn_13",
    "turn_18",
    "turn_3",
    "turn_40",
    "turn_5",
    "turn_7",
    "turn_80",
    "turn_9",
]

RECORDED_PAYLOAD = {
    "chain": {"name": "work style", "topic": "work_style"},
    "active_state": None,
    "fact_snapshot": [],
    "known_fact_keys": [],
    "event": {
        "evidence": ["I am open to small changes, but not to a huge overhaul."],
        "trigger_event_ref": "turn_17",
        "valid_from": "2022-01-03",
    },
    "candidate_claims": [
        {
            "aspect": "changes",
            "dimension": "behavior",
            "subject": "Wei Zhang",
            "value": "is open to trying small changes",
        }
    ],
    "local_event_refs": LOCAL_REFS,
}

RECORDED_ANSWER = json.dumps(
    {
        "status": "apply",
        "operations": [
            {
                "operation": "add",
                "fact_key": None,
                "subject": "Wei Zhang",
                "dimension": "behavior",
                "aspect": "changes",
                "value": "is open to trying small changes",
                "evidence_event_refs": ["turn_17"],
            }
        ],
        "full_summary": "Wei Zhang is open to trying small changes.",
        "reason": "The evidence supports the addition.",
    },
    ensure_ascii=False,
)


def messages(payload: dict, **extra: object) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Return only valid JSON."},
        {
            "role": "user",
            "content": json.dumps({**payload, **extra}, ensure_ascii=False),
        },
    ]


class RecordingClient:
    """Chat client that records the request and replays one answer."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.seen: list[list[dict[str, str]]] = []

    def chat(self, messages, json_mode=False, json_schema=None):  # noqa: ANN001
        self.seen.append([dict(message) for message in messages])
        return self.answer


class ReducerRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.pop(GUARD_ENV, None)

    def tearDown(self) -> None:
        os.environ.pop(GUARD_ENV, None)
        if self._saved is not None:
            os.environ[GUARD_ENV] = self._saved

    def test_an_unresolvable_trigger_ref_is_not_advertised(self) -> None:
        """The recorded payload asked the model to cite a ref it must not cite."""

        guarded, allowed = guard_request(messages(RECORDED_PAYLOAD))

        self.assertEqual(allowed, set(LOCAL_REFS))
        self.assertEqual(len(guarded), 2)
        payload = json.loads(guarded[-1]["content"])
        self.assertNotIn("trigger_event_ref", payload["event"])
        # The evidence text the reducer judges is untouched.
        self.assertEqual(payload["event"]["evidence"], RECORDED_PAYLOAD["event"]["evidence"])
        self.assertEqual(payload["candidate_claims"], RECORDED_PAYLOAD["candidate_claims"])

    def test_a_resolvable_trigger_ref_is_left_alone(self) -> None:
        payload = {
            **RECORDED_PAYLOAD,
            "event": {**RECORDED_PAYLOAD["event"], "trigger_event_ref": "turn_18"},
        }
        original = messages(payload)

        guarded, allowed = guard_request(original)

        self.assertEqual(guarded, original)
        self.assertEqual(allowed, set(LOCAL_REFS))

    def test_a_payload_without_local_refs_is_untouched(self) -> None:
        original = messages({"task": "extract_memory_window", "turns": []})

        guarded, allowed = guard_request(original)

        self.assertEqual(guarded, original)
        self.assertIsNone(allowed)


class ReducerAnswerTests(unittest.TestCase):
    def test_the_recorded_answer_loses_only_the_unresolvable_ref(self) -> None:
        guarded = guard_answer(RECORDED_ANSWER, set(LOCAL_REFS))

        payload = json.loads(guarded)
        self.assertEqual(payload["status"], "apply")
        self.assertEqual(payload["operations"][0]["evidence_event_refs"], [])
        self.assertEqual(payload["operations"][0]["value"], "is open to trying small changes")
        self.assertEqual(payload["full_summary"], "Wei Zhang is open to trying small changes.")

    def test_a_legal_answer_is_returned_unchanged(self) -> None:
        legal = json.dumps(
            {
                "status": "apply",
                "operations": [
                    {"operation": "add", "evidence_event_refs": ["turn_18"]}
                ],
            }
        )
        self.assertIs(guard_answer(legal, set(LOCAL_REFS)), legal)
        self.assertIs(guard_answer("not json", set(LOCAL_REFS)), "not json")
        self.assertIs(guard_answer(RECORDED_ANSWER, None), RECORDED_ANSWER)

    def test_an_operation_trigger_ref_is_dropped_when_unresolvable(self) -> None:
        answer = json.dumps(
            {
                "status": "apply",
                "operations": [
                    {
                        "operation": "add",
                        "evidence_event_refs": ["turn_17", "turn_18"],
                        "trigger_event_ref": "turn_17",
                    }
                ],
            }
        )

        operation = json.loads(guard_answer(answer, set(LOCAL_REFS)))["operations"][0]

        self.assertEqual(operation["evidence_event_refs"], ["turn_18"])
        self.assertNotIn("trigger_event_ref", operation)


class ReducerClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.pop(GUARD_ENV, None)
        install_reducer_guard()

    def tearDown(self) -> None:
        os.environ.pop(GUARD_ENV, None)
        if self._saved is not None:
            os.environ[GUARD_ENV] = self._saved

    def test_the_wrapper_cleans_both_halves_of_the_call(self) -> None:
        client = RecordingClient(RECORDED_ANSWER)

        answer = ReducerGuardedChatClient(client).chat(messages(RECORDED_PAYLOAD))

        seen = json.loads(client.seen[0][-1]["content"])
        self.assertNotIn("trigger_event_ref", seen["event"])
        cited = [
            ref
            for operation in json.loads(answer)["operations"]
            for ref in operation["evidence_event_refs"]
        ]
        self.assertEqual(cited, [])
        self.assertTrue(set(cited) <= set(LOCAL_REFS))

    def test_the_guard_can_be_switched_off(self) -> None:
        os.environ[GUARD_ENV] = "0"
        client = RecordingClient(RECORDED_ANSWER)
        original = messages(RECORDED_PAYLOAD)

        answer = ReducerGuardedChatClient(client).chat(original)

        self.assertEqual(client.seen[0], original)
        self.assertEqual(answer, RECORDED_ANSWER)

    def test_install_is_idempotent_and_reaches_the_memory_system(self) -> None:
        from memory import clients
        from memory.v4 import memory_system

        self.assertTrue(install_reducer_guard())
        self.assertIs(clients.make_chat_client, memory_system.make_chat_client)


if __name__ == "__main__":
    unittest.main()
