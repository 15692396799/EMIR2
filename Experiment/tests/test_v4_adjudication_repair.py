"""Cross-window adjudication contract tests for the V4 memory builder.

These tests drive ``V4MemoryBuilder._adjudicate_cross_window`` with a scripted
chat client and a stub store, so no LLM API is ever called. They pin the two
properties the shard runs depended on and that upstream did not guarantee:

* a decision that swaps source and target still answers the same candidate pair;
* a candidate pair the model leaves out is a "no relation" vote, while
  structurally invalid answers are re-asked once with validator feedback before
  the build is aborted.

Run with::

    python -m unittest discover -s Experiment/tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RETRIVAL_MEM_SRC = EXPERIMENT_DIR.parent / "Retrival-Mem" / "src"
for _path in (str(EXPERIMENT_DIR), str(RETRIVAL_MEM_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from memory.v4.builder import V4MemoryBuilder  # noqa: E402
from memory.v4.config import FailurePolicyConfig  # noqa: E402
from memory.v4.failure import V4BuildStageError  # noqa: E402
from memory.v4.schemas import MemoryNode  # noqa: E402


class ScriptedChatClient:
    """Replay canned adjudication answers and record every prompt."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_schema: object | None = None,
    ) -> str:
        self.calls.append([dict(message) for message in messages])
        if not self.responses:
            raise AssertionError("adjudication client called more often than scripted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return str(response)


def make_node(node_id: str, title: str) -> MemoryNode:
    return MemoryNode(
        id=node_id,
        namespace="memconflict:test:v1",
        scope_id="scope_1",
        chain_id="chain_1",
        topic_id="topic_1",
        node_type="event",
        title=title,
        summary=title,
        text=title,
        evidence_refs=[{"content": title}],
    )


def make_builder(client: ScriptedChatClient) -> V4MemoryBuilder:
    """Build a real V4MemoryBuilder without touching storage or the network.

    ``_adjudicate_cross_window`` reaches the store only through ``getattr``
    probes (checkpoint load/record), so an empty namespace is a complete stub.
    """

    builder = object.__new__(V4MemoryBuilder)
    builder.store = SimpleNamespace()
    builder.adjudication_client = client
    builder.model_identities = {"adjudication": ("dashscope_bailian", "test-model")}
    # Zero backoff keeps the retry path instant in tests.
    builder.failure_policy = FailurePolicyConfig(
        max_attempts=3, retry_initial_delay_seconds=0.0, retry_backoff_multiplier=1.0
    )
    return builder


def decision(source_ref: str, target_ref: str, edge_type: str = "CONTEXT_FOR") -> dict:
    return {
        "source_ref": source_ref,
        "target_ref": target_ref,
        "edge_type": edge_type,
        "confidence": 0.8,
        "explanation": "grounded in the supplied evidence",
    }


def payload(response: dict) -> str:
    return json.dumps(response, ensure_ascii=False)


class AdjudicationPairingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.by_id = {"n1": make_node("n1", "first event"), "n2": make_node("n2", "second event")}
        self.candidates = [("n1", "n2", "CONTEXT_FOR", "shared significant entity: e1")]

    def test_reversed_orientation_is_accepted(self) -> None:
        """A swapped source/target answers the same candidate pair."""

        client = ScriptedChatClient([payload({"decisions": [decision("r2", "r1")]})])
        result = make_builder(client)._adjudicate_cross_window(self.candidates, self.by_id)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(result), 1)
        source, target, edge_type, confidence, trust, explanation = result[0]
        # The model's direction is kept; the proposal is only a hint.
        self.assertEqual((source, target), ("n2", "n1"))
        self.assertEqual(edge_type, "CONTEXT_FOR")
        self.assertAlmostEqual(confidence, 0.8)
        self.assertEqual(trust, "explicit")
        self.assertTrue(explanation)

    def test_null_edge_type_rejects_without_error(self) -> None:
        client = ScriptedChatClient([payload({"decisions": [decision("r1", "r2", "null")]})])
        result = make_builder(client)._adjudicate_cross_window(self.candidates, self.by_id)

        self.assertEqual(result, [])
        self.assertEqual(len(client.calls), 1)

    def test_omitted_pair_is_treated_as_no_relation(self) -> None:
        """A pair the answer skips does not abort the build any more."""

        by_id = {**self.by_id, "n3": make_node("n3", "third event")}
        candidates = [
            ("n1", "n2", "CONTEXT_FOR", "shared significant entity: e1"),
            ("n1", "n3", "CONTEXT_FOR", "shared significant entity: e2"),
        ]
        client = ScriptedChatClient([payload({"decisions": [decision("r1", "r2")]})])
        result = make_builder(client)._adjudicate_cross_window(candidates, by_id)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual([(item[0], item[1]) for item in result], [("n1", "n2")])

    def test_empty_decisions_list_rejects_every_pair(self) -> None:
        client = ScriptedChatClient([payload({"decisions": []})])
        result = make_builder(client)._adjudicate_cross_window(self.candidates, self.by_id)

        self.assertEqual(result, [])
        self.assertEqual(len(client.calls), 1)


class AdjudicationRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.by_id = {"n1": make_node("n1", "first event"), "n2": make_node("n2", "second event")}
        self.candidates = [("n1", "n2", "CONTEXT_FOR", "shared significant entity: e1")]

    def test_off_candidate_pair_is_reasked_with_feedback(self) -> None:
        """An invented pair is repaired by one feedback retry, not a fatal abort."""

        by_id = {**self.by_id, "n3": make_node("n3", "third event")}
        candidates = [
            ("n1", "n2", "CONTEXT_FOR", "shared significant entity: e1"),
            ("n1", "n3", "CONTEXT_FOR", "shared significant entity: e2"),
        ]
        client = ScriptedChatClient([
            payload({"decisions": [decision("r2", "r3")]}),
            payload({"decisions": [decision("r1", "r2"), decision("r1", "r3")]}),
        ])
        result = make_builder(client)._adjudicate_cross_window(candidates, by_id)

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(
            [(item[0], item[1]) for item in result], [("n1", "n2"), ("n1", "n3")]
        )

        feedback = self._feedback_payload(client.calls[1])
        self.assertIn("not a candidate pair", feedback["validator_feedback"]["message"])
        self.assertEqual(
            feedback["validator_feedback"]["required_candidate_pairs"],
            [
                {"source_ref": "r1", "target_ref": "r2"},
                {"source_ref": "r1", "target_ref": "r3"},
            ],
        )
        # The repair prompt must still carry the native schema contract.
        self.assertIn("output_schema", feedback)

    def test_unknown_ref_is_reasked_with_feedback(self) -> None:
        client = ScriptedChatClient([
            payload({"decisions": [decision("r2", "r9")]}),
            payload({"decisions": [decision("r1", "r2")]}),
        ])
        result = make_builder(client)._adjudicate_cross_window(self.candidates, self.by_id)

        self.assertEqual(len(client.calls), 2)
        self.assertEqual([(item[0], item[1]) for item in result], [("n1", "n2")])
        feedback = self._feedback_payload(client.calls[1])
        self.assertIn("unknown node", feedback["validator_feedback"]["message"])

    def test_unparsable_answer_is_reasked_with_feedback(self) -> None:
        client = ScriptedChatClient([
            "",
            payload({"decisions": [decision("r1", "r2")]}),
        ])
        result = make_builder(client)._adjudicate_cross_window(self.candidates, self.by_id)

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(result), 1)
        feedback = self._feedback_payload(client.calls[1])
        self.assertEqual(feedback["validator_feedback"]["previous_decisions"], None)

    def test_persistent_structural_error_still_fails(self) -> None:
        """Repair is a courtesy, not a licence to keep a broken answer."""

        bad = payload({"decisions": [decision("r1", "r2", "NOT_AN_EDGE")]})
        client = ScriptedChatClient([bad, bad])
        builder = make_builder(client)

        with self.assertRaises(V4BuildStageError) as raised:
            builder._adjudicate_cross_window(self.candidates, self.by_id)

        self.assertEqual(len(client.calls), 2)
        self.assertIn("invalid edge type", str(raised.exception))

    def test_repaired_prompt_replays_previous_decisions(self) -> None:
        """Feedback includes a compact projection of the previous answer."""

        client = ScriptedChatClient([
            payload({"decisions": [decision("r1", "r2", "NOT_AN_EDGE")]}),
            payload({"decisions": [decision("r1", "r2")]}),
        ])
        make_builder(client)._adjudicate_cross_window(self.candidates, self.by_id)

        feedback = self._feedback_payload(client.calls[1])
        self.assertEqual(
            feedback["validator_feedback"]["previous_decisions"],
            [{"source_ref": "r1", "target_ref": "r2", "edge_type": "NOT_AN_EDGE"}],
        )

    @staticmethod
    def _feedback_payload(messages: list[dict[str, str]]) -> dict:
        """Return the repair payload of the last user message of a prompt."""

        user_messages = [message for message in messages if message["role"] == "user"]
        payload_ = json.loads(user_messages[-1]["content"])
        if "validator_feedback" not in payload_:
            raise AssertionError("prompt carries no validator_feedback")
        return payload_


if __name__ == "__main__":
    unittest.main()
