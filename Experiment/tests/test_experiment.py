"""Offline tests for the MemConflict experiment harness.

Run with::

    python -m unittest discover -s Experiment/tests -v

The tests never call an LLM API: the memory system and the chat clients are
replaced with stubs, and the dataset assertions read the released JSONL as data
only.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from memconflict_eval import runtime  # noqa: E402
from memconflict_eval.answering import MemConflictAnswerer  # noqa: E402
from memconflict_eval.data import (  # noqa: E402
    CONFLICT_TYPES,
    dataset_summary,
    flatten_session_dialogue,
    load_personas,
    parse_persona,
)
from memconflict_eval.judging import (  # noqa: E402
    _coerce_accuracy,
    _coerce_flag,
    _coerce_rank,
)
from memconflict_eval.memory import MemConflictMemory, RetrievedMemory, store_dir_for  # noqa: E402
from memconflict_eval import ollama_units, parallel  # noqa: E402
from memconflict_eval.embedding import (  # noqa: E402
    BatchedEmbeddingClient,
    _is_transient,
    embed_batch_limit,
    wrap_embedding_client,
)
from memconflict_eval.metrics import (  # noqa: E402
    aggregate,
    render_detail_table,
    render_table3,
    srs_from_rank,
)
from memconflict_eval.prompts import (  # noqa: E402
    ANSWER_SYSTEM_PROMPTS,
    build_answer_messages,
    build_judge_messages,
    extract_json_object,
    judge_json_schema,
)


DATASET = runtime.default_dataset_path()


# ---------------------------------------------------------------------------
# data reading (point 7, input side)
# ---------------------------------------------------------------------------


class DialogueOrderingTests(unittest.TestCase):
    def test_turn_keys_are_sorted_numerically(self):
        dialogue = {
            "dialogue_turn_10": [{"role": "user", "content": "ten"}],
            "dialogue_turn_2": [{"role": "user", "content": "two"}],
            "dialogue_turn_1": [{"role": "user", "content": "one"}],
        }
        turns = flatten_session_dialogue(dialogue, "2022-01-03")
        self.assertEqual([turn.content for turn in turns], ["one", "two", "ten"])
        self.assertEqual([turn.turn_id for turn in turns], [1, 2, 3])

    def test_non_user_or_assistant_roles_are_dropped(self):
        dialogue = {
            "dialogue_turn_1": [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "ignored"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": ""},
            ]
        }
        turns = flatten_session_dialogue(dialogue, "2022-01-03")
        self.assertEqual([turn.role for turn in turns], ["user", "assistant"])

    def test_memory_turn_carries_the_session_date(self):
        dialogue = {"dialogue_turn_1": [{"role": "user", "content": "x"}]}
        turn = flatten_session_dialogue(dialogue, "2024-05-06")[0]
        payload = turn.to_memory_turn()
        self.assertEqual(payload["timestamp"], "2024-05-06")
        self.assertEqual(payload["session_timestamp"], "2024-05-06")
        self.assertEqual(payload["role"], "user")

    def test_memory_turn_id_is_a_string(self):
        """A numeric turn_id makes the builder model invent a prefix.

        The extracted ``evidence_turn_ids`` are validated against the supplied
        ids, so emitting ``33`` instead of ``"turn_33"`` makes every extracted
        item unrepairable. Keep this a string.
        """
        dialogue = {"dialogue_turn_1": [{"role": "user", "content": "x"}]}
        payload = flatten_session_dialogue(dialogue, "2024-05-06")[0].to_memory_turn()
        self.assertIsInstance(payload["turn_id"], str)
        self.assertEqual(payload["turn_id"], "turn_1")
        self.assertFalse(payload["turn_id"].isdigit())

    def test_unsupported_dialogue_shapes_are_empty(self):
        self.assertEqual(flatten_session_dialogue(None, "2022-01-03"), ())
        self.assertEqual(flatten_session_dialogue([], "2022-01-03"), ())
        self.assertEqual(flatten_session_dialogue({"dialogue_turn_1": "text"}, "2022-01-03"), ())


class QuestionIdentityTests(unittest.TestCase):
    def test_question_key_includes_the_conflict_type(self):
        _, _, question = _sample_question()
        self.assertEqual(question.key, "dynamic_conflict:Q_001")


class PersonaParsingTests(unittest.TestCase):
    def test_personas_are_sorted_by_date(self):
        raw = {
            "ID": "persona-1",
            "Full_Session_Chain": [
                {"Session_ID": 1, "Date": "2022-02-01", "Session_Dialogue": {}},
                {"Session_ID": 0, "Date": "2022-01-01", "Session_Dialogue": {}},
            ],
        }
        persona = parse_persona(raw)
        self.assertEqual([session.session_id for session in persona.sessions], [0, 1])

    def test_missing_session_id_falls_back_to_index(self):
        raw = {"ID": "p", "Full_Session_Chain": [{"Date": "2022-01-01"}]}
        persona = parse_persona(raw)
        self.assertEqual(persona.sessions[0].session_id, 0)

    def test_missing_chain_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_persona({"ID": "p"})


# ---------------------------------------------------------------------------
# released dataset regression (read-only)
# ---------------------------------------------------------------------------


@unittest.skipUnless(DATASET.is_file(), f"dataset not found at {DATASET}")
class ReleasedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.personas = load_personas(DATASET, end_index=1)

    def test_first_record_shape_matches_the_analysis(self):
        persona = self.personas[0]
        self.assertEqual(len(persona.sessions), 53)
        self.assertEqual(persona.question_count, 122)
        self.assertEqual(
            [session.session_id for session in persona.sessions],
            list(range(53)),
        )

    def test_question_type_distribution(self):
        persona = self.personas[0]
        self.assertEqual(
            persona.questions_by_type(),
            {
                "dynamic_conflict": 95,
                "static_conflict": 12,
                "conditional_conflict": 15,
            },
        )

    def test_every_question_has_a_known_conflict_type(self):
        for session in self.personas[0].sessions:
            for question in session.questions:
                self.assertIn(question.conflict_type, CONFLICT_TYPES)
                self.assertTrue(question.question)
                self.assertTrue(question.answer)

    def test_dataset_summary_counts_sessions_and_triggers(self):
        summary = dataset_summary(self.personas)
        self.assertEqual(summary["Session_Count"], 53)
        self.assertEqual(summary["Question_Count"], 122)
        self.assertEqual(summary["Triggered_Session_Count"], 42)


# ---------------------------------------------------------------------------
# prompts (point 8)
# ---------------------------------------------------------------------------


def _sample_question():
    from memconflict_eval.data import Question

    question = Question(
        question_id="Q_001",
        question="Did the user's residence change recently?",
        answer="Yes.",
        conflict_type="dynamic_conflict",
        ability_target="track_state_over_time",
        difficulty="easy",
    )
    return "dynamic_conflict", "Yes.", question


class AnswerPromptTests(unittest.TestCase):
    def test_each_conflict_type_has_its_own_prompt(self):
        self.assertEqual(set(ANSWER_SYSTEM_PROMPTS), set(CONFLICT_TYPES))
        self.assertEqual(len(set(ANSWER_SYSTEM_PROMPTS.values())), 3)

    def test_prompts_are_not_the_locomo_answer_prompt(self):
        for prompt in ANSWER_SYSTEM_PROMPTS.values():
            self.assertNotIn("multi-hop", prompt.lower())
            self.assertNotIn("answer type", prompt.lower())

    def test_dynamic_prompt_mentions_update_direction(self):
        prompt = ANSWER_SYSTEM_PROMPTS["dynamic_conflict"]
        self.assertIn("CURRENT", prompt)
        self.assertIn("previous", prompt)

    def test_static_prompt_demands_the_true_value(self):
        prompt = ANSWER_SYSTEM_PROMPTS["static_conflict"]
        self.assertIn("true value", prompt)
        self.assertIn("Never adopt the contradicting value", prompt)

    def test_conditional_prompt_demands_the_condition(self):
        prompt = ANSWER_SYSTEM_PROMPTS["conditional_conflict"]
        self.assertIn("condition", prompt)

    def test_build_answer_messages_uses_the_routed_prompt(self):
        messages = build_answer_messages("Q?", "ctx", "conditional_conflict")
        self.assertEqual(messages[0]["content"], ANSWER_SYSTEM_PROMPTS["conditional_conflict"])
        self.assertIn("Q?", messages[1]["content"])
        self.assertIn("ctx", messages[1]["content"])

    def test_unknown_conflict_type_is_rejected(self):
        with self.assertRaises(ValueError):
            build_answer_messages("Q?", "ctx", "temporal")


class JudgePromptTests(unittest.TestCase):
    def _messages(self, conflict_type="dynamic_conflict"):
        memories = [
            {"rank": 1, "memory": "moved to Melbourne", "created_at": "2022-02-25", "score": 0.9},
            {"rank": 2, "memory": "lived in Darwin", "created_at": "2022-01-03", "score": 0.5},
        ]
        return build_judge_messages(
            question="Did the user's residence change recently?",
            gold_answer="Yes.",
            model_answer="Yes, they moved to Melbourne.",
            conflict_type=conflict_type,
            memories=memories,
            top_k=3,
        )

    def test_judge_prompt_is_not_the_mem0_binary_prompt(self):
        body = self._messages()[1]["content"]
        self.assertNotIn("CORRECT or WRONG", body)
        self.assertNotIn("generous", body)

    def test_judge_prompt_carries_all_three_inputs(self):
        body = self._messages()[1]["content"]
        self.assertIn("Did the user's residence change recently?", body)
        self.assertIn("Yes.", body)
        self.assertIn("moved to Melbourne", body)

    def test_judge_prompt_asks_for_the_support_rank(self):
        body = self._messages()[1]["content"]
        self.assertIn("support_rank", body)
        self.assertIn("Top-3", body)

    def test_conditional_judge_prompt_states_binary_scoring(self):
        body = self._messages("conditional_conflict")[1]["content"]
        self.assertIn("no partial credit", body)

    def test_judge_schema_matches_the_metric_set(self):
        schema = judge_json_schema(3)
        self.assertEqual(
            set(schema["properties"]),
            {"answer_accuracy", "conflict_handling", "support_rank", "reasoning"},
        )
        self.assertEqual(schema["properties"]["support_rank"]["maximum"], 3)


class JudgeJsonExtractionTests(unittest.TestCase):
    def test_plain_json(self):
        payload = extract_json_object('{"answer_accuracy": 1.0}')
        self.assertEqual(payload["answer_accuracy"], 1.0)

    def test_fenced_json(self):
        payload = extract_json_object('```json\n{"a": 1}\n```')
        self.assertEqual(payload["a"], 1)

    def test_json_wrapped_in_prose(self):
        payload = extract_json_object('Here is the score: {"a": 2} thanks.')
        self.assertEqual(payload["a"], 2)

    def test_nested_braces_are_preserved(self):
        payload = extract_json_object('{"reasoning": "uses { and } inside"}')
        self.assertEqual(payload["reasoning"], "uses { and } inside")

    def test_invalid_response_raises(self):
        with self.assertRaises(ValueError):
            extract_json_object("no json here")
        with self.assertRaises(ValueError):
            extract_json_object("")


# ---------------------------------------------------------------------------
# judge coercion
# ---------------------------------------------------------------------------


class JudgeCoercionTests(unittest.TestCase):
    def test_dynamic_and_static_keep_partial_credit(self):
        self.assertEqual(_coerce_accuracy(1.0, "dynamic_conflict"), 1.0)
        self.assertEqual(_coerce_accuracy(0.5, "dynamic_conflict"), 0.5)
        self.assertEqual(_coerce_accuracy(0.0, "dynamic_conflict"), 0.0)
        self.assertEqual(_coerce_accuracy(0.5, "static_conflict"), 0.5)

    def test_conditional_is_all_or_nothing(self):
        self.assertEqual(_coerce_accuracy(1.0, "conditional_conflict"), 1.0)
        self.assertEqual(_coerce_accuracy(0.5, "conditional_conflict"), 0.0)
        self.assertEqual(_coerce_accuracy(0.0, "conditional_conflict"), 0.0)

    def test_invalid_accuracy_is_zero(self):
        self.assertEqual(_coerce_accuracy("nonsense", "dynamic_conflict"), 0.0)
        self.assertEqual(_coerce_accuracy(None, "dynamic_conflict"), 0.0)

    def test_flag_coercion(self):
        self.assertEqual(_coerce_flag(True), 1)
        self.assertEqual(_coerce_flag(False), 0)
        self.assertEqual(_coerce_flag(1), 1)
        self.assertEqual(_coerce_flag("0"), 0)

    def test_rank_is_clamped_to_the_top_k_window(self):
        self.assertEqual(_coerce_rank(1, 3), 1)
        self.assertEqual(_coerce_rank(3, 3), 3)
        self.assertEqual(_coerce_rank(4, 3), 0)
        self.assertEqual(_coerce_rank(0, 3), 0)
        self.assertEqual(_coerce_rank(None, 3), 0)


# ---------------------------------------------------------------------------
# metrics (point 9)
# ---------------------------------------------------------------------------


class MetricTests(unittest.TestCase):
    def test_srs_formula(self):
        self.assertEqual(srs_from_rank(0), 0.0)
        self.assertAlmostEqual(srs_from_rank(1), 1.0)
        self.assertAlmostEqual(srs_from_rank(3), 1.0 / 2.0)

    def _questions(self):
        rows = []
        # dynamic: one full hit, one partial miss
        rows.append({"conflict_type": "dynamic_conflict", "answer_accuracy": 1.0,
                     "support_rank": 1, "conflict_handling": 1})
        rows.append({"conflict_type": "dynamic_conflict", "answer_accuracy": 0.5,
                     "support_rank": 4, "conflict_handling": 0})
        # static: one wrong
        rows.append({"conflict_type": "static_conflict", "answer_accuracy": 0.0,
                     "support_rank": 2, "conflict_handling": 0})
        # conditional: one right
        rows.append({"conflict_type": "conditional_conflict", "answer_accuracy": 1.0,
                     "support_rank": 3, "conflict_handling": 1})
        return rows

    def test_aggregate_by_conflict_type(self):
        metrics = aggregate(self._questions())
        self.assertEqual(metrics.question_count, 4)
        dynamic = metrics.by_conflict_type["dynamic_conflict"]
        self.assertEqual(dynamic.question_count, 2)
        self.assertAlmostEqual(dynamic.answer_accuracy, 0.75)
        self.assertAlmostEqual(dynamic.seh_at_3, 0.5)  # rank 4 is a miss
        static = metrics.by_conflict_type["static_conflict"]
        self.assertAlmostEqual(static.answer_accuracy, 0.0)
        self.assertAlmostEqual(static.seh_at_3, 1.0)  # rank 2 is a hit
        conditional = metrics.by_conflict_type["conditional_conflict"]
        self.assertAlmostEqual(conditional.answer_accuracy, 1.0)

    def test_average_aa_is_the_mean_of_the_three_types(self):
        metrics = aggregate(self._questions())
        expected = (0.75 + 0.0 + 1.0) / 3.0
        self.assertAlmostEqual(metrics.average_aa, expected)

    def test_missing_conflict_type_reports_none_not_zero(self):
        rows = [row for row in self._questions() if row["conflict_type"] == "dynamic_conflict"]
        metrics = aggregate(rows)
        self.assertIsNone(metrics.by_conflict_type["conditional_conflict"].answer_accuracy)
        self.assertAlmostEqual(metrics.average_aa, 0.75)

    def test_judge_errors_are_counted(self):
        rows = self._questions()
        rows[0]["judge_error"] = "ValueError: boom"
        metrics = aggregate(rows)
        self.assertEqual(metrics.judge_error_count, 1)
        self.assertEqual(metrics.by_conflict_type["dynamic_conflict"].judge_error_count, 1)

    def test_table3_rendering_matches_the_paper_columns(self):
        table = render_table3(aggregate(self._questions()), "EMIR²")
        header, separator, row = table.splitlines()
        self.assertEqual(header.count("|"), 9)
        self.assertIn("Dynamic SEH@3↑", header)
        self.assertIn("Conditional SEH@3↑", header)
        self.assertIn("Average AA↑", header)
        self.assertTrue(row.startswith("| EMIR² |"))
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        self.assertEqual(len(cells), 8)
        self.assertEqual(cells[0], "EMIR²")
        self.assertEqual(cells[-1], f"{aggregate(self._questions()).average_aa:.4f}")

    def test_empty_metrics_render_placeholder(self):
        table = render_table3(aggregate([]), "EMIR²")
        self.assertIn("–", table)

    def test_detail_table_lists_every_conflict_type(self):
        detail = render_detail_table(aggregate(self._questions()))
        for label in ("Dynamic", "Static", "Conditional"):
            self.assertIn(label, detail)


# ---------------------------------------------------------------------------
# memory adapter (point 7, without the real backend)
# ---------------------------------------------------------------------------


class _FakeMemoryItem:
    def __init__(self, title, text, observed_at, score, node_type="event", semantic=()):
        self.title = title
        self.text = text
        self.summary = text
        self.timestamp_start = observed_at
        self.metadata = {"absolute_time_start": observed_at}
        self.score = score
        self.node_type = node_type
        self.semantic_memories = list(semantic)


class _FakeSemantic:
    def __init__(self, subject, predicate, obj):
        self.subject = subject
        self.predicate = predicate
        self.object = obj


class _FakeRetrieval:
    def __init__(self, items):
        self.episodic_memories = items
        self.trace = [{"round": 1}]


class _FakeMemorySystem:
    """Records ingestion order and the memory state seen at each retrieval."""

    def __init__(self, *_args, **_kwargs):
        self.ingested = []
        self.retrieval_snapshots = []
        self.closed = False

    def ingest_conversation(self, namespace, conversation, metadata=None, *, participants=None):
        for blob in conversation:
            self.ingested.append(str(blob["session_id"]))

    def retrieve(self, question, namespace):
        self.retrieval_snapshots.append(tuple(self.ingested))
        return _FakeRetrieval(
            [
                _FakeMemoryItem(
                    "Residence update",
                    "The user relocated to Melbourne, Australia.",
                    "2022-02-25",
                    0.91,
                    semantic=[_FakeSemantic("user", "lives_in", "Melbourne")],
                ),
                _FakeMemoryItem(
                    "Earlier residence",
                    "The user lived in Darwin, Australia.",
                    "2022-01-03",
                    0.42,
                ),
            ]
        )

    def is_namespace_ready(self, namespace):
        return True

    def close(self):
        self.closed = True


class _FakeChatClient:
    def __init__(self, response="answer"):
        self.response = response
        self.calls = []

    def chat(self, messages, json_mode=False, json_schema=None):
        self.calls.append({"json_mode": json_mode, "messages": messages})
        return self.response


def _patch_runtime(fake_system, answer_client, judge_client):
    class _FakeEmbeddingClient:
        def embed_texts(self, texts):
            return [[0.0] for _ in texts]

    runtime.import_retrival_mem = lambda root=None: SimpleNamespace(
        MemorySystem=lambda config, api_history_logger=None, embedding_client=None: fake_system,
        ApiHistoryLogger=lambda _path: None,
        configure_backend_output_paths=lambda *_a, **_k: None,
        make_embedding_client=lambda model_config, resilient=False: _FakeEmbeddingClient(),
        make_chat_client=lambda model_config: (
            judge_client if getattr(model_config, "role", "") == "judge" else answer_client
        ),
    )
    runtime.load_memory_config = lambda config_path=None: SimpleNamespace(
        embedding=SimpleNamespace(provider="ollama", model="fake-embed"),
        answer_model=SimpleNamespace(role="answer", model="fake-answer"),
        judge_model=SimpleNamespace(role="judge", model="fake-judge"),
    )


class MemoryAdapterTests(unittest.TestCase):
    def test_retrieved_memory_dict_shape(self):
        memory = RetrievedMemory(rank=1, memory="m", created_at="2022-02-25", score=0.9)
        self.assertEqual(
            memory.to_dict(),
            {"rank": 1, "memory": "m", "created_at": "2022-02-25", "score": 0.9, "node_type": ""},
        )

    def test_store_dir_is_per_persona(self):
        raw = {"ID": "abcdef123456", "Full_Session_Chain": [{"Date": "2022-01-01"}]}
        persona = parse_persona(raw)
        directory = store_dir_for("out", persona, "v1")
        self.assertIn("abcdef123456", str(directory))
        self.assertEqual(directory.parent.name, "Memory")

    def test_store_dir_keeps_the_whole_persona_id(self):
        """A short suffix could collide; the store must key on the full id."""
        uuids = [
            "3c2e5fe5-a0fc-7e3c-b05c-7104ad748705",
            "75b15745-6d32-726b-be79-d765808bbfe7",
        ]
        slugs = set()
        for persona_id in uuids:
            persona = parse_persona({"ID": persona_id, "Full_Session_Chain": []})
            directory = store_dir_for("out", persona, "v1")
            slugs.add(directory.name)
            self.assertIn(persona_id, directory.name)
        self.assertEqual(len(slugs), len(uuids))

    def test_persona_namespaces_do_not_collide(self):
        first = parse_persona({"ID": "persona-aaaaaa", "Full_Session_Chain": []})
        second = parse_persona({"ID": "persona-bbbbbb", "Full_Session_Chain": []})
        self.assertNotEqual(
            MemConflictMemory.build_namespace(first, "v1"),
            MemConflictMemory.build_namespace(second, "v1"),
        )


class EndToEndOrderingTests(unittest.TestCase):
    """The core of point 7: QA must happen session by session."""

    def _persona(self):
        raw = {
            "ID": "persona-endtoend",
            "Full_Session_Chain": [
                {
                    "Session_ID": 0,
                    "Date": "2022-01-03",
                    "Session_Type": "initial_reveal",
                    "Session_Dialogue": {
                        "dialogue_turn_1": [{"role": "user", "content": "I live in Darwin."}],
                    },
                    "Session_Questions": [],
                },
                {
                    "Session_ID": 1,
                    "Date": "2022-02-25",
                    "Session_Type": "update",
                    "Session_Dialogue": {
                        "dialogue_turn_1": [
                            {"role": "user", "content": "I moved to Melbourne."},
                            {"role": "assistant", "content": "Noted."},
                        ],
                    },
                    "Session_Questions": [
                        {
                            "question_id": "Q_001",
                            "question": "Did the user's residence change recently?",
                            "answer": "Yes.",
                            "conflict_type": "dynamic_conflict",
                            "ability_target": "track_state_over_time",
                            "difficulty": "easy",
                        }
                    ],
                },
            ],
        }
        return parse_persona(raw)

    def test_question_never_sees_a_later_session(self):
        from run_experiment import run_persona

        fake_system = _FakeMemorySystem()
        answer_client = _FakeChatClient("Yes, they moved to Melbourne.")
        _patch_runtime(fake_system, answer_client, _FakeChatClient())

        with tempfile.TemporaryDirectory() as tmp:
            record = run_persona(
                persona=self._persona(),
                output_dir=Path(tmp),
                config_path=Path("unused.yaml"),
                answerer=MemConflictAnswerer(),
                top_k=2,
                stored_top_k=5,
                version="v1",
                keep_memory=False,
            )

        self.assertEqual(fake_system.ingested, ["0", "1"])
        # The single question belongs to session 1, so retrieval must have seen
        # sessions {0, 1} and nothing more.
        self.assertEqual(fake_system.retrieval_snapshots, [("0", "1")])
        self.assertEqual(record["Answered_Question_Count"], 1)
        self.assertEqual(record["Session_Count"], 2)
        answered = record["Sessions"][1]["Questions"][0]
        self.assertEqual(answered["Model_Answer"], "Yes, they moved to Melbourne.")
        self.assertEqual(answered["Memory_System"], runtime.MEMORY_SYSTEM_NAME)

    def test_retrieved_memories_are_persisted_with_ranks(self):
        from run_experiment import run_persona

        fake_system = _FakeMemorySystem()
        _patch_runtime(fake_system, _FakeChatClient("ans"), _FakeChatClient())

        with tempfile.TemporaryDirectory() as tmp:
            record = run_persona(
                persona=self._persona(),
                output_dir=Path(tmp),
                config_path=Path("unused.yaml"),
                answerer=MemConflictAnswerer(),
                top_k=2,
                stored_top_k=5,
                version="v1",
                keep_memory=False,
            )

        memories = record["Sessions"][1]["Questions"][0]["Retrieved_Memories"]
        self.assertEqual([m["rank"] for m in memories], [1, 2])
        self.assertIn("Melbourne", memories[0]["memory"])
        self.assertIn("semantic: user lives_in Melbourne", memories[0]["memory"])
        self.assertEqual(memories[0]["created_at"], "2022-02-25")

    def test_answer_prompt_uses_the_question_conflict_type(self):
        from run_experiment import run_persona

        fake_system = _FakeMemorySystem()
        answer_client = _FakeChatClient("ans")
        _patch_runtime(fake_system, answer_client, _FakeChatClient())

        with tempfile.TemporaryDirectory() as tmp:
            run_persona(
                persona=self._persona(),
                output_dir=Path(tmp),
                config_path=Path("unused.yaml"),
                answerer=MemConflictAnswerer(),
                top_k=2,
                stored_top_k=5,
                version="v1",
                keep_memory=False,
            )

        system_message = answer_client.calls[0]["messages"][0]["content"]
        self.assertEqual(system_message, ANSWER_SYSTEM_PROMPTS["dynamic_conflict"])

    def test_every_completed_session_is_reported_incrementally(self):
        """A persona that dies part-way must not lose its finished sessions."""
        from run_experiment import run_persona

        fake_system = _FakeMemorySystem()
        _patch_runtime(fake_system, _FakeChatClient("ans"), _FakeChatClient())
        rows = []

        with tempfile.TemporaryDirectory() as tmp:
            run_persona(
                persona=self._persona(),
                output_dir=Path(tmp),
                config_path=Path("unused.yaml"),
                answerer=MemConflictAnswerer(),
                top_k=2,
                stored_top_k=5,
                version="v1",
                keep_memory=False,
                on_session=rows.append,
            )

        self.assertEqual([row["Session_ID"] for row in rows], [0, 1])
        self.assertEqual(rows[0]["Questions"], [])
        self.assertEqual(len(rows[1]["Questions"]), 1)
        self.assertEqual(rows[1]["Questions"][0]["Model_Answer"], "ans")


class SessionFallbackTests(unittest.TestCase):
    """Scoring must survive a run that never finished a persona."""

    def test_personas_are_rebuilt_from_the_session_log(self):
        from run_scoring import rebuild_personas_from_sessions

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.jsonl"
            rows = [
                {
                    "Persona_ID": "p1",
                    "Memory_System": "retrival_mem_v4",
                    "Session_ID": 0,
                    "Date": "2022-01-03",
                    "Questions": [],
                },
                {
                    "Persona_ID": "p1",
                    "Session_ID": 5,
                    "Date": "2022-02-25",
                    "Questions": [{"question_id": "Q_001"}],
                },
                {
                    "Persona_ID": "p2",
                    "Session_ID": 0,
                    "Date": "2022-03-01",
                    "Questions": [{"question_id": "Q_001"}, {"question_id": "Q_002"}],
                },
            ]
            with open(path, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            personas = rebuild_personas_from_sessions(path)

        self.assertEqual([p["Persona_ID"] for p in personas], ["p1", "p2"])
        self.assertEqual(len(personas[0]["Sessions"]), 2)
        self.assertEqual(personas[0]["Answered_Question_Count"], 1)
        self.assertEqual(personas[1]["Answered_Question_Count"], 2)
        self.assertTrue(personas[0]["Rebuilt_From_Sessions_JSONL"])


class ScoringPipelineTests(unittest.TestCase):
    def test_score_persona_produces_table3_rows(self):
        from run_scoring import score_persona

        judge_response = json.dumps(
            {
                "answer_accuracy": 1.0,
                "conflict_handling": 1,
                "support_rank": 1,
                "reasoning": "correct",
            }
        )
        judge = _FakeChatClient(judge_response)

        class _JudgeStub:
            top_k = 3

            def judge(self, question, model_answer, memories):
                from memconflict_eval.judging import JudgeResult

                payload = json.loads(judge_response)
                return JudgeResult(
                    answer_accuracy=payload["answer_accuracy"],
                    conflict_handling=payload["conflict_handling"],
                    support_rank=payload["support_rank"],
                    reasoning=payload["reasoning"],
                    duration_ms=1.0,
                    raw_response=judge_response,
                )

        persona = {
            "Persona_ID": "p",
            "Sessions": [
                {
                    "Session_ID": 1,
                    "Questions": [
                        {
                            "question_id": "Q_001",
                            "question": "Did the user's residence change recently?",
                            "answer": "Yes.",
                            "conflict_type": "dynamic_conflict",
                            "Model_Answer": "Yes.",
                            "Retrieved_Memories": [
                                {"rank": 1, "memory": "moved to Melbourne",
                                 "created_at": "2022-02-25", "score": 0.9}
                            ],
                        }
                    ],
                }
            ],
        }

        scored, flat = score_persona(persona, judge=_JudgeStub(), top_k=3)
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0]["conflict_type"], "dynamic_conflict")
        self.assertEqual(flat[0]["support_rank"], 1)
        self.assertIn("Evaluation", scored["Sessions"][0]["Questions"][0])

        metrics = aggregate(flat)
        self.assertAlmostEqual(metrics.by_conflict_type["dynamic_conflict"].seh_at_3, 1.0)
        self.assertAlmostEqual(metrics.average_aa, 1.0)


# ---------------------------------------------------------------------------
# credential preflight
# ---------------------------------------------------------------------------


class EmbeddingBatchTests(unittest.TestCase):
    """Cloud embedding endpoints cap the batch; Ollama does not."""

    class _Recorder:
        def __init__(self):
            self.calls = []

        def embed_texts(self, texts):
            self.calls.append(list(texts))
            return [[float(len(str(t)))] for t in texts]

    def test_dashscope_limit_is_ten(self):
        self.assertEqual(embed_batch_limit("dashscope_bailian"), 10)

    def test_ollama_and_openai_are_not_chunked(self):
        self.assertIsNone(embed_batch_limit("ollama"))
        self.assertIsNone(embed_batch_limit("openai"))

    def test_small_batch_passes_through_unchanged(self):
        inner = self._Recorder()
        client = BatchedEmbeddingClient(inner, 10)
        out = client.embed_texts(["a", "b", "c"])
        self.assertEqual(len(out), 3)
        self.assertEqual(len(inner.calls), 1)

    def test_large_batch_is_split_and_order_is_preserved(self):
        inner = self._Recorder()
        client = BatchedEmbeddingClient(inner, 10)
        texts = [f"t{i}" for i in range(25)]
        out = client.embed_texts(texts)
        self.assertEqual([len(call) for call in inner.calls], [10, 10, 5])
        self.assertEqual(len(out), 25)
        self.assertEqual(out[0], [2.0])      # "t0"
        self.assertEqual(out[24], [3.0])     # "t24"
        self.assertEqual(client.batch_count, 3)

    def test_empty_input_makes_no_request(self):
        inner = self._Recorder()
        self.assertEqual(BatchedEmbeddingClient(inner, 10).embed_texts([]), [])
        self.assertEqual(inner.calls, [])

    def test_wrap_only_applies_to_limited_providers(self):
        inner = self._Recorder()
        self.assertIs(wrap_embedding_client(inner, "ollama"), inner)
        wrapped = wrap_embedding_client(inner, "dashscope_bailian")
        self.assertIsInstance(wrapped, BatchedEmbeddingClient)
        self.assertIs(wrap_embedding_client(wrapped, "dashscope_bailian"), wrapped)

    def test_transient_errors_are_retried_then_succeed(self):
        import requests

        class Flaky:
            def __init__(self):
                self.calls = 0

            def embed_texts(self, texts):
                self.calls += 1
                if self.calls < 3:
                    raise requests.exceptions.ChunkedEncodingError("ended prematurely")
                return [[1.0] for _ in texts]

        client = BatchedEmbeddingClient(Flaky(), 10, backoff_seconds=0.0)
        self.assertEqual(client.embed_texts(["a"]), [[1.0]])
        self.assertEqual(client.retry_count, 2)

    def test_hard_client_errors_are_not_retried(self):
        import requests

        class BadRequest:
            def __init__(self):
                self.calls = 0

            def embed_texts(self, texts):
                self.calls += 1
                response = requests.Response()
                response.status_code = 400
                raise requests.exceptions.HTTPError("400 Client Error", response=response)

        inner = BadRequest()
        client = BatchedEmbeddingClient(inner, 10, backoff_seconds=0.0)
        with self.assertRaises(requests.exceptions.HTTPError):
            client.embed_texts(["a"])
        self.assertEqual(inner.calls, 1)
        self.assertEqual(client.retry_count, 0)

    def test_transient_classification(self):
        import requests

        self.assertTrue(_is_transient(requests.exceptions.Timeout("slow")))
        self.assertTrue(_is_transient(requests.exceptions.ChunkedEncodingError("cut")))
        response = requests.Response()
        response.status_code = 503
        self.assertTrue(_is_transient(requests.exceptions.HTTPError("503", response=response)))
        response.status_code = 400
        self.assertFalse(_is_transient(requests.exceptions.HTTPError("400", response=response)))


class CredentialPreflightTests(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            embedding=SimpleNamespace(provider="ollama"),
            memory_builder=SimpleNamespace(provider="dashscope_bailian"),
            answer_model=SimpleNamespace(provider="dashscope_bailian"),
            judge_model=SimpleNamespace(provider="openrouter"),
            controller=SimpleNamespace(provider="ollama"),
            window_planner=SimpleNamespace(provider="ollama"),
            entity_judge=SimpleNamespace(provider="ollama"),
            adjudication_model=SimpleNamespace(provider="dashscope_bailian"),
            slm=SimpleNamespace(provider="ollama"),
            decomposition_gate=SimpleNamespace(provider="ollama"),
        )

    def test_runner_roles_need_ollama_and_dashscope_but_not_openrouter(self):
        names = runtime.required_env_names(self._config(), runtime.RUNNER_ROLES)
        self.assertIn("OLLAMA_CHAT_ENDPOINT", names)
        self.assertIn("OLLAMA_EMBED_ENDPOINT", names)
        self.assertIn("OLLAMA_LEGACY_EMBEDDINGS_ENDPOINT", names)
        self.assertIn("DASHSCOPE_API_KEY", names)
        self.assertIn("DASHSCOPE_CHAT_COMPLETIONS_ENDPOINT", names)
        self.assertNotIn("OPENROUTER_API_KEY", names)

    def test_scoring_roles_need_only_the_judge_provider(self):
        names = runtime.required_env_names(self._config(), runtime.SCORING_ROLES)
        self.assertEqual(
            names, ["OPENROUTER_API_KEY", "OPENROUTER_CHAT_COMPLETIONS_ENDPOINT"]
        )

    def test_embedding_endpoints_are_not_required_for_chat_roles(self):
        names = runtime.required_env_names(self._config(), ("judge_model",))
        self.assertNotIn("OLLAMA_EMBED_ENDPOINT", names)

    def test_missing_names_reflect_the_environment(self):
        import os

        config = self._config()
        saved = {name: os.environ.pop(name, None) for name in runtime.required_env_names(config)}
        try:
            missing = runtime.missing_env_names(config, runtime.RUNNER_ROLES)
            self.assertEqual(missing, runtime.required_env_names(config, runtime.RUNNER_ROLES))
            os.environ["DASHSCOPE_API_KEY"] = "set"
            missing = runtime.missing_env_names(config, runtime.RUNNER_ROLES)
            self.assertNotIn("DASHSCOPE_API_KEY", missing)
        finally:
            for name in runtime.required_env_names(config):
                os.environ.pop(name, None)
            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value

    def test_env_path_prefers_the_experiment_directory(self):
        # Experiment/.env may or may not exist; the fallback must be the checkout.
        path = runtime.default_env_path()
        self.assertTrue(
            path.name == ".env"
            and (
                path.parent == EXPERIMENT_DIR
                or path.parent == runtime.retrival_mem_root()
            )
        )


def _save_env(*names):
    return {name: os.environ.get(name) for name in names}


def _restore_env(saved):
    for name, value in saved.items():
        os.environ.pop(name, None)
        if value is not None:
            os.environ[name] = value


def _parallel_worker(payload):
    """Module-level so the process pool can pickle it (spawn re-imports)."""
    payload["progress"].put({"job": payload["job"], "pid": os.getpid()})
    return {"job": payload["job"], "pid": os.getpid()}


def _sleepy_worker(payload):
    """Stands in for a persona: the LLM calls are what make it slow."""
    time.sleep(float(payload["seconds"]))
    payload["progress"].put({"job": payload["job"], "pid": os.getpid()})
    return payload["job"]


class _UnpicklableError(RuntimeError):
    """Mimics V4StageError: rebuilt by pickle with the wrong arguments."""

    def __init__(self, context, attempts, cause):
        self.context = context
        self.attempts = attempts
        self.cause = cause
        super().__init__(f"stage failed: {context}")


def _failing_worker(payload):
    """Fails like a bad builder completion does, with an unpicklable error."""
    if payload["job"] == 1:
        raise _UnpicklableError("memory_builder", 2, ValueError("invalid JSON"))
    payload["progress"].put({"job": payload["job"], "pid": os.getpid()})
    return {"job": payload["job"]}


class _RowCollector:
    """Minimal stand-in for the progress sink a job receives."""

    def __init__(self):
        self.rows = []

    def put(self, row):
        self.rows.append(row)


class OllamaUnitTests(unittest.TestCase):
    """Point 16: one Ollama container per GPU, one container per worker."""

    def test_unit_list_is_split_deduplicated_and_ordered(self):
        units = ollama_units.parse_base_urls(
            " http://gpu:41133, http://gpu:41134 ; http://gpu:41133 "
        )
        self.assertEqual(units, ["http://gpu:41133", "http://gpu:41134"])

    def test_base_url_must_be_a_service_root(self):
        for bad in ("gpu:41133", "http://gpu:41133/api/chat", "http://u:p@gpu:1", ""):
            with self.assertRaises(ValueError):
                ollama_units.normalize_base_url(bad)

    def test_trailing_slash_is_stripped(self):
        self.assertEqual(
            ollama_units.normalize_base_url("http://gpu:41133/"), "http://gpu:41133"
        )

    def test_units_prefer_the_list_variable_over_the_single_one(self):
        saved = _save_env("OLLAMA_BASE_URLS", "OLLAMA_BASE_URL")
        try:
            os.environ["OLLAMA_BASE_URLS"] = "http://gpu:1,http://gpu:2"
            os.environ["OLLAMA_BASE_URL"] = "http://gpu:9"
            self.assertEqual(
                ollama_units.configured_units(), ["http://gpu:1", "http://gpu:2"]
            )
            os.environ.pop("OLLAMA_BASE_URLS")
            self.assertEqual(ollama_units.configured_units(), ["http://gpu:9"])
            os.environ.pop("OLLAMA_BASE_URL")
            self.assertEqual(ollama_units.configured_units(), [])
        finally:
            _restore_env(saved)

    def test_apply_unit_rewrites_every_endpoint_the_clients_read(self):
        saved = _save_env(*ollama_units.ENDPOINT_ENV_KEYS, ollama_units.ACTIVE_UNIT_ENV)
        try:
            values = ollama_units.apply_unit("http://gpu:41134/")
            self.assertEqual(values["OLLAMA_BASE_URL"], "http://gpu:41134")
            self.assertEqual(values["OLLAMA_CHAT_ENDPOINT"], "http://gpu:41134/api/chat")
            self.assertEqual(values["OLLAMA_EMBED_ENDPOINT"], "http://gpu:41134/api/embed")
            self.assertEqual(
                values["OLLAMA_LEGACY_EMBEDDINGS_ENDPOINT"],
                "http://gpu:41134/api/embeddings",
            )
            self.assertEqual(ollama_units.active_unit(), "http://gpu:41134")
            ollama_units.clear_unit()
            self.assertIsNone(ollama_units.active_unit())
        finally:
            _restore_env(saved)

    def test_personas_rotate_over_the_units(self):
        self.assertEqual(
            ollama_units.assign_units(["http://gpu:1", "http://gpu:2"], 5),
            ["http://gpu:1", "http://gpu:2", "http://gpu:1", "http://gpu:2", "http://gpu:1"],
        )
        self.assertEqual(ollama_units.assign_units([], 2), [None, None])


class UnitRoutingTests(unittest.TestCase):
    def test_load_memory_config_reapplies_the_unit_after_the_env_file(self):
        """Retrival-Mem loads .env with override=True, so the unit must win."""
        import importlib

        # Earlier tests stub runtime.load_memory_config; reload the real one.
        importlib.reload(runtime)
        saved = _save_env(*ollama_units.ENDPOINT_ENV_KEYS, ollama_units.ACTIVE_UNIT_ENV)
        calls = []
        original_import = runtime.import_retrival_mem

        def fake_load_config(path, env_path=None):
            # What load_dotenv(override=True) does to os.environ.
            os.environ["OLLAMA_CHAT_ENDPOINT"] = "http://dotenv:1/api/chat"
            calls.append((path, env_path))
            return SimpleNamespace()

        runtime.import_retrival_mem = lambda root=None: SimpleNamespace(
            load_config=fake_load_config
        )
        try:
            os.environ[ollama_units.ACTIVE_UNIT_ENV] = "http://gpu:41136"
            runtime.load_memory_config(Path("config.yaml"))
            self.assertEqual(
                os.environ["OLLAMA_CHAT_ENDPOINT"], "http://gpu:41136/api/chat"
            )
            self.assertEqual(calls, [(Path("config.yaml"), runtime.default_env_path())])
        finally:
            runtime.import_retrival_mem = original_import
            _restore_env(saved)


class ParallelPersonaTests(unittest.TestCase):
    """Point 17: personas run side by side, results keep the dataset order."""

    def test_single_worker_keeps_input_order_and_streams_rows(self):
        rows, delivered = [], []
        results = parallel.run_jobs(
            _parallel_worker,
            [{"job": index} for index in range(4)],
            workers=1,
            on_row=rows.append,
            on_result=lambda index, _value: delivered.append(index),
        )
        self.assertEqual([row["job"] for row in results], [0, 1, 2, 3])
        self.assertEqual([row["job"] for row in rows], [0, 1, 2, 3])
        self.assertEqual(delivered, [0, 1, 2, 3])

    def test_process_pool_spreads_personas_over_processes(self):
        rows = []
        results = parallel.run_jobs(
            _parallel_worker,
            [{"job": index} for index in range(4)],
            workers=2,
            on_row=rows.append,
        )
        self.assertEqual([row["job"] for row in results], [0, 1, 2, 3])
        self.assertGreater(len({row["pid"] for row in results}), 1)
        self.assertNotIn(os.getpid(), {row["pid"] for row in results})
        self.assertEqual(len(rows), 4)

    def test_empty_job_list_is_a_no_op(self):
        self.assertEqual(parallel.run_jobs(_parallel_worker, [], workers=3), [])

    def test_unpicklable_worker_error_becomes_a_job_error(self):
        """V4StageError cannot be pickled; the pool must not die over that."""
        with self.assertRaises(parallel.JobError) as caught:
            parallel.run_jobs(_failing_worker, [{"job": 0}, {"job": 1}], workers=2)
        self.assertIn("_UnpicklableError", str(caught.exception))

    def test_a_failed_persona_does_not_stop_the_others(self):
        results = parallel.run_jobs(
            _failing_worker,
            [{"job": index} for index in range(3)],
            workers=2,
            raise_errors=False,
        )
        self.assertEqual(results[0]["job"], 0)
        self.assertIsInstance(results[1], parallel.JobError)
        self.assertIn("memory_builder", str(results[1]))
        self.assertIn("_UnpicklableError", results[1].traceback_text)
        self.assertEqual(results[2]["job"], 2)

    def test_process_pool_cuts_the_wall_clock(self):
        """The scheduling, not the model, is what this proves.

        Four personas that each need one second: serial costs four, four
        workers cost about one. Replace ``_sleepy_worker`` with a real persona
        and the same arithmetic is what points 16+17 buy on the GPU server.
        """
        jobs = [{"job": index, "seconds": 1.0} for index in range(4)]
        started = time.perf_counter()
        parallel.run_jobs(_sleepy_worker, jobs, workers=1)
        serial = time.perf_counter() - started
        started = time.perf_counter()
        parallel.run_jobs(_sleepy_worker, jobs, workers=4)
        pooled = time.perf_counter() - started
        self.assertGreater(serial, 3.5)
        self.assertLess(pooled, serial * 0.6)
        self.assertLess(pooled, 3.0)


class RunnerMainWiringTests(unittest.TestCase):
    """The unit list lives in .env, so it must be read after that file loads."""

    def test_main_loads_the_env_file_before_it_reads_the_unit_list(self):
        import importlib

        importlib.reload(runtime)
        from run_experiment import main

        events = []
        saved = _save_env(
            ollama_units.UNITS_ENV,
            "OLLAMA_EMBED_ENDPOINT",
            "OLLAMA_LEGACY_EMBEDDINGS_ENDPOINT",
        )
        original_env_loader = runtime.load_env_file
        original_units = ollama_units.configured_units
        runtime.load_env_file = lambda: events.append("env-file")
        ollama_units.configured_units = lambda: (events.append("units"), [])[1]
        _patch_runtime(_FakeMemorySystem(), _FakeChatClient("ans"), _FakeChatClient())
        try:
            os.environ["OLLAMA_EMBED_ENDPOINT"] = "http://unit/api/embed"
            os.environ["OLLAMA_LEGACY_EMBEDDINGS_ENDPOINT"] = "http://unit/api/embeddings"
            with tempfile.TemporaryDirectory() as tmp:
                code = main(
                    [
                        "--output-dir",
                        tmp,
                        "--persona-limit",
                        "1",
                        "--max-sessions",
                        "2",
                    ]
                )
                self.assertEqual(code, 0)
                # The .env file is loaded first, otherwise OLLAMA_BASE_URLS is
                # invisible and every persona falls back to one container.
                self.assertEqual(events[:2], ["env-file", "units"])
                meta = json.loads(
                    (Path(tmp) / "run_meta.json").read_text(encoding="utf-8")
                )
                self.assertEqual(meta["Persona_Workers"], 1)
                self.assertEqual(meta["Ollama_Units"], [])
                self.assertTrue(
                    (Path(tmp) / "results.jsonl").read_text(encoding="utf-8").strip()
                )
                self.assertEqual(
                    len(
                        (Path(tmp) / "sessions.jsonl")
                        .read_text(encoding="utf-8")
                        .strip()
                        .splitlines()
                    ),
                    2,
                )
        finally:
            runtime.load_env_file = original_env_loader
            ollama_units.configured_units = original_units
            _restore_env(saved)


class PersonaJobTests(unittest.TestCase):
    def _persona(self):
        return parse_persona(
            {
                "ID": "persona-unit",
                "Full_Session_Chain": [
                    {
                        "Session_ID": 0,
                        "Date": "2022-01-03",
                        "Session_Type": "initial_reveal",
                        "Session_Dialogue": {
                            "dialogue_turn_1": [{"role": "user", "content": "I live in Darwin."}],
                        },
                        "Session_Questions": [],
                    }
                ],
            }
        )

    def test_persona_job_applies_its_unit_and_records_it(self):
        from run_experiment import persona_job

        _patch_runtime(_FakeMemorySystem(), _FakeChatClient("ans"), _FakeChatClient())
        saved = _save_env(*ollama_units.ENDPOINT_ENV_KEYS, ollama_units.ACTIVE_UNIT_ENV)
        collector = _RowCollector()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                record = persona_job(
                    {
                        "persona_index": 0,
                        "persona": self._persona(),
                        "output_dir": tmp,
                        "config_path": "unused.yaml",
                        "unit": "http://gpu:41136",
                        "top_k": 2,
                        "stored_top_k": 5,
                        "version": "v1",
                        "keep_memory": False,
                        "max_sessions": None,
                        "progress": collector,
                    }
                )
            self.assertEqual(record["Ollama_Unit"], "http://gpu:41136")
            self.assertEqual(
                os.environ["OLLAMA_EMBED_ENDPOINT"], "http://gpu:41136/api/embed"
            )
            # The progress channel carries the lifecycle event and then one row
            # per finished session (the log run_scoring.py falls back to).
            events = [row for row in collector.rows if row.get("Event")]
            sessions = [row for row in collector.rows if not row.get("Event")]
            self.assertEqual([row["Event"] for row in events], ["persona_start"])
            self.assertEqual([row["Session_ID"] for row in sessions], [0])
            self.assertEqual(record["Answered_Question_Count"], 0)
        finally:
            _restore_env(saved)

    def test_persona_job_without_a_unit_clears_the_assignment(self):
        from run_experiment import persona_job

        _patch_runtime(_FakeMemorySystem(), _FakeChatClient("ans"), _FakeChatClient())
        saved = _save_env(*ollama_units.ENDPOINT_ENV_KEYS, ollama_units.ACTIVE_UNIT_ENV)
        try:
            os.environ[ollama_units.ACTIVE_UNIT_ENV] = "http://gpu:41133"
            with tempfile.TemporaryDirectory() as tmp:
                record = persona_job(
                    {
                        "persona_index": 0,
                        "persona": self._persona(),
                        "output_dir": tmp,
                        "config_path": "unused.yaml",
                        "unit": None,
                        "top_k": 2,
                        "stored_top_k": 5,
                        "version": "v1",
                        "keep_memory": False,
                    }
                )
            self.assertEqual(record["Ollama_Unit"], "")
            self.assertIsNone(ollama_units.active_unit())
        finally:
            _restore_env(saved)

    def test_runners_expose_the_worker_flags(self):
        from run_experiment import build_arg_parser as runner_parser
        from run_scoring import build_arg_parser as scoring_parser

        self.assertEqual(runner_parser().parse_args([]).persona_workers, 1)
        self.assertEqual(
            runner_parser().parse_args(["--persona-workers", "4"]).persona_workers, 4
        )
        self.assertEqual(
            scoring_parser()
            .parse_args(["--run-dir", "runs/x", "--judge-workers", "3"])
            .judge_workers,
            3,
        )
        self.assertEqual(
            runner_parser()
            .parse_args(["--ollama-units", "http://gpu:1,http://gpu:2"])
            .ollama_units,
            "http://gpu:1,http://gpu:2",
        )


if __name__ == "__main__":
    unittest.main()
