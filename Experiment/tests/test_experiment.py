"""Offline tests for the MemConflict experiment harness.

Run with::

    python -m unittest discover -s Experiment/tests -v

The tests never call an LLM API: the memory system and the chat clients are
replaced with stubs, and the dataset assertions read the released JSONL as data
only.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
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
from memconflict_eval.progress import ProgressReporter  # noqa: E402
from memconflict_eval.embedding import (  # noqa: E402
    BatchedEmbeddingClient,
    _is_transient,
    embed_batch_limit,
    wrap_embedding_client,
)
from memconflict_eval.metrics import (  # noqa: E402
    aggregate,
    aggregate_by_k,
    render_detail_table,
    render_table3,
    render_table5,
    render_table6,
    render_white_box_by_k,
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


class EmptyAnswerGuardTests(unittest.TestCase):
    """Point 33: a blank completion is an Answer_Error, not a stored answer.

    The answering role is a reasoning model now, and such a model can spend its
    whole budget on reasoning tokens and return empty content. Storing that as
    the answer would show up as a wrong answer in AA with no trace of why.
    """

    def _answerer(self, response):
        from memconflict_eval.answering import MemConflictAnswerer

        answerer = MemConflictAnswerer.__new__(MemConflictAnswerer)
        answerer.config = SimpleNamespace(
            answer_model=SimpleNamespace(provider="openrouter")
        )
        answerer.client = _FakeChatClient(response)
        return answerer

    def test_blank_completion_raises_instead_of_storing_a_blank_answer(self):
        from memconflict_eval.answering import EmptyAnswerError

        _conflict_type, _gold, question = _sample_question()
        for blank in (None, "", "   \n"):
            with self.subTest(response=repr(blank)):
                with self.assertRaises(EmptyAnswerError):
                    self._answerer(blank).answer(question, [], memory_context="ctx")

    def test_a_real_answer_is_returned_trimmed(self):
        _conflict_type, _gold, question = _sample_question()
        result = self._answerer("  Yes.  ").answer(
            question, [], memory_context="ctx"
        )
        self.assertEqual(result.text, "Yes.")
        self.assertEqual(result.context, "ctx")


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


def _markdown_cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _sample_judge_rows():
    """Four judged questions covering all three conflict types."""
    return [
        {"conflict_type": "dynamic_conflict", "answer_accuracy": 1.0,
         "support_rank": 1, "conflict_handling": 1},
        {"conflict_type": "dynamic_conflict", "answer_accuracy": 0.5,
         "support_rank": 4, "conflict_handling": 0},
        {"conflict_type": "static_conflict", "answer_accuracy": 0.0,
         "support_rank": 2, "conflict_handling": 1},
        {"conflict_type": "conditional_conflict", "answer_accuracy": 1.0,
         "support_rank": 3, "conflict_handling": 0},
    ]


class Table5Table6Tests(unittest.TestCase):
    """Tables 5 and 6 are re-projections of the Table 3 judge pass."""

    def test_table5_reports_uocs_and_crs(self):
        metrics = aggregate(_sample_judge_rows())
        header, separator, row = render_table5(metrics, "EMIR²").splitlines()
        self.assertEqual(
            _markdown_cells(header),
            ["Method", "Dynamic AA↑", "Dynamic UOCS↑", "Static AA↑",
             "Static CRS↑", "Conditional AA↑", "Average AA↑"],
        )
        cells = _markdown_cells(row)
        self.assertEqual(cells[0], "EMIR²")
        # Dynamic UOCS is the dynamic conflict-handling mean ...
        self.assertEqual(cells[2], "0.5000")
        # ... and CRS the static one, not the static answer accuracy.
        self.assertEqual(cells[3], "0.0000")
        self.assertEqual(cells[4], "1.0000")

    def test_table6_reports_seh_and_srs_per_conflict_type(self):
        metrics = aggregate(_sample_judge_rows())
        header, _, row = render_table6(metrics, "EMIR²").splitlines()
        self.assertEqual(
            _markdown_cells(header),
            ["Method", "Dynamic SEH@3↑", "Dynamic SRS↑", "Static SEH@3↑",
             "Static SRS↑", "Conditional SEH@3↑", "Conditional SRS↑",
             "Average SEH@3↑", "Average SRS↑"],
        )
        cells = _markdown_cells(row)
        self.assertEqual(cells[1], "0.5000")  # rank 1 hit, rank 4 miss
        self.assertEqual(cells[2], "0.5000")  # (1.0 + 0.0) / 2
        self.assertEqual(cells[3], "1.0000")  # static rank 2 is a hit
        self.assertAlmostEqual(float(cells[4]), srs_from_rank(2), places=4)
        self.assertEqual(cells[7], "0.8333")  # average SEH@3

    def test_the_three_tables_share_their_overlapping_cells(self):
        metrics = aggregate(_sample_judge_rows())
        table3 = _markdown_cells(render_table3(metrics, "EMIR²").splitlines()[2])
        table5 = _markdown_cells(render_table5(metrics, "EMIR²").splitlines()[2])
        table6 = _markdown_cells(render_table6(metrics, "EMIR²").splitlines()[2])
        # AA columns of Table 5 must equal the AA columns of Table 3 (and the
        # average), because both read the same judged accuracy.
        self.assertEqual([table5[1], table5[3], table5[5], table5[6]],
                         [table3[1], table3[3], table3[5], table3[7]])
        # SEH@3 columns of Table 6 must equal Table 3's white-box columns.
        self.assertEqual([table6[1], table6[3], table6[5]], [table3[2], table3[4], table3[6]])

    def test_white_box_windows_are_cut_from_one_support_rank(self):
        rows = [
            {"conflict_type": "dynamic_conflict", "answer_accuracy": 1.0,
             "support_rank": 4, "conflict_handling": 1},
        ]
        by_k = aggregate_by_k(rows, [2, 3, 5])
        dynamic = {key: value.by_conflict_type["dynamic_conflict"] for key, value in by_k.items()}
        self.assertEqual(dynamic["2"].seh_at_3, 0.0)
        self.assertEqual(dynamic["3"].seh_at_3, 0.0)
        self.assertEqual(dynamic["5"].seh_at_3, 1.0)
        self.assertEqual(dynamic["3"].srs, 0.0)
        self.assertAlmostEqual(dynamic["5"].srs, srs_from_rank(4))
        # AA never moves with the white-box window.
        self.assertEqual(dynamic["2"].answer_accuracy, dynamic["5"].answer_accuracy)

    def test_by_k_table_prints_one_row_per_window(self):
        by_k = aggregate_by_k(_sample_judge_rows(), [2, 3, 5])
        lines = render_white_box_by_k(by_k, "EMIR²").splitlines()
        self.assertEqual(len(lines), 5)  # header + separator + three windows
        self.assertEqual([_markdown_cells(line)[1] for line in lines[2:]], ["@2", "@3", "@5"])

    def test_conditional_rows_have_no_conflict_handling_diagnostic(self):
        metrics = aggregate(_sample_judge_rows())
        self.assertIsNone(metrics.by_conflict_type["conditional_conflict"].conflict_handling)
        # Dynamic and static keep theirs (UOCS / CRS).
        self.assertIsNotNone(metrics.by_conflict_type["dynamic_conflict"].conflict_handling)
        self.assertIsNotNone(metrics.by_conflict_type["static_conflict"].conflict_handling)

    def test_scoring_judges_each_question_once_for_all_three_tables(self):
        from run_scoring import score_persona

        calls = []

        class _CountingJudge:
            top_k = 5

            def judge(self, question, model_answer, memories):
                from memconflict_eval.judging import JudgeResult

                calls.append(question.question_id)
                return JudgeResult(
                    answer_accuracy=1.0,
                    conflict_handling=1,
                    support_rank=2,
                    reasoning="ok",
                    duration_ms=1.0,
                    raw_response="{}",
                )

        persona = {
            "Persona_ID": "p",
            "Sessions": [
                {
                    "Session_ID": 1,
                    "Questions": [
                        {"question_id": "Q_1", "conflict_type": "dynamic_conflict",
                         "Model_Answer": "a", "Retrieved_Memories": []},
                        {"question_id": "Q_2", "conflict_type": "static_conflict",
                         "Model_Answer": "b", "Retrieved_Memories": []},
                        {"question_id": "Q_3", "conflict_type": "conditional_conflict",
                         "Model_Answer": "c", "Retrieved_Memories": []},
                    ],
                }
            ],
        }

        _, flat = score_persona(persona, judge=_CountingJudge(), top_k=5)
        self.assertEqual(len(calls), 3)  # one judge call per question, no more
        metrics = aggregate(flat, white_box_k=3)
        for table in (render_table3(metrics), render_table5(metrics), render_table6(metrics)):
            self.assertIn("EMIR²", table)

    def test_scoring_cli_defaults_keep_judging_at_the_primary_window(self):
        from run_scoring import build_arg_parser, parse_k_values

        defaults = build_arg_parser().parse_args(["--run-dir", "run"])
        self.assertEqual(defaults.top_k, 3)
        self.assertIsNone(defaults.white_box_k)
        self.assertIsNone(defaults.judge_top_k)
        self.assertEqual(parse_k_values("2,3,5"), [2, 3, 5])
        self.assertEqual(parse_k_values("5,3,3"), [3, 5])
        with self.assertRaises(ValueError):
            parse_k_values("0")

    def test_rebuild_tool_reproduces_the_scoring_pipeline(self):
        from tools.rebuild_tables import rows_from_scored_personas

        personas = [
            {
                "Persona_ID": "p",
                "Sessions": [
                    {
                        "Questions": [
                            {"question_id": "Q_1", "conflict_type": "dynamic_conflict",
                             "Evaluation": {"Answer_Accuracy": 1.0, "Conflict_Handling": 1,
                                            "Support_Rank": 1, "Judge_Error": None}},
                            {"question_id": "Q_2", "conflict_type": "static_conflict",
                             "Evaluation": {"Answer_Accuracy": 0.5, "Conflict_Handling": 0,
                                            "Support_Rank": 4, "Judge_Error": None}},
                        ]
                    }
                ],
            }
        ]
        rows = rows_from_scored_personas(personas)
        self.assertEqual(len(rows), 2)
        metrics = aggregate(rows, white_box_k=3)
        self.assertEqual(metrics.by_conflict_type["dynamic_conflict"].answer_accuracy, 1.0)
        # Rank 4 is outside the Top-3 window, so the static row is a miss.
        self.assertEqual(metrics.by_conflict_type["static_conflict"].seh_at_3, 0.0)
        # A question without an Evaluation is counted as a judged failure
        # instead of silently disappearing from the denominator.
        rows = rows_from_scored_personas(
            [{"Sessions": [{"Questions": [{"conflict_type": "dynamic_conflict"}]}]}]
        )
        self.assertEqual(aggregate(rows).by_conflict_type["dynamic_conflict"].question_count, 1)
        self.assertEqual(aggregate(rows).judge_error_count, 1)

    def test_rebuild_tool_writes_all_three_tables_from_scores_only(self):
        from tools import rebuild_tables

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            personas = [
                {
                    "Persona_ID": "p",
                    "Sessions": [
                        {
                            "Questions": [
                                {"question_id": f"Q_{index}", "conflict_type": conflict_type,
                                 "Evaluation": {"Answer_Accuracy": 1.0,
                                                "Conflict_Handling": 1, "Support_Rank": 2,
                                                "Judge_Error": None}}
                                for index, conflict_type in enumerate(CONFLICT_TYPES)
                            ]
                        }
                    ],
                }
            ]
            with open(run_dir / "scores.jsonl", "w", encoding="utf-8") as handle:
                for persona in personas:
                    handle.write(json.dumps(persona) + "\n")
            # A Top-3 judging pass, as recorded by run_scoring.py.
            (run_dir / "metrics.json").write_text(
                json.dumps({"Judge_Top_K": 3}), encoding="utf-8"
            )

            code = rebuild_tables.main(["--run-dir", str(run_dir), "--white-box-k", "2,3,5"])

            self.assertEqual(code, 0)
            for name in ("table3.md", "table5.md", "table6.md", "tables.md"):
                self.assertTrue((run_dir / name).is_file(), name)
            table3 = (run_dir / "table3.md").read_text(encoding="utf-8")
            table5 = (run_dir / "table5.md").read_text(encoding="utf-8")
            table6 = (run_dir / "table6.md").read_text(encoding="utf-8")
            self.assertIn("Dynamic AA↑", table3)
            self.assertIn("Dynamic UOCS↑", table5)
            self.assertIn("Dynamic SRS↑", table6)
            # Same judging pass, so the overlapping cells must agree exactly.
            row3 = _markdown_cells(table3.splitlines()[2])
            row5 = _markdown_cells(table5.splitlines()[2])
            row6 = _markdown_cells(table6.splitlines()[2])
            self.assertEqual([row5[1], row5[3], row5[5]], [row3[1], row3[3], row3[5]])
            self.assertEqual([row6[1], row6[3], row6[5]], [row3[2], row3[4], row3[6]])

    def test_rebuild_tool_refuses_rows_without_a_conflict_type(self):
        from tools import rebuild_tables

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            persona = {
                "Persona_ID": "p",
                "Sessions": [
                    {"Questions": [{"question_id": "Q_1", "conflict_type": "",
                                    "Evaluation": {"Answer_Accuracy": 1.0}}]}
                ],
            }
            (run_dir / "scores.jsonl").write_text(json.dumps(persona) + "\n", encoding="utf-8")
            code = rebuild_tables.main(["--run-dir", str(run_dir)])
            self.assertEqual(code, 2)
            self.assertFalse((run_dir / "table3.md").is_file())

    def test_rebuild_tool_reports_the_recorded_judge_window(self):
        from tools.rebuild_tables import recorded_judge_window

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            scores = run_dir / "scores.jsonl"
            scores.write_text("", encoding="utf-8")
            self.assertIsNone(recorded_judge_window(scores))
            (run_dir / "metrics.json").write_text(
                json.dumps({"White_Box_Top_K": 3}), encoding="utf-8"
            )
            self.assertEqual(recorded_judge_window(scores), 3)
            (run_dir / "metrics.json").write_text(
                json.dumps({"Judge_Top_K": 5}), encoding="utf-8"
            )
            self.assertEqual(recorded_judge_window(scores), 5)


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

    def test_a_resumed_record_gets_its_earlier_sessions_back(self):
        """--resume rewrites a persona with only the sessions it replayed.

        The already-answered earlier sessions live in sessions.jsonl; dropping
        them shrank the scored sample silently (2026-09-20: 44 of 46 questions).
        """
        from run_scoring import merge_personas_with_sessions

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.jsonl"
            rows = [
                {
                    "Persona_ID": "p1",
                    "Memory_System": "retrival_mem_v4",
                    "Session_ID": session_id,
                    "Date": f"2022-01-{session_id + 1:02d}",
                    "Ingest": {"Session_ID": session_id},
                    "Questions": (
                        [{"question_id": "Q_001"}]
                        if session_id in (5, 7)
                        else []
                    ),
                }
                for session_id in range(0, 11)
            ]
            with open(path, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            resumed = {
                "Persona_ID": "p1",
                "Memory_System": "retrival_mem_v4",
                "Session_Count": 5,
                "Answered_Question_Count": 1,
                "Sessions": [
                    {
                        "Session_ID": session_id,
                        "Date": f"2022-01-{session_id + 1:02d}",
                        "Questions": ([{"question_id": "Q_001"}] if session_id == 7 else []),
                    }
                    for session_id in (6, 7, 8, 9, 10)
                ],
            }
            merged, recovered = merge_personas_with_sessions([resumed], path)

        persona = merged[0]
        self.assertEqual(recovered, [])  # it is not a rebuilt persona
        self.assertEqual([s["Session_ID"] for s in persona["Sessions"]], list(range(11)))
        self.assertEqual(persona["Answered_Question_Count"], 2)
        self.assertEqual(persona["Session_Count"], 11)
        self.assertEqual(persona["Merged_Sessions_From_JSONL"], [0, 1, 2, 3, 4, 5])

    def test_merging_twice_changes_nothing(self):
        from run_scoring import merge_personas_with_sessions

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.jsonl"
            with open(path, "w", encoding="utf-8") as handle:
                for session_id in (0, 1):
                    handle.write(
                        json.dumps(
                            {
                                "Persona_ID": "p1",
                                "Session_ID": session_id,
                                "Date": f"2022-01-0{session_id + 1}",
                                "Questions": [],
                            }
                        )
                        + "\n"
                    )
            complete = {
                "Persona_ID": "p1",
                "Sessions": [{"Session_ID": 0}, {"Session_ID": 1}],
                "Answered_Question_Count": 0,
                "Session_Count": 2,
            }
            once, _ = merge_personas_with_sessions([complete], path)
            twice, _ = merge_personas_with_sessions(once, path)

        self.assertEqual([s["Session_ID"] for s in twice[0]["Sessions"]], [0, 1])
        self.assertNotIn("Merged_Sessions_From_JSONL", twice[0])


class ScoringMainTests(unittest.TestCase):
    """run_scoring writes the tables plus the traceability fields."""

    def test_metrics_record_the_judge_and_partial_coverage(self):
        from run_scoring import main as scoring_main

        judge_reply = json.dumps(
            {
                "answer_accuracy": 1.0,
                "conflict_handling": 1,
                "support_rank": 1,
                "reasoning": "supported by the first memory",
            }
        )
        saved = _save_env("OPENROUTER_API_KEY", "OPENROUTER_CHAT_COMPLETIONS_ENDPOINT")
        original_loader = runtime.load_memory_config
        original_builder = runtime.build_chat_client
        runtime.load_memory_config = lambda config_path=None: SimpleNamespace(
            judge_model=SimpleNamespace(
                provider="openrouter", model="openai/gpt-4o-mini", role="judge"
            )
        )
        runtime.build_chat_client = lambda model_config: _FakeChatClient(judge_reply)
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        os.environ["OPENROUTER_CHAT_COMPLETIONS_ENDPOINT"] = "https://example.invalid/chat"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "merged"
                run_dir.mkdir()
                (run_dir / "results.jsonl").write_text(
                    json.dumps(
                        {
                            "Persona_ID": "persona-0",
                            "Memory_System": "retrival_mem_v4",
                            "Sessions": [
                                {
                                    "Session_ID": 0,
                                    "Questions": [
                                        {
                                            "question_id": "Q_001",
                                            "question": "Where does the user live?",
                                            "answer": "Melbourne.",
                                            "conflict_type": "dynamic_conflict",
                                            "Model_Answer": "Melbourne.",
                                            "Retrieved_Memories": [
                                                {
                                                    "rank": 1,
                                                    "memory": "moved to Melbourne",
                                                    "created_at": "2022-02-25",
                                                    "score": 0.9,
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (run_dir / "run_meta.json").write_text(
                    json.dumps(
                        {
                            "Persona_Unit_Assignments": {"persona-0": None},
                            "Personas_Expected_From_Dataset": 30,
                            "Partial_Merge": True,
                            "Shards_Merged": [str(run_dir.parent / "shard_1")],
                        }
                    ),
                    encoding="utf-8",
                )
                with contextlib.redirect_stderr(io.StringIO()):
                    code = scoring_main(["--run-dir", str(run_dir)])
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

            self.assertEqual(code, 0)
            self.assertEqual(metrics["Judge_Model"], "openai/gpt-4o-mini")
            self.assertEqual(metrics["Judge_Provider"], "openrouter")
            self.assertTrue(metrics["Partial_Merge"])
            self.assertEqual(metrics["Dataset_Personas_Expected"], 30)
            self.assertEqual(metrics["Personas_Scored"], 1)
        finally:
            runtime.load_memory_config = original_loader
            runtime.build_chat_client = original_builder
            _restore_env(saved)


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

    def test_split_reachable_separates_the_dead_lanes(self):
        """Point 16: a down container must be visible before the run starts."""

        def fake_probe(base_url, timeout=5.0):
            return (False, "URLError: refused") if "41134" in base_url else (True, "8 model(s)")

        original = ollama_units.probe_unit
        ollama_units.probe_unit = fake_probe
        try:
            reachable, unreachable = ollama_units.split_reachable(
                ["http://gpu:41133", "http://gpu:41134", "http://gpu:41135"]
            )
        finally:
            ollama_units.probe_unit = original
        self.assertEqual(reachable, ["http://gpu:41133", "http://gpu:41135"])
        self.assertEqual(unreachable, [("http://gpu:41134", "URLError: refused")])

    def test_preflight_stops_on_a_dead_lane_and_ignores_it_on_request(self):
        """A dead lane used to kill its personas hours in; now it stops the run."""

        from run_experiment import preflight_units

        original = ollama_units.split_reachable
        ollama_units.split_reachable = lambda units, timeout=5.0: (
            [unit for unit in units if unit != "http://gpu:41134"],
            [("http://gpu:41134", "URLError: refused")],
        )
        units = ["http://gpu:41133", "http://gpu:41134"]
        try:
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertIsNone(preflight_units(units))
            self.assertIn("41134", stderr.getvalue())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    preflight_units(units, allow_missing=True), ["http://gpu:41133"]
                )
            ollama_units.split_reachable = lambda units, timeout=5.0: (
                [],
                [(unit, "down") for unit in units],
            )
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertIsNone(preflight_units(units, allow_missing=True))
        finally:
            ollama_units.split_reachable = original


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
                        "--no-progress",
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


class _FakeRetrievalReport:
    def __init__(self, memories):
        self.memories = memories
        self.round_count = 1
        self.duration_ms = 5.0


class _FakeAnswer:
    def __init__(self, text="ans"):
        self.text = text
        self.duration_ms = 3.0


class _FakeRetrievalMemory:
    """Minimal stand-in for MemConflictMemory used by the QA scheduler."""

    namespace = "memconflict:test:v1"

    def __init__(self, delay=0.0, fail_on=()):
        self.delay = delay
        self.fail_on = set(fail_on)
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def retrieve(self, question, *, keep=None):
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay)
            if question in self.fail_on:
                raise RuntimeError(f"retrieval exploded on {question}")
            return _FakeRetrievalReport(
                [
                    RetrievedMemory(
                        rank=1, memory=question, created_at="2022-01-01", score=1.0
                    )
                ]
            )
        finally:
            with self._lock:
                self.in_flight -= 1


class _FakeAnswerer:
    def answer(self, question, memories, *, namespace="", memory_context=None):
        return _FakeAnswer(f"answer to {question.question}")


def _questions_persona(count=4):
    raw = {
        "ID": "persona-qa",
        "Full_Session_Chain": [
            {
                "Session_ID": 0,
                "Date": "2022-01-03",
                "Session_Type": "initial_reveal",
                "Session_Dialogue": {
                    "dialogue_turn_1": [{"role": "user", "content": "hello"}],
                },
                "Session_Questions": [
                    {
                        "question_id": f"Q_{index:03d}",
                        "question": f"question {index}",
                        "answer": "gold",
                        "conflict_type": "dynamic_conflict",
                        "ability_target": "track_state_over_time",
                        "difficulty": "easy",
                    }
                    for index in range(count)
                ],
            }
        ],
    }
    return parse_persona(raw)


class ParallelAnsweringTests(unittest.TestCase):
    """Point 17b: one session's questions answer side by side."""

    def test_serial_worker_count_keeps_the_original_loop(self):
        from run_experiment import answer_session_questions

        memory = _FakeRetrievalMemory(delay=0.05)
        session = _questions_persona(3).sessions[0]
        rows = answer_session_questions(
            memory=memory,
            answerer=_FakeAnswerer(),
            session=session,
            top_k=3,
            stored_top_k=5,
            workers=1,
        )
        self.assertEqual([row["question_id"] for row in rows], ["Q_000", "Q_001", "Q_002"])
        self.assertEqual(memory.max_in_flight, 1)

    def test_questions_run_concurrently_and_keep_their_order(self):
        from run_experiment import answer_session_questions

        memory = _FakeRetrievalMemory(delay=0.2)
        session = _questions_persona(4).sessions[0]
        started = time.perf_counter()
        rows = answer_session_questions(
            memory=memory,
            answerer=_FakeAnswerer(),
            session=session,
            top_k=3,
            stored_top_k=5,
            workers=4,
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(
            [row["question_id"] for row in rows],
            ["Q_000", "Q_001", "Q_002", "Q_003"],
        )
        self.assertEqual(memory.max_in_flight, 4)
        self.assertLess(elapsed, 0.6)  # serial would need >= 0.8 s

    def test_one_bad_question_does_not_lose_the_others(self):
        from run_experiment import answer_session_questions

        memory = _FakeRetrievalMemory(delay=0.0, fail_on={"question 1"})
        session = _questions_persona(3).sessions[0]
        rows = answer_session_questions(
            memory=memory,
            answerer=_FakeAnswerer(),
            session=session,
            top_k=3,
            stored_top_k=5,
            workers=3,
        )
        self.assertEqual(rows[1]["Model_Answer"], "")
        self.assertIn("retrieval exploded", rows[1]["Answer_Error"])
        self.assertEqual(rows[0]["Model_Answer"], "answer to question 0")
        self.assertEqual(rows[2]["Model_Answer"], "answer to question 2")


class ResumeTests(unittest.TestCase):
    def test_completed_sessions_are_read_from_the_progress_log(self):
        from run_experiment import load_completed_sessions

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"Persona_ID": "p1", "Session_ID": 0}),
                        json.dumps({"Persona_ID": "p1", "Session_ID": 1}),
                        json.dumps({"Persona_ID": "p2", "Session_ID": 0}),
                        "{not json",
                        json.dumps({"Persona_ID": "p3", "Session_ID": "7"}),
                    ]
                ),
                encoding="utf-8",
            )
            completed = load_completed_sessions(path)
            self.assertEqual(completed["p1"], {0, 1})
            self.assertEqual(completed["p2"], {0})
            self.assertNotIn("p3", completed)  # non-integer session ids are ignored
        self.assertEqual(load_completed_sessions(Path("does-not-exist.jsonl")), {})

    def test_run_persona_skips_finished_sessions(self):
        from run_experiment import run_persona

        fake_system = _FakeMemorySystem()
        _patch_runtime(fake_system, _FakeChatClient("ans"), _FakeChatClient())
        persona = EndToEndOrderingTests()._persona()
        with tempfile.TemporaryDirectory() as tmp:
            record = run_persona(
                persona=persona,
                output_dir=Path(tmp),
                config_path=Path("unused.yaml"),
                answerer=MemConflictAnswerer(),
                top_k=2,
                stored_top_k=5,
                version="v1",
                keep_memory=True,
                skip_session_ids=(0,),
            )
        self.assertEqual(record["Sessions_Skipped"], 1)
        self.assertEqual(record["Session_Count"], 1)
        self.assertEqual([session["Session_ID"] for session in record["Sessions"]], [1])
        # Only the non-skipped session reached the memory system.
        self.assertEqual(fake_system.ingested, ["1"])


class ConcurrencyOverrideTests(unittest.TestCase):
    def test_overrides_reach_both_thread_pools(self):
        saved = _save_env(
            runtime.EXTRACTION_WORKERS_ENV, runtime.ENTITY_JUDGE_WORKERS_ENV
        )
        try:
            os.environ[runtime.EXTRACTION_WORKERS_ENV] = "2"
            os.environ[runtime.ENTITY_JUDGE_WORKERS_ENV] = "1"
            config = SimpleNamespace(
                memory=SimpleNamespace(
                    memory_extraction_workers=8,
                    backends={"v4": {"entity_judge_workers": 2}},
                )
            )
            applied = runtime.apply_concurrency_overrides(config)
            self.assertEqual(config.memory.memory_extraction_workers, 2)
            self.assertEqual(config.memory.backends["v4"]["entity_judge_workers"], 1)
            self.assertEqual(
                applied,
                {"memory_extraction_workers": 2, "entity_judge_workers": 1},
            )
        finally:
            _restore_env(saved)

    def test_no_override_leaves_the_config_alone(self):
        saved = _save_env(
            runtime.EXTRACTION_WORKERS_ENV, runtime.ENTITY_JUDGE_WORKERS_ENV
        )
        try:
            config = SimpleNamespace(
                memory=SimpleNamespace(
                    memory_extraction_workers=8,
                    backends={"v4": {"entity_judge_workers": 2}},
                )
            )
            self.assertEqual(runtime.apply_concurrency_overrides(config), {})
            self.assertEqual(config.memory.memory_extraction_workers, 8)
            self.assertEqual(config.memory.backends["v4"]["entity_judge_workers"], 2)
        finally:
            _restore_env(saved)


class ScoringMergeTests(unittest.TestCase):
    """P0-2: a failed persona's finished sessions still reach the table."""

    def test_duplicate_persona_records_are_collapsed(self):
        """--resume appends; the richer record must win, not both."""
        from run_scoring import load_personas_from_results

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"Persona_ID": "p1", "Answered_Question_Count": 1, "Sessions": []}
                        ),
                        json.dumps(
                            {"Persona_ID": "p1", "Answered_Question_Count": 5, "Sessions": []}
                        ),
                        json.dumps(
                            {"Persona_ID": "p2", "Answered_Question_Count": 2, "Sessions": []}
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            rows = load_personas_from_results(path)
        self.assertEqual([row["Persona_ID"] for row in rows], ["p1", "p2"])
        self.assertEqual(rows[0]["Answered_Question_Count"], 5)

    @staticmethod
    def _session_row(persona_id, session_id, questions=1):
        return {
            "Persona_ID": persona_id,
            "Memory_System": "retrival_mem_v4",
            "Session_ID": session_id,
            "Date": "2022-01-03",
            "Session_Type": "update",
            "Questions": [
                {
                    "question_id": f"Q_{index}",
                    "conflict_type": "dynamic_conflict",
                    "Model_Answer": "ans",
                }
                for index in range(questions)
            ],
        }

    def test_missing_personas_are_rebuilt_and_merged(self):
        from run_scoring import merge_personas_with_sessions

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(self._session_row("p1", 0)),
                        json.dumps(self._session_row("p2", 0)),
                        json.dumps(self._session_row("p2", 1, questions=2)),
                    ]
                ),
                encoding="utf-8",
            )
            merged, recovered = merge_personas_with_sessions(
                [{"Persona_ID": "p1", "Sessions": [{"Session_ID": 0, "Questions": []}]}],
                path,
            )
        self.assertEqual([row["Persona_ID"] for row in merged], ["p1", "p2"])
        self.assertEqual(recovered, ["p2"])
        p2 = merged[1]
        self.assertTrue(p2["Rebuilt_From_Sessions_JSONL"])
        self.assertTrue(p2["Partial_Persona"])
        self.assertEqual(len(p2["Sessions"]), 2)
        self.assertEqual(p2["Answered_Question_Count"], 3)

    def test_nothing_to_merge_keeps_the_results_untouched(self):
        from run_scoring import merge_personas_with_sessions

        rows = [{"Persona_ID": "p1", "Sessions": []}]
        merged, recovered = merge_personas_with_sessions(
            rows, Path("missing-sessions.jsonl")
        )
        self.assertEqual(merged, rows)
        self.assertEqual(recovered, [])


class ShardMergeTests(unittest.TestCase):
    """P1-3: sharded runs merge into one scoreable directory."""

    def _shard(self, root: Path, name: str, persona_ids: list[str], wall: float):
        shard = root / name
        shard.mkdir(parents=True, exist_ok=True)
        with open(shard / "results.jsonl", "w", encoding="utf-8") as handle:
            for persona_id in persona_ids:
                handle.write(
                    json.dumps(
                        {"Persona_ID": persona_id, "Memory_System": "retrival_mem_v4"}
                    )
                    + "\n"
                )
        with open(shard / "sessions.jsonl", "w", encoding="utf-8") as handle:
            for persona_id in persona_ids:
                handle.write(
                    json.dumps({"Persona_ID": persona_id, "Session_ID": 0}) + "\n"
                )
        (shard / "run_meta.json").write_text(
            json.dumps(
                {
                    "Wall_Clock_s": wall,
                    "Persona_Workers": 4,
                    "Ollama_Units": ["http://unit:1"],
                }
            ),
            encoding="utf-8",
        )
        return shard

    def test_clean_shards_merge(self):
        from tools.merge_shards import merge

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._shard(root, "shard_0", ["p1", "p2"], 100.0)
            second = self._shard(root, "shard_1", ["p3"], 200.0)
            merged, status = merge([first, second], dataset=None)
        self.assertEqual(status, 0)
        self.assertEqual(merged["summary"]["Persona_Count"], 3)
        self.assertEqual(merged["summary"]["Session_Count"], 3)
        self.assertEqual(merged["summary"]["Shard_Wall_Clock_Sum_s"], 300.0)
        self.assertEqual(merged["summary"]["Duplicate_Personas"], [])

    def test_overlapping_shards_are_reported(self):
        from tools.merge_shards import merge

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._shard(root, "shard_0", ["p1", "p2"], 100.0)
            second = self._shard(root, "shard_1", ["p2", "p3"], 100.0)
            merged, status = merge([first, second], dataset=None)
        self.assertEqual(status, 1)
        self.assertEqual(merged["summary"]["Duplicate_Personas"], ["p2"])
        self.assertEqual(merged["summary"]["Persona_Count"], 3)


def _http_error(status):
    import requests

    response = SimpleNamespace(status_code=status)
    return requests.exceptions.HTTPError(f"{status} error", response=response)


class RetryWrapperTests(unittest.TestCase):
    """OpenRouter's geo 403 arrives in bursts; the judge must ride it out."""

    def test_retryable_statuses(self):
        from memconflict_eval.retrying import is_retryable

        import requests

        for status in (403, 408, 429, 500, 502, 503):
            self.assertTrue(is_retryable(_http_error(status)), status)
        for status in (400, 401, 404, 422):
            self.assertFalse(is_retryable(_http_error(status)), status)
        self.assertTrue(is_retryable(requests.exceptions.ConnectionError("proxy down")))

    def test_403_then_success_is_retried(self):
        from memconflict_eval.retrying import RetryingChatClient

        class Flaky:
            def __init__(self):
                self.calls = 0

            def chat(self, messages, json_mode=False, json_schema=None):
                self.calls += 1
                if self.calls == 1:
                    raise _http_error(403)
                return "ok"

        sleeps = []
        inner = Flaky()
        client = RetryingChatClient(inner, attempts=3, backoff_seconds=0.5, sleeper=sleeps.append)
        self.assertEqual(client.chat([{"role": "user", "content": "hi"}]), "ok")
        self.assertEqual(inner.calls, 2)
        self.assertEqual(client.retry_count, 1)
        self.assertEqual(sleeps, [0.5])

    def test_a_hard_error_is_not_retried(self):
        from memconflict_eval.retrying import RetryingChatClient

        class Broken:
            def __init__(self):
                self.calls = 0

            def chat(self, messages, json_mode=False, json_schema=None):
                self.calls += 1
                raise _http_error(400)

        inner = Broken()
        client = RetryingChatClient(inner, attempts=3, backoff_seconds=0.0, sleeper=lambda _s: None)
        with self.assertRaises(Exception):
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(inner.calls, 1)

    def test_giving_up_after_the_last_attempt(self):
        from memconflict_eval.retrying import RetryingChatClient

        class Always:
            def __init__(self):
                self.calls = 0

            def chat(self, messages, json_mode=False, json_schema=None):
                self.calls += 1
                raise _http_error(403)

        inner = Always()
        client = RetryingChatClient(inner, attempts=2, backoff_seconds=0.0, sleeper=lambda _s: None)
        with self.assertRaises(Exception):
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(inner.calls, 2)

    def test_env_can_disable_the_wrapper(self):
        from memconflict_eval.retrying import RetryingChatClient, wrap_retrying

        saved = _save_env("MEMCONFLICT_TEST_CHAT_RETRIES")
        try:
            os.environ["MEMCONFLICT_TEST_CHAT_RETRIES"] = "1"
            plain = object()
            self.assertIs(wrap_retrying(plain, prefix="MEMCONFLICT_TEST"), plain)
            os.environ["MEMCONFLICT_TEST_CHAT_RETRIES"] = "4"
            self.assertIsInstance(
                wrap_retrying(plain, prefix="MEMCONFLICT_TEST"), RetryingChatClient
            )
        finally:
            _restore_env(saved)

    def test_the_judge_uses_the_retrying_client(self):
        from memconflict_eval.judging import MemConflictJudge

        _patch_runtime(_FakeMemorySystem(), _FakeChatClient("ans"), _FakeChatClient("{}"))
        config = runtime.load_memory_config()
        judge = MemConflictJudge(config, top_k=3)
        self.assertTrue(
            type(judge.client).__name__ == "RetryingChatClient",
            type(judge.client).__name__,
        )


class AzureChannelTests(unittest.TestCase):
    """The judge can leave OpenRouter's geo gate for Azure OpenAI."""

    def _client(self, **overrides):
        from memconflict_eval.azure_client import AzureChatClient

        params = dict(
            endpoint="https://my-res.openai.azure.com",
            deployment="gpt-4o-mini",
            api_version="2024-10-21",
            api_key="secret",
            temperature=0.0,
            timeout=30,
            max_tokens=512,
        )
        params.update(overrides)
        return AzureChatClient(**params)

    def test_endpoint_carries_the_deployment_and_api_version(self):
        from memconflict_eval.azure_client import build_azure_endpoint

        url = build_azure_endpoint(
            "https://my-res.openai.azure.com/", "gpt-4o-mini", "2024-10-21"
        )
        self.assertEqual(
            url,
            "https://my-res.openai.azure.com/openai/deployments/gpt-4o-mini/"
            "chat/completions?api-version=2024-10-21",
        )

    def test_an_endpoint_that_already_has_the_path_is_kept(self):
        from memconflict_eval.azure_client import build_azure_endpoint

        url = build_azure_endpoint(
            "https://gw.example.com/openai/deployments/judge/chat/completions",
            "",
            "2024-10-21",
        )
        self.assertTrue(url.endswith("?api-version=2024-10-21"))
        self.assertNotIn("/deployments//", url)

    def test_chat_uses_the_api_key_header(self):
        import requests

        client = self._client()
        captured = {}

        class _Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.update(url=url, headers=headers, json=json, timeout=timeout)
            return _Response()

        original = requests.post
        requests.post = fake_post
        try:
            text = client.chat([{"role": "user", "content": "hi"}], json_mode=True)
        finally:
            requests.post = original

        self.assertEqual(text, "ok")
        self.assertEqual(captured["headers"]["api-key"], "secret")
        self.assertNotIn("Authorization", captured["headers"])
        self.assertIn("/openai/deployments/gpt-4o-mini/", captured["url"])
        self.assertEqual(captured["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(captured["json"]["max_tokens"], 512)

    def test_bearer_mode_is_available_for_gateways(self):
        import requests

        client = self._client(auth_header="authorization")
        captured = {}

        class _Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        original = requests.post
        requests.post = lambda url, headers=None, json=None, timeout=None: (
            captured.update(headers=headers) or _Response()
        )
        try:
            client.chat([{"role": "user", "content": "hi"}])
        finally:
            requests.post = original
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertNotIn("api-key", captured["headers"])

    def test_missing_endpoint_is_reported_clearly(self):
        from memconflict_eval.azure_client import (
            AzureConfigurationError,
            build_azure_chat_client,
        )

        saved = _save_env("AZURE_OPENAI_ENDPOINT", "AZURE_API_KEY")
        try:
            os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
            os.environ["AZURE_API_KEY"] = "secret"
            with self.assertRaises(AzureConfigurationError):
                build_azure_chat_client(SimpleNamespace(provider="azure", extra={}))
        finally:
            _restore_env(saved)

    def test_judge_and_answerer_route_azure_through_our_client(self):
        calls = []
        original = runtime.build_chat_client
        runtime.build_chat_client = lambda model_config: (
            calls.append(getattr(model_config, "role", "")) or _FakeChatClient("ans")
        )
        _patch_runtime(_FakeMemorySystem(), _FakeChatClient("ans"), _FakeChatClient())
        try:
            from memconflict_eval.judging import MemConflictJudge

            config = runtime.load_memory_config()
            MemConflictJudge(config, top_k=3)
            MemConflictAnswerer(config)
        finally:
            runtime.build_chat_client = original
        self.assertEqual(calls, ["judge", "answer"])

    def test_preflight_follows_the_env_names_a_config_declares(self):
        config = SimpleNamespace(
            memory_builder=SimpleNamespace(
                provider="modelscope_openai_compatible",
                api_key_env="QIANFAN_API_KEY",
                chat_completions_endpoint_env="QIANFAN_CHAT_COMPLETIONS_ENDPOINT",
            ),
            judge_model=SimpleNamespace(provider="azure"),
            embedding=None,
        )
        names = runtime.required_env_names(
            config, ("memory_builder", "judge_model")
        )
        self.assertIn("QIANFAN_API_KEY", names)
        self.assertIn("QIANFAN_CHAT_COMPLETIONS_ENDPOINT", names)
        self.assertIn("AZURE_API_KEY", names)
        self.assertIn("AZURE_OPENAI_ENDPOINT", names)


class NewFlagTests(unittest.TestCase):
    def test_runner_flags_parse(self):
        from run_experiment import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--answer-workers",
                "8",
                "--extraction-workers",
                "2",
                "--entity-judge-workers",
                "1",
                "--resume",
                "--ollama-units",
                "http://unit:41134",
            ]
        )
        self.assertEqual(args.answer_workers, 8)
        self.assertEqual(args.extraction_workers, 2)
        self.assertEqual(args.entity_judge_workers, 1)
        self.assertTrue(args.resume)
        self.assertEqual(args.ollama_units, "http://unit:41134")
        defaults = build_arg_parser().parse_args([])
        self.assertEqual(defaults.answer_workers, 4)
        self.assertFalse(defaults.resume)

    def test_unit_probe_accepts_an_explicit_list(self):
        from tools.check_ollama_units import parse_base_urls as probe_units

        self.assertEqual(
            probe_units("http://172.26.94.12:41134,http://172.26.94.12:41135"),
            ["http://172.26.94.12:41134", "http://172.26.94.12:41135"],
        )

    def test_unit_probe_reads_ps_with_get(self):
        """``/api/ps`` is a GET endpoint; POSTing it returns HTTP 405."""
        from tools import check_ollama_units as probe_tool

        calls = []

        class _FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def read(self):
                return json.dumps(self._payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        def fake_urlopen(url, timeout=None):
            calls.append(url if isinstance(url, str) else "REQUEST-OBJECT")
            if isinstance(url, str) and url.endswith("/api/ps"):
                return _FakeResponse(
                    {"models": [{"name": "qwen3.5:latest", "size_vram": 8_720_000_000}]}
                )
            return _FakeResponse({"eval_count": 10, "eval_duration": 1_000_000_000})

        original = probe_tool.urllib.request.urlopen
        probe_tool.urllib.request.urlopen = fake_urlopen
        try:
            row = probe_tool.probe("http://unit:1", "qwen3.5:latest", timeout=5)
        finally:
            probe_tool.urllib.request.urlopen = original

        self.assertEqual(row["tokens_per_second"], 10.0)
        self.assertEqual(row["size_vram_gb"], 8.72)
        self.assertIn("http://unit:1/api/ps", calls)
        self.assertTrue(all(isinstance(call, str) for call in calls))


# ---------------------------------------------------------------------------
# progress reporting
# ---------------------------------------------------------------------------


class _Clock:
    """Injectable monotonic clock, so the ETA maths is testable."""

    def __init__(self, start=1_000.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += float(seconds)


class _TtyStream(io.StringIO):
    def isatty(self) -> bool:  # noqa: D102 - mirrors the real stream API
        return True


def _last_line(stream: io.StringIO) -> str:
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    return lines[-1] if lines else ""


class ProgressReporterTests(unittest.TestCase):
    def test_line_reports_counts_percentage_and_eta(self):
        clock = _Clock()
        stream = io.StringIO()
        progress = ProgressReporter(
            4,
            label="sessions",
            counters={"questions": 0, "errors": 0},
            totals={"questions": 6, "personas": 4},
            stream=stream,
            tty=False,
            heartbeat=None,
            log_interval=0.0,
            clock=clock,
        )
        progress.start()
        clock.tick(60)
        progress.advance(1, questions=2, errors=0)
        line = _last_line(stream)

        self.assertIn("[progress]", line)
        self.assertIn("1/4 sessions", line)
        self.assertIn("25%", line)
        self.assertIn("questions 2/6", line)
        self.assertIn("errors 0", line)
        # 60 s for one of four sessions leaves three at the same rate.
        self.assertIn("eta 3m00s", line)

    def test_eta_is_unknown_before_the_first_session(self):
        stream = io.StringIO()
        progress = ProgressReporter(
            3, stream=stream, tty=False, heartbeat=None, log_interval=0.0
        )
        progress.start()
        line = _last_line(stream)
        self.assertIn("0/3", line)
        self.assertNotIn("eta", line)

    def test_tty_rewrites_one_line_and_ends_with_a_newline(self):
        clock = _Clock()
        stream = _TtyStream()
        progress = ProgressReporter(
            2, stream=stream, tty=True, heartbeat=None, interval=0.0, clock=clock
        )
        progress.start()
        clock.tick(5)
        progress.advance(1)
        clock.tick(5)
        progress.finish()
        text = stream.getvalue()

        self.assertTrue(text.startswith("\r"))
        self.assertTrue(text.endswith("\n"))
        self.assertIn("1/2", text)
        self.assertNotIn("\r\n", text)

    def test_log_lines_are_throttled_to_the_log_interval(self):
        clock = _Clock()
        stream = io.StringIO()
        progress = ProgressReporter(
            10,
            stream=stream,
            tty=False,
            heartbeat=None,
            log_interval=60.0,
            clock=clock,
        )
        progress.start()  # the opening line is always written
        progress.advance(1)
        progress.advance(1)
        self.assertEqual(len(stream.getvalue().strip().splitlines()), 1)

        clock.tick(61)
        progress.advance(1)
        lines = stream.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("3/10", lines[-1])

    def test_disabled_reporting_writes_nothing(self):
        stream = io.StringIO()
        progress = ProgressReporter(
            3, stream=stream, enabled=False, heartbeat=None, tty=False
        )
        progress.start()
        progress.advance(1, questions=2)
        progress.note(last="persona s1")
        progress.finish()
        self.assertEqual(stream.getvalue(), "")

    def test_a_zero_sized_phase_stays_silent(self):
        stream = io.StringIO()
        progress = ProgressReporter(0, stream=stream, heartbeat=None, tty=False)
        progress.start()
        progress.finish()
        self.assertEqual(stream.getvalue(), "")

    def test_notes_are_carried_into_the_next_line(self):
        stream = io.StringIO()
        progress = ProgressReporter(
            2, stream=stream, tty=False, heartbeat=None, log_interval=0.0
        )
        progress.start()
        progress.note(running="2/4", last="75345e85 s3")
        line = _last_line(stream)
        self.assertIn("running 2/4", line)
        self.assertIn("last 75345e85 s3", line)

    def test_duration_formatting(self):
        from memconflict_eval.progress import format_duration

        self.assertEqual(format_duration(9), "9s")
        self.assertEqual(format_duration(95), "1m35s")
        self.assertEqual(format_duration(3661), "1h01m")
        self.assertEqual(format_duration(None), "--")
        self.assertEqual(format_duration(-1), "--")

    def test_a_narrow_terminal_sheds_the_least_important_pieces(self):
        stream = io.StringIO()
        progress = ProgressReporter(
            44,
            label="sessions",
            counters={"questions": 0, "errors": 0, "personas": 0, "failed": 0},
            totals={"questions": 46, "personas": 4},
            stream=stream,
            tty=False,
            heartbeat=None,
            log_interval=0.0,
        )
        progress.start()
        progress.advance(12, questions=9)
        progress.note(running="3/4", last="75345e85 s3")

        wide = progress.render()
        narrow = progress.render(width=len(wide) - 30)
        self.assertIn("last 75345e85 s3", wide)
        self.assertNotIn("last 75345e85 s3", narrow)
        self.assertLessEqual(len(narrow), len(wide) - 30)
        # The counter, the question tally and the ETA survive the fitting.
        self.assertIn("12/44 sessions", narrow)
        self.assertIn("questions 9/46", narrow)

    def test_fitting_never_drops_the_session_counter(self):
        stream = io.StringIO()
        progress = ProgressReporter(
            10, stream=stream, tty=False, heartbeat=None, log_interval=0.0
        )
        progress.start()
        progress.note(last="persona s9")
        narrow = progress.render(width=5)
        self.assertIn("0/10", narrow)

    def test_heartbeat_refreshes_between_sessions_and_stops_on_finish(self):
        stream = io.StringIO()
        progress = ProgressReporter(
            5, stream=stream, tty=False, heartbeat=0.05, log_interval=0.0
        )
        progress.start()
        before = len(stream.getvalue().splitlines())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if len(stream.getvalue().splitlines()) > before:
                break
            time.sleep(0.02)
        after = len(stream.getvalue().splitlines())
        progress.finish()
        self.assertGreater(after, before)
        self.assertFalse(progress._thread.is_alive())

    def test_log_keeps_the_progress_line_intact_on_a_terminal(self):
        stream = _TtyStream()
        progress = ProgressReporter(
            2, stream=stream, tty=True, heartbeat=None, interval=0.0
        )
        progress.start()
        progress.log("[persona 1/2] persona-aaaa sessions=2 questions=1")
        progress.advance(1)

        # \r is a line separator for splitlines(), so drop the blank chunks the
        # in-place rewriting and the padding leave behind.
        lines = [line for line in stream.getvalue().splitlines() if line.strip()]
        self.assertIn("[persona 1/2] persona-aaaa sessions=2 questions=1", lines)
        # The ordinary line is followed by a fresh progress line, not by the
        # remains of the one it had to clear.
        self.assertIn("1/2 items", lines[-1])
        self.assertTrue(all(line.startswith(("[progress]", "[persona")) for line in lines))

    def test_log_still_prints_when_reporting_is_off(self):
        stream = io.StringIO()
        progress = ProgressReporter(2, stream=stream, enabled=False, heartbeat=None)
        progress.log("[persona 1/2] xyz")
        self.assertEqual(stream.getvalue(), "[persona 1/2] xyz\n")


class ExpectedWorkTests(unittest.TestCase):
    """The progress denominator must match what the runner will replay."""

    def _sessions(self):
        return [
            SimpleNamespace(session_id=index, questions=tuple(range(index)))
            for index in range(5)
        ]

    def test_full_window(self):
        from memconflict_eval.progress import expected_work

        self.assertEqual(expected_work(self._sessions()), (5, 10))

    def test_max_sessions_truncates_the_denominator(self):
        from memconflict_eval.progress import expected_work

        self.assertEqual(expected_work(self._sessions(), max_sessions=3), (3, 3))
        self.assertEqual(expected_work(self._sessions(), max_sessions=0), (0, 0))

    def test_resume_skips_already_finished_sessions(self):
        from memconflict_eval.progress import expected_work

        self.assertEqual(
            expected_work(self._sessions(), skip_session_ids={1, 3}), (3, 6)
        )
        self.assertEqual(
            expected_work(self._sessions(), max_sessions=4, skip_session_ids={1, 3}),
            (2, 2),
        )

    def test_runner_totals_aggregate_over_personas(self):
        from run_experiment import expected_run_work, short_persona_id

        def persona(persona_id, question_counts):
            return parse_persona(
                {
                    "ID": persona_id,
                    "Full_Session_Chain": [
                        {
                            "Session_ID": index,
                            "Date": f"2022-01-{index + 1:02d}",
                            "Session_Type": "update",
                            "Session_Dialogue": {
                                "dialogue_turn_1": [{"role": "user", "content": "hi"}]
                            },
                            "Session_Questions": [
                                {
                                    "question_id": f"Q_{index}_{q}",
                                    "question": "q",
                                    "answer": "a",
                                    "conflict_type": "dynamic_conflict",
                                }
                                for q in range(count)
                            ],
                        }
                        for index, count in enumerate(question_counts)
                    ],
                }
            )

        first = persona("persona-aaaaaaaa", [0, 2, 1])
        second = persona("persona-bbbbbbbb", [1])
        self.assertEqual(expected_run_work([first, second], max_sessions=None), (4, 4))
        self.assertEqual(expected_run_work([first, second], max_sessions=2), (3, 3))
        self.assertEqual(
            expected_run_work(
                [first, second], max_sessions=None, resume_state={"persona-aaaaaaaa": {1}}
            ),
            (3, 2),
        )
        self.assertEqual(short_persona_id("75345e85-6427-eb51"), "75345e85")
        self.assertEqual(short_persona_id("persona-progress"), "persona")
        self.assertEqual(short_persona_id("ab"), "ab")
        self.assertEqual(short_persona_id(""), "?")


class SemanticValidationError(Exception):
    """Stand-in for ``memory.v4.semantic.SemanticValidationError``."""


class V4BuildStageError(Exception):
    """Stand-in for ``memory.v4.failure.V4BuildStageError``."""


class StaleCheckpointGuardTests(unittest.TestCase):
    """A stale V4 build-cache entry must not kill a whole persona.

    ``load_build_checkpoint`` matches on the key and ``status='succeeded'``
    only, so a reducer output validated against an older semantic state is
    replayed as-is and its ``reinforce <fact_key>`` fails validation. The
    harness drops those entries (preventively, and once more on failure).
    """

    def _store(self, tmp: str, *, revision: int, rows) -> Path:
        db = Path(tmp) / "memory.sqlite3"
        connection = sqlite3.connect(db)
        connection.execute(
            "CREATE TABLE v4_participant_scopes (id TEXT PRIMARY KEY, revision INTEGER)"
        )
        connection.execute(
            "CREATE TABLE v4_build_checkpoints ("
            "checkpoint_key TEXT PRIMARY KEY, namespace TEXT, scope_id TEXT,"
            "scope_revision INTEGER, stage TEXT, unit_id TEXT, status TEXT,"
            "error_json TEXT)"
        )
        connection.execute(
            "INSERT INTO v4_participant_scopes VALUES ('scope_x', ?)", (revision,)
        )
        for key, row_revision, stage, status in rows:
            connection.execute(
                "INSERT INTO v4_build_checkpoints VALUES (?,?,?,?,?,?,?,NULL)",
                (key, "ns", "scope_x", row_revision, stage, key, status),
            )
        connection.commit()
        connection.close()
        return db

    def _memory(self, tmp: str) -> MemConflictMemory:
        memory = object.__new__(MemConflictMemory)
        memory.store_dir = Path(tmp)
        memory.namespace = "ns"
        memory.persona = SimpleNamespace(persona_id="persona-guard")
        return memory

    def _statuses(self, db: Path) -> dict[str, str]:
        connection = sqlite3.connect(db)
        try:
            return {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT checkpoint_key, status FROM v4_build_checkpoints"
                )
            }
        finally:
            connection.close()

    def setUp(self):
        self._saved_guard = os.environ.pop("MEMCONFLICT_CHECKPOINT_GUARD", None)

    def tearDown(self):
        os.environ.pop("MEMCONFLICT_CHECKPOINT_GUARD", None)
        if self._saved_guard is not None:
            os.environ["MEMCONFLICT_CHECKPOINT_GUARD"] = self._saved_guard

    def test_checkpoints_from_an_older_scope_revision_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._store(
                tmp,
                revision=6,
                rows=[
                    ("stale", 5, "semantic_update", "succeeded"),
                    ("fresh", 6, "semantic_update", "succeeded"),
                    ("already_failed", 4, "semantic_update", "failed"),
                ],
            )
            dropped = self._memory(tmp).drop_stale_checkpoints()
            statuses = self._statuses(db)

        self.assertEqual(dropped, 1)
        self.assertEqual(statuses["stale"], "failed")
        self.assertEqual(statuses["fresh"], "succeeded")
        self.assertEqual(statuses["already_failed"], "failed")

    def test_invalidation_can_target_one_stage_or_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._store(
                tmp,
                revision=3,
                rows=[
                    ("sem", 3, "semantic_update", "succeeded"),
                    ("win", 3, "window_extraction", "succeeded"),
                ],
            )
            memory = self._memory(tmp)
            self.assertEqual(
                memory.invalidate_checkpoints(stage="semantic_update", reason="test"), 1
            )
            middle = self._statuses(db)
            self.assertEqual(middle["sem"], "failed")
            self.assertEqual(middle["win"], "succeeded")
            self.assertEqual(memory.invalidate_checkpoints(stage=None, reason="test"), 1)
            self.assertEqual(self._statuses(db)["win"], "failed")

    def test_guard_can_be_switched_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._store(
                tmp, revision=6, rows=[("stale", 5, "semantic_update", "succeeded")]
            )
            os.environ["MEMCONFLICT_CHECKPOINT_GUARD"] = "0"
            dropped = self._memory(tmp).drop_stale_checkpoints()
            statuses = self._statuses(db)

        self.assertEqual(dropped, 0)
        self.assertEqual(statuses["stale"], "succeeded")

    def test_a_missing_store_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = self._memory(tmp)
            self.assertEqual(memory.drop_stale_checkpoints(), 0)
            self.assertEqual(memory.invalidate_checkpoints(stage=None, reason="x"), 0)

    def test_only_stale_cache_errors_are_recognised(self):
        from memconflict_eval.memory import _is_stale_checkpoint_error, _stage_from_error

        self.assertTrue(
            _is_stale_checkpoint_error(
                SemanticValidationError("reinforce references unknown fact key")
            )
        )
        self.assertTrue(
            _is_stale_checkpoint_error(
                V4BuildStageError(
                    "V4 operation failed (stage=semantic_update, attempts=3, "
                    "cause=SemanticValidationError: reinforce references unknown fact key)"
                )
            )
        )
        self.assertFalse(_is_stale_checkpoint_error(RuntimeError("proxy is down")))
        self.assertFalse(
            _is_stale_checkpoint_error(SemanticValidationError("some other failure"))
        )
        self.assertEqual(
            _stage_from_error(
                V4BuildStageError("V4 operation failed (stage=window_extraction, ...)")
            ),
            "window_extraction",
        )
        self.assertIsNone(_stage_from_error(RuntimeError("no stage here")))

    def test_a_stale_cache_failure_is_invalidated_and_retried_once(self):
        class FakeSystem:
            def __init__(self):
                self.calls = 0

            def ingest_conversation(self, namespace, conversation, metadata=None, **_k):
                self.calls += 1
                if self.calls == 1:
                    raise SemanticValidationError("reinforce references unknown fact key")

            def is_namespace_ready(self, namespace):
                return True

        with tempfile.TemporaryDirectory() as tmp:
            self._store(
                tmp,
                revision=6,
                rows=[("stale", 5, "semantic_update", "succeeded")],
            )
            memory = self._memory(tmp)
            memory.system = FakeSystem()
            session = SimpleNamespace(
                session_id=7,
                date="2022-03-09",
                session_type="chitchat",
                dialogue=("a", "b"),
                to_memory_session=lambda: {"session_id": "7", "turns": []},
            )
            report = memory.ingest_session(session)

        self.assertEqual(memory.system.calls, 2)
        self.assertTrue(report.retried_after_validation_error)
        self.assertEqual(report.stale_checkpoints_dropped, 1)
        self.assertEqual(report.session_id, 7)
        self.assertTrue(report.to_dict()["Retried_After_Validation_Error"])

    def test_unrelated_failures_are_not_retried(self):
        class FakeSystem:
            def __init__(self):
                self.calls = 0

            def ingest_conversation(self, namespace, conversation, metadata=None, **_k):
                self.calls += 1
                raise RuntimeError("proxy is down")

        with tempfile.TemporaryDirectory() as tmp:
            self._store(tmp, revision=1, rows=[])
            memory = self._memory(tmp)
            memory.system = FakeSystem()
            session = SimpleNamespace(
                session_id=1,
                date="2022-01-10",
                session_type="initial_reveal",
                dialogue=("a",),
                to_memory_session=lambda: {"session_id": "1", "turns": []},
            )
            with self.assertRaises(RuntimeError):
                memory.ingest_session(session)

        self.assertEqual(memory.system.calls, 1)


class PersonaShardTests(unittest.TestCase):
    """Point 27: random persona shards, merged later (possibly on other hosts)."""

    def _write_dataset(self, path: Path, personas: int = 4) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for index in range(personas):
                record = {
                    "ID": f"persona-{index}",
                    "Full_Session_Chain": [
                        {
                            "Session_ID": 0,
                            "Date": "2022-01-03",
                            "Session_Type": "initial_reveal",
                            "Session_Dialogue": {
                                "dialogue_turn_1": [
                                    {"role": "user", "content": f"I am persona {index}."}
                                ]
                            },
                            "Session_Questions": [
                                {
                                    "question_id": "Q_001",
                                    "question": "Who is the user?",
                                    "answer": f"persona {index}",
                                    "conflict_type": "dynamic_conflict",
                                    "ability_target": "track_state_over_time",
                                    "difficulty": "easy",
                                }
                            ],
                        }
                    ],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_indices_are_parsed_and_deduplicated(self):
        from run_experiment import parse_persona_indices

        self.assertEqual(parse_persona_indices("0,4,18,27,28"), [0, 4, 18, 27, 28])
        self.assertEqual(parse_persona_indices("0-4,10,10"), [0, 1, 2, 3, 4, 10])
        for bad in ("x", "5-2", "", "-1"):
            with self.assertRaises(ValueError):
                parse_persona_indices(bad)

    def test_selection_keeps_dataset_order_and_checks_bounds(self):
        from run_experiment import select_personas_by_index

        personas = [SimpleNamespace(persona_id=f"p{index}") for index in range(5)]
        selected, indices = select_personas_by_index(personas, [4, 0, 2])
        self.assertEqual([persona.persona_id for persona in selected], ["p0", "p2", "p4"])
        self.assertEqual(indices, [0, 2, 4])
        with self.assertRaises(ValueError):
            select_personas_by_index(personas, [9])

    def test_main_runs_only_the_requested_personas(self):
        from run_experiment import main

        _patch_runtime(_FakeMemorySystem(), _FakeChatClient("ans"), _FakeChatClient())
        saved = _save_env(
            ollama_units.UNITS_ENV, "OLLAMA_BASE_URL", *ollama_units.ENDPOINT_ENV_KEYS
        )
        original_env_loader = runtime.load_env_file
        runtime.load_env_file = lambda: None
        for name in (ollama_units.UNITS_ENV, "OLLAMA_BASE_URL"):
            os.environ.pop(name, None)
        os.environ["OLLAMA_EMBED_ENDPOINT"] = "http://unit/api/embed"
        os.environ["OLLAMA_LEGACY_EMBEDDINGS_ENDPOINT"] = "http://unit/api/embeddings"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dataset = Path(tmp) / "data.jsonl"
                self._write_dataset(dataset)
                run_dir = Path(tmp) / "shard"
                with contextlib.redirect_stderr(io.StringIO()):
                    code = main(
                        [
                            "--input",
                            str(dataset),
                            "--output-dir",
                            str(run_dir),
                            "--persona-indices",
                            "0,2",
                            "--no-progress",
                        ]
                    )
                self.assertEqual(code, 0)
                records = [
                    json.loads(line)
                    for line in (run_dir / "results.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ]
                meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))

            self.assertEqual(
                [record["Persona_ID"] for record in records], ["persona-0", "persona-2"]
            )
            self.assertEqual([record["Dataset_Index"] for record in records], [0, 2])
            self.assertEqual(meta["Persona_Indices"], [0, 2])
            self.assertIn("answer_model", meta["Models"])
        finally:
            runtime.load_env_file = original_env_loader
            _restore_env(saved)

    def test_indices_cannot_be_combined_with_the_contiguous_window(self):
        from run_experiment import main

        _patch_runtime(_FakeMemorySystem(), _FakeChatClient("ans"), _FakeChatClient())
        saved = _save_env(
            ollama_units.UNITS_ENV, "OLLAMA_BASE_URL", *ollama_units.ENDPOINT_ENV_KEYS
        )
        original_env_loader = runtime.load_env_file
        runtime.load_env_file = lambda: None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dataset = Path(tmp) / "data.jsonl"
                self._write_dataset(dataset)
                with contextlib.redirect_stderr(io.StringIO()):
                    code = main(
                        [
                            "--input",
                            str(dataset),
                            "--output-dir",
                            str(Path(tmp) / "run"),
                            "--persona-indices",
                            "0",
                            "--start-index",
                            "1",
                        ]
                    )
            self.assertEqual(code, 2)
        finally:
            runtime.load_env_file = original_env_loader
            _restore_env(saved)

    def test_shard_plan_is_deterministic_disjoint_and_complete(self):
        from tools import shard_plan

        personas = [
            SimpleNamespace(sessions=[object()] * (51 + index % 4), question_count=100 + index)
            for index in range(30)
        ]
        plan = shard_plan.build_plan(personas, shards=6, per_shard=5, seed=27)
        again = shard_plan.build_plan(personas, shards=6, per_shard=5, seed=27)
        different = shard_plan.build_plan(personas, shards=6, per_shard=5, seed=28)

        self.assertEqual(plan, again)  # same seed -> same plan
        self.assertNotEqual(
            [shard["personas"] for shard in plan],
            [shard["personas"] for shard in different],
        )
        covered = [index for shard in plan for index in shard["personas"]]
        self.assertEqual(sorted(covered), list(range(30)))
        self.assertEqual(len(covered), len(set(covered)))  # disjoint
        self.assertTrue(all(len(shard["personas"]) == 5 for shard in plan))
        # consecutive slices must not stay consecutive after the shuffle
        self.assertNotEqual(plan[0]["personas"], list(range(5)))

    def test_shard_plan_rejects_a_mismatched_split(self):
        from tools import shard_plan

        personas = [SimpleNamespace(sessions=[], question_count=0) for _ in range(30)]
        with self.assertRaises(ValueError):
            shard_plan.build_plan(personas, shards=6, per_shard=4, seed=1)

    def test_dataset_digest_identifies_the_plan_input(self):
        """Same dataset hash + same seed -> the same plan on another host."""
        from tools import shard_plan

        digest = shard_plan.dataset_digest(DATASET)
        self.assertIsNotNone(digest)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, shard_plan.dataset_digest(DATASET))
        self.assertIsNone(shard_plan.dataset_digest(DATASET.parent / "does_not_exist.jsonl"))

    def test_partial_merge_is_allowed_for_incremental_tables(self):
        from tools.merge_shards import merge

        persona = {"Persona_ID": "persona-0", "Sessions": [], "Memory_System": "m"}
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "shard_1"
            shard.mkdir()
            (shard / "results.jsonl").write_text(
                json.dumps(persona) + "\n", encoding="utf-8"
            )
            (shard / "run_meta.json").write_text(json.dumps({"Wall_Clock_s": 10.0}), encoding="utf-8")
            dataset = Path(tmp) / "data.jsonl"
            with open(dataset, "w", encoding="utf-8") as handle:
                for index in range(2):
                    handle.write(
                        json.dumps(
                            {
                                "ID": f"persona-{index}",
                                "Full_Session_Chain": [
                                    {"Session_ID": 0, "Date": "2022-01-01", "Session_Dialogue": {}}
                                ],
                            }
                        )
                        + "\n"
                    )
            strict, strict_status = merge([shard], dataset=dataset, allow_partial=False)
            partial, partial_status = merge([shard], dataset=dataset, allow_partial=True)

        self.assertEqual(strict_status, 1)
        self.assertEqual(partial_status, 0)
        self.assertTrue(partial["summary"]["Partial_Merge"])
        self.assertEqual(partial["summary"]["Personas_Expected_From_Dataset"], 2)
        self.assertEqual(partial["summary"]["Persona_Count"], 1)
        self.assertEqual(strict["summary"]["Partial_Merge"], True)


class ApiRateReportTests(unittest.TestCase):
    """The quota check must read the real v4 API logs and project correctly."""

    def _write_log(self, root: Path) -> Path:
        run_dir = root / "shard_1"
        store = run_dir / "Memory" / "retrival_mem_v4_p1_v1"
        store.mkdir(parents=True)
        rows = [
            {
                "provider": "dashscope_bailian",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "success": True,
                "token_usage": {"prompt_tokens": 1000, "completion_tokens": 100},
                "module": "v4_memory_builder",
            },
            {
                "provider": "dashscope_bailian",
                "timestamp": "2026-01-01T00:01:00+00:00",
                "success": False,
                "token_usage": {"prompt_tokens": 2000, "completion_tokens": 0},
                "module": "v4_semantic_reducer",
            },
            {
                "provider": "ollama",
                "timestamp": "2026-01-01T00:00:30+00:00",
                "success": True,
                "token_usage": {"prompt_tokens": 500, "completion_tokens": 50},
                "module": "v4_window_planner",
            },
        ]
        (store / "v4_memory_builder.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )
        return run_dir

    def test_rates_are_computed_per_provider(self):
        from tools import api_rate_report

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_log(Path(tmp))
            stats = api_rate_report.collect([run_dir])

        bailian = stats["dashscope_bailian"]
        self.assertEqual(bailian.calls, 2)
        self.assertEqual(bailian.failures, 1)
        self.assertEqual(bailian.total_tokens, 3100)
        self.assertAlmostEqual(bailian.minutes, 1.0, places=3)
        self.assertAlmostEqual(bailian.per_minute(bailian.total_tokens), 3100.0, places=3)
        self.assertEqual(bailian.modules["v4_semantic_reducer"], 1)
        self.assertEqual(stats["ollama"].calls, 1)

    def test_cli_reports_the_fraction_of_a_quota(self):
        import contextlib
        import io as _io

        from tools import api_rate_report

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_log(Path(tmp))
            stdout = _io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = api_rate_report.main(
                    [
                        "--run-dir",
                        str(run_dir),
                        "--provider",
                        "dashscope_bailian",
                        "--tpm",
                        "1000000",
                        "--rpm",
                        "500",
                        "--machines",
                        "2",
                    ]
                )
            text = stdout.getvalue()

        self.assertEqual(code, 0)
        self.assertIn("3,100 tokens/min", text)
        self.assertIn("x2 machines", text)
        self.assertIn("vs TPM 1,000,000", text)
        self.assertIn("fits", text)


class ShardChainTests(unittest.TestCase):
    """Point 27 watchdog: chain shards, never fight over one store."""

    def _run_dir(self, tmp: str, name: str = "shard_1") -> Path:
        run_dir = Path(tmp) / name
        run_dir.mkdir(parents=True)
        return run_dir

    def test_status_of_an_untouched_directory_is_idle(self):
        from tools.run_shards_chain import shard_status

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(shard_status(self._run_dir(tmp)), "idle")

    def test_status_is_waiting_while_the_first_run_is_in_flight(self):
        from tools.run_shards_chain import shard_status

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            (run_dir / "sessions.jsonl").write_text("{}\n", encoding="utf-8")
            (run_dir / "results.jsonl").write_text("", encoding="utf-8")
            self.assertEqual(shard_status(run_dir), "waiting")

    def test_status_is_waiting_while_a_resume_is_in_flight(self):
        """run_meta from the failed attempt is stale; sessions.jsonl grew again."""
        from tools.run_shards_chain import shard_status

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            sessions = run_dir / "sessions.jsonl"
            meta = run_dir / "run_meta.json"
            sessions.write_text("{}\n", encoding="utf-8")
            meta.write_text(json.dumps({"Failed_Personas": ["p1"]}), encoding="utf-8")
            now = time.time()
            os.utime(meta, (now - 600, now - 600))
            os.utime(sessions, (now, now))
            self.assertEqual(shard_status(run_dir), "waiting")

            os.utime(meta, (now + 600, now + 600))
            self.assertEqual(shard_status(run_dir), "failed")

    def test_status_is_done_only_without_failed_personas(self):
        from tools.run_shards_chain import shard_status

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            (run_dir / "run_meta.json").write_text(
                json.dumps({"Failed_Personas": []}), encoding="utf-8"
            )
            self.assertEqual(shard_status(run_dir), "done")

    def test_plan_shards_are_run_scored_and_merged_in_order(self):
        from tools.run_shards_chain import Chain

        plan = {
            "plan": [
                {"name": "shard_1", "personas": [0, 1]},
                {"name": "shard_2", "personas": [2, 3]},
            ]
        }
        calls: list[str] = []

        def fake_step(command: list[str], log_path: Path) -> int:
            script = next(Path(part).name for part in command if part.endswith(".py"))
            calls.append(script)
            if script == "run_experiment.py":
                run_dir = Path(command[command.index("--output-dir") + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "sessions.jsonl").write_text("{}\n", encoding="utf-8")
                (run_dir / "run_meta.json").write_text(
                    json.dumps({"Failed_Personas": []}), encoding="utf-8"
                )
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            chain = Chain(
                plan=plan,
                runs_root=Path(tmp) / "runs",
                config=Path("cfg.yaml"),
                start_if_idle=True,
                step_runner=fake_step,
                sleep=lambda _seconds: None,
                printer=lambda _line: None,
            )
            code = chain.run(plan["plan"])

        self.assertEqual(code, 0)
        self.assertEqual(
            calls,
            [
                "run_experiment.py",
                "run_scoring.py",
                "merge_shards.py",
                "run_scoring.py",
                "run_experiment.py",
                "run_scoring.py",
                "merge_shards.py",
                "run_scoring.py",
            ],
        )

    def test_chain_waits_for_a_running_shard_instead_of_starting_it(self):
        from tools.run_shards_chain import Chain

        plan = {"plan": [{"name": "shard_1", "personas": [0]}]}
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / "runs"
            run_dir = runs_root / "shard_1"
            run_dir.mkdir(parents=True)
            (run_dir / "sessions.jsonl").write_text("{}\n", encoding="utf-8")

            def fake_sleep(_seconds: float) -> None:
                # the other terminal finishes while we wait
                (run_dir / "run_meta.json").write_text(
                    json.dumps({"Failed_Personas": []}), encoding="utf-8"
                )
                os.utime(run_dir / "run_meta.json", None)
                future = time.time() + 600
                os.utime(run_dir / "run_meta.json", (future, future))

            def fake_step(command: list[str], log_path: Path) -> int:
                return 0

            chain = Chain(
                plan=plan,
                runs_root=runs_root,
                config=Path("cfg.yaml"),
                step_runner=fake_step,
                sleep=fake_sleep,
                printer=lambda _line: None,
            )
            code = chain.run(plan["plan"])

        self.assertEqual(code, 0)

    def test_each_shard_can_get_its_own_ollama_container(self):
        from tools.run_shards_chain import Chain

        plan = {
            "plan": [
                {"name": "shard_1", "personas": [0], "_order": 0},
                {"name": "shard_2", "personas": [1], "_order": 1},
                {"name": "shard_3", "personas": [2], "_order": 2},
                {"name": "shard_4", "personas": [3], "_order": 3},
            ]
        }
        chain = Chain(
            plan=plan,
            runs_root=Path("runs"),
            config=Path("cfg.yaml"),
            units_map={"shard_1": "http://gpu:41135", "shard_3": "http://gpu:41136"},
            units_pool=["http://gpu:41133", "http://gpu:41134"],
            printer=lambda _line: None,
        )
        self.assertEqual(chain.unit_for(plan["plan"][0], 0), "http://gpu:41135")
        self.assertEqual(chain.unit_for(plan["plan"][2], 2), "http://gpu:41136")
        # no explicit entry -> round robin over --ollama-units
        self.assertEqual(chain.unit_for(plan["plan"][1], 1), "http://gpu:41134")
        self.assertEqual(chain.unit_for(plan["plan"][3], 3), "http://gpu:41134")
        command = chain.run_command(plan["plan"][0])
        self.assertIn("--ollama-units", command)
        self.assertEqual(command[command.index("--ollama-units") + 1], "http://gpu:41135")

    def test_parallel_mode_runs_every_shard_and_merges_under_a_lock(self):
        from tools.run_shards_chain import Chain

        plan = {
            "plan": [
                {"name": f"shard_{index}", "personas": [index]} for index in (1, 2, 3)
            ]
        }
        calls: list[str] = []
        lock = threading.Lock()

        def fake_step(command: list[str], log_path: Path) -> int:
            script = next(Path(part).name for part in command if part.endswith(".py"))
            with lock:
                calls.append(script)
            if script == "run_experiment.py":
                run_dir = Path(command[command.index("--output-dir") + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "sessions.jsonl").write_text("{}\n", encoding="utf-8")
                (run_dir / "run_meta.json").write_text(
                    json.dumps({"Failed_Personas": []}), encoding="utf-8"
                )
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            chain = Chain(
                plan=plan,
                runs_root=Path(tmp) / "runs",
                config=Path("cfg.yaml"),
                start_if_idle=True,
                parallel=3,
                step_runner=fake_step,
                sleep=lambda _seconds: None,
                printer=lambda _line: None,
            )
            code = chain.run(plan["plan"])

        self.assertEqual(code, 0)
        self.assertEqual(calls.count("run_experiment.py"), 3)
        self.assertEqual(calls.count("merge_shards.py"), 3)  # once per finished shard
        self.assertEqual(calls.count("run_scoring.py"), 6)  # per shard + per merged


class ResumeModelGuardTests(unittest.TestCase):
    """--resume must not continue a store built by different models."""

    def _prepare(self, tmp: str, recorded_models) -> tuple[Path, Path]:
        root = Path(tmp)
        dataset = root / "data.jsonl"
        dataset.write_text(
            json.dumps(
                {
                    "ID": "persona-0",
                    "Full_Session_Chain": [
                        {
                            "Session_ID": 0,
                            "Date": "2022-01-03",
                            "Session_Type": "initial_reveal",
                            "Session_Dialogue": {
                                "dialogue_turn_1": [{"role": "user", "content": "hi"}]
                            },
                            "Session_Questions": [],
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        run_dir = root / "run"
        run_dir.mkdir()
        (run_dir / "sessions.jsonl").write_text(
            json.dumps(
                {
                    "Persona_ID": "persona-0",
                    "Session_ID": 0,
                    "Date": "2022-01-03",
                    "Questions": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        meta = {"Persona_Unit_Assignments": {"persona-0": None}}
        if recorded_models is not None:
            meta["Models"] = recorded_models
        (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return dataset, run_dir

    def _run_resume(self, dataset: Path, run_dir: Path, extra=None):
        from run_experiment import main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(
                [
                    "--input",
                    str(dataset),
                    "--output-dir",
                    str(run_dir),
                    "--resume",
                    "--no-progress",
                    *(extra or []),
                ]
            )
        return code, stdout.getvalue(), stderr.getvalue()

    def setUp(self):
        _patch_runtime(_FakeMemorySystem(), _FakeChatClient("ans"), _FakeChatClient())
        self._saved = _save_env(
            ollama_units.UNITS_ENV, "OLLAMA_BASE_URL", *ollama_units.ENDPOINT_ENV_KEYS
        )
        self._loader = runtime.load_env_file
        runtime.load_env_file = lambda: None
        for name in (ollama_units.UNITS_ENV, "OLLAMA_BASE_URL"):
            os.environ.pop(name, None)
        os.environ["OLLAMA_EMBED_ENDPOINT"] = "http://unit/api/embed"
        os.environ["OLLAMA_LEGACY_EMBEDDINGS_ENDPOINT"] = "http://unit/api/embeddings"

    def tearDown(self):
        runtime.load_env_file = self._loader
        _restore_env(self._saved)

    def test_resume_continues_when_the_models_are_unchanged(self):
        # Must match exactly what ``_patch_runtime`` reports for the fake config.
        recorded = {
            "embedding": "ollama/fake-embed",
            "answer_model": "fake-answer",
            "judge_model": "fake-judge",
        }
        with tempfile.TemporaryDirectory() as tmp:
            dataset, run_dir = self._prepare(tmp, recorded)
            code, _stdout, stderr = self._run_resume(dataset, run_dir)
        self.assertEqual(code, 0, stderr)

    def test_resume_refuses_a_changed_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, run_dir = self._prepare(tmp, {"embedding": "ollama/qwen3-embedding"})
            code, _stdout, stderr = self._run_resume(dataset, run_dir)
            allowed_code, _stdout2, allowed_stderr = self._run_resume(
                dataset, run_dir, ["--allow-model-change"]
            )

        self.assertEqual(code, 2)
        self.assertIn("built with different models", stderr)
        self.assertIn("ollama/qwen3-embedding -> ollama/fake-embed", stderr)
        self.assertEqual(allowed_code, 0, allowed_stderr)
        self.assertIn("--allow-model-change", allowed_stderr)

    def test_resume_of_an_older_run_without_a_fingerprint_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, run_dir = self._prepare(tmp, None)
            code, stdout, stderr = self._run_resume(dataset, run_dir)

        self.assertEqual(code, 0, stderr)
        self.assertIn("no model fingerprint", stdout)


class BailianConfigTests(unittest.TestCase):
    """Point 33: gpt-4o-mini builds the memory, gpt-5-mini answers and judges."""

    def test_eval_large_pins_the_point33_roles(self):
        import importlib

        from run_experiment import model_summary

        # Other tests stub ``runtime`` globally; reload so this test reads the
        # real loader (the same pattern RunnerMainWiringTests uses).
        importlib.reload(runtime)
        config = runtime.load_memory_config(
            EXPERIMENT_DIR / "configs" / "eval_large.yaml"
        )
        models = model_summary(config)

        self.assertEqual(models["memory_builder"], "openrouter/openai/gpt-4o-mini")
        self.assertEqual(models["adjudication_model"], "openrouter/openai/gpt-4o-mini")
        self.assertEqual(models["answer_model"], "openrouter/openai/gpt-5-mini")
        self.assertEqual(models["judge_model"], "openrouter/openai/gpt-5-mini")
        self.assertEqual(models["embedding"], "ollama/qwen3-embedding")

        # gpt-4o-mini rejects a completion budget above its 16384-token cap, so
        # the builder stage must not ask for upstream's 32768.
        builder_cap = config.memory.backends["v4"]["prompt_output_tokens"]
        self.assertEqual(builder_cap["memory_builder"], 16384)

    def test_bailian_config_is_role_identical_to_eval_large(self):
        import importlib

        from run_experiment import model_summary

        importlib.reload(runtime)
        config_path = EXPERIMENT_DIR / "configs" / "eval_large_bailian.yaml"
        config = runtime.load_memory_config(config_path)
        models = model_summary(config)

        # Bailian serves no OpenAI model, and point 33 puts every cloud role on
        # one, so nothing is left to route through DashScope here.
        self.assertEqual(models["memory_builder"], "openrouter/openai/gpt-4o-mini")
        self.assertEqual(models["adjudication_model"], "openrouter/openai/gpt-4o-mini")
        self.assertEqual(models["answer_model"], "openrouter/openai/gpt-5-mini")
        self.assertEqual(models["judge_model"], "openrouter/openai/gpt-5-mini")
        self.assertEqual(models["embedding"], "ollama/qwen3-embedding")

        runner_env = runtime.required_env_names(config, runtime.RUNNER_ROLES)
        self.assertNotIn("DASHSCOPE_API_KEY", runner_env)
        self.assertIn("OPENROUTER_API_KEY", runner_env)
        self.assertIn("OPENROUTER_CHAT_COMPLETIONS_ENDPOINT", runner_env)
        self.assertEqual(
            runtime.required_env_names(config, runtime.SCORING_ROLES),
            ["OPENROUTER_API_KEY", "OPENROUTER_CHAT_COMPLETIONS_ENDPOINT"],
        )

    def test_all_bailian_config_removes_the_ollama_dependency(self):
        """A cloud host with no campus network can run this config end to end."""
        import importlib

        from run_experiment import model_summary

        importlib.reload(runtime)
        config = runtime.load_memory_config(
            EXPERIMENT_DIR / "configs" / "eval_large_bailian_all.yaml"
        )
        models = model_summary(config)

        self.assertEqual(models["embedding"], "dashscope_bailian/text-embedding-v4")
        self.assertEqual(models["memory_builder"], "openrouter/openai/gpt-4o-mini")
        self.assertEqual(models["answer_model"], "openrouter/openai/gpt-5-mini")
        self.assertEqual(models["judge_model"], "openrouter/openai/gpt-5-mini")
        self.assertEqual(models["controller"], "dashscope_bailian/qwen3.5-flash")
        self.assertEqual(models["window_planner"], "dashscope_bailian/qwen3.5-flash")

        runner_env = runtime.required_env_names(config, runtime.RUNNER_ROLES)
        self.assertIn("DASHSCOPE_API_KEY", runner_env)
        self.assertIn("DASHSCOPE_CHAT_COMPLETIONS_ENDPOINT", runner_env)
        self.assertIn("DASHSCOPE_EMBEDDINGS_ENDPOINT", runner_env)
        self.assertIn("OPENROUTER_API_KEY", runner_env)
        self.assertNotIn("OLLAMA_CHAT_ENDPOINT", runner_env)
        self.assertNotIn("OLLAMA_EMBED_ENDPOINT", runner_env)


class RunSummaryTests(unittest.TestCase):
    """The batch launcher must not carry printf-style `%` into cmd.exe."""

    def test_summary_lines_report_clock_workers_and_failures(self):
        from tools import run_summary

        lines = run_summary.summarize(
            {
                "Wall_Clock_s": 2050.6,
                "Persona_Workers": 4,
                "Answer_Workers": 4,
                "Failed_Personas": [],
            }
        )
        self.assertIn("2050.6 s (34.2 min)", lines[0])
        self.assertIn("4 persona / 4 answer", lines[1])
        self.assertIn("none", lines[2])

    def test_summary_tool_reads_a_run_directory(self):
        from tools import run_summary

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "run_meta.json").write_text(
                json.dumps(
                    {
                        "Wall_Clock_s": 60.0,
                        "Persona_Workers": 2,
                        "Answer_Workers": 1,
                        "Failed_Personas": ["persona-x"],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(run_summary.main(["--run-dir", str(run_dir)]), 0)
            self.assertEqual(run_summary.main(["--run-dir", str(run_dir / "missing")]), 1)

    def test_the_batch_launcher_has_no_bare_percent_signs(self):
        """A bare `%` is expanded by cmd.exe before Python sees it.

        That is what turned the run summary of ``run_scale1h.bat`` into a
        ``SyntaxError`` on 2026-09-20.
        """
        import re

        launcher = Path(__file__).resolve().parents[2] / "run_scale1h.bat"
        if not launcher.is_file():
            self.skipTest("launcher not present")
        assertable = re.compile(r"%[A-Za-z_][A-Za-z0-9_]*%|%%|%~[a-zA-Z]*[0-9*]|%[0-9*]")
        for number, line in enumerate(
            launcher.read_text(encoding="utf-8").splitlines(), start=1
        ):
            residual = assertable.sub("", line)
            self.assertNotIn(
                "%",
                residual,
                f"run_scale1h.bat:{number} contains a bare percent sign: {line.strip()}",
            )


class ReproToolTests(unittest.TestCase):
    """The bug-reproduction tool must keep reproducing the upstream failure."""

    def test_repro_tool_reproduces_the_exception(self):
        tool = EXPERIMENT_DIR / "tools" / "repro_stale_checkpoint_bug.py"
        candidates = list(
            (EXPERIMENT_DIR / "runs").glob(
                "*/Memory/retrival_mem_v4_90e98aa7*/memory.sqlite3"
            )
        )
        if not candidates:
            self.skipTest("no crashed persona store in this workspace")
        result = subprocess.run(
            [sys.executable, str(tool)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("REPRODUCED", result.stdout)
        self.assertIn("reinforce references unknown fact key", result.stdout)
        self.assertIn("conversation:{len(turns)}", result.stdout)
        self.assertIn("collision", result.stdout.lower())


class RunProgressIntegrationTests(unittest.TestCase):
    """A run must show progress without anyone opening sessions.jsonl."""

    def _write_dataset(self, path: Path) -> None:
        record = {
            "ID": "persona-progress",
            "Full_Session_Chain": [
                {
                    "Session_ID": 0,
                    "Date": "2022-01-03",
                    "Session_Type": "initial_reveal",
                    "Session_Dialogue": {
                        "dialogue_turn_1": [{"role": "user", "content": "I live in Darwin."}]
                    },
                    "Session_Questions": [],
                },
                {
                    "Session_ID": 1,
                    "Date": "2022-02-03",
                    "Session_Type": "update",
                    "Session_Dialogue": {
                        "dialogue_turn_1": [{"role": "user", "content": "I moved to Melbourne."}]
                    },
                    "Session_Questions": [
                        {
                            "question_id": "Q_001",
                            "question": "Where does the user live?",
                            "answer": "Melbourne, Australia.",
                            "conflict_type": "dynamic_conflict",
                            "ability_target": "track_state_over_time",
                            "difficulty": "easy",
                        }
                    ],
                },
            ],
        }
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    def _run(self, tmp: str, extra: list[str]) -> tuple[int, str]:
        from run_experiment import main

        dataset = Path(tmp) / "data.jsonl"
        self._write_dataset(dataset)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(
                [
                    "--input",
                    str(dataset),
                    "--output-dir",
                    str(Path(tmp) / "run"),
                    "--max-sessions",
                    "2",
                    *extra,
                ]
            )
        return code, stderr.getvalue()

    def setUp(self):
        self._patch = _patch_runtime(
            _FakeMemorySystem(), _FakeChatClient("answer"), _FakeChatClient()
        )
        self._saved_env = _save_env(
            ollama_units.UNITS_ENV,
            "OLLAMA_BASE_URL",
            *ollama_units.ENDPOINT_ENV_KEYS,
        )
        self._original_env_loader = runtime.load_env_file
        runtime.load_env_file = lambda: None
        for name in (ollama_units.UNITS_ENV, "OLLAMA_BASE_URL"):
            os.environ.pop(name, None)
        # The fake config only declares an Ollama embedding role, so the
        # credential preflight looks for these two endpoint variables.
        os.environ["OLLAMA_EMBED_ENDPOINT"] = "http://unit/api/embed"
        os.environ["OLLAMA_LEGACY_EMBEDDINGS_ENDPOINT"] = "http://unit/api/embeddings"

    def tearDown(self):
        runtime.load_env_file = self._original_env_loader
        _restore_env(self._saved_env)

    def test_main_prints_progress_without_opening_sessions_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, stderr = self._run(tmp, [])
        self.assertEqual(code, 0)
        self.assertIn("[progress]", stderr)
        self.assertIn("sessions", stderr)
        self.assertIn("questions 1/1", stderr)
        self.assertIn("2/2 sessions", stderr)

    def test_no_progress_silences_the_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, stderr = self._run(tmp, ["--no-progress"])
        self.assertEqual(code, 0)
        self.assertNotIn("[progress]", stderr)


if __name__ == "__main__":
    unittest.main()
