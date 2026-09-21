"""Tests for the harness-side OpenRouter reasoning guard.

Point 33 answers and judges with ``openai/gpt-5-mini``. Upstream's
``disable_request_thinking`` rewrites every OpenRouter request to
``reasoning: {"enabled": false, "effort": "none"}``, which that endpoint rejects
with ``400 Reasoning is mandatory for this endpoint and cannot be disabled``.
The guard (``memconflict_eval/openrouter_reasoning.py``) keeps upstream's intent
in a shape the endpoint accepts, and these tests pin both halves: the
reasoning-mandatory models get a legal effort, everything else keeps the
upstream request byte for byte.

Run with::

    python -m unittest discover -s Experiment/tests -v
"""

from __future__ import annotations

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
from memconflict_eval.openrouter_reasoning import (  # noqa: E402
    EFFORT_ENV,
    GUARD_ENV,
    install_reasoning_guard,
    reasoning_effort,
    requires_reasoning,
)

# Puts Retrival-Mem/src on sys.path and installs the guard, exactly like a run.
runtime.import_retrival_mem()

from memory.clients import disable_request_thinking  # noqa: E402


class OpenRouterReasoningGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            name: os.environ.pop(name, None) for name in (GUARD_ENV, EFFORT_ENV)
        }
        install_reasoning_guard()

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value

    @staticmethod
    def payload(model: str = "openai/gpt-5-mini", **extra: object) -> dict:
        return {"model": model, "messages": [], "temperature": 1.0, **extra}

    def test_a_reasoning_mandatory_model_gets_a_legal_effort(self) -> None:
        payload = self.payload(reasoning_effort="none", thinking=False)

        disable_request_thinking(payload, "openrouter")

        self.assertEqual(payload["reasoning"], {"effort": "minimal"})
        # The caller's own "do not think" fields are still gone.
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("thinking", payload)

    def test_other_models_keep_the_upstream_shape(self) -> None:
        payload = self.payload(model="openai/gpt-4o-mini")

        disable_request_thinking(payload, "openrouter")

        self.assertEqual(payload["reasoning"], {"enabled": False, "effort": "none"})

    def test_other_providers_are_untouched(self) -> None:
        payload = self.payload()

        disable_request_thinking(payload, "dashscope_bailian")

        self.assertIs(payload["enable_thinking"], False)
        self.assertNotIn("reasoning", payload)

    def test_the_effort_is_configurable_and_can_be_dropped(self) -> None:
        os.environ[EFFORT_ENV] = "low"
        payload = self.payload()
        disable_request_thinking(payload, "openrouter")
        self.assertEqual(payload["reasoning"], {"effort": "low"})

        os.environ[EFFORT_ENV] = "off"
        payload = self.payload()
        disable_request_thinking(payload, "openrouter")
        self.assertNotIn("reasoning", payload)

    def test_the_guard_can_be_switched_off(self) -> None:
        os.environ[GUARD_ENV] = "0"
        payload = self.payload()

        disable_request_thinking(payload, "openrouter")

        self.assertEqual(payload["reasoning"], {"enabled": False, "effort": "none"})

    def test_requires_reasoning_covers_the_gpt5_and_o_series(self) -> None:
        self.assertTrue(requires_reasoning("openai/gpt-5-mini"))
        self.assertTrue(requires_reasoning("GPT-5"))
        self.assertTrue(requires_reasoning("openai/o3-mini"))
        self.assertFalse(requires_reasoning("openai/gpt-4o-mini"))
        self.assertFalse(requires_reasoning("z-ai/glm-5.1"))
        self.assertFalse(requires_reasoning(None))
        self.assertEqual(reasoning_effort(), "minimal")


if __name__ == "__main__":
    unittest.main()
