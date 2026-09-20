# Smoke run archive — 2026-09-19

First end-to-end pass of the MemConflict experiment against the real memory
system, the school GPU server's Ollama and the OpenRouter large models.

Source run directory: `Experiment/runs/large_20260919_232108/` (git-ignored).
Only the small artefacts are archived here; the per-persona memory store
(`memory.sqlite3`, 23 MB), the FAISS index and the raw `v4_*.jsonl` API logs
stay in the run directory.

## Setup

| Role | Provider | Model |
| --- | --- | --- |
| memory_builder, adjudication_model, answer_model | OpenRouter | `z-ai/glm-5.1` |
| judge_model | OpenRouter | `openai/gpt-4o-mini` |
| embedding | Ollama (GPU server) | `qwen3-embedding` |
| controller, window_planner, entity_judge, decomposition_gate, slm | Ollama (GPU server) | `qwen3.5:latest` |

Config: `Experiment/configs/eval_large.yaml`.
Ollama endpoint: `http://172.26.94.12:41135` (`ollama021-1`, RTX 3090).

Command:

```powershell
python Experiment\run_experiment.py --config Experiment\configs\eval_large.yaml `
  --start-index 1 --persona-limit 1 --max-sessions 4 --top-k 3 --stored-top-k 5
python Experiment\run_scoring.py --run-dir <run dir>
```

Scope: persona `9841f645-49ad-3de3-9010-cef938781102`, sessions 0-3 (the first
four of 51). Session 3 carries the first questions, so 2 of the 122 questions
for this persona were answered.

## Result

```
| Method | Dynamic AA | Dynamic SEH@3 | Static AA | Static SEH@3 | Conditional AA | Conditional SEH@3 | Average AA |
| EMIR²  | 0.5000     | 0.5000        | –         | –            | –              | –                 | 0.5000     |
```

Two dynamic questions: one answered correctly, one answered "cannot confirm"
even though the supporting memory was retrieved — a retrieval-utilization
failure of exactly the kind MemConflict is designed to expose. Two questions is
far too small a sample to draw any conclusion; this run only proves the
pipeline.

## Measured cost and time

Memory-building stage (from the `v4_*.jsonl` API logs):

| Stage | Calls | In | Out | Reasoning | Cost | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| memory_builder | 4 | 37,308 | 27,053 | 0 | $0.15694 | 465.1 s |
| adjudication | 18 | 61,012 | 16,626 | 3,237 | $0.12982 | 307.6 s |
| semantic_reducer | 28 | 38,682 | 6,819 | 1,082 | $0.06621 | 182.3 s |
| window_planner | 47 | 51,915 | 2,169 | 0 | free (Ollama) | 205.0 s |
| controller | 18 | 100,074 | 1,975 | 0 | free (Ollama) | 64.7 s |
| semantic_slimming | 11 | 9,276 | 1,164 | 0 | free (Ollama) | 20.5 s |
| rerank | 3 | 8,077 | 782 | 0 | free (Ollama) | 12.7 s |
| entity_judge | 1 | 1,101 | 118 | 0 | free (Ollama) | 6.2 s |
| decomposition_gate | 2 | 636 | 20 | 0 | free (Ollama) | 1.2 s |
| **total** | 132 | 308,081 | 56,726 | 4,319 | **$0.35296** | 1,265.2 s |

* Persona wall clock: 1,155 s for 4 sessions (~289 s/session).
* Per-session API cost: ~$0.088.
* Extrapolated full run (1,579 sessions, 3,750 questions): **~127 h serial,
  ~$146** including answer generation and judging. The bottleneck is wall
  clock, not spend; sessions must be replayed serially per persona because each
  session's questions are answered before the next session is ingested.
* Answer-generation tokens are not in the table above: the experiment's answerer
  is not wrapped in the API-history logger, so that stage is not recorded.

## Problems hit and how they were resolved

1. **`turn_id` must be a string.** With a JSON number the builder model invents a
   `turn_` prefix (`"turn_33"`), and the extracted `evidence_turn_ids` then fail
   validation because `turn_33` is not a supplied id. Verified by replaying the
   same session with `33`, `"33"` and `"turn_33"`: 0/30, 25/25 and 49/49 ids
   matched respectively.
2. **`prompt_output_tokens.memory_builder` defaults to 8192.** That cap was sized
   for LoCoMo sessions (~670-1,360 tokens); MemConflict sessions are ~3,840 and
   their extraction exceeds 8192, returning truncated JSON. Raised to 32768.
   Note V4's `with_output_token_limit()` overrides any `max_tokens` set in the
   model's `extra_body`.
3. **Cloud embedding endpoints cap the batch.** Bailian rejects more than 10
   inputs per request, while V4 embeds all entity aliases in one call. A batching
   wrapper is injected through `MemorySystem(embedding_client=...)`.
4. **`z-ai/glm-5.1` is a reasoning model.** It intermittently spends its whole
   output budget on reasoning tokens and returns empty content (measured: 4,091
   reasoning tokens out of 4,096, `content` = ""), which aborts adjudication.
   Raising `prompt_output_tokens.cross_window_adjudication` from 4096 to 16384
   leaves room for both. A replay of the exact failing request against four
   OpenRouter models gave: `z-ai/glm-5.1` OK (43.6 s), `google/gemini-2.5-flash`
   OK (9.1 s), `openai/gpt-4o-mini` OK (8.0 s), `anthropic/claude-sonnet-4.5`
   FAIL (output not valid JSON).
5. **A local 9.7B model cannot do adjudication.** `qwen3.5:latest` omits
   candidate pairs in roughly 20% of batches, and V4 does not retry validation
   failures (`validation_retries` defaults to 0), so the build aborts. Upstream
   `default.yaml` points this role at the same large model as the memory builder,
   which is why `configs/ollama_smoke.yaml` now uses `qwen3.5:27b` for it.

## GPU note

`ollama021-1` originally ran entirely on CPU (0.8 tok/s) because its container
had lost GPU access: it held only `/dev/nvidia5` while the host had all eight
nodes, and CUDA reported "no CUDA-capable device is detected". After the
container was recreated it reached 54.7 tok/s. **Checking `nvidia-smi` for used
VRAM is not a valid test** — Ollama unloads the model after 5 idle minutes. Send
a request and then check `/api/ps` for a non-zero `size_vram`.
