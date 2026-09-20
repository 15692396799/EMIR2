# Experiment: Retrival-Mem on MemConflict

Conflict-testing experiment for EMIR². It drives the memory system through the
MemConflict session chains and reports the conflict-aware metrics.

**The upstream checkouts are read-only.** Nothing under `Retrival-Mem/` or
`MemConflict/` is created, edited or deleted by this directory — both are
imported/read as dependencies. All experiment-specific code, prompts, metrics
and credentials live here.

## Why a separate directory

The memory code ships a LoCoMo harness. LoCoMo and MemConflict disagree on three
things, and each one is handled here instead of by patching the upstream code:

| # | Difference | LoCoMo behaviour | MemConflict behaviour | Where |
| --- | --- | --- | --- | --- |
| 7 | Evaluation schedule | Build memory for a whole example, then answer every question | Answer each session's questions right after that session is ingested | `memconflict_eval/data.py`, `memory.py`, `run_experiment.py` |
| 8 | Answer prompt | Routed by LoCoMo category (multi-hop / temporal / open-ended / single-hop) | Routed by `conflict_type` (dynamic / static / conditional) | `memconflict_eval/prompts.py` |
| 8 | Judge prompt | `MEM0_ACCURACY_PROMPT`: binary CORRECT/WRONG, deliberately lenient, no memory-level judgement | Graded accuracy + conflict handling + white-box support rank | `memconflict_eval/prompts.py`, `judging.py` |
| 9 | Metrics | LoCoMo accuracy / BLEU-style scoring | Table 3: AA and SEH@3 per conflict type + Average AA | `memconflict_eval/metrics.py` |

Personas are fully independent (separate store, namespace and directory); the
sessions inside one persona are sequential, so a later session can see every
earlier session and no question can ever see a later one.

## Layout

```text
Experiment/
├── run_experiment.py           # point 7: ingest session -> answer that session
├── run_scoring.py              # points 8+9: judge + Table 3 metrics
├── .env.example                # credentials (copy to Experiment/.env)
├── memconflict_eval/
│   ├── runtime.py              # import the unmodified Retrival-Mem checkout
│   ├── data.py                 # point 7: MemConflict dataset reading
│   ├── memory.py               # point 7: per-persona store, per-session ingest
│   ├── ollama_units.py         # point 16: one Ollama container per worker
│   ├── parallel.py             # point 17: personas over worker processes
│   ├── prompts.py              # point 8: answer + judge prompts
│   ├── answering.py            # point 8: answer generation
│   ├── judging.py              # point 8: LLM judge
│   └── metrics.py              # point 9: AA / SEH@3 / SRS / Average AA
├── tools/check_ollama_units.py # probe every container for GPU backing
└── tests/test_experiment.py    # 84 offline tests
```

## Setup

```powershell
Copy-Item Experiment\.env.example Experiment\.env
# then fill in DASHSCOPE_API_KEY, OPENROUTER_API_KEY and the Ollama endpoints
```

The Ollama service must serve `qwen3-embedding` (embeddings) and
`qwen3.5:latest` (controller / window planner / entity judge).

## Run

```powershell
# Smoke test: one persona (53 sessions, 122 questions).
python Experiment\run_experiment.py --persona-limit 1

# Score that run and print the Table 3 row.
python Experiment\run_scoring.py --run-dir Experiment\runs\<timestamp>

# Full benchmark: all 30 personas.
python Experiment\run_experiment.py
python Experiment\run_scoring.py --run-dir Experiment\runs\<timestamp>
```

### Points 16 + 17: several GPUs, several personas at once

The GPU server runs one Ollama container per GPU. List them once in
`Experiment/.env` (comma separated, base URLs only) and every persona worker
owns one of them, round robin:

```text
OLLAMA_BASE_URLS=http://172.26.94.12:41133,http://172.26.94.12:41134,http://172.26.94.12:41135,http://172.26.94.12:41136
```

```powershell
# Four personas side by side, one container each.
python Experiment\run_experiment.py --persona-workers 4

# Same parallelism on one container (or override .env for a single run).
python Experiment\run_experiment.py --persona-workers 4 --ollama-units http://172.26.94.12:41135

# A/B a change: identical workload, only the worker count differs.
python Experiment\run_experiment.py --persona-limit 4 --max-sessions 2 --persona-workers 1 --ollama-units http://172.26.94.12:41135
python Experiment\run_experiment.py --persona-limit 4 --max-sessions 2 --persona-workers 4 --ollama-units http://172.26.94.12:41135
```

`--persona-workers 1` is the original in-process loop, so results stay
comparable. Personas are independent (own store, namespace and directory), and
the session chain inside one persona is still replayed serially, so point 7 is
unaffected. `run_scoring.py` takes the mirror flags `--judge-workers` and
`--ollama-units`.

Check the containers before a long run (a container that lost its device still
answers, just at CPU speed):

```powershell
python Experiment\tools\check_ollama_units.py
```

`run_meta.json` records `Persona_Workers`, `Ollama_Units`, the per-persona
`Persona_Unit_Assignments` and the run's `Wall_Clock_s`; every persona record in
`results.jsonl` carries its `Ollama_Unit` and `Started_At_s` / `Finished_At_s`
relative to the start of the run, which is what shows the overlap.

The measured serial-vs-parallel numbers are in
*Serial baseline and parallel speed-up (measured)* below.

`run_experiment.py` writes `results.jsonl` (answers + retrieved memories) and
`run_meta.json`. `run_scoring.py` adds `scores.jsonl`, `metrics.json` and
`table3.md`. Run directories (`Experiment/runs/`, and any memory store they
contain) are git-ignored.

Useful flags:

* `--start-index` / `--end-index` / `--persona-limit` — select personas
* `--top-k` — memories placed in the answer prompt (default 3)
* `--stored-top-k` — retrieved memories persisted per question (default 5)
* `--keep-memory` — reuse an existing store instead of rebuilding it

## Serial baseline and parallel speed-up (measured)

Numbers below were measured on the first end-to-end pass with **one persona
worker**, so they are the reference point for the persona-parallel comparison.
Full artefacts and the failure write-ups are in
`archive/2026-09-19_smoke_eval_large/`.

**Workload.** Persona `9841f645-49ad-3de3-9010-cef938781102`, sessions 0-3
(4 of its 51 sessions, so 2 of its 122 questions). Config `configs/eval_large.yaml`:
OpenRouter `z-ai/glm-5.1` for memory building / adjudication / answering,
`openai/gpt-4o-mini` for judging, and the GPU server's Ollama
(`qwen3-embedding`, `qwen3.5:latest`) for embeddings and retrieval control.

```powershell
python Experiment\run_experiment.py --config Experiment\configs\eval_large.yaml `
  --start-index 1 --persona-limit 1 --max-sessions 4 --top-k 3 --stored-top-k 5 `
  --persona-workers 1 --ollama-units http://172.26.94.12:41135
python Experiment\run_scoring.py --run-dir Experiment\runs\<timestamp>
```

### Wall clock

| Quantity | Measured |
| --- | ---: |
| Persona wall clock (4 sessions) | **1,155.3 s** |
| Per session (wall clock) | **~289 s** |
| Ingest only, session 0 / 1 / 2 / 3 | 407.1 / 190.1 / 84.0 / 348.7 s |

Session 3 carries the two questions, so its 348.7 s includes retrieval and
answering; sessions 0-2 are ingest only.

### Per-stage cost and time

From the `v4_*.jsonl` API logs of that run (Ollama-served stages are free):

| Stage | Provider | Calls | Input | Output | Reasoning | Cost | Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| memory_builder | OpenRouter | 4 | 37,308 | 27,053 | 0 | $0.15694 | 465.1 s |
| adjudication | OpenRouter | 18 | 61,012 | 16,626 | 3,237 | $0.12982 | 307.6 s |
| semantic_reducer | OpenRouter | 28 | 38,682 | 6,819 | 1,082 | $0.06621 | 182.3 s |
| window_planner | Ollama | 47 | 51,915 | 2,169 | 0 | — | 205.0 s |
| controller | Ollama | 18 | 100,074 | 1,975 | 0 | — | 64.7 s |
| semantic_slimming | Ollama | 11 | 9,276 | 1,164 | 0 | — | 20.5 s |
| rerank | Ollama | 3 | 8,077 | 782 | 0 | — | 12.7 s |
| entity_judge | Ollama | 1 | 1,101 | 118 | 0 | — | 6.2 s |
| decomposition_gate | Ollama | 2 | 636 | 20 | 0 | — | 1.2 s |
| **total** | | **132** | **308,081** | **56,726** | 4,319 | **$0.35296** | 1,265.2 s |

Stage time sums to 1,265.2 s while the persona wall clock is 1,155.3 s, because
stages inside a session overlap (`memory_extraction_workers`,
`entity_judge_workers`). Compare wall clock, not the stage column.

Answer generation is not in the table: `MemConflictAnswerer` is built without
the API-history logger, so that stage is unmeasured (~$7 projected for the full
benchmark).

### Per-session and full-run projection

| Quantity | Measured / projected |
| --- | ---: |
| API cost per session | **$0.088** |
| API cost of the whole 4-session persona | $0.353 |
| Full run, serial (1,579 sessions, 3,750 questions) | **~$146** |
| Full run, wall clock, serial | **~127 h** |

The bottleneck is wall clock, not spend. Sessions inside a persona must be
replayed serially (point 7: a session's questions are answered before the next
session is ingested), so the only axis for speed-up is personas — which are
fully independent.

### Parallel speed-up (measured, 2026-09-20)

Same config, same workload (the first session of personas 2-5:
`--start-index 1 --persona-limit 4 --max-sessions 1`), same single container
(`ollama021-1` on `device6`, `http://172.26.94.12:41135`). Only
`--persona-workers` differs, so the pair isolates point 17:

```powershell
python Experiment\run_experiment.py --config Experiment\configs\eval_large.yaml `
  --start-index 1 --persona-limit 4 --max-sessions 1 --persona-workers 1 `
  --ollama-units http://172.26.94.12:41135
python Experiment\run_experiment.py --config Experiment\configs\eval_large.yaml `
  --start-index 1 --persona-limit 4 --max-sessions 1 --persona-workers 4 `
  --ollama-units http://172.26.94.12:41135
```

| Arm | Workers | Persona wall clock | Run wall clock |
| --- | ---: | --- | ---: |
| Serial | 1 | 815.7 / 465.8 / 557.9 / 600.4 s | **2,439.9 s** (40.7 min) |
| Parallel | 4 | 411.5 / 331.8 / 347.9 / 589.2 s | **589.7 s** (9.8 min) |

**4.14x wall-clock speed-up on a single GPU container.** Every persona in the
parallel arm also finished faster than it did when run alone (411 vs 816 s,
332 vs 466 s, 348 vs 558 s, 589 vs 600 s), so the gain is not merely trading
per-persona latency for throughput. `results.jsonl` shows the overlap directly:
the parallel arm starts all four personas within 0.05 s, the serial arm starts
each one where the previous ended.

Read `Wall_Clock_s`, `Persona_Workers` and `Persona_Unit_Assignments` from each
`run_meta.json`, and `Ollama_Unit` / `Started_At_s` / `Finished_At_s` per persona
from `results.jsonl`, to reproduce the table.

Two caveats worth knowing before quoting these numbers:

* **Today's machine/API state is ~2.1x slower per session** than on 2026-09-19
  (610 s vs 289 s per persona-session serially), so compare same-day runs only.
  By rate the change is **610 s -> 147 s per persona-session**.
* This workload is ingestion only (`--max-sessions 1` answers no questions, since
  these personas' first questions live in session 3+). The equivalence checks
  below still need a question-bearing run.

### What to check when re-running

* **Identical answers.** The two runs must produce the same `Model_Answer` and
  the same `Retrieved_Memories` for every question — parallelism must not change
  results, only the schedule.
* **Same token totals.** Compare the summed input/output tokens across the
  `v4_*.jsonl` logs; a large drift means the unit assignment changed the
  retrieval path.
* **Per-persona failures.** A failing persona is reported in `errors.jsonl` and
  makes the run exit non-zero; the others still keep their results.

Full artefacts, run directories and the `BrokenProcessPool` / `V4StageError`
failure that the first parallel attempt exposed are written up in
`archive/2026-09-20_multi_container_parallel/NOTES.md`.

## Metrics

Table 3 (the row EMIR² reports):

| Method | Dynamic AA↑ | Dynamic SEH@3↑ | Static AA↑ | Static SEH@3↑ | Conditional AA↑ | Conditional SEH@3↑ | Average AA↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |

* **AA** — mean judged answer accuracy. Dynamic and static answers take 0 / 0.5
  / 1; conditional answers are all-or-nothing.
* **SEH@3** — share of questions whose supporting memory is ranked in the top 3
  by the judge.
* **Average AA** — mean of the three conflict-type AA columns.

`table3.md` also prints a detail table with SRS (`1 / log2(rank + 1)`) and the
dynamic UOCS / static CRS diagnostics, which come from the same judge call.

## Tests

```powershell
python -m unittest discover -s Experiment\tests -t .
```

The tests never call an LLM API. The memory system and chat clients are stubbed,
and the dataset tests read `MemConflict/Data/Step4_4.jsonl` as data. They cover
the turn-ordering trap, per-session QA ordering, prompt routing, judge JSON
tolerance and the metric aggregation.
