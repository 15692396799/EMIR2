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
│   ├── prompts.py              # point 8: answer + judge prompts
│   ├── answering.py            # point 8: answer generation
│   ├── judging.py              # point 8: LLM judge
│   └── metrics.py              # point 9: AA / SEH@3 / SRS / Average AA
└── tests/test_experiment.py    # 50 offline tests
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

`run_experiment.py` writes `results.jsonl` (answers + retrieved memories) and
`run_meta.json`. `run_scoring.py` adds `scores.jsonl`, `metrics.json` and
`table3.md`. Run directories (`Experiment/runs/`, and any memory store they
contain) are git-ignored.

Useful flags:

* `--start-index` / `--end-index` / `--persona-limit` — select personas
* `--top-k` — memories placed in the answer prompt (default 3)
* `--stored-top-k` — retrieved memories persisted per question (default 5)
* `--keep-memory` — reuse an existing store instead of rebuilding it

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
