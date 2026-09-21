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
| 9 | Metrics | LoCoMo accuracy / BLEU-style scoring | Tables 3 / 5 / 6: AA, SEH@K, SRS, UOCS, CRS per conflict type | `memconflict_eval/metrics.py` |

Personas are fully independent (separate store, namespace and directory); the
sessions inside one persona are sequential, so a later session can see every
earlier session and no question can ever see a later one.

## Layout

```text
Experiment/
├── run_experiment.py           # point 7: ingest session -> answer that session
├── run_scoring.py              # points 8+9: judge + Tables 3/5/6
├── docs/table5_table6_design.md # black-box/white-box experiment design
├── .env.example                # credentials (copy to Experiment/.env)
├── memconflict_eval/
│   ├── runtime.py              # import the unmodified Retrival-Mem checkout
│   ├── data.py                 # point 7: MemConflict dataset reading
│   ├── memory.py               # point 7: per-persona store, per-session ingest
│   ├── ollama_units.py         # point 16: one Ollama container per worker
│   ├── parallel.py             # point 17: personas over worker processes
│   ├── progress.py             # the live progress line (run + scoring)
│   ├── prompts.py              # point 8: answer + judge prompts
│   ├── answering.py            # point 8: answer generation
│   ├── judging.py              # point 8: LLM judge
│   └── metrics.py              # point 9: AA / SEH@K / SRS / UOCS / CRS
├── tools/check_ollama_units.py # probe every container for GPU backing
├── tools/merge_shards.py       # P1-3: merge persona shards into one run
├── tools/rebuild_tables.py     # re-project scores.jsonl into Tables 3/5/6, no LLM
└── tests/test_experiment.py    # 108 offline tests
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

# Score that run: one judge pass, then Tables 3 / 5 / 6.
python Experiment\run_scoring.py --run-dir Experiment\runs\<timestamp>

# Same, but also report the white-box columns at Top-2 / Top-3 / Top-5.
python Experiment\run_scoring.py --run-dir Experiment\runs\<timestamp> --white-box-k 2,3,5

# Already scored and only the tables changed? Rebuild them offline (no LLM).
python Experiment\tools\rebuild_tables.py --run-dir Experiment\runs\<timestamp> --white-box-k 2,3,5

# Full benchmark: all 30 personas.
python Experiment\run_experiment.py
python Experiment\run_scoring.py --run-dir Experiment\runs\<timestamp>
```

### Point 33: which model does what

| Role | Model | Provider |
| --- | --- | --- |
| `memory_builder`, `adjudication_model` | `openai/gpt-4o-mini` | OpenRouter |
| `answer_model` | `openai/gpt-5-mini` | OpenRouter |
| `judge_model` | `openai/gpt-5-mini` | OpenRouter |
| `embedding` | `qwen3-embedding` | GPU server's Ollama |
| `controller`, `window_planner`, `entity_judge`, `decomposition_gate`, `slm` | `qwen3.5:latest` | GPU server's Ollama |

Point 33 moved all three cloud roles: memory construction left `z-ai/glm-5.1`
for `openai/gpt-4o-mini`, and both the answering role (was `z-ai/glm-5.1`) and
the judge (was `openai/gpt-4o-mini`) now run on `openai/gpt-5-mini`. Note what
that means for the artefacts already on disk: the pre-point-33 shards answered
with `z-ai/glm-5.1`, so their AA is not comparable with a post-point-33 run --
the answers themselves change, not just the grading. Re-run a shard before
mixing the two in one table (`merge_shards.py` warns, `run_meta.json` records
the roles).

`configs/eval_large_bailian*.yaml` follow the same roles. Bailian serves no
OpenAI model, so `eval_large_bailian.yaml` has no Bailian role left and is now
role-identical to `eval_large.yaml` (kept because the shard commands and the
chain state refer to that name); `eval_large_bailian_all.yaml` still exists for
a host with no campus network, but it now needs `OPENROUTER_*` for the builder,
the answerer *and* the judge, and only its embedding / control roles come from
Bailian.

Two consequences of the swap:

* `prompt_output_tokens.memory_builder` is 16384, not upstream's 32768:
  gpt-4o-mini caps completions at 16384 tokens and the provider rejects the
  larger value as an invalid `max_tokens`. The V4 extraction JSON is large, so a
  session that needs more than 16384 output tokens is truncated and fails
  validation -- watch `Ingest.Retried_After_Validation_Error` in
  `sessions.jsonl` on the first run with the new builder.
* Both gpt-5-mini roles are reasoning models now: 8192 output tokens and a 300 s
  timeout instead of the budgets that were sized for gpt-4o-mini (512 tokens)
  and for glm-5.1. A small budget on a reasoning model can be spent entirely on
  reasoning tokens and return empty content: `MemConflictAnswerer` raises
  `EmptyAnswerError` for a blank completion (recorded as `Answer_Error` on the
  question row, `Answer_Error_Count` in `metrics.json`) instead of storing a
  blank answer that AA would quietly count as wrong, and
  `MemConflictJudge` records a `Judge_Error` the same way. Neither is retried,
  and gpt-5-mini only accepts the default temperature (1.0, set in the config).
  The upstream client also always disables thinking
  (`reasoning: {enabled: false, effort: none}` for OpenRouter, from
  `Retrival-Mem`'s `disable_request_thinking`); if a serving endpoint rejects
  that field, the answering and judging lanes error on every question -- run
  `check_channels.py --roles answer_model,judge_model` once before a long run
  and score with `configs/eval_large_dsjudge.yaml` if the judge fails.

### Points 16 + 17: several GPUs, several personas at once

The GPU server runs one Ollama container per GPU, each publishing its own host
port. List them once in `Experiment/.env` (comma separated, base URLs only) and
every persona worker owns one of them, round robin. The shipped default lists
all six lanes; `run_experiment.py` probes them at startup and refuses to start
if one is down (`--allow-missing-units` runs on the rest instead).

| Host port | Container | GPU | Status (probed 2026-09-21) |
| --- | --- | --- | --- |
| 41137 | `ollama021-3-1` | device2 | new lane -- start it |
| 41138 | `ollama021-3-2` | device3 | new lane -- start it |
| 41133 | `ollama021-2-1` | device4 | GPU, 8.72 GB vram, 94.1 tok/s |
| 41134 | `ollama021-2-2` | device5 | **not running -- start it** |
| 41135 | `ollama021-1` | device6 | GPU, 8.72 GB vram, 48.2 tok/s while the live shards use it |
| 41136 | `ollama021` | device7 | GPU, 8.72 GB vram, 94.5 tok/s |

Start the missing containers on the GPU server (same shape as `ollama021-1`, and
idempotent):

```bash
docker run -d --name ollama021-3-1 --gpus '"device=2"' -p 0.0.0.0:41137:11434 \
  -v /data/ollama_models:/root/.ollama/models -e OLLAMA_CONTEXT_LENGTH=32768 \
  --restart unless-stopped ollama/ollama:0.21
docker run -d --name ollama021-3-2 --gpus '"device=3"' -p 0.0.0.0:41138:11434 \
  -v /data/ollama_models:/root/.ollama/models -e OLLAMA_CONTEXT_LENGTH=32768 \
  --restart unless-stopped ollama/ollama:0.21
docker run -d --name ollama021-2-2 --gpus '"device=5"' -p 0.0.0.0:41134:11434 \
  -v /data/ollama_models:/root/.ollama/models -e OLLAMA_CONTEXT_LENGTH=32768 \
  --restart unless-stopped ollama/ollama:0.21
docker run -d --name ollama021 --gpus '"device=7"' -p 0.0.0.0:41136:11434 \
  -v /data/ollama_models:/root/.ollama/models -e OLLAMA_CONTEXT_LENGTH=32768 \
  --restart unless-stopped ollama/ollama:0.21
```

One container per GPU keeps the GPUs independent: a lane is only as fast as its
own card, and a container that crashes takes only its own workers down. Host
ports 41137/41138 were free when this was written (`41133`-`41136` are the
older lanes).

`Experiment/tools/ollama_containers.sh status` lists what runs where, and
`Experiment/tools/ollama_containers.sh start` (re)creates the missing ones.

Why extra lanes pay off now: the 2026-09-20 A/B (four personas over two
containers: 723.5 s versus 589.7 s on one) was taken while the cloud models
carried the wall clock. The 2026-09-21 picture from shard_1's own API logs is
different -- retrieval costs **141 s per question and 99% of the answer stage**,
all of it Ollama, with five workers queueing on a single container. The run is
Ollama bound now, so every container you add removes real queueing.

```powershell
# One persona worker per container (six lanes in .env, six workers).
python Experiment\run_experiment.py --persona-workers 6

# Explicit lane list, one container per worker.
python Experiment\run_experiment.py --persona-workers 6 `
  --ollama-units http://172.26.94.12:41135,http://172.26.94.12:41136,http://172.26.94.12:41133,http://172.26.94.12:41134,http://172.26.94.12:41137,http://172.26.94.12:41138

# Two personas per container (more requests in flight per GPU; watch tok/s).
python Experiment\run_experiment.py --persona-workers 12

# Keep going while device5/2/3 come up: drop the dead lanes and use the rest.
python Experiment\run_experiment.py --persona-workers 6 --allow-missing-units
```

`--persona-workers 1` is the original in-process loop, so results stay
comparable. Personas are independent (own store, namespace and directory), and
the session chain inside one persona is still replayed serially, so point 7 is
unaffected. `run_scoring.py` takes the mirror flags `--judge-workers` and
`--ollama-units`.

Check the lanes before a long run. Two failure modes to catch: a container that
lost its device still answers, just at CPU speed, and a container that is not
running makes every worker assigned to it fail.

```powershell
python Experiment\tools\check_ollama_units.py
python Experiment\tools\check_ollama_units.py --units http://172.26.94.12:41133,http://172.26.94.12:41134,http://172.26.94.12:41135,http://172.26.94.12:41136,http://172.26.94.12:41137,http://172.26.94.12:41138
```

The startup probe answers the first question only ("is the lane up?"). It is
`GET /api/tags` per lane, which is why it is safe to run every time; a container
that is up but has lost its GPU still needs the tok/s check above.

`run_meta.json` records `Persona_Workers`, `Ollama_Units`, the per-persona
`Persona_Unit_Assignments` and the run's `Wall_Clock_s`; every persona record in
`results.jsonl` carries its `Ollama_Unit` and `Started_At_s` / `Finished_At_s`
relative to the start of the run, which is what shows the overlap.

The measured serial-vs-parallel numbers are in
*Serial baseline and parallel speed-up (measured)* below.

### Point 17b: answering one session's questions in parallel

A session is fully ingested before any of its questions run, so the questions
of that session read the same frozen memory state and only the schedule
differs. `--answer-workers N` (default 4) is therefore a pure speed-up over the
historical serial loop (`--answer-workers 1`):

```powershell
python Experiment\run_experiment.py --persona-workers 4 --answer-workers 4
```

It is safe by construction: the V4 store opens SQLite with
`check_same_thread=False` behind a re-entrant lock, FAISS searches and the
API-history log take their own locks, and every `retrieve()` builds its own
controller (the backend exposes `get_multi_round_controller`, not a cached
`get_controller`). A question that fails is recorded as `Answer_Error` and does
not stop the rest of the session; the count lands in the session row, the
persona record and `metrics.json`.

Concurrency is multiplicative, so cap the per-worker thread pools when several
persona workers share the same endpoints:

```powershell
python Experiment\run_experiment.py --persona-workers 4 --extraction-workers 2 --entity-judge-workers 1
```

Those two flags override `memory.memory_extraction_workers` and
`memory.backends.v4.entity_judge_workers` inside every worker (P1-2). Without
them, four workers at the configured eight extraction threads put 32 calls in
flight at once, which is what produced the builder validation failures in the
first parallel attempt.

### Watching a run: the live progress line

`run_experiment.py` writes one `sessions.jsonl` row per finished session, which
is complete but awkward to watch. It also renders a progress line on stderr, so
the state of a long parallel run is visible without opening that file:

```text
[progress] [#####---------------] 12/44 sessions 27% | 21m30s elapsed | eta 58m10s | questions 9/46 | errors 0 | personas 2/4 | failed 0 | running 2/4 | last 8cb3c9c6 s3
```

* The counters are fed by the same rows that are appended to `sessions.jsonl`,
  from the parent process, so they also work with `--persona-workers 4` (the
  child processes ship their rows back over the progress queue).
* The denominator is what this invocation will actually replay: `--max-sessions`
  truncates every chain and `--resume` subtracts the sessions that already have
  a row, so the percentage and the ETA describe the remaining work.
* On a terminal one line is rewritten in place (`\r`), and it drops its least
  important fields first when the window is narrow. When the output is
  redirected to a file it prints a new line every 60 s instead, which keeps
  `run_scale1h.bat > console.log 2>&1` readable.
* `--no-progress` disables it. `run_scoring.py` takes the same flag and renders
  a persona-level line (the judge only reports at persona granularity).

### The stale build-checkpoint guard

V4 caches every build stage in the store's `v4_build_checkpoints` table and
replays a `status='succeeded'` row whenever the same unit (same
`checkpoint_key`) is built again. The loader matches on that key and the status
only, so an output that was validated against an older semantic state can be
replayed as-is; a `reinforce <fact_key>` operation then references a key that no
longer exists and the persona dies:

```text
SemanticValidationError: reinforce references unknown fact key
```

Measured on 2026-09-20: persona `90e98aa7` stopped after 6 of its 11 sessions
while replaying a cached `semantic_update` output that had been recorded at
scope revision 5, with the scope already at revision 6. The replay path
(`builder.py:2023-2031`) applies the cached output with validation *enabled* and
has no feedback retry, so it cannot repair itself.

Reproduce it in seconds, offline, without touching the original store:

```powershell
python Experiment\tools\repro_stale_checkpoint_bug.py
```

The tool copies a crashed persona's store, shows three upstream design facts
(plan cache keyed by turn count -> cross-session collisions; the checkpoint
loader ignoring `scope_revision`; the replay path raising instead of
recomputing) and prints the same exception the run recorded in
`errors.jsonl`. It needs no model, no proxy and no credentials.

`memconflict_eval/memory.py` guards against this without touching the upstream
checkout:

* before every session ingest, succeeded checkpoints recorded against an older
  scope revision are marked `failed` — the cache still works inside one
  session's own retries, which is the only reuse V4 needs;
* if an ingest still fails with one of the stale-cache validation messages, the
  namespace's checkpoints are invalidated and the session is retried **once**;
  the retry is reported on stderr and in the session row
  (`Ingest.Retried_After_Validation_Error`, `Ingest.Stale_Checkpoints_Dropped`);
* `MEMCONFLICT_CHECKPOINT_GUARD=0` turns the guard off (the raw upstream
  behaviour, useful when comparing against an unguarded run).

### Interrupted runs: resume, and never lose finished work

* `results.jsonl` is written the moment a persona finishes (P0-1). The dataset
  order is recovered from `Persona_Index`, so a crash while other personas are
  still in flight keeps everything that already finished.
* `sessions.jsonl` stays the fine-grained log: one row per *finished* session.
* `errors.jsonl` holds the personas that failed; their finished sessions are
  still scored, because `run_scoring.py` now merges `results.jsonl` with
  `sessions.jsonl` instead of only falling back when results is empty (P0-2).
  `metrics.json` records `Personas_Scored`, `Personas_Expected`,
  `Persona_Ids_Missing`, `Personas_Rebuilt_From_Sessions`,
  `Personas_Merged_From_Sessions` and `Answer_Error_Count` so a shrunken
  sample is visible rather than silent.
* `--resume` rewrites a persona record with only the sessions that invocation
  replayed, so the already-finished earlier sessions (and their answered
  questions) are merged back in from `sessions.jsonl`, deduplicated by
  `Session_ID`. Without that merge the 2026-09-20 one-hour run would have been
  scored over 44 of its 46 questions.
* `--resume` continues an interrupted run in the same `--output-dir`: the
  sessions already present in `sessions.jsonl` are skipped and their memory
  store is kept (`--keep-memory` is implied).

```powershell
python Experiment\run_experiment.py --persona-workers 4 --resume --output-dir Experiment\runs\<run>
```

### Point 27: random persona shards (one host now, several hosts later)

The full benchmark is 30 personas / 1,579 sessions, which is ~172 h of worker
time on 2026-09-20 hardware — too long for one machine. Point 27 therefore runs
*random* shards of 5 personas and merges them:

```powershell
# 1. build the plan (6 shards x 5 personas, fixed seed) and print the commands
python Experiment\tools\shard_plan.py --only 1

# 2. run that shard: one terminal, 4 persona workers, Bailian for the GLM roles
python -u Experiment\run_experiment.py --config Experiment\configs\eval_large_bailian.yaml --persona-indices 0,4,18,27,28 --persona-workers 4 --answer-workers 4 --extraction-workers 2 --entity-judge-workers 1 --output-dir Experiment\runs\shard_1

# 3. score that shard, then refresh the merged tables after every finished shard
python Experiment\run_scoring.py --run-dir Experiment\runs\shard_1 --white-box-k 2,3,5
python Experiment\tools\merge_shards.py --allow-partial --out Experiment\runs\merged --shards Experiment\runs\shard_1
python Experiment\run_scoring.py --run-dir Experiment\runs\merged --white-box-k 2,3,5
```

What makes this work:

* `--persona-indices` selects arbitrary dataset indices (`0,4,18,27,28` or
  `0-4,10`) and cannot be combined with the contiguous
  `--start-index/--end-index/--persona-limit` window. Personas replay in
  dataset order; `run_meta.json` records `Persona_Indices` and every persona
  record carries `Dataset_Index`, so a shard is self-describing.
* `tools/shard_plan.py` shuffles with a fixed seed (default 27), cuts
  `shards x per-shard` personas, prints a balance table plus ready-to-run
  commands, and writes `Experiment/shards/shard_plan.json` — the record of which
  shard owns which persona. The personas are near-uniform (51–54 sessions,
  107–144 questions), so the six shards land within ±5% of each other and each
  shard is an unbiased sample, which is what makes the incremental tables
  meaningful.
* `merge_shards.py --allow-partial` merges whatever shards have finished and
  records `Personas_Expected_From_Dataset`, `Partial_Merge`, `Shards_Merged` and
  `Shard_Configs`; `run_scoring.py` copies the coverage fields into
  `metrics.json` next to `Judge_Model`, so a partial table cannot be mistaken
  for a full one. Merging shards produced by different configs or models prints
  `[warn] ... not a single-model result`.
* Shards share no state: each has its own `--output-dir`, memory store and
  namespace. Running the remaining shards on other hosts needs the same code,
  the same `.env` and the same `shard_plan.json`, then copying the three JSONL
  files (`results.jsonl`, `sessions.jsonl`, `errors.jsonl`) back before
  merging. Do not mix model configs in one merged table: the one-hour
  `runs/scale1h` window (personas 11–14, OpenRouter) is a calibration run, not
  one of the six shards.
* Two configs are available for the shards:
  `configs/eval_large_bailian.yaml` keeps the control roles and embeddings on
  the campus Ollama (after point 33 no role comes from Bailian at all: memory
  construction is `openai/gpt-4o-mini`, answering and judging are
  `openai/gpt-5-mini`, all on OpenRouter, exactly as in `eval_large.yaml`), while
  `configs/eval_large_bailian_all.yaml` moves every role the campus Ollama
  would otherwise serve to Bailian, so a host with no campus network can run a
  shard (it needs `DASHSCOPE_*` plus `OPENROUTER_*` for the builder, the answerer
  and the judge). The two are different systems: swapping the
  embedding model changes the retrieval rankings, so all shards of one merged
  table must use the same config. `run_experiment.py --resume` refuses to
  continue a store when `run_meta.json` records different models (override with
  `--allow-model-change`), because a store built by one embedding model cannot
  be continued by another.
* `tools/run_shards_chain.py` is the unattended watchdog: it waits for the shard
  that is currently running (an in-flight `--resume` is recognised by
  `sessions.jsonl` being newer than `run_meta.json`), then runs the remaining
  shards one by one, retries a shard whose `run_meta.json` lists
  `Failed_Personas` with `--resume` (bounded), scores every finished shard and
  refreshes the merged tables. Everything is echoed to
  `runs/<shard>.chain.log`, state lives in `shards/chain_state.json`, and a lock
  file stops two chains from writing one store.

  ```powershell
  python Experiment\tools\run_shards_chain.py --dry-run      # print the plan
  python Experiment\tools\run_shards_chain.py                # wait for shard_1, then 2..6
  python Experiment\tools\run_shards_chain.py --shards shard_1,shard_2 --persona-workers 4
  ```

  When several shards run at once (three terminals, or `--parallel`), give each
  shard its own GPU container so the Ollama queues stay separate (one container
  can also be shared by two shards, but then the two queues compete again):

  ```powershell
  python Experiment\tools\run_shards_chain.py --parallel 4 --persona-workers 1 `
    --units-map "shard_1=http://172.26.94.12:41135;shard_2=http://172.26.94.12:41136;shard_3=http://172.26.94.12:41133;shard_4=http://172.26.94.12:41134"
  ```

  `--units-map` pins one container per shard (shards without an entry fall back
  to `--ollama-units`, round robin); the merged tables are refreshed one shard at
  a time under a lock, and `run_meta.json` records the container each shard used.

### Full-scale sharding

The full benchmark is 30 personas / 1,579 sessions / 3,750 questions, which is
far too long for one run and too much work to leave in a single point of
failure. Split it by persona range, one shard per machine or GPU group:

```powershell
python Experiment\run_experiment.py --start-index 0  --end-index 8  --persona-workers 4 --output-dir Experiment\runs\shard_0
python Experiment\run_experiment.py --start-index 8  --end-index 16 --persona-workers 4 --output-dir Experiment\runs\shard_1
python Experiment\run_experiment.py --start-index 16 --end-index 24 --persona-workers 4 --output-dir Experiment\runs\shard_2
python Experiment\run_experiment.py --start-index 24                  --persona-workers 4 --output-dir Experiment\runs\shard_3

python Experiment\tools\merge_shards.py --out Experiment\runs\full_merged `
  --shards Experiment\runs\shard_0 Experiment\runs\shard_1 Experiment\runs\shard_2 Experiment\runs\shard_3
python Experiment\run_scoring.py --run-dir Experiment\runs\full_merged
```

`merge_shards.py` concatenates `results.jsonl` / `sessions.jsonl` /
`errors.jsonl`, restores dataset order, keeps the per-shard wall clocks and unit
assignments in the merged `run_meta.json`, and fails loudly if a persona appears
in two shards (overlapping ranges) or in none. Sharding across hosts needs no
shared state: each shard has its own store directory and its own API history.

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
These are the pre-point-33 roles, which is what the archived artefacts were
produced with; the current `eval_large.yaml` builds memory with
`openai/gpt-4o-mini` and both answers and judges with `openai/gpt-5-mini`. The
timings and costs below still hold as a shape, not as the cost of the current
role split.

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

One judging pass produces all three benchmark tables. `run_scoring.py` writes
`table3.md`, `table5.md`, `table6.md` and `tables.md`; `metrics.json` keeps the
numbers plus the white-box breakdown by Top-K. The full design (what each cell
is, which experiment to run, and the caveats) is in
`docs/table5_table6_design.md`.

Table 3 — conflict-aware evaluation (the row EMIR² reports):

| Method | Dynamic AA↑ | Dynamic SEH@3↑ | Static AA↑ | Static SEH@3↑ | Conditional AA↑ | Conditional SEH@3↑ | Average AA↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |

Table 5 — black-box performance (dynamic AA / UOCS, static AA / CRS, conditional AA):

| Method | Dynamic AA↑ | Dynamic UOCS↑ | Static AA↑ | Static CRS↑ | Conditional AA↑ | Average AA↑ |
| --- | --- | --- | --- | --- | --- | --- |

Table 6 — white-box retrieval and ranking:

| Method | Dynamic SEH@3↑ | Dynamic SRS↑ | Static SEH@3↑ | Static SRS↑ | Conditional SEH@3↑ | Conditional SRS↑ | Average SEH@3↑ | Average SRS↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

* **AA** — mean judged answer accuracy. Dynamic and static answers take 0 / 0.5
  / 1; conditional answers are all-or-nothing.
* **SEH@K** — share of questions whose supporting memory is ranked in the top K
  by the judge. `SEH@2 / SEH@3 / SEH@5` are all derived from one Top-5 ranking.
* **SRS** — `1 / log2(rank + 1)` inside the same window; 0 when the supporting
  memory is outside it.
* **Average AA** — mean of the three conflict-type AA columns.
* **UOCS / CRS** — the dynamic / static conflict-handling diagnostics. They are
  binary per question, come from the same judge call, and are `–` (undefined)
  for conditional questions.

The judge window must cover the largest white-box window: `--white-box-k 2,3,5`
raises `--judge-top-k` to 5 automatically, and the run has to have stored that
many memories per question (`run_experiment.py --stored-top-k 5`). A mismatch is
reported in `Questions_With_Short_Memory_Window` instead of being hidden.

## Tests

```powershell
python -m unittest discover -s Experiment\tests -t .
```

The tests never call an LLM API. The memory system and chat clients are stubbed,
and the dataset tests read `MemConflict/Data/Step4_4.jsonl` as data. They cover
the turn-ordering trap, per-session QA ordering, prompt routing, judge JSON
tolerance and the metric aggregation.
