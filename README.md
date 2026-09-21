# EMIR2

Workspace for evaluating the **Retrival-Mem (AutoRetri V4)** long-term memory
system on the **MemConflict** benchmark.

## Layout

| Path | Contents |
| --- | --- |
| `Retrival-Mem/` | Memory system under test (V4 backend, multi-round retrieval). |
| `MemConflict/` | Conflict benchmark, released data, and the evaluation harness. |
| `Experiment/` | EMIR² conflict-test experiment. Owns the runner, the MemConflict prompts, the judge and the Table 3 metrics. |
| `doc_by_human/` | Task notes and the `Step4_4.jsonl` data-structure analysis. |
| `doc_by_human/Step4_4_record1_analysis.md` | Field-by-field breakdown of the first record of the benchmark data. |
| `run_scale1h.bat` | Windows entry point for the one-hour scale test: preflight + parallel run + scoring in one command. |

`Retrival-Mem/` and `MemConflict/` are kept byte-identical to their upstream
commits and are treated as read-only dependencies: all experiment-specific code
lives in `Experiment/`. See [Experiment/README.md](Experiment/README.md) for the
LoCoMo-to-MemConflict adaptation and the run commands.

## Upstream sources

Both codebases are vendored here as plain directories so that this workspace can
be tracked as a single repository. The original clones' git metadata has been
moved aside into `.upstream_git/` (git-ignored) and is fully recoverable.

| Directory | Upstream | Branch | Commit |
| --- | --- | --- | --- |
| `Retrival-Mem/` | `http://10.245.4.83:3000/kiana/Retrival-Mem.git` | `mem_v4_end_clean` | `3916306ef8d1dd645d9cf211fc647b30b700137b` |
| `MemConflict/` | `https://github.com/TaoZhen1110/MemConflict.git` | `main` | `ec51d5d36e87f7665d1337f3a88cbde95fc2a964` |

To restore an original clone in place:

```powershell
Move-Item E:\code\github\EMIR2\.upstream_git\Retrival-Mem.git E:\code\github\EMIR2\Retrival-Mem\.git
Move-Item E:\code\github\EMIR2\.upstream_git\MemConflict.git   E:\code\github\EMIR2\MemConflict\.git
```

## Environment

Python 3.11 with `faiss-cpu`, `nltk`, `tiktoken`, `openai`, `python-dotenv`,
`tenacity`, `jsonlines`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`.

Credentials are read from `.env` files that are not committed:

* `Retrival-Mem/.env` — `DASHSCOPE_API_KEY`, `OPENROUTER_API_KEY`, and the
  Ollama endpoints (`OLLAMA_CHAT_ENDPOINT`, `OLLAMA_EMBED_ENDPOINT`,
  `OLLAMA_LEGACY_EMBEDDINGS_ENDPOINT`). Template: `Retrival-Mem/.env.example`.
* `MemConflict/.env` — `OPENAI_API_KEY` for answer generation and judging.
  Template: `MemConflict/.env.example`.

The Ollama service must serve the embedding model `qwen3-embedding` and the
controller / window-planner model `qwen3.5:latest`.

## Host environment check

The run happens on this Windows workstation (`E:\code\github\EMIR2`), not on the
GPU server, and it needs real outbound network access to three places: the
Ollama containers on the GPU server, OpenRouter through the local proxy, and
the answer / judge endpoints. A sandboxed Codex shell installs an offline
firewall (`codex_sandbox_offline_block_outbound`) that breaks every one of them
while `curl` still seems to work, so start the run from a plain PowerShell or
`codex -s danger-full-access`.

Check the environment in this order before a long run:

```powershell
# 1. Python 3.11 with the required packages
python -c "import sys, faiss, nltk, tiktoken, openai, dotenv, tenacity, jsonlines, numpy, pandas, matplotlib, seaborn, tqdm; print(sys.version)"

# 2. Credentials: both env files exist (Experiment/.env is the one loaded first)
Test-Path Experiment\.env, Retrival-Mem\.env

# 3. Proxy: the local proxy must be up and Python must see it
$env:NO_PROXY   # must print nothing
Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" | Select-Object ProxyEnable, ProxyServer, ProxyOverride
# expect ProxyEnable 1, ProxyServer 127.0.0.1:7897, and 172.26.* in ProxyOverride
python -c "import urllib.request; print(urllib.request.getproxies())"
# expect http/https -> http://127.0.0.1:7897 plus a 'no' entry for 172.26.*.
# An empty {} means Python cannot read the system proxy - typical inside the
# sandbox, and fatal here: OpenRouter would then be dialled directly and
# geo-blocked.

# 4. Ollama containers are GPU-backed (one line per URL in OLLAMA_BASE_URLS)
python Experiment\tools\check_ollama_units.py
python Experiment\tools\check_ollama_units.py --units http://172.26.94.12:41135

# 5. Every model role in the config really answers
python Experiment\tools\check_channels.py --config Experiment\configs\eval_large.yaml --roles memory_builder,answer_model,embedding
python Experiment\tools\check_channels.py --config Experiment\configs\eval_large.yaml --roles judge_model
```

How to read the results:

| Check | Passing | Failing |
| --- | --- | --- |
| `check_ollama_units.py` | `[ok] ... vram=NN GB NN tok/s` | `[warn] ... size_vram=0` — the container lost its GPU device and is answering at CPU speed; rebuild it on the host |
| `check_channels.py` runner roles | one `[ok]` line per role, a second or two each | `ProxyError` / `SSLError` / `ConnectionError` — the system proxy is down or unreachable, or `NO_PROXY` sneaked back into the shell; the run would otherwise fail inside `errors.jsonl` hours later |
| `check_channels.py --roles judge_model` | `[ok] judge_model openrouter openai/gpt-5-mini` | `403 ... not available in your region` — OpenRouter's OpenAI endpoint is geo-filtered; score with `--config Experiment\configs\eval_large_dsjudge.yaml` and say so when reporting the numbers |

`nvidia-smi` is not a valid container check: Ollama unloads a model after five
idle minutes, so an empty GPU proves nothing. `check_ollama_units.py` sends one
small request per container and then reads `/api/ps`, where `size_vram > 0` is
the actual evidence. `run_scale1h.bat` performs checks 4 and 5 itself and stops
before the run when either fails.

## Running the conflict evaluation

```powershell
cd E:\code\github\EMIR2\MemConflict\Evaluation

# Smoke test on one persona.
python eval_retrival_mem.py --start_idx 0 --end_idx 1 --top_k 3
python scoring_retrival_mem.py --input_file Results\retrival_mem_results.jsonl

# Full benchmark (30 personas, 122 questions).
python eval_retrival_mem.py --top_k 3
python scoring_retrival_mem.py --input_file Results\retrival_mem_results.jsonl
```

`eval_retrival_mem.py` ingests each session into a per-persona, isolated V4
store under `Results/Memory/`, then answers that session's questions from the
retrieved memory context. Scoring reports answer accuracy (AA), the dynamic
`UOCS` and static `CRS` diagnostics, and the white-box `SEH@2/3/5` / `SRS`
metrics.

## One-hour scale test: parallel personas + parallel answering

`run_scale1h.bat` (repo root) is the repeatable "how long and how expensive is
the full benchmark" run. It replays four personas side by side
(`--persona-workers 4`), answers each session's questions in parallel
(`--answer-workers 4`), keeps everything on the single GPU container
`ollama021-1`, and then runs one judge pass that writes Tables 3, 5 and 6 in the
same invocation.

```powershell
cd E:\code\github\EMIR2
run_scale1h.bat              # preflight + one-hour run + scoring -> Experiment\runs\scale1h
run_scale1h.bat preflight    # check containers and channels only, run nothing
run_scale1h.bat dryrun       # print the exact commands, run nothing
run_scale1h.bat quick        # 1 persona x 3 sessions smoke instead of the hour
run_scale1h.bat resume       # continue an interrupted run in the same folder
run_scale1h.bat noscore      # run only, no judge
run_scale1h.bat dsjudge      # run, then score with DeepSeek direct
```

While it runs, the runner keeps one progress line up to date on stderr, so
nobody has to tail `sessions.jsonl`:

```text
[progress] [#####---------------] 12/44 sessions 27% | 21m30s elapsed | eta 58m10s | questions 9/46 | errors 0 | personas 2/4 | failed 0 | running 2/4 | last 8cb3c9c6 s3
```

On a terminal that line is rewritten in place (and sheds its least important
fields when the window is narrow); redirected to a log it is reprinted every
60 s. `--no-progress`, accepted by both `run_experiment.py` and
`run_scoring.py`, turns it off — the JSONL logs are written either way.

Workload (the file's defaults): personas 12-15 (`--start-index 11
--persona-limit 4`) with the first 11 sessions of each — 44 persona-sessions,
14 of which carry questions, 46 questions answered. Those four personas hold
214 sessions / 499 questions in total, which is the sample the test is drawn
from. The equivalent explicit commands:

```powershell
python -u Experiment\run_experiment.py `
  --config Experiment\configs\eval_large.yaml `
  --start-index 11 --persona-limit 4 --max-sessions 11 `
  --persona-workers 4 --answer-workers 4 `
  --extraction-workers 2 --entity-judge-workers 1 `
  --ollama-units http://172.26.94.12:41135 `
  --output-dir Experiment\runs\scale1h
python -u Experiment\run_scoring.py --run-dir Experiment\runs\scale1h
```

`Experiment\runs\scale1h\` then holds:

| File | What it tells you |
| --- | --- |
| `run_meta.json` | `Wall_Clock_s`, `Persona_Workers`, `Answer_Workers`, `Extraction_Workers`, `Ollama_Units`, per-persona `Persona_Unit_Assignments`, `Failed_Personas` |
| `results.jsonl` / `sessions.jsonl` | per-persona answers plus per-session timings; each persona's `Started_At_s` / `Finished_At_s` is what shows the overlap |
| `errors.jsonl` | personas that failed. The run exits non-zero but the other personas keep their results |
| `table3.md`, `table5.md`, `table6.md`, `tables.md` | the three tables from that single judge pass |
| `metrics.json` / `scores.jsonl` | the numbers behind the tables, the white-box windows, and the coverage counters (`Personas_Scored`, `Answer_Error_Count`) |

Turn the run into a full-benchmark estimate from the measured rate, not from a
single number: seconds per persona-session is
`Wall_Clock_s × Persona_Workers / 44`, and the full benchmark is 1,579
persona-sessions (30 personas). Cost comes out of the per-persona API logs in
`Memory\retrival_mem_v4_<persona>_v1\v4_*.jsonl` (input/output tokens and
dollars per stage); the 2026-09-19 reference was $0.088 per session, ~$146 for
the full run.

Notes before believing the numbers:

* **Parallelism must not change answers.** Compare `Model_Answer` and
  `Retrieved_Memories` per question against a `--persona-workers 1
  --answer-workers 1` arm on the same workload. The measured A/B on one
  container is 2,439.9 s serially against 589.7 s with four workers (4.14x);
  the full write-up is in
  [Experiment/README.md](Experiment/README.md#serial-baseline-and-parallel-speed-up-measured).
* **The OpenRouter lane is the fragile part, not the GPU.** A 2026-09-20
  one-hour attempt lost all four personas to `ProxyError` / `SSLError` after
  334 s. Keep the proxy up for the whole run, leave `NO_PROXY` unset (the .bat
  clears it defensively), and continue with `run_scale1h.bat resume` — finished
  sessions are kept and skipped.
* **A persona can also die on a stale V4 build cache.** The second 2026-09-20
  attempt finished three personas and stopped the fourth with
  `SemanticValidationError: reinforce references unknown fact key`, raised while
  replaying a cached `semantic_update` output against a newer scope revision.
  `memconflict_eval/memory.py` now drops those entries before each session
  ingest and retries the session once; the retry shows up as
  `Ingest.Retried_After_Validation_Error` in `sessions.jsonl`. Set
  `MEMCONFLICT_CHECKPOINT_GUARD=0` to run unguarded. The upstream design flaws
  behind the crash (window plans cached by turn count, a checkpoint loader that
  ignores `scope_revision`, and a replay path that raises instead of
  recomputing) are reproduced offline in seconds by
  `python Experiment\tools\repro_stale_checkpoint_bug.py`.
* **Concurrency has to be capped.** Four persona workers at the config's 8
  extraction threads would put 32 calls in flight at once against the same
  endpoints, which is what produced the builder validation failures in the
  first parallel attempt; hence `--extraction-workers 2 --entity-judge-workers 1`.

## Full benchmark: random persona shards

The full run is 30 personas / 1,579 sessions / 3,750 questions, which measured
~172 h of worker time on 2026-09-20 hardware — ~43 h with `--persona-workers 4`
on one machine, or ~11 h spread over four hosts. Run it as random shards of five
personas and merge the results later:

```powershell
# plan: 6 shards x 5 personas, fixed seed, prints per-shard commands
python Experiment\tools\shard_plan.py --only 1

# run one shard (the GLM roles go through Bailian: no OpenRouter fee on them)
python -u Experiment\run_experiment.py --config Experiment\configs\eval_large_bailian.yaml --persona-indices 0,4,18,27,28 --persona-workers 4 --answer-workers 4 --extraction-workers 2 --entity-judge-workers 1 --output-dir Experiment\runs\shard_1
python Experiment\run_scoring.py --run-dir Experiment\runs\shard_1 --white-box-k 2,3,5

# refresh the merged tables after every finished shard (partial merges allowed)
python Experiment\tools\merge_shards.py --allow-partial --out Experiment\runs\merged --shards Experiment\runs\shard_1
python Experiment\run_scoring.py --run-dir Experiment\runs\merged --white-box-k 2,3,5
```

Shards share no state, so the remaining ones can run on other hosts and their
`results.jsonl` / `sessions.jsonl` / `errors.jsonl` are merged here afterwards.
`merge_shards.py` warns when the shards came from different model configs, and
`metrics.json` records `Partial_Merge`, `Personas_Expected_From_Dataset` and
`Judge_Model` so an incremental table cannot be mistaken for a full one.

To run the shards back to back unattended, use the watchdog (it waits for the
shard that is already running, retries failures with `--resume`, and refreshes
the tables after every shard):

```powershell
python Experiment\tools\run_shards_chain.py            # wait for shard_1, then 2..6
python Experiment\tools\run_shards_chain.py --dry-run  # print the commands only
```
