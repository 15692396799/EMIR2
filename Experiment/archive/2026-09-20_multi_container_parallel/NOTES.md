# Multi-container + persona parallelism (points 16/17) — first measured A/B

Two runs on 2026-09-20, same config (`Experiment/configs/eval_large.yaml`), same
workload (first session of personas 2-5, i.e. `--start-index 1 --persona-limit 4
--max-sessions 1`), same single Ollama container (`ollama021-1` on `device6`,
`http://172.26.94.12:41135`). The only difference is the worker count.

| Arm | Command difference | Workers | Persona wall clock | Run wall clock |
| --- | --- | ---: | --- | ---: |
| Serial | `--persona-workers 1` | 1 | 815.7 / 465.8 / 557.9 / 600.4 s | **2,439.9 s** (40.7 min) |
| Parallel | `--persona-workers 4` | 4 | 411.5 / 331.8 / 347.9 / 589.2 s | **589.7 s** (9.8 min) |

* **4.14x wall-clock speedup** on four personas, even though all four share one
  GPU container. Every persona in the parallel arm also finished *faster than it
  did serially* (411 vs 816 s, 332 vs 466 s, 348 vs 558 s, 589 vs 600 s), so the
  concurrency is not simply trading per-persona latency for throughput.
* The per-persona `Started_At_s` / `Finished_At_s` in `results.jsonl` show the
  overlap directly: the parallel arm starts all four within 0.05 s, the serial
  arm starts each one where the previous ended.
* Sessions answered no questions here: the questions of these personas start at
  session id 3-6, so `--max-sessions 1` exercises ingestion only.

## Reference point: the 2026-09-19 run

`Experiment/archive/2026-09-19_smoke_eval_large/` recorded 1 persona x 4 sessions
= 1,155.3 s (**289 s per session**) serially, same config and same container.
Today a serial persona-session costs 610 s on average, i.e. the machine/API state
is ~2.1x slower than it was on 09-19, so the two dates are not directly
comparable. What is comparable is the *same-day* pair above, and the rate:

| Metric | 09-19 serial | 09-20 serial | 09-20 parallel (4 workers) |
| --- | ---: | ---: | ---: |
| Seconds per persona-session | 289 | 610 | **147** |

## How to reproduce

```powershell
python Experiment\run_experiment.py --config Experiment\configs\eval_large.yaml `
  --start-index 1 --persona-limit 4 --max-sessions 1 --persona-workers 1 `
  --ollama-units http://172.26.94.12:41135
python Experiment\run_experiment.py --config Experiment\configs\eval_large.yaml `
  --start-index 1 --persona-limit 4 --max-sessions 1 --persona-workers 4 `
  --ollama-units http://172.26.94.12:41135
```

Run directories: `Experiment/runs/ab_serial_0920_105617/` and
`Experiment/runs/ab_parallel_0920_104623/` (git-ignored; `run_meta.json` is
archived next to this file).

## Failure found on the way (fixed)

The first parallel attempt died after six minutes with
`BrokenProcessPool: A process in the process pool was terminated abruptly`,
whose real cause was hidden:

```
TypeError: V4StageError.__init__() missing 2 required positional arguments:
'attempts' and 'cause'
```

`V4StageError` (raised when a V4 stage exhausts its retries, here: the memory
builder returned invalid output twice) cannot be pickled: Python rebuilds
exceptions as `Cls(*self.args)`, and this class needs more than the message.
Two consequences, both fixed in `memconflict_eval/parallel.py` and
`run_experiment.py`:

1. every worker exception is converted to a picklable `parallel.JobError` that
   carries the original message and traceback;
2. `run_experiment.py` runs with `raise_errors=False`, so one bad persona is
   written to `errors.jsonl` (and reported, with a non-zero exit code) while the
   remaining personas keep their results — a 4-way run must not lose three
   personas of work because the fourth hit one invalid completion.

`Retrival-Mem` itself is untouched: the builder keeps its own
`validation_retries=1`, so a flaky completion still fails that persona.
