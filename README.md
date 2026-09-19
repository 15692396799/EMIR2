# EMIR2

Workspace for evaluating the **Retrival-Mem (AutoRetri V4)** long-term memory
system on the **MemConflict** benchmark.

## Layout

| Path | Contents |
| --- | --- |
| `Retrival-Mem/` | Memory system under test (V4 backend, multi-round retrieval). |
| `MemConflict/` | Conflict benchmark, released data, and the evaluation harness. |
| `MemConflict/Evaluation/eval_retrival_mem.py` | Retrival-Mem runner: replays the session chain, ingestion + retrieval + answering. |
| `MemConflict/Evaluation/scoring_retrival_mem.py` | Scoring entry point (delegates to `eval_scoring.py`). |
| `doc_by_human/` | Task notes. |

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
