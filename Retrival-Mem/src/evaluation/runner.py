from __future__ import annotations

import copy
import json
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fall back to periodic prints
    tqdm = None

from api_history import ApiHistoryLogger, LoggedBatchChatClient, LoggedChatClient
from logging_utils import tee_output
from run_artifacts import snapshot_run_configs, update_manifest

try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except (
    ImportError
):  # pragma: no cover - exercised only when dependencies are not installed.
    nltk = None
    SmoothingFunction = None
    sentence_bleu = None

from agent.agent import LightweightMemoryAgent
from agent.answer_context import prepare_answer_context
from agent.evidence_answer import unpack_answer_output
from memory.clients import (
    ChatBatchRequest,
    ChatBatchResult,
    ChatClient,
    make_batch_chat_client,
    make_chat_client,
)
from memory.config import AppConfig, config_to_dict, configure_backend_output_paths
from memory import MemorySystem
from memory.base import RetrievalResult
from memory.results import EpisodicMemoryItem, SemanticMemoryItem
from memory.structured_output import (
    messages_with_json_schema,
    parse_json_object_with_schema,
)
from memory.v4.storage import V4SQLiteStore
from evaluation.checkpoints import QuestionStageLedger
from evaluation.prompts import (
    LOCOMO_JUDGE_JSON_SCHEMA,
    PROACTIVE_PRECISION_JSON_SCHEMA,
    PROACTIVE_RECALL_JSON_SCHEMA,
    build_locomo_judge_messages,
    build_proactive_precision_messages,
    build_proactive_recall_messages,
)
from evaluation.benchmarks import get_benchmark_evaluator


LOCOMO_PARALLEL_OPTION_KEYS = {"answer_workers", "answer_chunk_size"}

# These are the numeric IDs in the released locomo10.json. They are not the
# paper's prose enumeration; the official evaluator orders them as 4/1/2/3/5
# when presenting Single-hop/Multi-hop/Temporal/Open-domain/Adversarial.
LOCOMO_CATEGORY_NAMES = {
    "1": "Multi-hop",
    "2": "Temporal",
    "3": "Open-ended",
    "4": "Single-hop",
}

MODEL_ROLE_LABELS = {
    "embedding": "Embedding",
    "slm": "SLM",
    "answer_model": "Answer model",
    "judge_model": "LLM judge",
    "memory_builder": "Memory builder",
    "controller": "Controller",
    "decomposition_gate": "Decomposition gate",
    "window_planner": "Window planner",
    "adjudication_model": "Adjudication model",
    "entity_judge": "Entity judge",
}


@dataclass(frozen=True)
class LocomoAnswerJob:
    example_index: int
    qa_index: int
    namespace: str
    question: str
    reference: str
    category: str
    gold_evidence: tuple[str, ...] = ()


@dataclass
class LocomoChunkResult:
    results: list[tuple[dict[str, Any], dict[str, Any]]]
    errors: list[Exception]


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", str(text).lower()))


def mem0_simple_tokenize(text: str) -> list[str]:
    return (
        str(text)
        .lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
        .split()
    )


def mem0_calculate_bleu_scores(prediction: str, reference: str) -> dict[str, float]:
    _ensure_nltk_tokenizer()
    pred_tokens = nltk.word_tokenize(str(prediction).lower())
    ref_tokens = [nltk.word_tokenize(str(reference).lower())]

    weights_list = [
        (1, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (0.33, 0.33, 0.33, 0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    smooth = SmoothingFunction().method1

    scores: dict[str, float] = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(
                ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth
            )
        except Exception as exc:
            print(f"Error calculating BLEU score: {exc}")
            score = 0.0
        scores[f"bleu{n}"] = score
    return scores


def mem0_calculate_metrics(prediction: str, reference: str) -> dict[str, float]:
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
        }

    prediction = str(prediction).strip()
    reference = str(reference).strip()
    exact_match = int(prediction.lower() == reference.lower())
    pred_tokens = set(mem0_simple_tokenize(prediction))
    ref_tokens = set(mem0_simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    bleu_scores = mem0_calculate_bleu_scores(prediction, reference)
    return {
        "exact_match": exact_match,
        "f1": f1,
        **bleu_scores,
    }


def _ensure_nltk_tokenizer() -> None:
    if nltk is None or SmoothingFunction is None or sentence_bleu is None:
        raise RuntimeError(
            "Mem0-aligned LoCoMo BLEU scoring requires the `nltk` package."
        )
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


def concept_recall(candidates: list[str], retrieved_text: str) -> float:
    if not candidates:
        return 0.0
    retrieved = normalize_text(retrieved_text)
    hits = sum(1 for candidate in candidates if normalize_text(candidate) in retrieved)
    return hits / len(candidates)


def concept_precision(candidates: list[str], retrieved_units: list[str]) -> float:
    if not retrieved_units:
        return 0.0
    candidate_norm = [normalize_text(candidate) for candidate in candidates]
    hits = 0
    for unit in retrieved_units:
        norm_unit = normalize_text(unit)
        if any(candidate and candidate in norm_unit for candidate in candidate_norm):
            hits += 1
    return hits / len(retrieved_units)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_benchmark_artifact_dir(path: str | Path, benchmark: str) -> Path:
    artifact_dir = Path(path)
    if (artifact_dir / "memory.sqlite3").exists() or (
        artifact_dir / "predictions.jsonl"
    ).exists():
        return artifact_dir
    nested = artifact_dir / benchmark
    if (nested / "memory.sqlite3").exists() or (nested / "predictions.jsonl").exists():
        return nested
    return artifact_dir


def _copy_sqlite_database(source_dir: Path, output_dir: Path) -> None:
    source_db = source_dir / "memory.sqlite3"
    if not source_db.exists():
        raise FileNotFoundError(f"Reusable memory database not found at {source_db}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        source = Path(f"{source_db}{suffix}")
        destination = Path(f"{output_dir / 'memory.sqlite3'}{suffix}")
        if source.exists():
            if source.resolve() == destination.resolve():
                continue
            shutil.copy2(source, destination)
    source_faiss = source_dir / "faiss"
    if source_faiss.exists():
        destination_faiss = output_dir / "faiss"
        destination_faiss.mkdir(parents=True, exist_ok=True)
        for source in source_faiss.iterdir():
            if source.is_file():
                destination = destination_faiss / source.name
                if source.resolve() == destination.resolve():
                    continue
                shutil.copy2(source, destination)


def _copy_prediction_artifacts(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (source_dir / "predictions.jsonl").exists():
        raise FileNotFoundError(
            f"Reusable predictions not found at {source_dir / 'predictions.jsonl'}"
        )
    for name in ("predictions.jsonl", "retrieval_traces.jsonl", "prediction_meta.json"):
        source = source_dir / name
        if source.exists():
            destination = output_dir / name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)


def _write_memory_meta(output_dir: Path, benchmark: str, data_path: Path, config: AppConfig) -> None:
    (output_dir / "memory_meta.json").write_text(
        json.dumps({"benchmark": benchmark, "data_path": str(data_path), "backend": "v4"}, indent=2),
        encoding="utf-8",
    )


def _write_prediction_meta(output_dir: Path, benchmark: str, data_path: Path, config: AppConfig) -> None:
    (output_dir / "prediction_meta.json").write_text(
        json.dumps({"benchmark": benchmark, "data_path": str(data_path)}, indent=2),
        encoding="utf-8",
    )


def _validate_rejudge_source(source_dir: Path, benchmark: str, data_path: Path) -> None:
    for name in ("predictions.jsonl", "retrieval_traces.jsonl"):
        if not (source_dir / name).is_file():
            raise FileNotFoundError(f"Prediction source is missing {source_dir / name}")


def _prepare_reused_memory(
    benchmark: str, data_path: Path, config: AppConfig,
    output_dir: Path, source_value: str | Path | None,
) -> None:
    if not source_value:
        return
    source_dir = _resolve_benchmark_artifact_dir(source_value, benchmark)
    V4SQLiteStore(str(source_dir / "memory.sqlite3"), read_only=True).close()
    _copy_sqlite_database(source_dir, output_dir)
    V4SQLiteStore(str(output_dir / "memory.sqlite3"), read_only=True).close()
    print(f"[{benchmark}] reused memory from {source_dir}", flush=True)


def _logged_chat_client(
    client: ChatClient,
    logger: ApiHistoryLogger | None,
    module: str,
    model_config,
    context: dict[str, Any] | None = None,
) -> ChatClient:
    if logger is None:
        return client
    return LoggedChatClient(
        client,
        logger,
        module,
        model_config.provider,
        model_config.model,
        context=context,
    )


def _validate_benchmark_parallel_options(config: AppConfig) -> None:
    if "ollama_units" in config.evaluation.locomo:
        from evaluation.ollama_units import resolve_units
        resolve_units(config.evaluation.locomo["ollama_units"])
    proactive_options = config.evaluation.proactive_membench or {}
    invalid_keys = sorted(LOCOMO_PARALLEL_OPTION_KEYS & set(proactive_options))
    if invalid_keys:
        joined = ", ".join(invalid_keys)
        raise ValueError(
            f"{joined} are only supported under evaluation.locomo; "
            "remove them from evaluation.proactive_membench"
        )


_STAGE_INVALIDATION_DEPENDENCIES = {
    "memory": {"memory", "retrieval", "answer", "temporal_correction", "scoring"},
    "retrieval": {"retrieval", "answer", "temporal_correction", "scoring"},
    "answer": {"answer", "temporal_correction", "scoring"},
    "temporal_correction": {"temporal_correction", "scoring"},
    "scoring": {"scoring"},
}


def _expanded_invalidated_stages(stages: Sequence[str] | None) -> set[str]:
    expanded: set[str] = set()
    for raw_stage in stages or ():
        stage = str(raw_stage).strip().lower()
        if stage not in _STAGE_INVALIDATION_DEPENDENCIES:
            raise ValueError(f"Unsupported invalidation stage: {raw_stage}")
        expanded.update(_STAGE_INVALIDATION_DEPENDENCIES[stage])
    return expanded


def _invalidate_benchmark_artifacts(benchmark_dir: Path, stages: set[str]) -> None:
    if not benchmark_dir.exists() or not stages:
        return
    if "memory" in stages:
        shutil.rmtree(benchmark_dir)
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        return
    ledger_path = benchmark_dir / "question_stages.sqlite3"
    if ledger_path.exists():
        with QuestionStageLedger(ledger_path) as ledger:
            ledger.invalidate_stages(
                frozenset(stages).intersection(
                    {
                        "retrieval",
                        "answer",
                        "temporal_correction",
                        "scoring",
                    }
                )
            )
    if "retrieval" in stages or "answer" in stages or "temporal_correction" in stages:
        names = {
            "predictions.jsonl",
            "retrieval_traces.jsonl",
            "scored_predictions.jsonl",
            "prediction_meta.json",
            "metrics.json",
            "rerank_checkpoints.sqlite3",
        }
    elif "scoring" in stages:
        names = {"scored_predictions.jsonl", "metrics.json"}
    else:
        names = set()
    if "scoring" in stages:
        names.update(
            {
                "judge_batch_input.jsonl",
                "judge_batch_output.jsonl",
                "judge_batch_errors.jsonl",
                "judge_batch_meta.json",
                "judge_batch_errors.json",
            }
        )
    if "retrieval" in stages or "answer" in stages:
        names.update({f"answer_batch_{suffix}" for suffix in
                      ("input.jsonl", "output.jsonl", "meta.json", "errors.json")})
    for name in names:
        path = benchmark_dir / name
        if path.exists():
            path.unlink()


def _prepare_forked_run(
    source_run_dir: str | Path,
    destination_run_dir: Path,
    benchmarks: Sequence[str],
    invalidated_stages: set[str],
) -> Path:
    source = Path(source_run_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Fork source run directory not found: {source}")
    if source.resolve() == destination_run_dir.resolve():
        raise ValueError("Fork source and destination must be different directories")
    for benchmark in benchmarks:
        source_benchmark = source / benchmark
        if not source_benchmark.is_dir():
            continue
        destination_benchmark = destination_run_dir / benchmark
        shutil.copytree(source_benchmark, destination_benchmark, dirs_exist_ok=True)
        _invalidate_benchmark_artifacts(destination_benchmark, invalidated_stages)
    return source.resolve()


def run_evaluation(
    config: AppConfig,
    selected_benchmarks: list[str] | None = None,
    resume_run_dir: str | Path | None = None,
    config_input_path: str | Path | None = None,
    fork_run_from: str | Path | None = None,
    invalidate_stages: Sequence[str] | None = None,
) -> Path:
    from datetime import datetime

    _validate_benchmark_parallel_options(config)
    benchmarks = selected_benchmarks or config.evaluation.benchmarks
    if resume_run_dir and fork_run_from:
        raise ValueError("resume_run_dir and fork_run_from are mutually exclusive")
    invalidated = _expanded_invalidated_stages(invalidate_stages)
    if resume_run_dir:
        run_dir = Path(resume_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory not found: {run_dir}")
        if not run_dir.is_dir():
            raise NotADirectoryError(f"Resume run path is not a directory: {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = Path(config.evaluation.output_dir) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        if fork_run_from:
            _prepare_forked_run(fork_run_from, run_dir, benchmarks, invalidated)
    snapshot_run_configs(
        run_dir,
        config_to_dict(config),
        config_path=config_input_path,
        input_config=config.raw or config_to_dict(config),
        resume=bool(resume_run_dir),
        run_kind="evaluation",
    )
    if resume_run_dir:
        for benchmark in benchmarks:
            _invalidate_benchmark_artifacts(run_dir / benchmark, invalidated)
    if fork_run_from:
        update_manifest(
            run_dir,
            forked_from=str(Path(fork_run_from).resolve()),
            invalidated_stages=sorted(invalidated),
        )
    elif resume_run_dir and invalidated:
        update_manifest(run_dir, invalidated_stages=sorted(invalidated))
    api_history_logger = ApiHistoryLogger(run_dir / "api_history")

    with tee_output(run_dir / "console.log"):
        all_metrics: dict[str, dict[str, Any]] = {}
        benchmark_runners = {
            "proactive_membench": run_proactive_membench,
            "locomo": run_locomo,
        }
        try:
            for benchmark in benchmarks:
                evaluator = get_benchmark_evaluator(benchmark)
                api_calls_before = _api_history_call_count(run_dir / "api_history")
                all_metrics[benchmark] = benchmark_runners[evaluator.name](
                    config,
                    run_dir / benchmark,
                    api_history_logger=api_history_logger,
                )
                api_calls_after = _api_history_call_count(run_dir / "api_history")
                all_metrics[benchmark]["api_call_count"] = (
                    api_calls_after - api_calls_before
                )
            for benchmark, benchmark_metrics in all_metrics.items():
                metrics_path = run_dir / benchmark / "metrics.json"
                if metrics_path.parent.exists():
                    metrics_path.write_text(
                        json.dumps(benchmark_metrics, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            write_markdown_report(run_dir / "evaluation_report.md", config, all_metrics)
        except Exception as error:
            from evaluation.batch import BatchPendingError
            update_manifest(
                run_dir,
                status="pending" if isinstance(error, BatchPendingError) else "partial"
                if _evaluation_has_prediction_progress(run_dir)
                else "failed",
                benchmarks=benchmarks,
                failure={"type": type(error).__name__, "message": str(error)},
            )
            raise
        update_manifest(
            run_dir, status="completed", benchmarks=benchmarks, failure=None
        )
        print(f"Wrote evaluation outputs to {run_dir}", flush=True)
    return run_dir


def _api_history_call_count(directory: Path) -> int:
    return sum(
        1
        for path in directory.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _evaluation_has_prediction_progress(run_dir: Path) -> bool:
    for path in run_dir.glob("*/predictions.jsonl"):
        if any(line.strip() for line in path.read_text(encoding="utf-8").splitlines()):
            return True
    return False




def _run_question_stage(
    ledger: QuestionStageLedger,
    question_key: str,
    stage: str,
    operation,
) -> dict[str, Any]:
    cached = ledger.load_success(question_key, stage)
    if cached is not None:
        return cached
    try:
        output = operation()
        if not isinstance(output, dict):
            raise TypeError(f"{stage} checkpoint operation must return an object")
    except Exception as error:
        ledger.record_failure(
            question_key,
            stage,
            error,
            attempts=max(1, int(getattr(error, "attempts", 1))),
        )
        raise
    ledger.record_success(question_key, stage, output)
    return output


def _bundle_from_checkpoint(retrieval: dict[str, Any]) -> RetrievalResult:
    episodic_memories: list[EpisodicMemoryItem] = []
    for raw_item in retrieval.get("episodic_memories", []):
        item = dict(raw_item)
        item["semantic_memories"] = [
            SemanticMemoryItem(**dict(raw_semantic))
            for raw_semantic in item.get("semantic_memories", [])
        ]
        episodic_memories.append(EpisodicMemoryItem(**item))
    return RetrievalResult(
        question=str(retrieval.get("question") or ""),
        namespace=str(retrieval.get("namespace") or ""),
        episodic_memories=episodic_memories,
        trace=[],
    )


def _answer_with_question_checkpoints(
    agent: LightweightMemoryAgent,
    memory_system: MemorySystem,
    config: AppConfig,
    ledger: QuestionStageLedger,
    *,
    benchmark: str,
    question_key: str,
    question: str,
    namespace: str,
    category: str | int | None = None,
    regeneration_token: str | None = None,
) -> dict[str, Any]:
    normalized_category = None if category is None else str(category).strip()
    frozen_retrieval = benchmark == "locomo" and config.evaluation.locomo.get("answer_only_frozen_retrieval", False)
    if regeneration_token:
        ledger.invalidate_question(question_key, keep_retrieval=frozen_retrieval)
    retrieval_prompt_profile = (
        "locomo_open_ended_v1"
        if benchmark == "locomo" and normalized_category == "3"
        else None
    )
    def retrieve() -> dict[str, Any]:
        bundle, memory_context = agent.retrieve_for_answer(
            question,
            namespace,
            metadata={
                "retrieval_prompt_profile": retrieval_prompt_profile,
                "rerank_checkpoint_path": str(
                    ledger.path.parent / "rerank_checkpoints.sqlite3"
                ),
                "rerank_checkpoint_key": question_key,
                "rerank_reset": bool(regeneration_token),
            },
        )
        return {
            "retrieval": bundle.to_dict(),
            "memory_context": memory_context,
        }

    if frozen_retrieval:
        # Explicit answer ablation: preserve the source retrieval version and never
        # silently run retrieval when a requested source checkpoint is unavailable.
        state = ledger.get_state(question_key, "retrieval")
        if not state or state.get("status") != "succeeded":
            raise ValueError(f"Frozen retrieval checkpoint missing for {question_key}")
        retrieval_output = state["output"]
        frozen = retrieval_output.get("retrieval", {})
        if frozen.get("question") != question or frozen.get("namespace") != namespace:
            raise ValueError(f"Frozen retrieval identity mismatch for {question_key}")
    else:
        retrieval_output = _run_question_stage(
            ledger, question_key, "retrieval", retrieve
        )
    retrieval = dict(retrieval_output.get("retrieval") or {})
    bundle = _bundle_from_checkpoint(retrieval)
    memory_context = str(retrieval_output.get("memory_context") or "")
    answer_options = (config.evaluation.locomo if benchmark == "locomo"
                      else config.evaluation.proactive_membench)
    # Frozen checkpoints retain their original context; only answer input changes.
    memory_context = prepare_answer_context(bundle, memory_context, answer_options)

    def answer() -> dict[str, Any]:
        category_kwargs = (
            {"category": normalized_category} if normalized_category is not None else {}
        )
        response, answer_failed = agent.generate_answer(
            question,
            namespace,
            bundle,
            memory_context,
            **category_kwargs,
        )
        response, output_fields = unpack_answer_output(response, answer_options)
        return {"response": response, "answer_failed": bool(answer_failed), **output_fields}

    answer_output = _run_question_stage(
        ledger, question_key, "answer", answer
    )
    raw_response = str(answer_output.get("response") or "")
    def correct() -> dict[str, Any]:
        response, correction = agent.correct_answer(raw_response, bundle, question)
        return {"response": response, "temporal_correction": correction}

    temporal_output = _run_question_stage(
        ledger,
        question_key,
        "temporal_correction",
        correct,
    )
    return {
        "answer": str(temporal_output.get("response") or ""),
        "retrieval": retrieval,
        "memory_context": memory_context,
        "temporal_correction": dict(temporal_output.get("temporal_correction") or {}),
        "answer_failed": bool(answer_output.get("answer_failed", False)),
        **{key: answer_output[key] for key in ("answer_output", "raw_answer_output")
           if key in answer_output},
    }


def _proactive_row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("domain") or ""), str(row.get("id"))


def _paired_proactive_rows(
    predictions: list[dict[str, Any]], traces: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_map = {_proactive_row_key(row): row for row in predictions}
    trace_map = {_proactive_row_key(row): row for row in traces}
    keys = sorted(set(prediction_map).intersection(trace_map))
    return [prediction_map[key] for key in keys], [trace_map[key] for key in keys]


def _score_proactive_with_checkpoint(
    ledger: QuestionStageLedger,
    row: dict[str, Any],
    judge_client: ChatClient | None,
) -> dict[str, Any]:
    score_fields = {
        "recall",
        "precision",
        "recall_judge",
        "precision_judge",
        "judge",
    }
    unscored = {key: value for key, value in row.items() if key not in score_fields}
    domain, question_id = _proactive_row_key(unscored)
    question_key = f"proactive_membench:{domain}:{question_id}"
    output = _run_question_stage(
        ledger,
        question_key,
        "scoring",
        lambda: {"row": _score_proactive_prediction(unscored, judge_client)},
    )
    return dict(output["row"])


def run_proactive_membench(
    config: AppConfig,
    output_dir: Path,
    api_history_logger: ApiHistoryLogger | None = None,
) -> dict[str, Any]:
    benchmark_root = Path(config.evaluation.benchmark_dir) / "proactive_membench"
    data_root = benchmark_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(
            f"ProactiveMemBench data not found at {data_root}. Run scripts/download_benchmarks.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    traces_path = output_dir / "retrieval_traces.jsonl"
    options = config.evaluation.proactive_membench
    regenerate_answers = bool(options.get("regenerate_answers", False))
    reuse_predictions_from = options.get("reuse_predictions_from")
    rejudge_from = options.get("rejudge_from")
    force_rejudge = bool(options.get("force_rejudge", False))
    if reuse_predictions_from and rejudge_from:
        raise ValueError(
            "locomo.reuse_predictions_from and rejudge_from are mutually exclusive"
        )
    regeneration_token = str(time.time_ns()) if regenerate_answers else None
    local_config = copy.deepcopy(config)
    _configure_output_memory(local_config, output_dir)

    use_llm_judge = bool(
        config.evaluation.proactive_membench.get("use_llm_judge", True)
    )
    if reuse_predictions_from and not regenerate_answers:
        source_dir = _resolve_benchmark_artifact_dir(
            reuse_predictions_from, "proactive_membench"
        )
        _copy_prediction_artifacts(source_dir, output_dir)
        predictions = load_jsonl(predictions_path)
        traces = load_jsonl(traces_path)
        if force_rejudge:
            judge_client = (
                _logged_chat_client(
                    make_chat_client(config.judge_model),
                    api_history_logger,
                    "proactive_judge",
                    config.judge_model,
                )
                if use_llm_judge
                else None
            )
            predictions = [
                _score_proactive_prediction(row, judge_client) for row in predictions
            ]
        metrics = _aggregate_proactive_metrics(predictions)
        write_jsonl(predictions_path, predictions)
        write_jsonl(traces_path, traces)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[proactive_membench] reused predictions from {source_dir}", flush=True)
        return metrics

    _prepare_reused_memory(
        "proactive_membench",
        data_root,
        config,
        output_dir,
        options.get("reuse_memory_from"),
    )
    memory_system = MemorySystem(local_config, api_history_logger=api_history_logger)
    ledger = QuestionStageLedger(output_dir / "question_stages.sqlite3")
    try:
        agent = LightweightMemoryAgent(
            local_config,
            memory_system=memory_system,
            api_history_logger=api_history_logger,
        )
        judge_client = (
            _logged_chat_client(
                make_chat_client(config.judge_model),
                api_history_logger,
                "proactive_judge",
                config.judge_model,
            )
            if use_llm_judge
            else None
        )
        if force_rejudge or rejudge_from:
            ledger.invalidate_stages({"scoring"})
        loaded_predictions = [] if regenerate_answers else load_jsonl(predictions_path)
        loaded_traces = [] if regenerate_answers else load_jsonl(traces_path)
        predictions, traces = _paired_proactive_rows(loaded_predictions, loaded_traces)
        predictions = [
            _score_proactive_with_checkpoint(ledger, row, judge_client)
            for row in predictions
        ]
        write_jsonl(predictions_path, predictions)
        write_jsonl(traces_path, traces)
        completed = {_proactive_row_key(row) for row in predictions}
        if completed:
            print(
                f"[proactive_membench] resuming from {len(completed)} completed questions",
                flush=True,
            )

        domains = config.evaluation.proactive_membench.get("domains") or [
            path.name for path in data_root.iterdir() if path.is_dir()
        ]
        max_questions = config.evaluation.proactive_membench.get("max_questions")
        for domain in domains:
            domain_root = data_root / domain
            if not domain_root.exists():
                continue
            conversations = load_json(domain_root / "step4_conversations.json")
            questions = load_json(domain_root / "step5_proactive_questions.json")
            namespace = f"proactive_membench:{domain}"
            if memory_system.is_namespace_ready(namespace):
                print(
                    f"[proactive_membench] domain {domain}: reuse existing memory",
                    flush=True,
                )
            else:
                memory_system.ingest_conversation(
                    namespace,
                    conversations,
                    {
                        "dataset": "proactive_membench",
                        "domain": domain,
                        "collection": domain,
                    },
                )
            selected_questions = questions[
                : max_questions if max_questions is not None else None
            ]
            for question_index, question in enumerate(selected_questions):
                question_id = question.get("id")
                row_key = (domain, str(question_id))
                if row_key in completed:
                    continue
                stable_id = question_id if question_id is not None else question_index
                question_key = f"proactive_membench:{domain}:{stable_id}"
                question_text = str(question["question"])
                answer = _answer_with_question_checkpoints(
                    agent,
                    memory_system,
                    local_config,
                    ledger,
                    benchmark="proactive_membench",
                    question_key=question_key,
                    question=question_text,
                    namespace=namespace,
                    regeneration_token=regeneration_token,
                )
                retrieval = answer["retrieval"]
                candidates = [
                    str(item.get("concept") or item.get("memory_unit") or "")
                    for item in question.get("candidate_set", [])
                ]
                record = {
                    "domain": domain,
                    "id": question_id,
                    "question": question_text,
                    "trigger_type": question.get("trigger_type"),
                    "difficulty": question.get("difficulty"),
                    "candidate_set": candidates,
                    "answer": answer["answer"],
                    "retrieved_units": _retrieved_unit_strings(retrieval),
                    "answer_failed": bool(answer.get("answer_failed", False)),
                    "temporal_correction": answer.get("temporal_correction", {}),
                }
                scored = _score_proactive_with_checkpoint(
                    ledger, record, judge_client
                )
                trace = {
                    "domain": domain,
                    "id": question_id,
                    "trace": retrieval.get("trace", []),
                    "temporal_correction": answer.get("temporal_correction", {}),
                }
                predictions.append(scored)
                traces.append(trace)
                append_jsonl(predictions_path, scored)
                append_jsonl(traces_path, trace)
                completed.add(row_key)

        predictions, traces = _paired_proactive_rows(predictions, traces)
        metrics = _aggregate_proactive_metrics(predictions)
        write_jsonl(predictions_path, predictions)
        write_jsonl(traces_path, traces)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_prediction_meta(
            output_dir, "proactive_membench", data_root, local_config
        )
        _write_memory_meta(output_dir, "proactive_membench", data_root, config)
        return metrics
    finally:
        ledger.close()
        memory_system.close()


def _locomo_config_for_output(config: AppConfig, output_dir: Path) -> AppConfig:
    local_config = copy.deepcopy(config)
    _configure_output_memory(local_config, output_dir)
    return local_config


def _configure_output_memory(config: AppConfig, output_dir: Path) -> None:
    configure_backend_output_paths(
        config, str(output_dir / "memory.sqlite3"), str(output_dir / "faiss")
    )


def _locomo_answer_parallel_options(options: dict[str, Any]) -> tuple[int, int]:
    workers = max(1, int(options.get("answer_workers", 16)))
    if "ollama_units" in options:
        from evaluation.ollama_units import validate_units
        validate_units(options["ollama_units"])
        workers = sum(unit.get("workers", 1) for unit in options["ollama_units"])
    chunk_size = max(1, int(options.get("answer_chunk_size", 8)))
    return workers, chunk_size


def _prepare_locomo_memory(
    memory_system: MemorySystem,
    items: list[dict[str, Any]],
    example_indices: Sequence[int] | None = None,
) -> None:
    for example_index, example in enumerate(items):
        if example_indices is not None and example_index not in example_indices:
            continue
        namespace = f"locomo:{example_index}"
        conversation = _extract_locomo_conversation(example)
        qas = _extract_locomo_qas(example)
        print(
            f"[locomo] example {example_index + 1}/{len(items)}: ingest {len(conversation)} turns, {len(qas)} qas",
            flush=True,
        )
        if memory_system.is_namespace_ready(namespace):
            print(
                f"[locomo] example {example_index + 1}/{len(items)}: reuse existing memory",
                flush=True,
            )
        else:
            memory_system.ingest_conversation(
                namespace,
                conversation,
                {"dataset": "locomo", "collection": f"locomo_{example_index}"},
            )




def _collect_locomo_answer_jobs(
    items: list[dict[str, Any]],
    completed: set[tuple[int, int]],
    max_questions: int | None,
    categories: Sequence[str] | None = None,
    example_indices: Sequence[int] | None = None,
) -> list[LocomoAnswerJob]:
    jobs: list[LocomoAnswerJob] = []
    answerable_seen = 0
    for example_index, example in enumerate(items):
        if example_indices is not None and example_index not in example_indices:
            continue
        namespace = f"locomo:{example_index}"
        for qa_index, qa in enumerate(_extract_locomo_qas(example)):
            if max_questions is not None and answerable_seen >= max_questions:
                return jobs
            category = str(qa.get("category") or qa.get("type") or "unknown")
            if category == "5" or (categories is not None and category not in categories):
                continue
            answerable_seen += 1
            if (example_index, qa_index) in completed:
                continue
            jobs.append(
                LocomoAnswerJob(
                    example_index=example_index,
                    qa_index=qa_index,
                    namespace=namespace,
                    question=str(qa.get("question") or qa.get("query") or ""),
                    reference=str(qa.get("answer") or qa.get("reference") or ""),
                    category=category,
                    gold_evidence=tuple(str(value) for value in qa.get("evidence", [])),
                )
            )
    return jobs


def _chunked_locomo_jobs(
    jobs: list[LocomoAnswerJob],
    chunk_size: int,
) -> list[list[LocomoAnswerJob]]:
    return [
        jobs[index : index + chunk_size] for index in range(0, len(jobs), chunk_size)
    ]


def _locomo_question_failure_rows(
    job: LocomoAnswerJob,
    error: Exception,
    latency_ms: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    failure = {
        "type": type(error).__name__,
        "message": str(error),
    }
    record = {
        "example_index": job.example_index,
        "qa_index": job.qa_index,
        "question": job.question,
        "answer": job.reference,
        "response": "",
        "reference": job.reference,
        "generated_answer": "",
        "category": job.category,
        "latency_ms": latency_ms,
        "temporal_correction": {},
        "answer_failed": True,
        "failure": failure,
    }
    trace_record = {
        "example_index": job.example_index,
        "qa_index": job.qa_index,
        "gold_evidence": list(job.gold_evidence),
        "trace": [],
        "episodic_memories": [],
        "prompt_context": "",
        "retrieved_count": 0,
        "temporal_correction": {},
        "answer_failed": True,
        "latency_ms": latency_ms,
        "failure": failure,
    }
    return record, trace_record


def _answer_locomo_job_chunk(
    config: AppConfig,
    jobs: list[LocomoAnswerJob],
    api_history_logger: ApiHistoryLogger | None = None,
    *,
    ledger_path: Path | None = None,
    regeneration_token: str | None = None,
) -> LocomoChunkResult:
    if not jobs:
        return LocomoChunkResult(results=[], errors=[])
    worker_config = copy.deepcopy(config)
    memory_system = MemorySystem(
        worker_config,
        api_history_logger=api_history_logger,
        store_read_only=True,
    )
    ledger: QuestionStageLedger | None = None
    try:
        agent = LightweightMemoryAgent(
            worker_config,
            memory_system=memory_system,
            api_history_logger=api_history_logger,
        )
        ledger = QuestionStageLedger(ledger_path) if ledger_path is not None else None
        results: list[tuple[dict[str, Any], dict[str, Any]]] = []
        errors: list[Exception] = []
        for job in jobs:
            started = time.perf_counter()
            try:
                if ledger is None:
                    answer = agent.answer(
                        job.question,
                        job.namespace,
                        category=job.category,
                    )
                else:
                    answer = _answer_with_question_checkpoints(
                        agent,
                        memory_system,
                        worker_config,
                        ledger,
                        benchmark="locomo",
                        question_key=f"locomo:{job.example_index}:{job.qa_index}",
                        question=job.question,
                        namespace=job.namespace,
                        category=job.category,
                        regeneration_token=regeneration_token,
                    )
            except Exception as error:
                latency_ms = (time.perf_counter() - started) * 1000.0
                results.append(_locomo_question_failure_rows(job, error, latency_ms))
                print(
                    "[locomo] question failed "
                    f"example={job.example_index} qa={job.qa_index}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                continue
            latency_ms = (time.perf_counter() - started) * 1000.0
            record = {
                "example_index": job.example_index,
                "qa_index": job.qa_index,
                "question": job.question,
                "answer": job.reference,
                "response": answer["answer"],
                "reference": job.reference,
                "generated_answer": answer["answer"],
                "category": job.category,
                "latency_ms": latency_ms,
                "temporal_correction": answer.get("temporal_correction", {}),
                "answer_failed": bool(answer.get("answer_failed", False)),
            }
            retrieval = answer.get("retrieval", {})
            # Keep the existing answer-only scoring contract; audit JSON is separate.
            record.update({key: answer[key] for key in ("answer_output", "raw_answer_output")
                           if key in answer})
            trace_record = {
                "example_index": job.example_index,
                "qa_index": job.qa_index,
                "gold_evidence": list(job.gold_evidence),
                "trace": retrieval.get("trace", []),
                "episodic_memories": retrieval.get("episodic_memories", []),
                "prompt_context": answer.get("memory_context", ""),
                "retrieved_count": len(retrieval.get("episodic_memories", [])),
                "temporal_correction": answer.get("temporal_correction", {}),
                "answer_failed": bool(answer.get("answer_failed", False)),
                "latency_ms": latency_ms,
            }
            results.append((record, trace_record))
        return LocomoChunkResult(results=results, errors=errors)
    finally:
        if ledger is not None:
            ledger.close()
        memory_system.close()


def _run_locomo_answer_jobs(
    config: AppConfig,
    jobs: list[LocomoAnswerJob],
    workers: int,
    chunk_size: int,
    api_history_logger: ApiHistoryLogger | None = None,
    ledger_path: Path | None = None,
    regeneration_token: str | None = None,
):
    answer_mode = str(config.evaluation.locomo.get("answer_mode", "single")).lower()
    if (answer_mode == "batch" or
            config.answer_model.provider.lower() == "openrouter" and config.answer_model.model.endswith(":batch")):
        from evaluation.batch_answers import run_locomo_batch_answers
        if ledger_path is None:
            raise ValueError("Batch answers require an output question ledger")
        yield from run_locomo_batch_answers(
            config, jobs, workers, chunk_size, ledger_path, api_history_logger,
            regeneration_token=regeneration_token,
        )
        return
    if answer_mode != "single":
        raise ValueError(f"Unsupported LoCoMo answer_mode: {answer_mode}")
    unit_config = config.evaluation.locomo.get("ollama_units")
    if unit_config is not None:
        from evaluation.ollama_units import resolve_units, run_unit_chunks
        units = resolve_units(unit_config)
        print("[locomo] Ollama units: " + ", ".join(
            f"{name} ({count} workers)" for name, count, _ in units
        ), flush=True)
        errors = []
        def answer_chunk(chunk):
            return _answer_locomo_job_chunk(
                config, chunk, api_history_logger,
                ledger_path=ledger_path, regeneration_token=regeneration_token,
            )
        for result in run_unit_chunks(units, _chunked_locomo_jobs(jobs, chunk_size), answer_chunk):
            if isinstance(result, Exception):
                errors.append(result)
                continue
            yield from result.results
            errors.extend(result.errors)
        if errors:
            raise errors[0]
        return
    if workers <= 1:
        chunk_result = _answer_locomo_job_chunk(
            config,
            jobs,
            api_history_logger,
            ledger_path=ledger_path,
            regeneration_token=regeneration_token,
        )
        for result in chunk_result.results:
            yield result
        if chunk_result.errors:
            raise chunk_result.errors[0]
        return
    chunks = _chunked_locomo_jobs(jobs, chunk_size)
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _answer_locomo_job_chunk,
                config,
                chunk,
                api_history_logger,
                ledger_path=ledger_path,
                regeneration_token=regeneration_token,
            )
            for chunk in chunks
        ]
        for future in as_completed(futures):
            try:
                chunk_result = future.result()
            except Exception as error:
                errors.append(error)
                continue
            for result in chunk_result.results:
                yield result
            errors.extend(chunk_result.errors)
    if errors:
        raise errors[0]


def _sort_locomo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("example_index", -1)),
            int(row.get("qa_index", -1)),
        ),
    )


def _paired_locomo_rows(
    predictions: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deduped_predictions = _sort_locomo_rows(_dedupe_locomo_rows(predictions))
    deduped_traces = _sort_locomo_rows(_dedupe_locomo_rows(traces))
    paired_keys = {_locomo_row_key(row) for row in deduped_predictions}.intersection(
        _locomo_row_key(row) for row in deduped_traces
    )
    return (
        [row for row in deduped_predictions if _locomo_row_key(row) in paired_keys],
        [row for row in deduped_traces if _locomo_row_key(row) in paired_keys],
    )


def _score_locomo_with_checkpoints(
    config: AppConfig, output_dir: Path, predictions: list[dict[str, Any]], *,
    force_rejudge: bool, api_history_logger: ApiHistoryLogger | None,
    force_token: str | None = None,
) -> list[dict[str, Any]]:
    with QuestionStageLedger(output_dir / "question_stages.sqlite3") as ledger:
        if force_rejudge or force_token:
            ledger.invalidate_stages({"scoring"})
        cached_rows = {}
        pending = []
        for row in predictions:
            key = _locomo_row_key(row)
            cached = ledger.load_success(f"locomo:{key[0]}:{key[1]}", "scoring")
            if cached is None:
                pending.append(row)
            else:
                cached_rows[key] = dict(cached["row"])
        if pending:
            try:
                scored = score_locomo_predictions(
                    config, output_dir, pending,
                    force_rejudge=force_rejudge or bool(force_token),
                    api_history_logger=api_history_logger,
                )
            except Exception as error:
                for row in pending:
                    key = _locomo_row_key(row)
                    ledger.record_failure(f"locomo:{key[0]}:{key[1]}", "scoring", error)
                raise
            for row in scored:
                key = _locomo_row_key(row)
                ledger.record_success(f"locomo:{key[0]}:{key[1]}", "scoring", {"row": row})
                cached_rows[key] = row
        return [cached_rows[_locomo_row_key(row)] for row in predictions]

def run_locomo(
    config: AppConfig,
    output_dir: Path,
    api_history_logger: ApiHistoryLogger | None = None,
) -> dict[str, Any]:
    locomo_path = Path(config.evaluation.benchmark_dir) / "locomo" / "locomo10.json"
    if not locomo_path.exists():
        raise FileNotFoundError(
            f"LoCoMo data not found at {locomo_path}. Run scripts/download_benchmarks.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    scored_predictions_path = output_dir / "scored_predictions.jsonl"
    traces_path = output_dir / "retrieval_traces.jsonl"
    options = config.evaluation.locomo
    categories = options.get("categories")
    if categories is not None:
        if not isinstance(categories, (list, tuple)) or not categories:
            raise ValueError("locomo.categories must be a nonempty list")
        categories = tuple(str(value) for value in categories)
        if set(categories) - {"1", "2", "3", "4"}:
            raise ValueError("locomo.categories must contain only 1, 2, 3, or 4")
    regenerate_answers = bool(options.get("regenerate_answers", False))
    reuse_predictions_from = options.get("reuse_predictions_from")
    rejudge_from = options.get("rejudge_from")
    force_rejudge = bool(options.get("force_rejudge", False))

    regeneration_token = str(time.time_ns()) if regenerate_answers else None
    force_rejudge_token = str(time.time_ns()) if force_rejudge else None

    data = load_json(locomo_path)
    items = (
        data if isinstance(data, list) else data.get("data", data.get("examples", []))
    )
    example_indices = options.get("example_indices")
    if example_indices is not None:
        if (not isinstance(example_indices, (list, tuple)) or not example_indices
                or any(type(value) is not int or not 0 <= value < len(items)
                       for value in example_indices)):
            raise ValueError("locomo.example_indices must be a nonempty list of valid zero-based integers")
        example_indices = tuple(dict.fromkeys(example_indices))
    selected_items = [example for index, example in enumerate(items)
                      if example_indices is None or index in example_indices]
    max_questions_value = config.evaluation.locomo.get("max_questions")
    max_questions = (
        int(max_questions_value) if max_questions_value is not None else None
    )
    selected_keys = {
        (job.example_index, job.qa_index)
        for job in _collect_locomo_answer_jobs(items, set(), max_questions, categories, example_indices)
    }
    prediction_source = rejudge_from or reuse_predictions_from
    if prediction_source and not regenerate_answers:
        source_dir = _resolve_benchmark_artifact_dir(prediction_source, "locomo")
        _validate_rejudge_source(source_dir, "locomo", locomo_path)
        _copy_prediction_artifacts(source_dir, output_dir)
        predictions = _sort_locomo_rows(_dedupe_locomo_rows([
            row for row in load_jsonl(predictions_path)
            if _is_mem0_locomo_answer(row) and _locomo_row_key(row) in selected_keys
        ]))
        traces = _sort_locomo_rows([
            row for row in load_jsonl(traces_path)
            if _locomo_row_key(row) in selected_keys
        ])
        skipped_category_5_count = _count_locomo_skipped_category_5(
            selected_items, max_questions
        )
        _validate_full_locomo_outputs(
            items,
            max_questions,
            predictions,
            traces,
            protocol=config.evaluation.protocol,
            categories=categories,
            example_indices=example_indices,
        )
        scored_predictions = _score_locomo_with_checkpoints(
            config,
            output_dir,
            predictions,
            force_rejudge=force_rejudge or bool(rejudge_from),
            api_history_logger=api_history_logger,
            force_token=force_rejudge_token,
        )
        metrics = _aggregate_locomo_metrics(
            scored_predictions, skipped_category_5_count, traces
        )
        write_jsonl(predictions_path, predictions)
        write_jsonl(scored_predictions_path, scored_predictions)
        write_jsonl(traces_path, traces)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        operation = "rejudged" if rejudge_from else "reused"
        print(f"[locomo] {operation} predictions from {source_dir}", flush=True)
        return metrics

    _prepare_reused_memory(
        "locomo",
        locomo_path,
        config,
        output_dir,
        options.get("reuse_memory_from"),
    )
    local_config = _locomo_config_for_output(config, output_dir)
    ledger_path = output_dir / "question_stages.sqlite3"
    with QuestionStageLedger(ledger_path):
        pass
    memory_system = MemorySystem(
        local_config,
        api_history_logger=api_history_logger,
    )
    try:
        _prepare_locomo_memory(memory_system, items, example_indices)
    finally:
        memory_system.close()

    loaded_predictions = [
        row
        for row in ([] if regenerate_answers else load_jsonl(predictions_path))
        if _is_mem0_locomo_answer(row) and _locomo_row_key(row) in selected_keys
    ]
    loaded_traces = [] if regenerate_answers else [
        row for row in load_jsonl(traces_path) if _locomo_row_key(row) in selected_keys
    ]
    predictions, traces = _paired_locomo_rows(loaded_predictions, loaded_traces)
    # A crash can occur between the two per-question appends. Canonicalize both
    # files before scheduling work so an unpaired row is rerun without leaving a
    # stale duplicate that could later be paired with the wrong trace.
    write_jsonl(predictions_path, predictions)
    write_jsonl(traces_path, traces)
    completed = {_locomo_row_key(row) for row in predictions}
    question_count = len(completed)
    skipped_category_5_count = _count_locomo_skipped_category_5(selected_items, max_questions)
    if question_count:
        print(
            f"[locomo] resuming from {question_count} completed questions", flush=True
        )

    jobs = _collect_locomo_answer_jobs(items, completed, max_questions, categories, example_indices)
    workers, chunk_size = _locomo_answer_parallel_options(options)
    if jobs:
        print(
            f"[locomo] answering {len(jobs)} questions with {workers} workers, chunk size {chunk_size}",
            flush=True,
        )
    progress = None
    if tqdm is not None and jobs:
        progress = tqdm(
            total=len(jobs),
            initial=0,
            desc="[locomo] answering",
            unit="q",
            file=sys.stderr,
            dynamic_ncols=True,
        )
    try:
        for record, trace_record in _run_locomo_answer_jobs(
            local_config,
            jobs,
            workers,
            chunk_size,
            api_history_logger=api_history_logger,
            ledger_path=ledger_path,
            regeneration_token=regeneration_token,
        ):
            predictions.append(record)
            traces.append(trace_record)
            append_jsonl(predictions_path, record)
            append_jsonl(traces_path, trace_record)
            completed.add((int(record["example_index"]), int(record["qa_index"])))
            question_count += 1
            if progress is not None:
                progress.update(1)
            elif question_count % 50 == 0:
                print(f"[locomo] processed {question_count} questions", flush=True)
    finally:
        if progress is not None:
            progress.close()

    predictions = _sort_locomo_rows(_dedupe_locomo_rows(predictions))
    traces = _sort_locomo_rows(_dedupe_locomo_rows(traces))
    scored_predictions = _score_locomo_with_checkpoints(
        local_config,
        output_dir,
        predictions,
        force_rejudge=force_rejudge or regenerate_answers,
        api_history_logger=api_history_logger,
        force_token=force_rejudge_token,
    )
    metrics = _aggregate_locomo_metrics(
        scored_predictions, skipped_category_5_count, traces
    )
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(scored_predictions_path, scored_predictions)
    write_jsonl(output_dir / "retrieval_traces.jsonl", traces)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_prediction_meta(output_dir, "locomo", locomo_path, local_config)
    _write_memory_meta(output_dir, "locomo", locomo_path, local_config)
    _validate_full_locomo_outputs(
        items,
        max_questions,
        predictions,
        traces,
        protocol=config.evaluation.protocol,
        categories=categories,
        example_indices=example_indices,
    )
    return metrics


def write_markdown_report(
    path: Path, config: AppConfig, metrics: dict[str, dict[str, Any]]
) -> None:
    v4 = config.memory.backends.get("v4", {})
    resolved_models = config_to_dict(config)["models"]
    budget_summary = (
        f"{v4.get('simple_probe_budget', 8)}/{v4.get('complex_probe_budget', 12)} initial, "
        f"{v4.get('followup_probe_budget', 6)} follow-up, "
        f"{v4.get('expansion_budget_per_anchor', 3)} expansion, "
        f"{v4.get('total_candidate_budget', 32)} total, "
        f"{v4.get('final_top_k', 16)} final"
    )
    lines = [
        "# Evaluation Report",
        "",
        "## Configuration",
        "",
        "| Component | Config key | Provider | Model |",
        "|---|---|---:|---:|",
    ]
    for role, model_config in resolved_models.items():
        component = MODEL_ROLE_LABELS.get(role, role.replace("_", " ").capitalize())
        lines.append(
            f"| {component} | `models.{role}` | {model_config['provider']} | {model_config['model']} |"
        )
    lines.extend(
        [
            "",
            f"- Memory backend: `{config.memory.backend}`",
            f"- V4 retrieval budgets: `{budget_summary}`",
            f"- Include retrieval trace in answer prompt: `{config.memory.include_retrieval_trace_in_answer_prompt}`",
            f"- Answer context mode: `{config.memory.answer_context_mode}`",
            f"- ProactiveMemBench LLM judge: `{config.evaluation.proactive_membench.get('use_llm_judge', True)}`",
            f"- LoCoMo LLM judge: `{config.evaluation.locomo.get('use_llm_judge', True)}`",
            f"- LoCoMo answer workers: `{config.evaluation.locomo.get('answer_workers', 16)}`",
            f"- LoCoMo answer chunk size: `{config.evaluation.locomo.get('answer_chunk_size', 8)}`",
            f"- LoCoMo judge mode: `{config.evaluation.locomo.get('judge_mode', 'single')}`",
            f"- LoCoMo judge runs: `{config.evaluation.locomo.get('judge_runs', 1)}`",
            "",
        ]
    )
    if "proactive_membench" in metrics:
        proactive = metrics["proactive_membench"]
        lines.extend(
            [
                "## ProactiveMemBench",
                "",
                "| Split | Count | Recall@k | Precision |",
                "|---|---:|---:|---:|",
                f"| Overall | {proactive.get('count', 0)} | {proactive.get('overall_recall', 0):.4f} | {proactive.get('overall_precision', 0):.4f} |",
            ]
        )
        for key, value in sorted(proactive.get("by_trigger_type", {}).items()):
            lines.append(
                f"| Trigger: {key} | {value['count']} | {value['recall']:.4f} | {value['precision']:.4f} |"
            )
        for key, value in sorted(proactive.get("by_difficulty", {}).items()):
            lines.append(
                f"| Difficulty: {key} | {value['count']} | {value['recall']:.4f} | {value['precision']:.4f} |"
            )
        lines.append("")
    if "locomo" in metrics:
        locomo = metrics["locomo"]
        has_llm_score = locomo.get("overall_llm_score") is not None
        header = (
            "| Category ID | Category name | Count | F1 | BLEU-1 | LLM Judge | LLM Judge Std |"
            if has_llm_score
            else "| Category ID | Category name | Count | F1 | BLEU-1 |"
        )
        separator = (
            "|---|---|---:|---:|---:|---:|---:|"
            if has_llm_score
            else "|---|---|---:|---:|---:|"
        )
        overall = (
            f"| Overall | All categories | {locomo.get('count', 0)} | {locomo.get('overall_f1', 0):.4f} | "
            f"{locomo.get('overall_bleu1', 0):.4f} | {locomo.get('overall_llm_score', 0):.4f} | "
            f"{locomo.get('overall_llm_score_std', 0):.4f} |"
            if has_llm_score
            else f"| Overall | All categories | {locomo.get('count', 0)} | {locomo.get('overall_f1', 0):.4f} | {locomo.get('overall_bleu1', 0):.4f} |"
        )
        lines.extend(
            [
                "## LoCoMo",
                "",
                "- Category `5` is skipped to match Mem0 LoCoMo evaluation.",
                f"- Skipped category `5` questions: `{locomo.get('skipped_category_5_count', 0)}`",
                "",
                header,
                separator,
                overall,
            ]
        )
        for key, value in sorted(locomo.get("by_category", {}).items()):
            category_name = LOCOMO_CATEGORY_NAMES.get(str(key), "Unknown")
            if has_llm_score:
                lines.append(
                    f"| {key} | {category_name} | {value['count']} | {value['f1']:.4f} | {value['bleu1']:.4f} | "
                    f"{value.get('llm_score', 0):.4f} | {value.get('llm_score_std', 0):.4f} |"
                )
            else:
                lines.append(
                    f"| {key} | {category_name} | {value['count']} | "
                    f"{value['f1']:.4f} | {value['bleu1']:.4f} |"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _retrieved_unit_strings(retrieval: dict[str, Any]) -> list[str]:
    units: list[str] = []
    for item in retrieval.get("episodic_memories", []):
        units.append(
            " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("summary", "")),
                    str(item.get("text", "")),
                ]
            )
        )
        for semantic in item.get("semantic_memories", []):
            units.append(
                " ".join(
                    [
                        str(semantic.get("subject", "")),
                        str(semantic.get("predicate", "")),
                        str(semantic.get("object", "")),
                    ]
                )
            )
    return [unit for unit in units if unit.strip()]


def _score_proactive_prediction(
    row: dict[str, Any], judge_client: ChatClient | None
) -> dict[str, Any]:
    scored = dict(row)
    candidates = [str(item) for item in scored.get("candidate_set", [])]
    retrieved_units = [str(item) for item in scored.get("retrieved_units", [])]
    judged_recall = None
    judged_precision = None
    if judge_client is not None:
        judged_recall = judge_proactive_recall(
            judge_client,
            str(scored.get("question", "")),
            candidates,
            retrieved_units,
        )
        judged_precision = judge_proactive_precision(
            judge_client,
            str(scored.get("trigger_type", "")),
            str(scored.get("question", "")),
            "\n".join(retrieved_units),
            retrieved_units,
        )
    retrieved_text = "\n".join(retrieved_units + [str(scored.get("answer", ""))])
    scored["recall"] = (
        judged_recall
        if judged_recall is not None
        else concept_recall(candidates, retrieved_text)
    )
    scored["precision"] = (
        judged_precision
        if judged_precision is not None
        else concept_precision(candidates, retrieved_units)
    )
    scored["recall_judge"] = "llm" if judged_recall is not None else "lexical_fallback"
    scored["precision_judge"] = (
        "llm" if judged_precision is not None else "lexical_fallback"
    )
    if judged_recall is not None and judged_precision is not None:
        scored["judge"] = "llm"
    elif judged_recall is None and judged_precision is None:
        scored["judge"] = "lexical_fallback"
    else:
        scored["judge"] = "mixed"
    return scored


def judge_proactive_recall(
    judge_client: ChatClient,
    question: str,
    candidate_units: list[str],
    retrieved_units: list[str],
) -> float | None:
    try:
        raw = judge_client.chat(
            messages_with_json_schema(
                build_proactive_recall_messages(
                    question, candidate_units, retrieved_units
                ),
                PROACTIVE_RECALL_JSON_SCHEMA,
            ),
            json_mode=True,
            json_schema=PROACTIVE_RECALL_JSON_SCHEMA,
        )
        data = parse_json_object_with_schema(
            raw,
            PROACTIVE_RECALL_JSON_SCHEMA,
            source="Proactive recall judge",
        )
        matches = data.get("matches")
        if not isinstance(matches, list) or not candidate_units:
            return None
        hit_count = sum(
            1
            for item in matches
            if isinstance(item, dict) and bool(item.get("matched"))
        )
        return hit_count / len(candidate_units)
    except Exception:
        return None


def judge_proactive_precision(
    judge_client: ChatClient,
    trigger_type: str,
    question: str,
    context_str: str,
    retrieved_units: list[str],
) -> float | None:
    if not retrieved_units:
        return 0.0
    try:
        raw = judge_client.chat(
            messages_with_json_schema(
                build_proactive_precision_messages(
                    trigger_type, question, context_str, retrieved_units
                ),
                PROACTIVE_PRECISION_JSON_SCHEMA,
            ),
            json_mode=True,
            json_schema=PROACTIVE_PRECISION_JSON_SCHEMA,
        )
        data = parse_json_object_with_schema(
            raw,
            PROACTIVE_PRECISION_JSON_SCHEMA,
            source="Proactive precision judge",
        )
        judgments = data.get("judgments")
        if not isinstance(judgments, list) or not judgments:
            return None
        scores = []
        for item in judgments:
            if not isinstance(item, dict):
                scores.append(0.0)
                continue
            try:
                scores.append(max(0.0, min(5.0, float(item.get("score", 0)))))
            except (TypeError, ValueError):
                scores.append(0.0)
        if not scores:
            return None
        return sum(scores) / (5.0 * len(judgments))
    except Exception:
        return None


def judge_locomo_answer(
    judge_client: ChatClient,
    question: str,
    gold_answer: str,
    generated_answer: str,
) -> int:
    raw = judge_client.chat(
        messages_with_json_schema(
            build_locomo_judge_messages(question, gold_answer, generated_answer),
            LOCOMO_JUDGE_JSON_SCHEMA,
        ),
        json_mode=True,
        json_schema=LOCOMO_JUDGE_JSON_SCHEMA,
    )
    data = parse_json_object_with_schema(
        raw, LOCOMO_JUDGE_JSON_SCHEMA, source="LoCoMo judge"
    )
    label = data["label"]
    return 1 if str(label).strip().upper() == "CORRECT" else 0


def score_locomo_predictions(
    config: AppConfig,
    output_dir: Path,
    rows: list[dict[str, Any]],
    force_rejudge: bool = False,
    api_history_logger: ApiHistoryLogger | None = None,
) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        response = str(row.get("response", row.get("generated_answer", "")))
        answer = str(row.get("answer", row.get("reference", "")))
        metrics = mem0_calculate_metrics(response, answer)
        scored = dict(row)
        scored["f1_score"] = metrics["f1"]
        scored["bleu_score"] = metrics["bleu1"]
        scored_rows.append(scored)

    if not bool(config.evaluation.locomo.get("use_llm_judge", True)):
        for row in scored_rows:
            for key in ("llm_scores", "llm_score_mean", "llm_score_std", "llm_score"):
                row.pop(key, None)
        return scored_rows

    judge_runs = max(1, int(config.evaluation.locomo.get("judge_runs", 1)))
    judge_rows = [row for row in scored_rows if not row.get("answer_failed")]
    judge_mode = str(config.evaluation.locomo.get("judge_mode", "single")).lower()
    if config.judge_model.provider.lower() == "openrouter" and config.judge_model.model.endswith(":batch"):
        judge_mode = "batch"
    score_lists: dict[tuple[int, int], list[int]] = {}
    if judge_rows:
        if judge_mode == "single":
            score_lists = _score_locomo_judge_single(
                config, judge_rows, judge_runs, api_history_logger
            )
        elif judge_mode == "batch":
            score_lists = _score_locomo_judge_batch(
                config,
                output_dir,
                judge_rows,
                judge_runs,
                force_rejudge,
                api_history_logger,
            )
        else:
            raise ValueError(f"Unsupported LoCoMo judge_mode: {judge_mode}")
    score_lists.update(
        {
            _locomo_row_key(row): [0] * judge_runs
            for row in scored_rows
            if row.get("answer_failed")
        }
    )
    _attach_locomo_judge_scores(scored_rows, score_lists)
    return scored_rows


def _score_locomo_judge_single(
    config: AppConfig,
    rows: list[dict[str, Any]],
    judge_runs: int,
    api_history_logger: ApiHistoryLogger | None = None,
) -> dict[tuple[int, int], list[int]]:
    workers = max(1, int(config.evaluation.locomo.get("judge_workers", 1)))
    retries = max(0, int(config.evaluation.locomo.get("judge_max_retries", 2)))
    backoff = max(
        0.0, float(config.evaluation.locomo.get("judge_retry_backoff_seconds", 1.0))
    )

    def score_one(
        row: dict[str, Any],
        run_index: int,
        judge_client: ChatClient,
        unit_name: str | None = None,
    ) -> tuple[tuple[int, int], int, int]:
        key = _locomo_row_key(row)
        context = {
            "example_index": key[0],
            "qa_index": key[1],
            "run_index": run_index,
        }
        if unit_name is not None:
            context["ollama_unit"] = unit_name
        client = _logged_chat_client(
            judge_client,
            api_history_logger,
            "locomo_judge",
            config.judge_model,
            context=context,
        )
        for retry in range(retries + 1):
            try:
                score = judge_locomo_answer(
                    client,
                    str(row.get("question", "")),
                    str(row.get("answer", "")),
                    str(row.get("response", "")),
                )
                return key, run_index, score
            except Exception:
                if retry >= retries:
                    raise
                time.sleep(backoff * (2**retry))
        raise AssertionError("unreachable")

    jobs = [(row, run_index) for row in rows for run_index in range(judge_runs)]
    unit_config = config.evaluation.locomo.get("ollama_units")
    use_ollama_units = (
        config.judge_model.provider.lower() == "ollama" and unit_config is not None
    )
    if use_ollama_units:
        from evaluation.ollama_units import resolve_units, run_unit_jobs

        units = resolve_units(unit_config)
        print(
            "[locomo] Ollama judge units: "
            + ", ".join(f"{name} ({count} workers)" for name, count, _ in units),
            flush=True,
        )

        def score_unit_job(unit_name, job):
            row, run_index = job
            judge_client = make_chat_client(config.judge_model)
            return score_one(row, run_index, judge_client, unit_name)

        outcomes = list(run_unit_jobs(units, jobs, score_unit_job))
        errors = [result for _name, result in outcomes if isinstance(result, Exception)]
        if errors:
            raise errors[0]
        results = [result for _name, result in outcomes]
    else:
        judge_client = make_chat_client(config.judge_model)
        if workers <= 1:
            results = [
                score_one(row, run_index, judge_client) for row, run_index in jobs
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(score_one, row, run_index, judge_client)
                    for row, run_index in jobs
                ]
                results = [future.result() for future in as_completed(futures)]
    scores: dict[tuple[int, int], list[int | None]] = {
        _locomo_row_key(row): [None] * judge_runs for row in rows
    }
    for key, run_index, score in results:
        scores[key][run_index] = score
    return {
        key: [int(value) for value in values if value is not None]
        for key, values in scores.items()
    }


def _score_locomo_judge_batch(
    config: AppConfig,
    output_dir: Path,
    rows: list[dict[str, Any]],
    judge_runs: int,
    force_rejudge: bool = False,
    api_history_logger: ApiHistoryLogger | None = None,
) -> dict[tuple[int, int], list[int]]:
    if config.judge_model.provider.lower() == "openrouter":
        from evaluation.batch import run_chat_batch
        results = run_chat_batch(
            config.judge_model, _build_locomo_judge_batch_requests(rows, judge_runs),
            output_dir, "judge_batch", config.evaluation.locomo, api_history_logger,
            force=force_rejudge,
        )
        return _locomo_scores_from_batch_results(results, rows, judge_runs)
    batch_client = make_batch_chat_client(config.judge_model)
    if api_history_logger is not None:
        batch_client = LoggedBatchChatClient(
            batch_client,
            api_history_logger,
            "locomo_batch_judge",
            config.judge_model.provider,
            config.judge_model.model,
            context={"judge_runs": judge_runs, "row_count": len(rows)},
        )
    input_path = output_dir / "judge_batch_input.jsonl"
    meta_path = output_dir / "judge_batch_meta.json"
    output_path = output_dir / "judge_batch_output.jsonl"
    errors_path = output_dir / "judge_batch_errors.jsonl"
    expected_request_count = len(rows) * judge_runs

    job = None
    meta = load_json(meta_path) if meta_path.exists() else None
    meta_matches = (
        not force_rejudge
        and isinstance(meta, dict)
        and meta.get("request_count") == expected_request_count
    )
    if output_path.exists() and meta_matches:
        output_lines = load_jsonl(output_path)
    else:
        output_lines = []
        if meta_matches and isinstance(meta, dict) and meta.get("batch_id"):
            job = batch_client.get_batch(str(meta["batch_id"]))
        else:
            requests_ = _build_locomo_judge_batch_requests(rows, judge_runs)
            batch_client.write_chat_batch_input(input_path, requests_)
            job = batch_client.submit_chat_batch(
                input_path,
                str(config.evaluation.locomo.get("batch_completion_window", "24h")),
                metadata={
                    "benchmark": "locomo",
                    "purpose": "llm_judge",
                },
            )
        _write_locomo_batch_meta(
            meta_path,
            config,
            job,
            expected_request_count,
        )

        wait_for_batch = bool(config.evaluation.locomo.get("batch_wait", True))
        terminal_statuses = {"completed", "failed", "cancelled", "expired"}
        while wait_for_batch and job.status not in terminal_statuses:
            time.sleep(
                max(
                    1,
                    int(
                        config.evaluation.locomo.get("batch_poll_interval_seconds", 60)
                    ),
                )
            )
            job = batch_client.get_batch(job.id)
            _write_locomo_batch_meta(
                meta_path,
                config,
                job,
                expected_request_count,
            )

        if job.status != "completed":
            if wait_for_batch:
                raise RuntimeError(
                    f"LoCoMo batch judge did not complete successfully: {job.status}"
                )
            print(
                f"[locomo] batch judge is {job.status}; scored predictions will omit LLM judge for now",
                flush=True,
            )
            return {}
        if job.output_file_id:
            output_lines = batch_client.download_batch_file(job.output_file_id)
            write_jsonl(output_path, output_lines)
        if job.error_file_id:
            error_lines = batch_client.download_batch_file(job.error_file_id)
            write_jsonl(errors_path, error_lines)

    batch_results = batch_client.parse_chat_batch_results(output_lines)
    return _locomo_scores_from_batch_results(batch_results, rows, judge_runs)


def _build_locomo_judge_batch_requests(
    rows: list[dict[str, Any]], judge_runs: int
) -> list[ChatBatchRequest]:
    requests_: list[ChatBatchRequest] = []
    for row in rows:
        example_index, qa_index = _locomo_row_key(row)
        for run_index in range(judge_runs):
            requests_.append(
                ChatBatchRequest(
                    custom_id=_locomo_batch_custom_id(
                        example_index, qa_index, run_index
                    ),
                    messages=build_locomo_judge_messages(
                        str(row.get("question", "")),
                        str(row.get("answer", "")),
                        str(row.get("response", "")),
                    ),
                    json_mode=True,
                )
            )
    return requests_


def _locomo_scores_from_batch_results(
    batch_results: list[ChatBatchResult],
    rows: list[dict[str, Any]],
    judge_runs: int,
) -> dict[tuple[int, int], list[int]]:
    scores = {_locomo_row_key(row): [None] * judge_runs for row in rows}
    for result in batch_results:
        parsed = _parse_locomo_batch_custom_id(result.custom_id)
        if parsed is None or result.content is None or result.error:
            continue
        example_index, qa_index, run_index = parsed
        if run_index >= judge_runs:
            continue
        key = (example_index, qa_index)
        if key not in scores or scores[key][run_index] is not None:
            raise RuntimeError("LoCoMo batch judge returned unexpected or duplicate result IDs")
        data = parse_json_object_with_schema(result.content, LOCOMO_JUDGE_JSON_SCHEMA)
        scores[key][run_index] = 1 if data["label"] == "CORRECT" else 0

    missing = [
        key
        for key, values in scores.items()
        if len(values) != judge_runs or any(value is None for value in values)
    ]
    if missing:
        raise RuntimeError(
            f"LoCoMo batch judge returned incomplete results for {len(missing)} questions"
        )
    return {
        key: [int(value) for value in values if value is not None]
        for key, values in scores.items()
    }


def _attach_locomo_judge_scores(
    rows: list[dict[str, Any]],
    score_lists: dict[tuple[int, int], list[int]],
) -> None:
    for row in rows:
        scores = score_lists.get(_locomo_row_key(row))
        if scores is None:
            continue
        row["llm_scores"] = scores
        row["llm_score_mean"] = _mean(scores)
        row["llm_score_std"] = _std(scores)
        row["llm_score"] = row["llm_score_mean"]


def _write_locomo_batch_meta(
    path: Path,
    config: AppConfig,
    job,
    request_count: int,
) -> None:
    data = {
        "provider": config.judge_model.provider,
        "model": config.judge_model.model,
        "batch_id": job.id,
        "status": job.status,
        "input_file_id": job.input_file_id,
        "output_file_id": job.output_file_id,
        "error_file_id": job.error_file_id,
        "request_count": request_count,
        "raw": job.raw,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _locomo_batch_custom_id(example_index: int, qa_index: int, run_index: int) -> str:
    return f"locomo-{example_index}-{qa_index}-run-{run_index}"




def _parse_locomo_batch_custom_id(custom_id: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"locomo-(\d+)-(\d+)-run-(\d+)", custom_id)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _locomo_row_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["example_index"]), int(row["qa_index"])


def _count_locomo_skipped_category_5(
    items: list[dict[str, Any]], max_questions: int | None
) -> int:
    skipped = 0
    answered = 0
    for example in items:
        for qa in _extract_locomo_qas(example):
            if max_questions is not None and answered >= max_questions:
                return skipped
            category = str(qa.get("category") or qa.get("type") or "unknown")
            if category == "5":
                skipped += 1
                continue
            answered += 1
    return skipped


def _aggregate_proactive_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "overall_recall": _mean(row["recall"] for row in rows),
        "overall_precision": _mean(row["precision"] for row in rows),
        "answer_failure_rate": _mean(
            int(bool(row.get("answer_failed"))) for row in rows
        ),
        "by_trigger_type": _group_metric(rows, "trigger_type", ("recall", "precision")),
        "by_difficulty": _group_metric(rows, "difficulty", ("recall", "precision")),
    }


def _aggregate_locomo_metrics(
    rows: list[dict[str, Any]],
    skipped_category_5_count: int = 0,
    traces: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    has_llm_score = any(_row_llm_scores(row) for row in rows)
    by_category: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category", "unknown"))].append(row)
    for key, values in grouped.items():
        item = {
            "count": len(values),
            "f1": _mean(value["f1_score"] for value in values),
            "bleu1": _mean(value["bleu_score"] for value in values),
        }
        if has_llm_score:
            item["llm_score"] = _mean(
                _row_llm_mean(value) for value in values if _row_llm_scores(value)
            )
            item["llm_score_mean"] = item["llm_score"]
            item["llm_score_std"] = _judge_run_std(values)
        by_category[key] = item

    metrics: dict[str, Any] = {
        "count": len(rows),
        "skipped_category_5_count": skipped_category_5_count,
        "overall_f1": _mean(row["f1_score"] for row in rows),
        "overall_bleu1": _mean(row["bleu_score"] for row in rows),
        "overall_llm_score": None,
        "overall_llm_score_mean": None,
        "overall_llm_score_std": None,
        "answer_failure_rate": _mean(
            int(bool(row.get("answer_failed"))) for row in rows
        ),
        "by_category": by_category,
    }
    if has_llm_score:
        overall_llm_score = _mean(
            _row_llm_mean(row) for row in rows if _row_llm_scores(row)
        )
        metrics["overall_llm_score"] = overall_llm_score
        metrics["overall_llm_score_mean"] = overall_llm_score
        metrics["overall_llm_score_std"] = _judge_run_std(rows)
    if traces is not None:
        metrics.update(_aggregate_v2_retrieval_metrics(traces))
    return metrics


def _aggregate_v2_retrieval_metrics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [
        int(row.get("retrieved_count", len(row.get("episodic_memories", []))))
        for row in traces
    ]
    limits = [_trace_final_top_k(row) for row in traces]
    evidence_recalls = []
    evidence_precisions = []
    facet_coverages = []
    residual = 0
    end_to_end_latencies = []
    retrieval_latencies = []
    router_calls = 0
    retrieval_calls = 0
    router_failures = 0
    rounds = []
    hops = []
    for row in traces:
        gold = {str(value) for value in row.get("gold_evidence", [])}
        retrieved_refs = {
            str(ref.get("turn_id"))
            for memory in row.get("episodic_memories", [])
            for ref in memory.get("source_refs", [])
            if ref.get("turn_id") is not None
        }
        if gold:
            matched = len(gold.intersection(retrieved_refs))
            evidence_recalls.append(matched / len(gold))
            evidence_precisions.append(
                matched / len(retrieved_refs) if retrieved_refs else 0.0
            )
        correction = row.get("temporal_correction") or {}
        residual += int(bool(correction.get("residual")))
        end_to_end_latency = _trace_metric_float(row.get("latency_ms"))
        if end_to_end_latency is not None:
            end_to_end_latencies.append(end_to_end_latency)
        for trace in row.get("trace", []):
            plan = trace.get("query_plan") or {}
            plan_router_calls = _trace_metric_int(
                plan.get("router_calls")
                if "router_calls" in plan
                else plan.get("controller_calls")
            )
            if plan_router_calls is not None:
                router_calls += plan_router_calls
            if "router_failures" in plan:
                plan_router_failures = _trace_metric_int(plan.get("router_failures"))
            else:
                controller_errors = plan.get("controller_errors")
                plan_router_failures = (
                    len(controller_errors)
                    if isinstance(controller_errors, list)
                    else None
                )
            if plan_router_failures is not None:
                router_failures += plan_router_failures
            plan_retrieval_calls = _trace_metric_int(plan.get("retrieval_calls"))
            if plan_retrieval_calls is not None:
                retrieval_calls += plan_retrieval_calls
            plan_rounds = plan.get("rounds")
            if isinstance(plan_rounds, list):
                rounds.append(len(plan_rounds))
            else:
                round_count = _trace_metric_int(plan_rounds)
                if round_count is not None:
                    rounds.append(round_count)
            graph_hops = _trace_metric_int(plan.get("graph_hops"))
            if graph_hops is not None:
                hops.append(graph_hops)
            latency = _trace_metric_float(plan.get("latency_ms"))
            if latency is not None:
                retrieval_latencies.append(latency)
            facets = list(plan.get("evidence_facets") or [])
            covered = list(plan.get("covered_facets") or [])
            if facets and "covered_facets" in plan:
                facet_coverages.append(
                    len(set(facets).intersection(covered)) / len(set(facets))
                )
    return {
        "evidence_recall_at_n": _mean(evidence_recalls),
        "evidence_precision_at_n": _mean(evidence_precisions),
        "facet_coverage": _mean(facet_coverages),
        "empty_retrieval_rate": _mean(int(value == 0) for value in counts),
        "average_rounds": _mean(rounds),
        "average_graph_hops": _mean(hops),
        "router_calls": router_calls,
        "router_failures": router_failures,
        "router_failure_rate": router_failures / router_calls if router_calls else 0.0,
        "retrieval_calls": retrieval_calls,
        "retrieval_latency_ms_p50": _percentile(retrieval_latencies, 50),
        "retrieval_latency_ms_p95": _percentile(retrieval_latencies, 95),
        # Retained as an alias for older result-analysis notebooks.
        "evidence_coverage": _mean(evidence_recalls),
        "retrieval_count_mean": _mean(counts),
        "retrieval_count_max": max(counts, default=0),
        "retrieval_count_at_limit_rate": _mean(
            int(value == limit) for value, limit in zip(counts, limits)
        ),
        "relative_time_answer_residual_rate": residual / len(traces) if traces else 0.0,
        "end_to_end_latency_ms_p50": _percentile(end_to_end_latencies, 50),
        "end_to_end_latency_ms_p95": _percentile(end_to_end_latencies, 95),
    }


def _trace_metric_int(value: Any) -> int | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _trace_metric_float(value: Any) -> float | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _trace_final_top_k(row: dict[str, Any]) -> int:
    for trace in row.get("trace", []):
        plan = trace.get("query_plan") or {}
        for key in ("final_top_k", "max_final_top_k"):
            value = plan.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return 32


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _validate_full_locomo_outputs(
    items: list[dict[str, Any]],
    max_questions: int | None,
    predictions: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    *,
    protocol: str = "research",
    categories: Sequence[str] | None = None,
    example_indices: Sequence[int] | None = None,
) -> None:
    if categories is not None or example_indices is not None:
        expected_keys = {
            (job.example_index, job.qa_index)
            for job in _collect_locomo_answer_jobs(items, set(), max_questions, categories, example_indices)
        }
        for label, rows in (("predictions", predictions), ("traces", traces)):
            keys = {_locomo_row_key(row) for row in rows}
            if keys != expected_keys or len(rows) != len(expected_keys):
                raise RuntimeError(f"Selected LoCoMo {label} do not match requested questions")
        return
    if protocol != "canonical_mem0":
        return
    if max_questions is not None:
        return
    prediction_keys = {_locomo_row_key(row) for row in predictions}
    trace_keys = {_locomo_row_key(row) for row in traces}
    if len(prediction_keys) != len(predictions) or len(trace_keys) != len(traces):
        raise RuntimeError(
            "Full LoCoMo acceptance requires unique prediction and trace rows"
        )
    if any(
        int(row.get("retrieved_count", len(row.get("episodic_memories", [])))) > 32
        for row in traces
    ):
        raise RuntimeError(
            "Full LoCoMo acceptance requires at most 32 retrieved memories per question"
        )
    expected = sum(
        1
        for example in items
        for qa in _extract_locomo_qas(example)
        if str(qa.get("category") or qa.get("type") or "unknown") != "5"
    )
    if len(predictions) != expected or len(traces) != expected:
        raise RuntimeError(
            f"Full LoCoMo acceptance expected {expected} predictions/traces, got {len(predictions)}/{len(traces)}"
        )
    if len(items) == 10 and expected != 1540:
        raise RuntimeError(
            f"Expected canonical full LoCoMo to contain 1540 answerable questions, found {expected}"
        )


def _is_mem0_locomo_answer(row: dict[str, Any]) -> bool:
    if str(row.get("category")) == "5":
        return False
    required = ("example_index", "qa_index", "question", "answer", "response")
    return not any(key not in row for key in required)


def _dedupe_locomo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        if "example_index" not in row or "qa_index" not in row:
            continue
        key = (int(row["example_index"]), int(row["qa_index"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _group_metric(
    rows: list[dict[str, Any]], group_key: str, metric_keys: tuple[str, ...]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, "unknown"))].append(row)
    result: dict[str, Any] = {}
    for key, values in grouped.items():
        result[key] = {"count": len(values)}
        for metric_key in metric_keys:
            result[key][metric_key] = _mean(value[metric_key] for value in values)
    return result


def _mean(values) -> float:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _std(values) -> float:
    values = [float(value) for value in values if value is not None]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def _row_llm_scores(row: dict[str, Any]) -> list[float]:
    scores = row.get("llm_scores")
    if isinstance(scores, list):
        return [float(score) for score in scores if score is not None]
    if row.get("llm_score_mean") is not None:
        return [float(row["llm_score_mean"])]
    if row.get("llm_score") is not None:
        return [float(row["llm_score"])]
    return []


def _row_llm_mean(row: dict[str, Any]) -> float | None:
    scores = _row_llm_scores(row)
    if scores:
        return _mean(scores)
    return None


def _judge_run_std(rows: list[dict[str, Any]]) -> float:
    max_runs = max((len(_row_llm_scores(row)) for row in rows), default=0)
    run_means = []
    for run_index in range(max_runs):
        run_values = []
        for row in rows:
            scores = _row_llm_scores(row)
            if run_index < len(scores):
                run_values.append(scores[run_index])
        if run_values:
            run_means.append(_mean(run_values))
    return _std(run_means)


def _extract_locomo_conversation(example: dict[str, Any]) -> list[dict[str, Any]]:
    from evaluation.datasets import normalize_locomo_conversation

    return normalize_locomo_conversation(example)


def _extract_locomo_qas(example: dict[str, Any]) -> list[dict[str, Any]]:
    from evaluation.datasets import normalize_locomo_qas

    return normalize_locomo_qas(example)


def _normalize_message_list(messages: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            message = {"content": str(message)}
        normalized.append(
            {
                "session_id": message.get(
                    "session_id", message.get("session", "default")
                ),
                "turn_id": message.get("turn_id", message.get("turn", index + 1)),
                "role": message.get("role", message.get("speaker", "unknown")),
                "content": message.get(
                    "content", message.get("text", message.get("message", ""))
                ),
                "timestamp": message.get("timestamp", message.get("time")),
                "mentioned_entities": message.get(
                    "mentioned_entities", message.get("entities", [])
                ),
            }
        )
    return normalized


def _normalize_locomo_session_dict(
    conversation: dict[str, Any],
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    session_keys = [
        key
        for key, value in conversation.items()
        if re.fullmatch(r"session_\d+", str(key)) and isinstance(value, list)
    ]
    for session_key in sorted(session_keys, key=_session_sort_key):
        session_time = conversation.get(f"{session_key}_date_time")
        for index, message in enumerate(conversation.get(session_key, [])):
            if not isinstance(message, dict):
                message = {"text": str(message)}
            turns.append(
                _normalize_locomo_turn(message, session_key, index + 1, session_time)
            )
    return turns


def _session_sort_key(key: str) -> int:
    match = re.search(r"(\d+)$", key)
    return int(match.group(1)) if match else 0


def _normalize_locomo_turn(
    message: dict[str, Any],
    session_id: str,
    turn_index: int,
    session_time: Any,
) -> dict[str, Any]:
    content_parts = [
        str(
            message.get("content")
            or message.get("text")
            or message.get("message")
            or ""
        )
    ]
    if message.get("blip_caption"):
        content_parts.append(f"Image caption: {message['blip_caption']}")
    if message.get("query"):
        content_parts.append(f"Image query: {message['query']}")
    content = " ".join(part.strip() for part in content_parts if part and part.strip())
    return {
        "session_id": session_id,
        "turn_id": message.get("turn_id", message.get("dia_id", turn_index)),
        "role": message.get("role", message.get("speaker", "unknown")),
        "content": content,
        "timestamp": message.get("timestamp", message.get("time", session_time)),
        "mentioned_entities": message.get(
            "mentioned_entities", message.get("entities", [])
        ),
        "source": {
            "dia_id": message.get("dia_id"),
            "img_url": message.get("img_url"),
        },
    }
