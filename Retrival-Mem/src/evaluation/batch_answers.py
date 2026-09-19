"""LoCoMo retrieval, asynchronous compact answers, then normal scoring."""
from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent.agent import LightweightMemoryAgent
from agent.answer_context import prepare_answer_context
from agent.compact_answer import compact_messages, finish_compact_answer
from agent.evidence_answer import unpack_answer_output
from agent.temporal_answer import is_structured_duration_question
from evaluation.batch import run_chat_batch
from evaluation.checkpoints import QuestionStageLedger
from memory import MemorySystem
from memory.clients import ChatBatchRequest, NoopChatClient


def _prepare_chunk(config, jobs, ledger_path, logger, regeneration_token):
    from evaluation.runner import _bundle_from_checkpoint, _run_question_stage

    config = copy.deepcopy(config)
    options = config.evaluation.locomo
    memory = MemorySystem(config, api_history_logger=logger, store_read_only=True)
    try:
        agent = LightweightMemoryAgent(config, memory_system=memory,
                                       answer_client=NoopChatClient())
        prepared = []
        with QuestionStageLedger(ledger_path) as ledger:
            for job in jobs:
                started = time.perf_counter()
                key = f"locomo:{job.example_index}:{job.qa_index}"
                frozen = bool(options.get("answer_only_frozen_retrieval", False))
                if regeneration_token:
                    ledger.invalidate_question(key, keep_retrieval=frozen)

                def retrieve():
                    bundle, context = agent.retrieve_for_answer(
                        job.question, job.namespace,
                        metadata={
                            "retrieval_prompt_profile": "locomo_open_ended_v1" if job.category == "3" else None,
                            "rerank_checkpoint_path": str(ledger_path.parent / "rerank_checkpoints.sqlite3"),
                            "rerank_checkpoint_key": key,
                            "rerank_reset": bool(regeneration_token),
                        },
                    )
                    return {"retrieval": bundle.to_dict(), "memory_context": context}

                if frozen:
                    output = ledger.load_success(key, "retrieval")
                    if output is None:
                        raise ValueError(f"Frozen retrieval checkpoint missing for {key}")
                    if (output["retrieval"].get("question") != job.question
                            or output["retrieval"].get("namespace") != job.namespace):
                        raise ValueError(f"Frozen retrieval identity mismatch for {key}")
                else:
                    output = _run_question_stage(ledger, key, "retrieval", retrieve)
                bundle = _bundle_from_checkpoint(output["retrieval"])
                context = prepare_answer_context(bundle, str(output.get("memory_context") or ""), options)
                messages, context = agent.prepare_answer_messages(job.question, context, job.category)
                messages = compact_messages(
                    messages, duration=is_structured_duration_question(job.question),
                    temporal=str(job.category) == "2", single_hop=str(job.category) == "4",
                )
                prepared.append((job, key, output["retrieval"], context, messages,
                                 (time.perf_counter() - started) * 1000))
        return prepared
    finally:
        memory.close()


def run_locomo_batch_answers(config, jobs, workers: int, chunk_size: int,
                             ledger_path: Path, logger=None, *, regeneration_token=None):
    from evaluation.runner import _bundle_from_checkpoint, _run_question_stage

    if not jobs:
        return
    options = config.evaluation.locomo
    if not options.get("answer_evidence_first") or not options.get("answer_compact_json", True):
        raise ValueError("Batch answering requires answer_evidence_first and answer_compact_json")
    chunks = [jobs[start:start + max(1, chunk_size)]
              for start in range(0, len(jobs), max(1, chunk_size))]

    def prepare(chunk):
        return _prepare_chunk(config, chunk, ledger_path, logger, regeneration_token)

    prepared = []
    units_by_key = {}
    unit_config = options.get("ollama_units")
    if unit_config is not None:
        from evaluation.ollama_units import resolve_units, run_unit_jobs
        for name, result in run_unit_jobs(resolve_units(unit_config), chunks,
                                          lambda _name, chunk: prepare(chunk)):
            if isinstance(result, Exception):
                raise result
            units_by_key.update({item[1]: name for item in result})
            prepared.extend(result)
    else:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for result in executor.map(prepare, chunks):
                prepared.extend(result)
    prepared.sort(key=lambda item: (item[0].example_index, item[0].qa_index))
    with QuestionStageLedger(ledger_path) as ledger:
        pending = [ChatBatchRequest(key, messages, json_mode=True)
                   for _, key, _, _, messages, _ in prepared
                   if ledger.load_success(key, "answer") is None]
        started = time.perf_counter()
        results = run_chat_batch(
            config.answer_model, pending, ledger_path.parent, "answer_batch", options, logger,
            force=bool(regeneration_token),
        )
        by_id = {result.custom_id: result for result in results}
        batch_ms = (time.perf_counter() - started) * 1000
        # No memory operations are needed after retrieval; avoid reopening stores.
        agent = LightweightMemoryAgent(config, memory_system=object(), answer_client=NoopChatClient())
        for job, key, retrieval, context, _, retrieval_ms in prepared:
            bundle = _bundle_from_checkpoint(retrieval)

            def answer():
                raw = by_id[key].content
                value = finish_compact_answer(raw, context, job.question)
                response, fields = unpack_answer_output(value, options)
                return {"response": response, "answer_failed": False, **fields}

            output = _run_question_stage(ledger, key, "answer", answer)

            def correct():
                response, correction = agent.correct_answer(output["response"], bundle, job.question)
                return {"response": response, "temporal_correction": correction}

            corrected = _run_question_stage(ledger, key, "temporal_correction", correct)
            latency = retrieval_ms + (batch_ms if key in by_id else 0)
            record = {
                "example_index": job.example_index, "qa_index": job.qa_index,
                "question": job.question, "answer": job.reference, "reference": job.reference,
                "response": corrected["response"], "generated_answer": corrected["response"],
                "category": job.category, "latency_ms": latency, "answer_failed": False,
                "temporal_correction": corrected["temporal_correction"],
                **{name: output[name] for name in ("answer_output", "raw_answer_output") if name in output},
            }
            trace = {
                "example_index": job.example_index, "qa_index": job.qa_index,
                "gold_evidence": list(job.gold_evidence), "trace": retrieval.get("trace", []),
                "episodic_memories": retrieval.get("episodic_memories", []),
                "prompt_context": context, "retrieved_count": len(retrieval.get("episodic_memories", [])),
                "temporal_correction": corrected["temporal_correction"],
                "answer_failed": False, "latency_ms": latency,
            }
            if key in units_by_key:
                record["ollama_unit"] = units_by_key[key]
                trace["ollama_unit"] = units_by_key[key]
            yield record, trace
