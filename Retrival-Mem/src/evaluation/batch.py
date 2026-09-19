"""Persist asynchronous batch jobs so answer/judge stages can resume."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from api_history import LoggedBatchChatClient
from memory.clients import ChatBatchRequest, ChatBatchResult, make_batch_chat_client
from memory.config import ModelConfig


class BatchPendingError(RuntimeError):
    """The remote job is still running; resume this run to collect its results."""


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_chat_batch(model: ModelConfig, requests_: list[ChatBatchRequest], directory: Path,
                   prefix: str, options: dict[str, Any], logger=None,
                   *, force: bool = False) -> list[ChatBatchResult]:
    if not requests_:
        return []
    client = make_batch_chat_client(model)
    if logger is not None:
        client = LoggedBatchChatClient(client, logger, prefix, model.provider, model.model)
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / f"{prefix}_input.jsonl"
    meta_path = directory / f"{prefix}_meta.json"
    output_path = directory / f"{prefix}_output.jsonl"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() and not force else None
    expected = {request.custom_id for request in requests_}
    if len(expected) != len(requests_):
        raise ValueError("Duplicate batch request IDs")
    if meta is not None:
        if (meta["model"] != model.model or meta["provider"] != model.provider
                or not expected.issubset(set(meta["request_ids"]))):
            raise ValueError(f"{prefix} already contains another job; use a new run or regenerate stage")
    if meta is not None and meta.get("results_saved") and output_path.exists():
        lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    else:
        if meta is None:
            client.write_chat_batch_input(input_path, requests_)
            # Do not automatically retry submission: a lost response may still create a paid job.
            job = client.submit_chat_batch(input_path, str(options.get("batch_completion_window", "24h")))
            meta = {"provider": model.provider, "model": model.model,
                    "batch_id": job.id, "request_ids": sorted(expected)}
        else:
            job = client.get_batch(meta["batch_id"])
        started = time.monotonic()
        timeout = float(options.get("batch_timeout_seconds", 90000))
        interval = max(1.0, min(60.0, float(options.get("batch_poll_interval_seconds", 30))))
        terminal = {"completed", "failed", "expired", "cancelled"}
        while True:
            meta.update(status=job.status, raw={key: value for key, value in job.raw.items()
                                              if key != "results"})
            _save(meta_path, meta)
            if job.status in terminal:
                break
            if not options.get("batch_wait", True) or time.monotonic() - started >= timeout:
                raise BatchPendingError(f"{prefix} {job.id} is {job.status}; resume the same run later")
            time.sleep(min(interval, max(0.0, timeout - (time.monotonic() - started))))
            for attempt in range(3):
                try:
                    job = client.get_batch(job.id)
                    break
                except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as error:
                    status = getattr(getattr(error, "response", None), "status_code", None)
                    if attempt == 2 or (status is not None and status != 429 and status < 500):
                        raise
                    time.sleep(2 ** attempt)
        if job.status != "completed":
            raise RuntimeError(f"{prefix} {job.id} ended with {job.status}; see {meta_path}")
        if job.results is not None:
            lines = job.results
        elif job.output_file_id:
            lines = client.download_batch_file(job.output_file_id)
        else:
            raise RuntimeError(f"{prefix} {job.id} completed without results")
        temporary = output_path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
                             encoding="utf-8")
        temporary.replace(output_path)
        meta["results_saved"] = True
        _save(meta_path, meta)
        errors = [line for line in lines if line.get("error")
                  or (line.get("response") or {}).get("status_code", 200) >= 300]
        if job.error_file_id:
            errors.extend(client.download_batch_file(job.error_file_id))
        if errors:
            _save(directory / f"{prefix}_errors.json", errors)
    results = client.parse_chat_batch_results(lines)
    selected = [result for result in results if result.custom_id in expected]
    ids = [result.custom_id for result in selected]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise RuntimeError(f"{prefix} returned missing or duplicate results; see {output_path}")
    failed = [result.custom_id for result in selected if result.error or not result.content]
    if failed:
        raise RuntimeError(f"{prefix} has {len(failed)} failed requests; see {output_path}")
    return selected
