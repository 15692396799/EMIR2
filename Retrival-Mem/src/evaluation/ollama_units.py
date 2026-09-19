"""Optional Ollama inference units for LoCoMo's existing question scheduler."""
from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from memory.ollama_routing import ollama_routes


def load_ollama_units(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"units"}:
        raise ValueError("Ollama unit config requires only a 'units' list")
    validate_units(payload["units"])
    return payload


def validate_units(units):
    if not isinstance(units, list) or not units:
        raise ValueError("Ollama units must be a nonempty list")
    names = set()
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("Each Ollama unit must be a mapping")
        name = unit.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Ollama unit names must be nonempty and unique")
        names.add(name)
        mode = unit.get("mode", "single")
        if mode not in {"single", "dual"}:
            raise ValueError(f"Unit {name}: mode must be single or dual")
        allowed = {"name", "mode", "workers"} | (
            {"base_url_env"} if mode == "single" else {"chat_base_url_env", "embedding_base_url_env"}
        )
        if set(unit) - allowed:
            raise ValueError(f"Unit {name}: unsupported options {sorted(set(unit) - allowed)}")
        workers = unit.get("workers", 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError(f"Unit {name}: workers must be a positive integer")
        if mode == "dual" and not all(unit.get(k) for k in ("chat_base_url_env", "embedding_base_url_env")):
            raise ValueError(f"Unit {name}: dual mode requires chat and embedding base URL environment variables")
        for key in allowed - {"name", "mode", "workers"}:
            if key in unit and (not isinstance(unit[key], str) or not unit[key].strip()):
                raise ValueError(f"Unit {name}: {key} must be an environment variable name")


def _base(env: str) -> str:
    base = os.environ.get(env, "").strip().rstrip("/")
    url = urlsplit(base)
    if url.scheme not in {"http", "https"} or not url.hostname or url.path or url.query or url.fragment or url.username or url.password:
        raise ValueError(f"{env} must contain an HTTP(S) service base URL without credentials or /api paths")
    return base


def resolve_units(units):
    validate_units(units)
    result = []
    for unit in units:
        routes = {}
        if unit.get("mode", "single") == "dual":
            chat = _base(unit["chat_base_url_env"])
            embed = _base(unit["embedding_base_url_env"])
        elif unit.get("base_url_env"):
            chat = embed = _base(unit["base_url_env"])
        else:
            # A single service without an override uses existing endpoint settings.
            result.append((unit["name"], unit.get("workers", 1), routes))
            continue
        routes = {"chat": chat + "/api/chat", "embed": embed + "/api/embed", "legacy_embed": embed + "/api/embeddings"}
        result.append((unit["name"], unit.get("workers", 1), routes))
    return result


def run_unit_jobs(units, jobs, run_job):
    """Run jobs under worker-local routes and refill the unit that finishes."""
    pending_jobs = iter(jobs)

    def invoke(unit, job):
        name, _, routes = unit
        with ollama_routes(routes):
            return run_job(name, job)

    with ThreadPoolExecutor(max_workers=sum(unit[1] for unit in units)) as pool:
        active = {}

        def submit(unit):
            job = next(pending_jobs, None)
            if job is not None:
                active[pool.submit(invoke, unit, job)] = unit

        for unit in units:
            for _ in range(unit[1]):
                submit(unit)
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                unit = active.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = error
                submit(unit)
                yield unit[0], result


def run_unit_chunks(units, chunks, answer_chunk):
    """Run answer chunks across units and annotate their prediction rows."""

    def run_chunk(_unit_name, chunk):
        return answer_chunk(chunk)

    for name, result in run_unit_jobs(units, chunks, run_chunk):
        if isinstance(result, Exception):
            yield result
            continue
        for record, trace in result.results:
            record["ollama_unit"] = name
            trace["ollama_unit"] = name
        yield result
