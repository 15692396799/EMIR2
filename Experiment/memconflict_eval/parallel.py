"""Point 17: run personas in parallel, one worker (and one GPU) per persona.

Personas are fully independent (separate memory store, namespace and output
directory), so the only thing that has to stay serial is the session chain
inside one persona. That makes the persona the natural unit of parallelism.

A single-worker run keeps the original in-process loop, so it behaves exactly
as it did before this module existed. With more workers the personas are spread
over child processes.

Progress rows travel over a manager queue so the parent can keep appending
sessions.jsonl while the run is still going: a run that dies half way through a
persona is still scoreable, because run_scoring.py falls back to that log.
"""

from __future__ import annotations

import traceback
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable

# Sentinel that tells the writer thread to stop draining the queue. It travels
# through a manager queue, which pickles every row, so it is compared by value
# instead of by identity.
_STOP = {"Event": "__memconflict_stop__"}


class JobError(RuntimeError):
    """A worker failure that is guaranteed to survive pickling.

    Retrival-Mem raises its own exception types (``V4StageError`` and friends)
    whose ``__init__`` needs more than the message, so the default
    ``BaseException`` pickling rebuilds them with the wrong arguments and the
    pool dies with ``BrokenProcessPool`` instead of reporting the stage that
    failed. Every worker exception is therefore converted into this type with
    the original text and traceback attached.
    """

    def __init__(self, message: str, *, traceback_text: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.traceback_text = traceback_text

    @property
    def summary(self) -> str:
        return self.message


class _InlineSink:
    """Stand-in for the progress queue when everything runs in one process."""

    def __init__(self, on_row: Callable[[dict[str, Any]], None] | None) -> None:
        self._on_row = on_row

    def put(self, row: dict[str, Any]) -> None:
        if self._on_row is not None:
            self._on_row(row)


def _execute(worker: Callable[[dict[str, Any]], Any], payload: dict[str, Any]) -> Any:
    """Run one job, turning any failure into a picklable :class:`JobError`."""
    try:
        return worker(payload)
    except BaseException as error:  # noqa: BLE001 - reported to the parent
        raise JobError(
            f"{type(error).__name__}: {error}",
            traceback_text=traceback.format_exc(),
        ) from None


def run_jobs(
    worker: Callable[[dict[str, Any]], Any],
    jobs: Iterable[dict[str, Any]],
    *,
    workers: int = 1,
    on_row: Callable[[dict[str, Any]], None] | None = None,
    on_result: Callable[[int, Any], None] | None = None,
    raise_errors: bool = True,
) -> list[Any]:
    """Run ``worker(job)`` for every job, at most ``workers`` at a time.

    ``worker`` must be picklable (a module-level function) because the
    multi-worker path ships it to child processes. Every job receives an extra
    ``progress`` entry: an object with ``put(row)``, which the worker uses to
    push progress rows back to the parent's ``on_row``.

    Results come back in input order. ``on_result(index, value)`` is called in
    the parent process as soon as a job finishes, in whatever order that is.
    With ``raise_errors=False`` a failed job yields a :class:`JobError` in the
    result list instead of stopping the whole run, so one bad persona cannot
    throw away the work of the others.
    """
    payloads = [dict(job) for job in jobs]
    results: list[Any] = [None] * len(payloads)
    if not payloads:
        return results

    def finish(index: int, value: Any) -> None:
        if isinstance(value, JobError) and raise_errors:
            raise value
        results[index] = value
        if on_result is not None:
            on_result(index, value)

    if int(workers) <= 1:
        sink = _InlineSink(on_row)
        for index, payload in enumerate(payloads):
            payload["progress"] = sink
            finish(index, _execute(worker, payload))
        return results

    from multiprocessing import Manager

    manager = Manager()
    queue = manager.Queue()
    writer_error: list[BaseException] = []

    def drain() -> None:
        while True:
            row = queue.get()
            if row == _STOP:
                return
            if on_row is None:
                continue
            try:
                on_row(row)
            except BaseException as error:  # noqa: BLE001 - re-raised in the parent
                writer_error.append(error)
                return

    writer = threading.Thread(target=drain, name="memconflict-progress", daemon=True)
    writer.start()
    try:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {}
            for index, payload in enumerate(payloads):
                payload["progress"] = queue
                futures[pool.submit(_execute, worker, payload)] = index
            for future, index in list(futures.items()):
                error = future.exception()
                if error is None:
                    finish(index, future.result())
                    continue
                if not isinstance(error, JobError):
                    error = JobError(f"{type(error).__name__}: {error}")
                finish(index, error)
    finally:
        queue.put(_STOP)
        writer.join(timeout=30)
    manager.shutdown()
    if writer_error:
        raise writer_error[0]
    return results


__all__ = ["JobError", "run_jobs"]
