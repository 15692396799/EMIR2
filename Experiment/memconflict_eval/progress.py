"""Live progress for the long phases of the experiment.

The runner already writes one ``sessions.jsonl`` row per finished session, but
reading that file to see how far a four-persona run has come is awkward, and a
single session can take several minutes. This module renders one compact
progress line instead:

* on a terminal the same line is rewritten in place (sessions done/total,
  percentage, elapsed, ETA, questions, errors, how many personas are done and
  running, and which session finished last);
* in a redirected log it prints that line every ``log_interval`` seconds, so
  ``run_scale1h.bat > console.log 2>&1`` stays readable.

Nothing here touches the network or the memory system, and the counters are
updated from the parent process only (the personas' ``sessions.jsonl`` rows
already travel back to it), so the report is safe while workers run in child
processes.

The class is deliberately dependency-free: ``tqdm`` would render nothing into a
redirected file by default, and a long run is exactly when the log matters.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import IO, Iterable, Mapping, Sequence

#: Filled / empty cells of the bar. ASCII on purpose: a Windows console in a
#: non-UTF-8 code page must not turn the bar into mojibake.
BAR_FILLED = "#"
BAR_EMPTY = "-"

DEFAULT_INTERVAL = 15.0
DEFAULT_LOG_INTERVAL = 60.0
DEFAULT_HEARTBEAT = 30.0
DEFAULT_BAR_WIDTH = 20


def format_duration(seconds: float | None) -> str:
    """``3661`` -> ``1h01m``, ``95`` -> ``1m35s``, unknown -> ``--``."""
    if seconds is None:
        return "--"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "--"
    if value != value or value < 0:  # NaN or negative
        return "--"
    total = int(round(value))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class ProgressReporter:
    """One progress line for a countable phase (sessions, personas, ...).

    Counters are free-form: ``advance(1, questions=2)`` moves the main counter
    by one and adds two questions. ``totals`` gives a denominator for a counter
    so it renders as ``questions 9/46`` instead of ``questions 9``; ``note()``
    sets free-form text such as ``last 8cb3c9c6 s3``.
    """

    def __init__(
        self,
        total: int,
        *,
        label: str = "items",
        counters: Mapping[str, int] | None = None,
        totals: Mapping[str, int] | None = None,
        stream: IO[str] | None = None,
        enabled: bool = True,
        tty: bool | None = None,
        interval: float = DEFAULT_INTERVAL,
        log_interval: float = DEFAULT_LOG_INTERVAL,
        heartbeat: float | None = DEFAULT_HEARTBEAT,
        bar_width: int = DEFAULT_BAR_WIDTH,
        clock=time.monotonic,
    ) -> None:
        self.total = max(0, int(total))
        self.label = str(label)
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = bool(enabled) and self.total > 0
        self.tty = _is_tty(self.stream) if tty is None else bool(tty)
        self.interval = max(0.0, float(interval))
        self.log_interval = max(0.0, float(log_interval))
        self.heartbeat = None if heartbeat is None else max(0.0, float(heartbeat))
        self.bar_width = max(0, int(bar_width))
        self._clock = clock

        self._counters: dict[str, int] = {str(k): int(v) for k, v in (counters or {}).items()}
        self._totals: dict[str, int] = {str(k): int(v) for k, v in (totals or {}).items()}
        self._notes: dict[str, str] = {}

        self._done = 0
        self._started_at: float | None = None
        self._last_emit = 0.0
        self._rendered_length = 0
        self._finished = False
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- state -------------------------------------------------------------

    @property
    def done(self) -> int:
        return self._done

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def elapsed(self) -> float:
        with self._lock:
            if self._started_at is None:
                return 0.0
            return max(0.0, self._clock() - self._started_at)

    def eta(self) -> float | None:
        """Seconds left at the average rate so far, or ``None`` before that."""
        elapsed = self.elapsed()
        with self._lock:
            done, total = self._done, self.total
        if done <= 0 or elapsed <= 0 or total <= done:
            return None
        rate = done / elapsed
        if rate <= 0:
            return None
        return (total - done) / rate

    def percent(self) -> float:
        with self._lock:
            if self.total <= 0:
                return 100.0
            return 100.0 * self._done / self.total

    # -- public API --------------------------------------------------------

    def start(self) -> "ProgressReporter":
        if not self.enabled:
            return self
        with self._lock:
            if self._started_at is None:
                self._started_at = self._clock()
        self._emit(force=True)
        self._start_heartbeat()
        return self

    def advance(self, count: int = 1, **increments: int) -> None:
        """Move the main counter and any named counter forward."""
        with self._lock:
            self._done = min(self.total, max(0, self._done + int(count)))
            for key, value in increments.items():
                self._counters[str(key)] = self._counters.get(str(key), 0) + int(value)
        self._emit()

    def note(self, **fields: str) -> None:
        """Set free-form trailing text (``last``, ``running``, ...)."""
        with self._lock:
            for key, value in fields.items():
                self._notes[str(key)] = str(value)
        self._emit()

    def log(self, text: str = "") -> None:
        """Print an ordinary line without trampling the in-place progress line.

        On a terminal the progress line occupies the current row, so a plain
        ``print`` would land on top of it and be erased by the next refresh.
        This clears the row, prints, and lets the next refresh redraw below.
        """
        if not self.enabled or not self.tty:
            self._write(text + "\n")
            return
        with self._lock:
            padding = " " * self._rendered_length
            self._rendered_length = 0
            self._write("\r" + padding + "\r" + text + "\n")
        self._emit(force=True)

    def finish(self) -> None:
        """Stop the heartbeat and print the closing line once."""
        if not self.enabled or self._finished:
            return
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._emit(force=True)
        if self.tty:
            # Leave the cursor (and the next log line) on a fresh row.
            self._write("\n")
        self._finished = True

    def __enter__(self) -> "ProgressReporter":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.finish()

    # -- rendering ---------------------------------------------------------

    def bar(self) -> str:
        if self.bar_width <= 0:
            return ""
        with self._lock:
            ratio = 1.0 if self.total <= 0 else self._done / self.total
        filled = int(round(ratio * self.bar_width))
        filled = max(0, min(self.bar_width, filled))
        return "[" + BAR_FILLED * filled + BAR_EMPTY * (self.bar_width - filled) + "]"

    #: Drop priorities for a narrow terminal: the lowest number goes first, so
    #: "last session" disappears long before the session counter or the ETA.
    #: A counter that is still zero has no information yet and drops early.
    _PRIORITY = {
        "core": 100,
        "questions": 8,
        "personas": 8,
        "eta": 7,
        "elapsed": 6,
        "errors": 5,
        "failed": 4,
        "running": 3,
        "last": 2,
    }
    #: Extra demotion for a zero-valued counter.
    _ZERO_DEMOTION = 3

    def _segments(self) -> list[tuple[str, str, int]]:
        with self._lock:
            done = self._done
            counters = dict(self._counters)
            notes = dict(self._notes)
        head = f"{done}/{self.total} {self.label} {self.percent():.0f}%"
        bar = self.bar()
        pieces: list[tuple[str, str, int]] = [
            ("core", f"{bar} {head}".strip(), self._PRIORITY["core"])
        ]
        pieces.append(
            ("elapsed", f"{format_duration(self.elapsed())} elapsed", self._PRIORITY["elapsed"])
        )
        eta = self.eta()
        if eta is not None:
            pieces.append(("eta", f"eta {format_duration(eta)}", self._PRIORITY["eta"]))
        for key, value in counters.items():
            total = self._totals.get(key)
            text = f"{key} {value}/{total}" if total is not None else f"{key} {value}"
            priority = self._PRIORITY.get(key, 5)
            if value == 0:
                priority -= self._ZERO_DEMOTION
            pieces.append((key, text, priority))
        for key, value in notes.items():
            pieces.append((key, f"{key} {value}", self._PRIORITY.get(key, 4)))
        return pieces

    def segments(self) -> list[tuple[str, str]]:
        """Ordered ``(key, text)`` pieces of the line, before any fitting."""
        return [(key, text) for key, text, _priority in self._segments()]

    def parts(self) -> list[str]:
        return [text for _key, text in self.segments()]

    def render(self, width: int | None = None) -> str:
        """The plain one-line report, without carriage returns.

        With ``width`` the line sheds its least important pieces until it fits,
        instead of being cut in the middle: a narrow terminal keeps the session
        counter, the ETA and the question count, and loses "last"/"running".
        """
        # The prefix makes the line greppable in a redirected console.log and
        # keeps it distinguishable from the per-persona lines next to it.
        segments = self._segments()
        core = [text for key, text, _priority in segments if key == "core"]
        optional = [(key, text, priority) for key, text, priority in segments if key != "core"]

        def line_for(items: list[tuple[str, str, int]]) -> str:
            return "[progress] " + " | ".join(
                core + [text for _key, text, _priority in items]
            )

        if width is not None and width > 0:
            while optional and len(line_for(optional)) > width:
                index = min(
                    range(len(optional)),
                    key=lambda position: (optional[position][2], -position),
                )
                optional = optional[:index] + optional[index + 1 :]
        return line_for(optional)

    # -- internals ---------------------------------------------------------

    def _emit(self, *, force: bool = False) -> None:
        if not self.enabled or self._finished:
            return
        with self._lock:
            now = self._clock()
            if self._started_at is None:
                self._started_at = now
            wait = self.interval if self.tty else self.log_interval
            if not force and (now - self._last_emit) < wait:
                return
            self._last_emit = now
        if not self.tty:
            self._write(self.render() + "\n")
            return
        text = self.render(width=max(1, _terminal_width() - 1))
        padding = " " * max(0, self._rendered_length - len(text))
        self._write("\r" + text + padding)
        self._rendered_length = len(text)

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except (ValueError, OSError):  # a closed stream must not kill the run
            pass

    def _start_heartbeat(self) -> None:
        if self.heartbeat is None or self.heartbeat <= 0:
            return
        if self._thread is not None:
            return

        def beat() -> None:
            while not self._stop.wait(self.heartbeat or 0.0):
                self._emit()

        self._thread = threading.Thread(target=beat, name="memconflict-progress", daemon=True)
        self._thread.start()


def expected_work(
    sessions: Sequence[object],
    *,
    max_sessions: int | None = None,
    skip_session_ids: Iterable[int] = (),
) -> tuple[int, int]:
    """``(sessions, questions)`` this invocation will really replay.

    ``--max-sessions`` truncates the chain and ``--resume`` skips the sessions
    that already have a row; the progress bar has to know both up front, or its
    percentage and ETA would describe work nobody is going to do.
    """
    window = list(sessions)
    if max_sessions is not None:
        window = window[: max(0, int(max_sessions))]
    skipped = {int(value) for value in skip_session_ids}
    kept = [session for session in window if getattr(session, "session_id", None) not in skipped]
    questions = sum(len(getattr(session, "questions", ()) or ()) for session in kept)
    return len(kept), questions


def _is_tty(stream: IO[str]) -> bool:
    checker = getattr(stream, "isatty", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except (ValueError, OSError):
        return False


def _terminal_width() -> int:
    try:
        import shutil

        return int(shutil.get_terminal_size((120, 20)).columns)
    except Exception:  # noqa: BLE001 - a missing size must not break the run
        return 120


__all__ = ["ProgressReporter", "expected_work", "format_duration"]
