from __future__ import annotations

import re

# Matches an ISO datetime whose time-of-day is exactly midnight, e.g.
# "2024-04-01T00:00:00" or "2024-04-01T00:00:00Z". Such values are almost
# always a date mis-serialized as a datetime; display them as a bare date.
_MIDNIGHT_RE = re.compile(r"T00:00:00(?:Z|[+-]\d{2}:?\d{2})?$")


def _endpoint(value: str) -> str:
    return value[:10] if len(value) >= 10 else value


def format_time_display(
    start: str | None, end: str | None, precision: str | None
) -> str | None:
    """Canonical display string for an absolute-time interval.

    Shared by ``TimeNormalization.display``, the builder, and the agent's
    answer-correction layer so display behavior cannot drift between them.
    For ``datetime`` precision, a midnight time-of-day is collapsed to a date
    to avoid injecting spurious ``00:00:00`` into answers.
    """
    if not start:
        return None
    start_text = str(start)
    if precision == "year":
        return start_text[:4]
    if precision == "month":
        return start_text[:7]
    if precision == "date":
        return start_text[:10]
    if precision == "range" and end:
        return f"{_endpoint(start_text)} to {_endpoint(str(end))}"
    if precision == "datetime" and _MIDNIGHT_RE.search(start_text):
        return start_text[:10]
    return start_text
