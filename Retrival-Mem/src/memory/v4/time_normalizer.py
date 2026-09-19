from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Mapping

from memory.time_utils import format_time_display  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class TimeNormalization:
    raw_time_expression: str | None
    time_anchor: str | None
    absolute_time_start: str | None
    absolute_time_end: str | None
    time_precision: str | None
    normalization_status: str
    normalization_method: str | None

    def display(self) -> str | None:
        return format_time_display(self.absolute_time_start, self.absolute_time_end, self.time_precision)


class TimeNormalizer:
    """Deterministic normalizer; it never uses the wall clock as an anchor."""

    WEEKDAYS = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
    MONTHS.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if name})
    NUMBERS = {
        "a": 1,
        "an": 1,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    APPROXIMATE_QUANTIFIERS = {"few", "several", "couple", "a couple"}

    def normalize(
        self,
        expression: str | None,
        anchor: str | datetime | date | None,
        references: Mapping[str, str | datetime | date] | None = None,
    ) -> TimeNormalization:
        raw = str(expression).strip() if expression is not None else ""
        anchor_dt = _parse_datetime(anchor)
        anchor_text = _anchor_text(anchor_dt) if anchor_dt else (str(anchor) if anchor else None)
        if not raw:
            return self._unresolved(None, anchor_text, "missing_expression")

        absolute = self._absolute(raw, anchor_text)
        if absolute:
            return absolute

        lowered = re.sub(r"\s+", " ", raw.lower()).strip(" .,;:")
        reference_anchors = self._reference_anchors(lowered, references)
        if len(reference_anchors) > 1:
            return self._unresolved(raw, anchor_text, "ambiguous_reference_anchor")
        if len(reference_anchors) == 1:
            anchor_dt = reference_anchors[0]
            anchor_text = _anchor_text(reference_anchors[0])
        weekday_reference = self._weekday_before_after_date(raw, lowered, anchor_text)
        if weekday_reference:
            return weekday_reference
        if anchor_dt is None:
            return self._unresolved(raw, anchor_text, "missing_anchor")

        relative = self._relative(raw, lowered, anchor_dt, anchor_text)
        if relative:
            return relative
        return self._unresolved(raw, anchor_text, "ambiguous_or_unsupported")

    # (pattern, priority) — lower priority value wins. Specific dates/ranges
    # outrank coarse years and bare weekdays so multi-expression windows keep
    # the most precise anchor rather than whichever phrase appears first.
    _EXTRACTION_PATTERNS: list[tuple[str, int]] = [
        (r"\b\d{4}-\d{1,2}-\d{1,2}(?:[T ][0-2]?\d:[0-5]\d(?::[0-5]\d)?)?\b", 0),
        (r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b", 0),
        (r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,?\s+\d{4})?\b", 0),
        (r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+(?:before|after)\s+\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4}\b", 1),
        (r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b", 1),
        (r"\b(?:last|next|this|previous|following)\s+(?:year|month|week|weekend|day|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", 1),
        (r"\b(?:yesterday|today|tomorrow)\b", 1),
        (r"\b(?:the\s+day\s+(?:before\s+yesterday|after\s+tomorrow))\b", 1),
        (r"\b(?:day\s+(?:before\s+yesterday|after\s+tomorrow))\b", 1),
        (r"\b(?:in\s+)?(?:(?:a\s+couple|couple|few|several|a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+)?(?:years?|months?|weeks?|days?)\s+(?:ago|before|after|later|from\s+now)\b", 1),
        (r"\bin\s+(?:(?:a\s+couple|couple|few|several|a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+)(?:years?|months?|weeks?|days?)\b", 1),
        (r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", 2),
        (r"\b(?:19|20)\d{2}\b", 2),
    ]

    def extract_expressions(self, text: str) -> list[str]:
        """Return every time expression in ``text``, most precise first."""
        spans: list[tuple[int, int, int, str]] = []  # (priority, start, end, matched)
        for pattern, priority in self._EXTRACTION_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                spans.append((priority, match.start(), match.end(), match.group(0)))
        # Drop any match whose span is a strict sub-span of another match
        # (e.g. the bare-year pattern matching "2024" inside "2024-03-01").
        filtered = [
            (priority, start, end, matched)
            for index, (priority, start, end, matched) in enumerate(spans)
            if not any(
                other_index != index and other_start <= start and end <= other_end
                and (other_start < start or end < other_end)
                for other_index, (_, other_start, other_end, _) in enumerate(spans)
            )
        ]
        # Deduplicate identical spans, then order by precision then position.
        seen: set[tuple[int, int]] = set()
        unique: list[tuple[int, int, str]] = []
        for priority, start, end, matched in sorted(filtered, key=lambda item: (item[0], item[1])):
            if (start, end) in seen:
                continue
            seen.add((start, end))
            unique.append((priority, start, matched))
        return [matched for _, _, matched in unique]

    def _absolute(self, raw: str, anchor_text: str | None) -> TimeNormalization | None:
        text = raw.strip()
        iso_range = re.fullmatch(
            r"(\d{4}-\d{1,2}-\d{1,2})\s+(?:to|through|until|-)\s+(\d{4}-\d{1,2}-\d{1,2})",
            text,
            flags=re.IGNORECASE,
        )
        if iso_range:
            start = _parse_datetime(iso_range.group(1))
            end = _parse_datetime(iso_range.group(2))
            if start and end and end >= start:
                return self._resolved(raw, anchor_text, start, end, "range", "absolute", "iso_range")

        iso_datetime = re.fullmatch(
            r"\d{4}-\d{1,2}-\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)",
            text,
        )
        if iso_datetime:
            parsed = _parse_datetime(text)
            if parsed:
                return self._resolved(raw, anchor_text, parsed, parsed, "datetime", "absolute", "iso_datetime")

        locomo_datetime = re.fullmatch(
            r"\d{1,2}:\d{2}\s*[ap]m\s+on\s+\d{1,2}(?:st|nd|rd|th)?\s+"
            r"[A-Za-z]+,?\s+\d{4}",
            text,
            flags=re.IGNORECASE,
        )
        if locomo_datetime:
            parsed = _parse_datetime(text)
            if parsed:
                return self._resolved(
                    raw, anchor_text, parsed, parsed, "datetime", "absolute", "locomo_datetime"
                )

        iso_date = re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", text)
        if iso_date:
            parsed = _parse_datetime(text)
            if parsed:
                return self._resolved(raw, anchor_text, parsed, parsed, "date", "absolute", "iso_date")

        year_month = re.fullmatch(r"(\d{4})-(\d{1,2})", text)
        if year_month:
            year, month = map(int, year_month.groups())
            if 1 <= month <= 12:
                start, end = _month_bounds(year, month)
                return self._resolved(raw, anchor_text, start, end, "month", "absolute", "iso_month")

        year = re.fullmatch(r"(?:in\s+)?((?:19|20)\d{2})", text, flags=re.IGNORECASE)
        if year:
            start, end = _year_bounds(int(year.group(1)))
            return self._resolved(raw, anchor_text, start, end, "year", "absolute", "year")

        named = re.fullmatch(
            r"(?:on\s+)?([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?",
            text,
            flags=re.IGNORECASE,
        )
        if named:
            month = self.MONTHS.get(named.group(1).lower())
            year_value = int(named.group(3)) if named.group(3) else None
            if month and year_value:
                try:
                    value = datetime(year_value, month, int(named.group(2)))
                except ValueError:
                    return None
                return self._resolved(raw, anchor_text, value, value, "date", "absolute", "named_date")

        day_named = re.fullmatch(
            r"(?:on\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})",
            text,
            flags=re.IGNORECASE,
        )
        if day_named:
            month = self.MONTHS.get(day_named.group(2).lower())
            if month:
                try:
                    value = datetime(int(day_named.group(3)), month, int(day_named.group(1)))
                except ValueError:
                    return None
                return self._resolved(raw, anchor_text, value, value, "date", "absolute", "named_date_dmy")

        named_month = re.fullmatch(r"(?:in\s+)?([A-Za-z]+)\s+(\d{4})", text, flags=re.IGNORECASE)
        if named_month:
            month = self.MONTHS.get(named_month.group(1).lower())
            if month:
                start, end = _month_bounds(int(named_month.group(2)), month)
                return self._resolved(raw, anchor_text, start, end, "month", "absolute", "named_month")
        return None

    def _relative(
        self,
        raw: str,
        lowered: str,
        anchor: datetime,
        anchor_text: str,
    ) -> TimeNormalization | None:
        # "the day before yesterday" / "day after tomorrow" — two-day offsets.
        day_offset = re.fullmatch(
            r"(?:the\s+)?day\s+(before\s+yesterday|after\s+tomorrow)", lowered
        )
        if day_offset:
            amount = -2 if "before" in day_offset.group(1) else 2
            method = "relative_day_before_yesterday" if amount < 0 else "relative_day_after_tomorrow"
            return self._offset_result(raw, anchor_text, anchor, amount, "day", method)

        if lowered in {"yesterday", "today", "tomorrow"}:
            amount = {"yesterday": -1, "today": 0, "tomorrow": 1}[lowered]
            return self._offset_result(raw, anchor_text, anchor, amount, "day", f"relative_{lowered}")

        keyword = re.fullmatch(r"(?:the\s+)?(last|previous|next|following|this)\s+(year|month|week|day)", lowered)
        if keyword:
            direction, unit = keyword.groups()
            amount = -1 if direction in {"last", "previous"} else 1 if direction in {"next", "following"} else 0
            return self._offset_result(raw, anchor_text, anchor, amount, unit, f"relative_{direction}_{unit}")

        weekend = re.fullmatch(r"(?:the\s+)?(last|previous|next|following|this)\s+weekend", lowered)
        if weekend:
            direction = weekend.group(1)
            start, end = self._weekend_bounds(anchor, direction)
            return self._resolved(
                raw, anchor_text, start, end, "range",
                "resolved_relative", f"relative_{direction}_weekend",
            )

        number_alt = r"a\s+couple|couple|few|several|a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+"
        # "in <num> <unit>" — future with no explicit direction word.
        in_quantified = re.fullmatch(rf"in\s+({number_alt})\s+(years?|months?|weeks?|days?)\s*", lowered)
        if in_quantified:
            amount, approximate = self._parse_amount(in_quantified.group(1))
            if approximate or amount is None:
                return self._unresolved(raw, anchor_text, "ambiguous_quantifier")
            return self._offset_result(
                raw, anchor_text, anchor, amount, in_quantified.group(2).rstrip("s"),
                "relative_in", approximate=approximate,
            )

        # "<num> <unit> <direction>" — direction explicit, includes "from now".
        quantified = re.search(
            rf"\b({number_alt})\s+(years?|months?|weeks?|days?)\s+(ago|before|after|later|from\s+now)\b",
            lowered,
        )
        if quantified:
            amount, approximate = self._parse_amount(quantified.group(1))
            if approximate or amount is None:
                return self._unresolved(raw, anchor_text, "ambiguous_quantifier")
            direction = quantified.group(3)
            if direction in {"ago", "before"}:
                amount *= -1
            method = "relative_from_now" if direction == "from now" else f"relative_{direction}"
            return self._offset_result(
                raw, anchor_text, anchor, amount, quantified.group(2).rstrip("s"),
                method, approximate=approximate,
            )

        simple = re.fullmatch(r"(?:the\s+)?(day|week|month|year)\s+(before|after)", lowered)
        if simple:
            amount = -1 if simple.group(2) == "before" else 1
            return self._offset_result(raw, anchor_text, anchor, amount, simple.group(1), "relative_before_after")

        weekday = re.fullmatch(
            r"(?:on\s+)?(?:(last|previous|next|following|this)\s+)?"
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            lowered,
        )
        if weekday:
            direction, day_name = weekday.groups()
            target = self.WEEKDAYS[day_name]
            delta = target - anchor.weekday()
            defaulted = False
            if direction in {"last", "previous"}:
                delta = delta - 7 if delta >= 0 else delta
            elif direction in {"next", "following"}:
                delta = delta + 7 if delta <= 0 else delta
            elif direction == "this":
                pass
            else:
                # An unqualified weekday is genuinely ambiguous; default to
                # "this <weekday>" (this week's occurrence) and flag it so the
                # answer-correction layer can treat it as low confidence.
                direction = "this"
                defaulted = True
            value = anchor + timedelta(days=delta)
            if defaulted:
                return self._resolved(
                    raw, anchor_text, value, value, "date",
                    "resolved_relative_defaulted", "relative_weekday_defaulted",
                )
            return self._resolved(raw, anchor_text, value, value, "date", "resolved_relative", "relative_weekday")
        return None

    def _parse_amount(self, token: str) -> tuple[int | None, bool]:
        """Resolve a quantifier token to (amount, approximate).

        Vague quantifiers return ``(None, True)`` so callers demote them to an
        unresolved time instead of fabricating a fixed numeric offset.
        """
        if token in self.APPROXIMATE_QUANTIFIERS:
            return None, True
        if token in self.NUMBERS:
            return self.NUMBERS[token], False
        return int(token), False

    def _weekday_before_after_date(
        self,
        raw: str,
        lowered: str,
        anchor_text: str | None,
    ) -> TimeNormalization | None:
        match = re.fullmatch(
            r"(?:the\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
            r"(before|after)\s+(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+),?\s+(\d{4})",
            lowered,
        )
        if match:
            day_name, direction, day_value, month_name, year_value = match.groups()
            month = self.MONTHS.get(month_name)
            if not month:
                return None
            try:
                reference = datetime(int(year_value), month, int(day_value))
            except ValueError:
                return None
            target = self.WEEKDAYS[day_name]
            delta = target - reference.weekday()
            if direction == "before":
                delta = delta - 7 if delta >= 0 else delta
            else:
                delta = delta + 7 if delta <= 0 else delta
            value = reference + timedelta(days=delta)
            return self._resolved(raw, anchor_text, value, value, "date", "resolved_relative", "weekday_before_after_date")
        return None

    @staticmethod
    def _weekend_bounds(anchor: datetime, direction: str) -> tuple[datetime, datetime]:
        """Saturday..Sunday bounds for ``direction`` in a Monday-based week."""
        monday = anchor - timedelta(days=anchor.weekday())
        if direction in {"last", "previous"}:
            monday -= timedelta(days=7)
        elif direction in {"next", "following"}:
            monday += timedelta(days=7)
        saturday = monday + timedelta(days=5)
        sunday = monday + timedelta(days=6)
        return (
            datetime(saturday.year, saturday.month, saturday.day),
            datetime(sunday.year, sunday.month, sunday.day, 23, 59, 59),
        )

    def _offset_result(
        self,
        raw: str,
        anchor_text: str,
        anchor: datetime,
        amount: int,
        unit: str,
        method: str,
        *,
        approximate: bool = False,
    ) -> TimeNormalization:
        if unit == "year":
            value = _shift_months(anchor, amount * 12)
            start, end = _year_bounds(value.year)
            precision = "year"
        elif unit == "month":
            value = _shift_months(anchor, amount)
            start, end = _month_bounds(value.year, value.month)
            precision = "month"
        elif unit == "week":
            value = anchor + timedelta(weeks=amount)
            start = value - timedelta(days=value.weekday())
            end = start + timedelta(days=6)
            precision = "range"
        else:
            value = anchor + timedelta(days=amount)
            start = end = value
            precision = "date"
        status = "resolved_relative_approximate" if approximate else "resolved_relative"
        suffix = "_approximate" if approximate else ""
        return self._resolved(raw, anchor_text, start, end, precision, status, method + suffix)

    def _reference_anchors(
        self,
        expression: str,
        references: Mapping[str, str | datetime | date] | None,
    ) -> list[datetime]:
        if not references or not re.search(r"\b(?:before|after)\b", expression):
            return []
        matched = []
        for label, value in references.items():
            if str(label).lower() in expression:
                parsed = _parse_datetime(value)
                if parsed:
                    matched.append(parsed)
        return matched

    def _resolved(
        self,
        raw: str,
        anchor: str | None,
        start: datetime,
        end: datetime,
        precision: str,
        status: str,
        method: str,
    ) -> TimeNormalization:
        return TimeNormalization(
            raw_time_expression=raw,
            time_anchor=anchor,
            absolute_time_start=_format_datetime(start, precision),
            absolute_time_end=_format_datetime(end, precision),
            time_precision=precision,
            normalization_status=status,
            normalization_method=method,
        )

    def _unresolved(self, raw: str | None, anchor: str | None, method: str) -> TimeNormalization:
        return TimeNormalization(raw, anchor, None, None, "unknown", "unresolved", method)


def _parse_datetime(value: str | datetime | date | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        result = datetime.fromisoformat(normalized)
        if result.tzinfo:
            result = result.astimezone(timezone.utc).replace(tzinfo=None)
        return result
    except ValueError:
        pass
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)
    formats = (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %B %Y",
        "%d %B, %Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%B %d %Y",
        "%Y/%m/%d",
    )
    for format_ in formats:
        try:
            return datetime.strptime(cleaned, format_)
        except ValueError:
            continue
    return None


def _shift_months(value: datetime, months: int) -> datetime:
    total = value.year * 12 + value.month - 1 + months
    year, month_index = divmod(total, 12)
    month = month_index + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _year_bounds(year: int) -> tuple[datetime, datetime]:
    return datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59)


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    last_day = calendar.monthrange(year, month)[1]
    return datetime(year, month, 1), datetime(year, month, last_day, 23, 59, 59)


def _format_datetime(value: datetime, precision: str) -> str:
    if precision in {"year", "month", "date", "range"}:
        return value.date().isoformat()
    return value.isoformat(timespec="seconds")


def _anchor_text(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def parse_iso_datetime(value: str | datetime | date | None) -> datetime | None:
    """Strict ISO 8601 parser. Rejects natural-language datetimes.

    Unlike ``_parse_datetime``, this only accepts ISO 8601 forms so that a
    relative expression like "last week" or "1:56 pm on 8 May, 2023" returns
    None and the caller can demote it to the deterministic normalizer.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        result = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if result.tzinfo:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result
