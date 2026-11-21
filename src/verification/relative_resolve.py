"""
Relative dates as real dates (results/round2/relative_resolve_plan.md, committed a0e3284; v2 PASS,
results/round2/relative_resolve_v2_report.md). Setting CAMPUS_VERIFY_RESOLVE_RELATIVE, on by default since v2.0
(owner, 2026-10-27); "0" switches it off. English messages only; wired in src/api/main.py before routing.

"today", "tonight", "tomorrow" ("tmrw", "tmr"), "the day after tomorrow" and "yesterday" are replaced by the calendar date
they mean, in Philippine time (UTC+8, fixed: the Philippines has no daylight saving), so the frozen date check
(date_check.py) can hold the message to the post's own dates: "Classes are suspended tomorrow." asked on 27 October
2026 is checked as "Classes are suspended on October 28, 2026."

Left to the v1.3 relative-date rule: possessives ("tomorrow's"), weekday words, and any message that already names an
explicit date (the student's own date wins).

v2 (results/round2/relative_resolve_v2_plan.md, committed 99cc53f) adds guard(): for a resolved message, a CONTRADICTED
verdict stays only when the post used names a date within NEAR_DAYS of a resolved date; otherwise it becomes
INSUFFICIENT_EVIDENCE. A post about a day weeks away cannot settle a message about today or tomorrow.
"""
import datetime as _dt
import os
import re
from dataclasses import replace as _replace
from typing import Optional, Tuple

from src.verification.date_check import extract_dates

ENV = "CAMPUS_VERIFY_RESOLVE_RELATIVE"
VERSION = "relative-resolve-v2"
NEAR_DAYS = 7
MANILA = _dt.timezone(_dt.timedelta(hours=8))
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
OFFSETS = (("the day after tomorrow", 2), ("tomorrow", 1), ("tmrw", 1), ("tmr", 1), ("tonight", 0), ("today", 0),
           ("yesterday", -1))
PREPOSITIONS = frozenset("on by until till before after from for since".split())
_RX = re.compile(r"(?<![\w'’])(" + "|".join(re.escape(w).replace(r"\ ", r"\s+") for w, _ in OFFSETS) + r")(?![\w'’])",
                 re.I)


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"         # on by default since v2.0 (owner, 2026-10-27)


def today() -> _dt.date:
    return _dt.datetime.now(MANILA).date()


def spoken(day: _dt.date) -> str:
    return f"{WEEKDAYS[day.weekday()]}, {day.day} {MONTHS[day.month - 1]} {day.year}"


def resolve(text: str, on: Optional[_dt.date] = None) -> Tuple[str, Optional[dict]]:
    """(text with relative words replaced, info), or (text unchanged, None) when nothing applies."""
    if not text or extract_dates(text):
        return text, None
    base = on or today()
    offsets = {w: n for w, n in OFFSETS}
    words, dates = [], []

    def replace(m: re.Match) -> str:
        word = re.sub(r"\s+", " ", m.group(1).lower())
        day = base + _dt.timedelta(days=offsets[word])
        words.append(word)
        dates.append(day)
        written = f"{MONTHS[day.month - 1]} {day.day}, {day.year}"
        before = re.findall(r"[A-Za-z]+", text[:m.start()])
        if before and before[-1].lower() in PREPOSITIONS:
            return written
        return ("On " if m.start() == 0 else "on ") + written

    out = _RX.sub(replace, text)
    if not words:
        return text, None
    return out, {"version": VERSION, "today": base.isoformat(), "words": words,
                 "dates": [d.isoformat() for d in dates], "read_as": [spoken(d) for d in dates], "checked_as": out}


def nearest_days(post_text: str, resolved: _dt.date) -> Optional[int]:
    """Smallest distance in days between a resolved date and any month-day date in the post (the post's year is not
    known, so the resolved year and the years either side are tried); None when the post names no date."""
    best = None
    for month, day in extract_dates(post_text or ""):
        for year in (resolved.year - 1, resolved.year, resolved.year + 1):
            try:
                gap = abs((_dt.date(year, month, day) - resolved).days)
            except ValueError:                                  # 29 February in a non-leap year
                continue
            best = gap if best is None else min(best, gap)
    return best


def guard(result, resolution: Optional[dict]):
    """(result, info). For a resolved message, CONTRADICTED needs the post used to name a date within NEAR_DAYS."""
    used = getattr(result, "used_evidence", None)
    if not resolution or result.verdict != "CONTRADICTED" or used is None:
        return result, None
    gaps = [nearest_days(used.text, _dt.date.fromisoformat(d)) for d in resolution["dates"]]
    known = [g for g in gaps if g is not None]
    nearest = min(known) if known else None
    info = {"near_days": NEAR_DAYS, "nearest_post_date_days": nearest, "post_used": used.document_id,
            "withheld": nearest is None or nearest > NEAR_DAYS, "original_verdict": result.verdict}
    if not info["withheld"]:
        return result, info
    reason = (" Date guard: the closest post is about a different time ("
              + ("it names no date" if nearest is None else f"its nearest date is {nearest} days away")
              + "), so it cannot settle this.")
    return _replace(result, verdict="INSUFFICIENT_EVIDENCE", explanation=result.explanation + reason), info
