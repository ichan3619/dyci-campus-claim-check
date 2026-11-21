"""
Class-day date guard (CAMPUS_VERIFY_CLASS_DAY_GUARD, on by default since v2.2, "0" switches it off; v1 results/round2/class_day_guard_plan.md FAIL,
v2 results/round2/class_day_guard_v2_plan.md PASS in class_day_guard_v2_report.md).

A message about class status on an explicit date ("There are no classes on March 16") is about that day. A DYCI post
about class status on a different day says nothing about it, but the verdict model can still match "classes" against
"classes". So a SUPPORTED or CONTRADICTED resting on a DYCI post stands only when the post names a date within
WINDOW_DAYS (v2: 1; v1's 7 let a post about 20 August answer for 26 August) of one of the message's dates; otherwise
it becomes INSUFFICIENT_EVIDENCE. Event messages (no class-status
word) are untouched: a wrong-date event is a real contradiction. Never creates or reverses a verdict.
"""
import datetime as _dt
import os
import re
from dataclasses import replace
from typing import List

from src.verification.date_check import extract_dates
from src.verification.relative_resolve import MONTHS, nearest_days

ENV = "CAMPUS_VERIFY_CLASS_DAY_GUARD"
VERSION = "class-day-guard-v2"
WINDOW_DAYS = 1
CLASS_WORDS = frozenset("class classes school pasok klase f2f ftf face-to-face in-person online suspended suspension "
                        "cancelled canceled holiday work".split())


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"         # on by default since v2.2 (owner, 2026-10-27)


def is_class_day(text: str) -> bool:
    return any(t in CLASS_WORDS for t in re.findall(r"[a-z][a-z-]*", (text or "").lower()))


def message_dates(text: str, today: _dt.date) -> List[_dt.date]:
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", text or "")]
    year = years[0] if len(set(years)) == 1 else today.year
    out = []
    for month, day in sorted(extract_dates(text or "")):
        try:
            out.append(_dt.date(year, month, day))
        except ValueError:
            continue
    return out


def check(message: str, result, today: _dt.date):
    """(result, info); info None when the guard does not apply."""
    used = getattr(result, "used_evidence", None)
    if result.verdict not in ("SUPPORTED", "CONTRADICTED") or used is None or not is_class_day(message):
        return result, None
    days = message_dates(message, today)
    if not days:
        return result, None
    gaps = [g for g in (nearest_days(used.text, d) for d in days) if g is not None]
    nearest = min(gaps) if gaps else None
    info = {"version": VERSION, "message_dates": [d.isoformat() for d in days], "post_used": used.document_id,
            "nearest_post_date_days": nearest, "window_days": WINDOW_DAYS,
            "withheld": nearest is None or nearest > WINDOW_DAYS, "original_verdict": result.verdict}
    if not info["withheld"]:
        return result, info
    shown = ", ".join(f"{MONTHS[d.month - 1]} {d.day}" for d in days)
    return replace(result, verdict="INSUFFICIENT_EVIDENCE",
                   explanation=result.explanation + f" Class-day guard: the closest DYCI post is about a different "
                                                    f"day, so it cannot settle classes on {shown}."), info
