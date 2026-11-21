"""
Philippine holiday calendar (CAMPUS_VERIFY_HOLIDAYS, on by default since v2.1, "0" switches it off;
results/round2/holidays_plan.md, PASS in holidays_report.md).

Source: Proclamation No. 1006, s. 2025 (signed 3 September 2025), the 2026 regular holidays and special days. Eid'l
Fitr and Eid'l Adha (separate proclamations) and local holidays are not listed; dates outside 2026 never match.

H1 (English path only): when every other step left INSUFFICIENT_EVIDENCE and the message names exactly one date that
is on the list, the frozen verdict model judges the message against a fixed holiday statement; a SUPPORTED or
CONTRADICTED with confidence >= 0.90 becomes the verdict, always with the H2 warning. A DYCI post that settled the
message always wins, because H1 only reads INSUFFICIENT_EVIDENCE. H3: a display-only note for any message whose date
is on the list, in any language.
"""
import datetime as _dt
import os
import re
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from src.verification.date_check import extract_dates

ENV = "CAMPUS_VERIFY_HOLIDAYS"
VERSION = "holidays-v1"
SOURCE = "Proclamation No. 1006, s. 2025"
YEAR = 2026
MIN_CONFIDENCE = 0.90
REGULAR, SPECIAL, WORKING = "regular holiday", "special (non-working) day", "special working day"
HOLIDAYS: Dict[Tuple[int, int], Tuple[str, str]] = {
    (1, 1): ("New Year's Day", REGULAR), (4, 2): ("Maundy Thursday", REGULAR), (4, 3): ("Good Friday", REGULAR),
    (4, 9): ("Araw ng Kagitingan", REGULAR), (5, 1): ("Labor Day", REGULAR), (6, 12): ("Independence Day", REGULAR),
    (8, 31): ("National Heroes Day", REGULAR), (11, 30): ("Bonifacio Day", REGULAR),
    (12, 25): ("Christmas Day", REGULAR), (12, 30): ("Rizal Day", REGULAR),
    (8, 21): ("Ninoy Aquino Day", SPECIAL), (11, 1): ("All Saints' Day", SPECIAL),
    (12, 8): ("Feast of the Immaculate Conception of Mary", SPECIAL), (12, 31): ("Last Day of the Year", SPECIAL),
    (2, 17): ("Chinese New Year", SPECIAL), (4, 4): ("Black Saturday", SPECIAL), (11, 2): ("All Souls' Day", SPECIAL),
    (12, 24): ("Christmas Eve", SPECIAL),
    (2, 25): ("EDSA People Power Revolution Anniversary", WORKING),
}
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
TAGALOG_RELATIVE = {"bukas": 1, "ngayon": 0, "kahapon": -1}
# Amendment H-A1: a holiday named in words counts as its date. Most specific names first; each match is blanked out so
# "Christmas Eve" is not also read as Christmas Day. "Undas" covers 1 and 2 November, so it names two dates.
NAMES = (
    (r"chinese\s+new\s+year", [(2, 17)]), (r"new\s+year'?s\s+eve|last\s+day\s+of\s+the\s+year", [(12, 31)]),
    (r"new\s+year'?s?(?:\s+day)?|bagong\s+taon", [(1, 1)]),
    (r"christmas\s+eve|noche\s+buena", [(12, 24)]), (r"christmas\s+(?:carols?|caroling|party|break|spirit)", []),
    (r"christmas(?:\s+day)?|\bpasko\b", [(12, 25)]),
    (r"maundy\s+thursday|huwebes\s+santo", [(4, 2)]), (r"good\s+friday|biyernes\s+santo", [(4, 3)]),
    (r"black\s+saturday|sabado\s+de\s+gloria", [(4, 4)]), (r"araw\s+ng\s+kagitingan|day\s+of\s+valou?r", [(4, 9)]),
    (r"labou?r\s+day|araw\s+ng\s+paggawa", [(5, 1)]), (r"independence\s+day|araw\s+ng\s+kalayaan", [(6, 12)]),
    (r"ninoy\s+aquino\s+day", [(8, 21)]), (r"national\s+heroes\s+day|araw\s+ng\s+mga\s+bayani", [(8, 31)]),
    (r"all\s+saints'?\s+day", [(11, 1)]), (r"all\s+souls'?\s+day", [(11, 2)]), (r"\bundas\b", [(11, 1), (11, 2)]),
    (r"bonifacio\s+day", [(11, 30)]), (r"immaculate\s+conception", [(12, 8)]), (r"rizal\s+day", [(12, 30)]),
    (r"\bedsa\b", [(2, 25)]),
)
WARNING = (f"Based on the national holiday list ({SOURCE}), not on a DYCI announcement. DYCI may announce otherwise: "
           "check its official page.")


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"         # on by default since v2.1 (owner, 2026-10-27)


def _written(day: _dt.date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]} {day.year} ({WEEKDAYS[day.weekday()]})"


UNDAS_ENV = "CAMPUS_VERIFY_UNDAS_NO_CLASSES"
UNDAS = ((11, 1), (11, 2))


def undas_no_classes() -> bool:
    """Amendment H-A2 (results/round2/class_day_guard_plan.md): All Saints' and All Souls' Day settle classes like a
    regular holiday."""
    return os.environ.get(UNDAS_ENV, "1").strip() != "0"   # on by default since v2.2 (owner, 2026-10-27)


def statement(day: _dt.date) -> str:
    name, kind = HOLIDAYS[(day.month, day.day)]
    shown = "special (non-working) holiday" if kind == SPECIAL else kind          # amendment H-A1: the everyday word
    head = f"{_written(day)} is {'the ' if kind == WORKING else ''}{name}, a {shown} in the Philippines ({SOURCE})."
    if kind == REGULAR or ((day.month, day.day) in UNDAS and undas_no_classes()):
        return head + " There are no classes and no work nationwide on this day."
    if kind == WORKING:
        return head + " It is not a holiday."
    return head


def message_dates(text: str, resolution: Optional[dict], today: _dt.date) -> List[_dt.date]:
    """Calendar dates the message is about: explicit month-day dates (year from the message, else today's), the
    dates relative_resolve produced, Tagalog bukas/ngayon/kahapon, and holiday names (amendment H-A1)."""
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", text or "")]
    year = years[0] if len(set(years)) == 1 else today.year
    out = set()
    for month, d in extract_dates(text or ""):
        try:
            out.add(_dt.date(year, month, d))
        except ValueError:
            continue
    for iso in (resolution or {}).get("dates", []):
        out.add(_dt.date.fromisoformat(iso))
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if w in TAGALOG_RELATIVE:
            out.add(today + _dt.timedelta(days=TAGALOG_RELATIVE[w]))
    low = (text or "").lower().replace("’", "'")
    for rx, days in NAMES:
        if re.search(rx, low):
            out.update(_dt.date(year, m, d) for m, d in days)
            low = re.sub(rx, " ", low)
    return sorted(out)


def far_post_guard(message: str, result, resolution: Optional[dict], today: _dt.date):
    """Amendment H-A1: a SUPPORTED/CONTRADICTED for a message naming exactly one listed date stands only if the DYCI
    post used names a date within 7 days of it (relative_resolve.nearest_days); otherwise INSUFFICIENT_EVIDENCE, so
    that check() may judge it against the holiday list. (result, info)."""
    from src.verification.relative_resolve import NEAR_DAYS, nearest_days
    used = getattr(result, "used_evidence", None)
    if result.verdict not in ("SUPPORTED", "CONTRADICTED") or used is None:
        return result, None
    days = message_dates(message, resolution, today)
    if len(days) != 1 or not _on_list(days[0]):
        return result, None
    gap = nearest_days(used.text, days[0])
    info = {"post_used": used.document_id, "nearest_post_date_days": gap, "near_days": NEAR_DAYS,
            "withheld": gap is None or gap > NEAR_DAYS, "original_verdict": result.verdict}
    if not info["withheld"]:
        return result, info
    return replace(result, verdict="INSUFFICIENT_EVIDENCE",
                   explanation=result.explanation + " Date guard: the closest DYCI post is about a different time, so "
                                                    "it cannot settle this."), info


def _on_list(day: _dt.date) -> bool:
    return day.year == YEAR and (day.month, day.day) in HOLIDAYS


def note(text: str, resolution: Optional[dict], today: _dt.date) -> Optional[dict]:
    """H3: display-only note for every listed date the message names."""
    days = [d for d in message_dates(text, resolution, today) if _on_list(d)]
    if not days:
        return None
    lines = []
    for d in days:
        name, kind = HOLIDAYS[(d.month, d.day)]
        lines.append(f"{d.day} {MONTHS[d.month - 1]} {d.year} is {'the ' if kind == WORKING else ''}{name}, a {kind} "
                     f"in the Philippines ({SOURCE}).")
    return {"version": VERSION, "dates": [d.isoformat() for d in days], "text": " ".join(lines)}


def check(pipeline_text: str, message: str, result, resolution: Optional[dict], today: _dt.date, classify):
    """H1. (result, info); info None when H1 does not apply. `classify(claim, evidence)` is the frozen classifier."""
    if result.verdict != "INSUFFICIENT_EVIDENCE":
        return result, None
    days = message_dates(message, resolution, today)
    if len(days) != 1 or not _on_list(days[0]):
        return result, None
    day = days[0]
    premise = statement(day)
    c = classify(pipeline_text, premise)
    name, kind = HOLIDAYS[(day.month, day.day)]
    applied = c.label in ("SUPPORTED", "CONTRADICTED") and c.confidence >= MIN_CONFIDENCE
    info = {"version": VERSION, "date": day.isoformat(), "holiday": name, "kind": kind, "statement": premise,
            "source": SOURCE, "label": c.label, "confidence": round(float(c.confidence), 4), "applied": applied,
            "original_verdict": result.verdict, "warning": WARNING if applied else None}
    if not applied:
        return result, info
    return replace(result, verdict=c.label, used_evidence=None, classification_result=c,
                   explanation=f"Holiday calendar: judged against \"{premise}\" {WARNING}"), info
