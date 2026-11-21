"""
Header-only advisories, rule H (CAMPUS_VERIFY_HEADER_DATE, default off; results/round2/header_date_plan.md). Wired in
src/api/main.py right after verify_claim and the name-align step; English messages only.

The frozen date check withholds a SUPPORTED when the claimed date is not among the post's BODY dates (the header date
is left out). An advisory that gives its date only in the header ("DYCI ADVISORY November 12, 2026 ... suspended
immediately") can therefore never confirm a message about that day, even word for word. Rule H restores the
classifier's SUPPORTED only when:
    1. the date check withheld it (date_check == "claimed_date_not_in_evidence");
    2. the classifier's label was SUPPORTED and the value check did not act;
    3. the post used names no body date and exactly one header date;
    4. the message names exactly one explicit date, equal to that header date (and year).
It never touches any other verdict and never creates CONTRADICTED; every later safeguard still runs.
"""
import datetime as _dt
import os
import re
from dataclasses import replace

from src.verification.date_check import HEADER, MONTH_NAMES, event_dates, extract_dates, header_dates

ENV = "CAMPUS_VERIFY_HEADER_DATE"
VERSION = "header-date-v1"
REASON = (" Header date: the advisory is dated {day} and says this happens today or immediately, so its date counts "
          "as the day it is about.")


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"          # default off (header_date_plan.md)


def _year_of(text: str, published: str = ""):
    mt = HEADER.match(text or "")
    if mt:
        y = re.search(r"\b(\d{4})\b", mt.group("date"))
        if y:
            return int(y.group(1))
    return int(published[:4]) if re.match(r"\d{4}", published or "") else None


def check(message: str, result, today: _dt.date, sources=None):
    """(result, info); info None when the rule does not apply."""
    used = getattr(result, "used_evidence", None)
    cls = getattr(result, "classification_result", None)
    if (result.verdict != "INSUFFICIENT_EVIDENCE" or used is None or cls is None
            or getattr(result, "date_check", None) != "claimed_date_not_in_evidence"
            or getattr(cls, "label", None) != "SUPPORTED" or getattr(result, "value_check", None)):
        return result, None
    text = used.text or ""
    if event_dates(text):
        return result, None
    head = header_dates(text)
    claimed = extract_dates(message or "")
    if len(head) != 1 or len(claimed) != 1 or head != claimed:
        return result, None
    years = {int(y) for y in re.findall(r"\b(20\d\d)\b", message or "")}
    message_year = years.pop() if len(years) == 1 else today.year
    info_src = (sources or {}).get(used.document_id)
    post_year = _year_of(text, getattr(info_src, "published_date", "") or "")
    if post_year is not None and post_year != message_year:
        return result, None
    (month, day), = head
    day_name = f"{MONTH_NAMES[month - 1]} {day}, {post_year or message_year}"
    info = {"version": VERSION, "post_used": used.document_id, "header_date": day_name,
            "original_verdict": result.verdict, "verdict": "SUPPORTED"}
    return replace(result, verdict="SUPPORTED", explanation=result.explanation + REASON.format(day=day_name)), info
