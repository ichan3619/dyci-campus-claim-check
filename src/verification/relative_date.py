"""
Relative dates ("tomorrow", "bukas", "today" ...) against the date of the post shown. Added 2026-10-12 after manual
testing: "is there no classes tomorrow" was answered SUPPORTED from the 20 August advisory ("no classes tomorrow,
August 21"). The system does not know which day the student means by "tomorrow"; an old post's "tomorrow" is another day.

    relative_words()   the relative day words in the message (RELATIVE_WORDS, from the router's day words; weekdays and
                       "next week" are not included)
    post_date()        the date the post was made, as recorded: the original DYCI date if recorded; else the DYCI
                       posting date of a hand-entered post; else the date in the post's header line (day and month
                       only); else unknown. A test-Page copy's re-post date is never used: it is not DYCI's date.
    notice()           display only (A): one sentence per shown post asking the reader to check the day
    check()            CAMPUS_VERIFY_RELATIVE_DATE_CHECK (B, default off): withholds a SUPPORTED/CONTRADICTED verdict on
                       a message with a relative day word unless the post used was made today or yesterday (server date).
                       It only ever changes SUPPORTED/CONTRADICTED to INSUFFICIENT_EVIDENCE.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import replace
from typing import Iterable, List, Optional, Tuple

from src.retrieval.bm25 import word_tokens
from src.verification.date_check import MONTH_NAMES, header_dates

VERSION = "relative-date-v1"
RELATIVE_WORDS = ("today", "tonight", "tomorrow", "tmrw", "tmr", "yesterday",
                  "bukas", "ngayon", "mamaya", "kahapon", "kanina")
SC = ("SUPPORTED", "CONTRADICTED")
PostDate = Tuple[str, object]           # ("full", date) or ("monthday", (month, day))


def relative_words(text: str) -> List[str]:
    seen = []
    for w in word_tokens(text or ""):
        if w in RELATIVE_WORDS and w not in seen:
            seen.append(w)
    return seen


def _iso(value: str) -> Optional[_dt.date]:
    try:
        return _dt.date.fromisoformat((value or "")[:10])
    except ValueError:
        return None


def post_date(text: str, info=None) -> Optional[PostDate]:
    if info is not None:
        original = _iso(getattr(info, "original_date", ""))
        if original:
            return ("full", original)
        if getattr(info, "provenance", "") == "manual_entry":
            posted = _iso(getattr(info, "published_date", ""))
            if posted:
                return ("full", posted)
    headers = sorted(header_dates(text or ""))
    if headers:
        return ("monthday", headers[0])
    return None


def describe(pd: Optional[PostDate]) -> Optional[str]:
    if pd is None:
        return None
    if pd[0] == "full":
        d = pd[1]
        return f"{d.day} {MONTH_NAMES[d.month - 1]} {d.year}"
    m, d = pd[1]
    return f"{d} {MONTH_NAMES[m - 1]}"


def is_recent(pd: Optional[PostDate], today: _dt.date) -> bool:
    """Made today or yesterday (by `today`). Unknown dates are not recent."""
    if pd is None:
        return False
    days = (today, today - _dt.timedelta(days=1))
    if pd[0] == "full":
        return pd[1] in days
    return pd[1] in {(d.month, d.day) for d in days}


def _sentence(word: str, pd: Optional[PostDate]) -> str:
    when = f"This post is from {describe(pd)}" if pd else "This post’s date is not recorded"
    return f"Your message says “{word}”. {when}, so check that it is about the day you mean."


def notice(message: str, posts: Iterable, sources=None) -> Optional[dict]:
    """Display only: {"words": [...], "posts": {document_id: {"post_date": ..., "text": ...}}}, or None when the message
    has no relative day word or no post is shown."""
    words = relative_words(message)
    if not words:
        return None
    out = {}
    for p in posts:
        if p is None or p.document_id in out:
            continue
        pd = post_date(p.text, (sources or {}).get(p.document_id))
        out[p.document_id] = {"post_date": describe(pd), "text": _sentence(words[0], pd)}
    return {"words": words, "posts": out} if out else None


def check(message: str, result, sources=None, today: Optional[_dt.date] = None):
    """(result, info). Withholds SUPPORTED/CONTRADICTED -> INSUFFICIENT_EVIDENCE when the message has a relative day
    word and the post used was not made today or yesterday. Nothing else changes."""
    words = relative_words(message)
    if not words or result.verdict not in SC or result.used_evidence is None:
        return result, None
    today = today or _dt.date.today()
    used = result.used_evidence
    pd = post_date(used.text, (sources or {}).get(used.document_id))
    recent = is_recent(pd, today)
    info = {"version": VERSION, "words": words, "post_date": describe(pd), "today": today.isoformat(),
            "withheld": not recent, "original_verdict": result.verdict}
    if recent:
        return result, info
    reason = (f" Relative date check: the message says “{words[0]}”, and the post used is "
              + (f"from {describe(pd)}" if pd else "undated") + ", not from today or yesterday, so the verdict was withheld.")
    return replace(result, verdict="INSUFFICIENT_EVIDENCE", explanation=result.explanation + reason), info
