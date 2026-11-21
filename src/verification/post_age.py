"""
Post-age rule (improvement 1b; results/round2/post_age_plan.md, approved 2026-10-25: D = 30 days, withhold).

CAMPUS_VERIFY_POST_AGE (default ON since v1.4, owner option B 2026-10-25; "0" switches it off). It applies to a
SUPPORTED/CONTRADICTED verdict when the message speaks about now or the future (a word from MARKERS), names no explicit
date and no relative day word, and the post used is older than MAX_DAYS on the server's date (an unknown date counts as
old). Past-tense and dated messages are never touched.

Owner decision 2026-10-25 (after the 1b run): the app WARNS instead of withholding. notice() keeps the verdict and
returns the warning shown under it (post_age_notice in the response). check() is the withholding version measured by the
1b run (scripts/round2/post_age_run.py); it is kept for that record and is not used by the app.

v2 (CAMPUS_VERIFY_POST_AGE_V2, default ON since v1.6, owner 2026-10-25 after the PASS run; "0" switches it off;
results/round2/post_age_v2_plan.md, approved with Q1 = C): the
warning also applies to messages WITHOUT a word from MARKERS ("classes suspended"), whatever their tense; the other
conditions are unchanged. Those messages get a softer sentence (Q2); messages with a MARKERS word keep the v1.5 one.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import replace
from typing import Optional

from src.verification.date_check import extract_dates
from src.verification.relative_date import describe, post_date, relative_words

ENV = "CAMPUS_VERIFY_POST_AGE"
V2_ENV = "CAMPUS_VERIFY_POST_AGE_V2"
MAX_DAYS = 30
MARKERS = frozenset("is are am isn't aren't will won't has have hasn't haven't now currently still ongoing open "
                    "there's".split())
SC = ("SUPPORTED", "CONTRADICTED")


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"


def v2_enabled() -> bool:
    return os.environ.get(V2_ENV, "1").strip() != "0"


def today() -> _dt.date:
    return _dt.date.today()


def speaks_about_now(message: str) -> bool:
    return any(t in MARKERS for t in re.findall(r"[a-z']+", (message or "").lower().replace("’", "'")))


def names_a_date(message: str) -> bool:
    return bool(extract_dates(message or "")) or bool(relative_words(message or ""))


def age_in_days(pd, on: _dt.date) -> Optional[int]:
    """Days from the post's date to `on`; a day-and-month date counts from its most recent past occurrence."""
    if pd is None:
        return None
    if pd[0] == "full":
        return (on - pd[1]).days
    month, day = pd[1]
    try:
        when = _dt.date(on.year, month, day)
    except ValueError:
        return None
    if when > on:
        when = when.replace(year=on.year - 1)
    return (on - when).days


def _old_post(message: str, result, sources, on: _dt.date, any_tense: bool = False):
    """(post date, age in days) when the rule applies, else None. any_tense (v2): no MARKERS word needed."""
    if result.verdict not in SC or result.used_evidence is None:
        return None
    if names_a_date(message) or not (any_tense or speaks_about_now(message)):
        return None
    used = result.used_evidence
    pd = post_date(used.text, (sources or {}).get(used.document_id))
    age = age_in_days(pd, on)
    if age is not None and age <= MAX_DAYS:
        return None
    return pd, age


def notice(message: str, result, sources=None, on: Optional[_dt.date] = None, v2: bool = False) -> Optional[dict]:
    """The warning for an old (or undated) post behind a verdict, or None. Never changes the verdict. Without v2 only
    messages about now (a MARKERS word) qualify; with v2 any message that names no date does."""
    on = on or today()
    old = _old_post(message, result, sources, on, any_tense=v2)
    if old is None:
        return None
    pd, age = old
    when = f"is from {describe(pd)} ({age} days ago)" if age is not None else "has no date"
    if speaks_about_now(message):
        text = (f"Old post: the post this is based on {when}. It may be outdated or about a different event, so it may "
                f"not reflect what is true now.")
    else:                                                    # v2 only (Q2): no "now" word, so a softer sentence
        text = (f"Old post: the post this is based on {when}. If your message is about something recent, this post may "
                f"be outdated or about a different event.")
    return {"document_id": result.used_evidence.document_id, "post_date": describe(pd) if pd else None,
            "post_age_days": age, "today": on.isoformat(), "max_days": MAX_DAYS, "text": text}


def check(message: str, result, sources=None, on: Optional[_dt.date] = None):
    """(result, info). The 1b-run version: withholds a SUPPORTED/CONTRADICTED verdict as described above. Not used by the app."""
    on = on or today()
    old = _old_post(message, result, sources, on)
    if old is None:
        return result, None
    pd, age = old
    when = f"from {describe(pd)} ({age} days ago)" if age is not None else "undated"
    reason = (f" Post-age check: the matching post is {when}, so it can't confirm what is true now "
              f"(posts older than {MAX_DAYS} days are not used for messages about now).")
    info = {"original_verdict": result.verdict, "post_age_days": age, "today": on.isoformat()}
    return replace(result, verdict="INSUFFICIENT_EVIDENCE", explanation=result.explanation + reason), info
