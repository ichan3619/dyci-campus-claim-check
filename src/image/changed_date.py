"""
Rule F: picture check, a poster that closely matches a DYCI post but gives a different date
(results/round2/changed_date_plan.md). CAMPUS_VERIFY_CHANGED_DATE=1 (default off). Picture check only, after the level
is assigned; it changes the picture's level and reasons, never a verdict.

Fires when the level is 3 or 4 and, for the post the whole-text check used:
  F-a the picture names a date that is neither an event date nor the header date of the post;
  F-b the post names an event date at most 7 days from that date;
  F-c at least half of the picture's content words appear in the post (2 x found >= words, words >= 1);
  F-d the whole-text verdict is not CONTRADICTED.
Then: Level 2, with the reason first. The frozen date_check module is imported, not changed.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import unicodedata
from typing import Optional

from src.verification.date_check import MONTHS, event_dates, extract_dates, header_dates

ENV = "CAMPUS_VERIFY_CHANGED_DATE"
VERSION = "changed-date-v1"
MAX_DAYS = 7
NAMES = ("", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
         "November", "December")
SCHOOL = {"yanga", "yangas", "colleges", "college", "inc", "dyci", "bocaue", "bulacan"}
LABELS = {"announcement", "advisory", "notice", "reminder"}
STOP = {"the", "and", "for", "will", "this", "that", "with", "are", "has", "have", "was", "were", "not", "only", "until",
        "into", "from", "all", "our", "your", "you", "its", "their", "there", "been", "per", "due", "but", "also", "may",
        "can", "shall", "who", "which", "on", "at", "by", "to", "of", "in", "is", "be", "as", "an", "a"}
_WORD = re.compile(r"[a-z]+")


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii").lower()


def content_words(text: str) -> list:
    """Letters-only words of 3+ letters, without month names, stop words, the school name's words and labels.
    Digits split words ("34th" -> "th", dropped by length), so dates and times leave nothing behind."""
    return [w for w in _WORD.findall(_fold(text))
            if len(w) >= 3 and w not in MONTHS and w not in STOP and w not in SCHOOL and w not in LABELS]


def overlap(picture_text: str, post_text: str):
    words = content_words(picture_text)
    post = set(_WORD.findall(_fold(post_text)))
    found = [w for w in words if w in post]
    return len(found), len(words)


def _days(a, b) -> int:
    """Days between two (month, day) dates, the shorter way round the year."""
    try:
        da, db = dt.date(2001, *a), dt.date(2001, *b)
    except ValueError:                     # 29 February
        return 366
    d = abs((da - db).days)
    return min(d, 365 - d)


def _md(d) -> str:
    return f"{d[1]} {NAMES[d[0]]}"


def _get(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def check(picture_text: str, text_check, level: int) -> Optional[dict]:
    """The rule F finding, or None. `text_check` is the whole-text check (a dict or a VerificationResultModel)."""
    if level not in (3, 4) or text_check is None or _get(text_check, "verdict") == "CONTRADICTED":   # F-d
        return None
    used = _get(text_check, "used_evidence")
    post_text = _get(used, "text") if used is not None else None
    if not post_text:
        return None
    events = set(event_dates(post_text))
    known = events | set(header_dates(post_text))
    changed = sorted(d for d in extract_dates(picture_text) if d not in known)                       # F-a
    pairs = sorted((_days(c, e), c, e) for c in changed for e in events)
    if not pairs or pairs[0][0] > MAX_DAYS:                                                         # F-b
        return None
    found, words = overlap(picture_text, post_text)
    if words == 0 or 2 * found < words:                                                             # F-c
        return None
    _, picture_date, post_date = pairs[0]
    header = sorted(header_dates(post_text))
    return {"version": VERSION, "picture_date": _md(picture_date), "post_date": _md(post_date),
            "post_dated": _md(header[0]) if header else None,
            "post": _get(used, "document_id"), "post_published": (_get(used, "original_date")
                                                                  or _get(used, "published_date")),
            "overlap": f"{found} of {words}"}


def apply(card: dict, info: dict) -> dict:
    """The level card at Level 2 with rule F's reason first; nothing else in the response changes."""
    from src.image.levels import CHECK, NAMES as LEVEL_NAMES, _long_date
    post = (f"DYCI's announcement dated {info['post_dated']}" if info.get("post_dated")
            else f"DYCI's post of {_long_date(info.get('post_published'))}")
    reason = (f"This picture closely matches {post}, but it gives {info['picture_date']} where that announcement says "
              f"{info['post_date']}. It may have been edited.")
    return {**card, "level": 2, "name": LEVEL_NAMES[2], "reasons": [reason] + list(card.get("reasons") or []),
            "why_not_sure": None, "closest": [], "check": CHECK}
