"""
Option A version 2: a picture's result also shows its whole-text verdict (results/round2/picture_text_line_v2_plan.md).
CAMPUS_VERIFY_PICTURE_TEXT_LINE=1 (default off). Picture check only, after the level and rule F; it reads the
whole-text check (`text_check`) and the level, and changes neither. Version 1 (picture_text_line_plan.md, FAIL on its
"as typed text" wording) is reproducible at 414b1e8.

The post is named as rule F names it: by its own header date (the frozen date_check.header_dates, imported, not
changed), otherwise by original_date, then published_date. When rule G released the picture's verdict
(text_check.poster_language is set), the line says that typed as is, the words would get no verdict.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Optional

from src.verification.date_check import header_dates

ENV = "CAMPUS_VERIFY_PICTURE_TEXT_LINE"
VERSION = "picture-text-line-v2"
MONTHS = ("", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
LINES = {"SUPPORTED": "Checked as one piece of text, the words read from this picture match {post}.",
         "CONTRADICTED": "Checked as one piece of text, the words read from this picture conflict with {post}.",
         "INSUFFICIENT_EVIDENCE": "Checked as one piece of text, the words read from this picture could not be confirmed."}
RULE_G_NOTE = " (Seal and school-name text were set aside; typed as is, these words would get no verdict.)"
AGREES = {"SUPPORTED": (4, 5), "CONTRADICTED": (1,), "INSUFFICIENT_EVIDENCE": (3,)}
DIFFERS_NOTE = "The level above comes from checking each statement separately; see Details."


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def _readable(date: Optional[str]) -> Optional[str]:
    """'2031-07-16' (or '2026-09-29T06:31:23+0000') -> '16 July 2031'; anything else as given; None when empty."""
    if not date:
        return None
    try:
        d = dt.date.fromisoformat(str(date)[:10])
    except ValueError:
        return str(date)
    return f"{d.day} {MONTHS[d.month]} {d.year}"


def _http(url: Optional[str]) -> Optional[str]:
    return url if isinstance(url, str) and url.lower().startswith(("http://", "https://")) else None


def _header_date(post_text: Optional[str]) -> Optional[str]:
    """The post's own header date as 'day Month' (the first in date order), or None."""
    found = sorted(header_dates(post_text)) if post_text else []
    return f"{found[0][1]} {MONTHS[found[0][0]]}" if found else None


def post_name(ev) -> tuple:
    """(how the post is named, its header date or None, its original/posting date or None). Also used by option B."""
    post_dated = _header_date(getattr(ev, "text", None)) if ev is not None else None
    post_date = (_readable(getattr(ev, "original_date", None)) or _readable(getattr(ev, "published_date", None))
                 if ev is not None else None)
    name = (f"DYCI's announcement dated {post_dated}" if post_dated
            else f"DYCI's post of {post_date}" if post_date else "DYCI's post")
    return name, post_dated, post_date


def build(text_check, level: Optional[int]) -> Optional[dict]:
    """The text_line for a picture response, or None (no text check, no verdict, or the verdict was withheld)."""
    if text_check is None or getattr(text_check, "verdict_withheld", None):
        return None
    verdict = getattr(text_check, "verdict", None)
    if verdict not in LINES:
        return None
    ev = getattr(text_check, "used_evidence", None) if verdict != "INSUFFICIENT_EVIDENCE" else None
    post, post_dated, post_date = post_name(ev)
    url = (_http(getattr(ev, "original_url", None)) or _http(getattr(ev, "source_url", None))) if ev is not None else None
    rule_g = bool(getattr(text_check, "poster_language", None))
    return {"version": VERSION, "verdict": verdict, "post": ev.document_id if ev is not None else None,
            "post_dated": post_dated, "post_date": post_date, "post_url": url, "rule_g": rule_g,
            "line": LINES[verdict].format(post=post) + (RULE_G_NOTE if rule_g else ""),
            "differs": level not in AGREES[verdict]}
