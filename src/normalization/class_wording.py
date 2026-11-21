"""
Rule C: class wording written the standard way before checking (results/round2/class_wording_plan.md).
CAMPUS_VERIFY_CLASS_WORDING=1 (default off). A closed rewrite, no model; applied in _answer after the Tagalog
translation and the relative-date step.

Only for a message that talks about classes ("class", "classes", "pasok" or "klase" as a word):
- C-a plural: "no class" -> "no classes" (also "no f2f class", "no online class", "no face to face class");
  "class is suspended" -> "classes are suspended"; "class suspended" -> "classes suspended".
- C-b date first: a message starting with an explicit date (month or short month and day, optional year, optional
  comma, colon or dash) followed by at least two words has the date moved to the end as "on <date>".
Nothing else changes; a message already in the standard form is returned unchanged.
"""
from __future__ import annotations

import os
import re
from typing import Optional

ENV = "CAMPUS_VERIFY_CLASS_WORDING"
VERSION = "class-wording-v1"
_TOPIC = re.compile(r"\b(?:class|classes|pasok|klase)\b", re.IGNORECASE)
_MONTH = (r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
          r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?")
_LEADING_DATE = re.compile(r"^\s*(?P<date>" + _MONTH + r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?)\s*[,:\-–—]?\s+"
                           r"(?P<rest>\S+(?:\s+\S+)+?)\s*$", re.IGNORECASE)
_PLURAL = [
    (re.compile(r"\b(no\s+(?:f2f\s+|online\s+|face[\s-]to[\s-]face\s+)?)(class)\b", re.IGNORECASE),
     lambda m: m.group(1) + m.group(2) + ("ES" if m.group(2).isupper() else "es")),
    (re.compile(r"\b(class)(\s+)(is)(\s+suspended)\b", re.IGNORECASE),
     lambda m: m.group(1) + ("ES" if m.group(1).isupper() else "es") + m.group(2)
     + ("ARE" if m.group(3).isupper() else "are") + m.group(4)),
    (re.compile(r"\b(class)(\s+suspended)\b", re.IGNORECASE),
     lambda m: m.group(1) + ("ES" if m.group(1).isupper() else "es") + m.group(2)),
]


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def rewrite(message: str) -> Optional[dict]:
    """{original, rewritten, rules} when rule C changes the message, else None."""
    text = message or ""
    if not _TOPIC.search(text):
        return None
    rules, out = [], text
    for rx, repl in _PLURAL:
        new = rx.sub(repl, out)
        if new != out:
            rules.append("plural")
            out = new
    m = _LEADING_DATE.match(out)
    if m:
        out = f"{m.group('rest').rstrip(' .!')} on {m.group('date')}"
        rules.append("date_first")
    if out == text:
        return None
    return {"version": VERSION, "original": text, "rewritten": out, "rules": sorted(set(rules))}
