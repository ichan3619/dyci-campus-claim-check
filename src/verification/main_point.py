"""
Rule M version 2: the main point checked apart from a copied reason (results/round2/main_point_v2_plan.md).
CAMPUS_VERIFY_MAIN_POINT=1 (default off). Applied at the end of _answer in src/api/main.py, after rule P; typed messages
and picture text alike. Version 1 (results/round2/main_point_plan.md, FAIL) is reproducible at e3d4288.

A fake made from a real advisory by changing only its main point ("NO CLASSES" for "shortened class hours") and copying
its reason word for word is SUPPORTED by the frozen classifier: the long copied reason outweighs the short false main
point. Rule M re-checks a SUPPORTED answer's main point alone:
    V2-a reason markers (whole words, any case): MARKER below, with guards on "since", "following" and "due to";
         the first marker that passes its guard splits the message;
    V2-b main point: the text before it, trimmed of " ,;:-." and of a dangling "the closure is";
    V2-c the main point must have a content word (CONTENT below); otherwise nothing changes;
    V2-d main point not SUPPORTED (any post, any reason) -> INSUFFICIENT_EVIDENCE; SUPPORTED -> unchanged.
         Rule M never makes an answer CONTRADICTED.
Answers that are not SUPPORTED are never changed. Rule P (reason_parts.py) acts only on answers that are not SUPPORTED.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

ENV = "CAMPUS_VERIFY_MAIN_POINT"
VERSION = "main-point-v2"

MARKER = re.compile(
    r"\b(in\s+observance\s+of|due\s+to|pursuant\s+to|to\s+give\s+way\s+to|because\s+of|in\s+line\s+with|in\s+view\s+of|"
    r"in\s+connection\s+with|in\s+accordance\s+with|as\s+per|dahil\s+sa|"
    r"because|since|following|so\s+that|in\s+order\s+to|to\s+allow|to\s+make\s+way\s+for|this\s+is|this\s+was)\b",
    re.IGNORECASE)

MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
          "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun"}
DATE_WORDS = (MONTHS | WEEKDAYS | {"today", "tomorrow", "yesterday", "tonight", "morning", "afternoon", "evening"}
              | {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                 "twenty", "thirty", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
                 "ninth", "tenth", "twentieth", "thirtieth"})
LABELS = {"announcement", "advisory", "update", "notice", "reminder", "memo", "memorandum", "bulletin", "attention",
          "please", "note", "important", "urgent"}
SCHOOL = {"dr", "yanga", "yangas", "yanga's", "colleges", "inc", "dyci", "edu", "bocaue", "bulacan"}
FUNCTION = {"the", "and", "for", "this", "that", "are", "was", "were", "will", "with", "from", "has", "have", "been",
            "all", "our", "its", "their", "there"}
NOT_CONTENT = DATE_WORDS | LABELS | SCHOOL | FUNCTION

# "since" is not a reason before a date word, a number or time, "then", "last", "early" or "noon"
_SINCE_NOT = re.compile(r"\s*(\d|(" + "|".join(sorted(DATE_WORDS | {"then", "last", "early", "noon"}, key=len,
                                                       reverse=True)) + r")\b)", re.IGNORECASE)
_FOLLOWING_NOT_AFTER = re.compile(r"\bthe\s*$", re.IGNORECASE)
_FOLLOWING_NOT_BEFORE = re.compile(r"\s*(:|(is|are)\b)", re.IGNORECASE)
_DUE_TO_NOT_BEFORE = re.compile(r"\s+(resume|start|begin|end|open|close|return|be)\b", re.IGNORECASE)
_DANGLING = re.compile(r"(?:^|\s)(?:the|its|our|their|this|that|it)(?:\s+[\w'’-]+){0,2}\s+(?:is|are|was|were)$",
                       re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")

NOTE = {"main_not_supported": " Main point check: the part before the reason could not be confirmed on its own, so the "
                              "message is not confirmed."}


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def _marker(message: str):
    """V2-a: the first marker that passes its guard, or None."""
    for m in MARKER.finditer(message):
        word = " ".join(m.group(1).lower().split())
        after, before = message[m.end():], message[:m.start()]
        if word == "since" and _SINCE_NOT.match(after):
            continue
        if word == "following" and (_FOLLOWING_NOT_AFTER.search(before) or _FOLLOWING_NOT_BEFORE.match(after)):
            continue
        if word == "due to" and _DUE_TO_NOT_BEFORE.match(after):
            continue
        return m, word
    return None


def trim(main: str) -> str:
    """V2-b: strip " ,;:-." and a dangling clause subject ("THE CLOSURE IS")."""
    main = main.strip(" ,;:-.")
    main = _DANGLING.sub("", main)
    return main.strip(" ,;:-.")


def has_content_word(text: str) -> bool:
    """V2-c: a word of 3 or more letters that is not a label, date, school, place or function word."""
    for w in _WORD.findall(text or ""):
        w = w.lower().replace("’", "'")
        if sum(c.isalpha() for c in w) >= 3 and w not in NOT_CONTENT and re.sub(r"'s?$", "", w) not in NOT_CONTENT:
            return True
    return False


def split(message: str) -> Optional[Tuple[str, str]]:
    """(main point, reason marker), or None (no marker, or no content word before it)."""
    found = _marker(message or "")
    if found is None:
        return None
    m, word = found
    main = trim(message[:m.start()])
    return (main, word) if has_content_word(main) else None


def decide(inner_verdict: Optional[str]) -> str:
    """V2-d: main_supported / main_not_supported (never CONTRADICTED)."""
    return "main_supported" if inner_verdict == "SUPPORTED" else "main_not_supported"
