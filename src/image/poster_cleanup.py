"""
Rule R: poster clean-up for the picture check (results/round2/poster_cleanup_plan.md).
CAMPUS_VERIFY_POSTER_CLEANUP=1 (default off). Picture check only; /verify/text is unchanged.

Text read from an announcement poster carries layout that is not part of the announcement: the round seal's curved
text, the school-name line, the label ("ANNOUNCEMENT"), letter-spaced lines read as single letters, the issue date and
the footer (page name, website). These are set aside before the text is checked; every change is reported.

- R-b spaced-out letters: a run of at least 4 tokens, each 1-2 capitals or digits and at least half of them single
  characters, is rejoined ("1 0 NO VE M B E R 2 0 2 6" -> "10 NOVEMBER 2026").
- R-a seal and header: the first school-name line ("Dr. Yanga's Colleges, Inc." matched on letters only, or "DYCI"),
  if it starts within the first 15 words, is dropped with everything before it, and so is a label word right after it.
- R-c header date: a date directly after that point is the poster's issue date (reported as poster_date, not checked).
- R-d footer: a school-name line followed by at most 6 words, and everything after it, is dropped; website addresses
  are dropped anywhere.
- R-e language gate (language_ok): a single word from the verdict gate's extra Tagalog list in a text of 8 or more words
  does not withhold the verdict when the router's own Tagalog check does not fire.
"""
from __future__ import annotations

import contextvars
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

ENV = "CAMPUS_VERIFY_POSTER_CLEANUP"
VERSION = "poster-cleanup-v1"
SCHOOL_LETTERS = "dryangascollegesinc"
HEADER_WINDOW = 15            # the school-name line must start within the first 15 words to be a header
FOOTER_TAIL = 6               # at most 6 words may follow a footer school-name line
MAX_NAME_TOKENS = 30          # a letter-spaced school name can be read as many tokens
LABELS = ("PUBLIC ADVISORY", "ANNOUNCEMENT", "ADVISORY", "NOTICE", "REMINDER")
_MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
           "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
_DATE_AT_START = re.compile(rf"^(?:\d{{1,2}}\s+(?:{_MONTHS})\.?,?\s+\d{{4}}|(?:{_MONTHS})\.?\s+\d{{1,2}},?\s+\d{{4}})\b",
                            re.IGNORECASE)
_WEBSITE = re.compile(r"(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)*\.(?:edu\.ph|com\.ph|gov\.ph|com|ph|org|net)(?:/\S*)?",
                      re.IGNORECASE)
_SPACED_TOKEN = re.compile(r"[A-Z0-9]{1,2}")

# Set by the picture check while it runs its text checks with rule R on (read by language_ok's caller in main.py).
POSTER_ACTIVE = contextvars.ContextVar("poster_cleanup_active", default=False)


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


@dataclass
class PosterText:
    text: str
    changes: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    poster_date: Optional[str] = None
    header_found: bool = False

    def to_dict(self) -> dict:
        return {"version": VERSION, "changes": self.changes, "removed": self.removed,
                "poster_date": self.poster_date, "header_found": self.header_found}


def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def rejoin_spaced(text: str) -> str:
    """R-b on a whitespace-separated text."""
    toks, out, i = text.split(), [], 0
    while i < len(toks):
        j = i
        while j < len(toks) and _SPACED_TOKEN.fullmatch(toks[j]):
            j += 1
        run = toks[i:j]
        if len(run) >= 4 and sum(len(t) == 1 for t in run) * 2 >= len(run):
            joined = re.sub(r"(?<=[A-Z])(?=\d)|(?<=\d)(?=[A-Z])", " ", "".join(run))
            out.append(joined)
            i = j
        elif run:
            out.extend(run)
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def _school_spans(toks: List[str]):
    """(start, end) token spans that read as the school name, in order, not overlapping."""
    spans, i = [], 0
    while i < len(toks):
        hit = None
        if _letters(toks[i]) == "dyci" and not _WEBSITE.fullmatch(toks[i]):
            hit = (i, i + 1)
        else:
            acc = ""
            for j in range(i, min(len(toks), i + MAX_NAME_TOKENS)):
                acc += _letters(toks[j])
                if not SCHOOL_LETTERS.startswith(acc) or not acc:
                    break
                if acc == SCHOOL_LETTERS:
                    hit = (i, j + 1)
                    break
        if hit:
            end = hit[1]
            while end < len(toks) and not re.search(r"\w", toks[end]):   # a stray "." or "," read after the name
                end += 1
            hit = (hit[0], end)
            spans.append(hit)
            i = hit[1]
        else:
            i += 1
    return spans


def clean(text: str) -> PosterText:
    result = PosterText(text=text or "")
    t = _WEBSITE.sub(lambda m: (result.removed.append(m.group(0)) or ""), result.text)
    if t != result.text:
        result.changes.append("website")
    t = re.sub(r"\s+", " ", t).strip()
    joined = rejoin_spaced(t)
    if joined != t:
        result.changes.append("spaced_letters")
    t = joined
    toks = t.split()
    spans = _school_spans(toks)
    start = 0
    if spans and spans[0][0] < HEADER_WINDOW:
        result.header_found = True
        start = spans[0][1]
        rest = " ".join(toks[start:])
        m = re.match(r"^(?:(?:DYCI|COLLEGE)\s+)?(?:" + "|".join(LABELS) + r")\b[:\s]*", rest, re.IGNORECASE)
        if m:
            start += len(m.group(0).split())
        rest = " ".join(toks[start:])
        d = _DATE_AT_START.match(rest)
        if d:
            result.poster_date = d.group(0)
            start += len(d.group(0).split())
        if start:
            result.removed.append(" ".join(toks[:start]))
            result.changes.append("seal_and_header")
        spans = spans[1:]
    end = len(toks)
    for s, e in spans:
        if s > start and len(toks) - e <= FOOTER_TAIL:
            end = s
            result.removed.append(" ".join(toks[s:]))
            result.changes.append("footer")
            break
    result.text = " ".join(toks[start:end])
    return result


def language_ok(text: str) -> bool:
    """R-e: True when a verdict may be given although the verdict gate's extra Tagalog list matched: the router's own
    check does not fire, at most one extra-list word matches, and the text has at least 8 words."""
    from src.api.verdict_policy import EXTRA_TAGALOG_WORDS, TAGALOG_PREFIXES
    from src.routing.router import is_tagalog_or_taglish, tokens
    if is_tagalog_or_taglish(text):
        return False
    toks = tokens(text if isinstance(text, str) else "")
    hits = sum(1 for t in toks if t in EXTRA_TAGALOG_WORDS
               or ("-" in t and t.split("-")[0] in TAGALOG_PREFIXES))
    return hits <= 1 and len(toks) >= 8
