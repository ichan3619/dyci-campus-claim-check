"""
Rule E: picture text, the poster's core is checked without its seal, header and footer
(results/round2/poster_core_plan.md). CAMPUS_VERIFY_POSTER_CORE=1 (default off). Picture check only, after rule D.

Rule R's clean-up steps its run did not fault (src/image/poster_cleanup.py: R-a seal and header, R-c issue date, R-d
footer, website removal), with one change: the school name is recognised with up to 2 letters missing, added or changed
when at least 15 of its letters are read ("colle g e $ Inc." for "Colleges, Inc."). Rule R's spaced-letter step (R-b)
and language step (R-e) are not used (rules D and G cover them).
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

from src.image.poster_cleanup import (_DATE_AT_START, _WEBSITE, FOOTER_TAIL, HEADER_WINDOW, LABELS, MAX_NAME_TOKENS,
                                      SCHOOL_LETTERS, _letters)

ENV = "CAMPUS_VERIFY_POSTER_CORE"
VERSION = "poster-core-v1"
MAX_EDITS = 2
MIN_LETTERS = 15


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def _distance(a: str, b: str) -> int:
    """Levenshtein distance (short strings)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _school_spans(toks: List[str]) -> List[Tuple[int, int]]:
    """(start, end) spans reading as the school name (or "DYCI"), with up to MAX_EDITS letter errors, not overlapping."""
    spans, i = [], 0
    while i < len(toks):
        hit = None
        if _letters(toks[i]) == "dyci" and not _WEBSITE.fullmatch(toks[i]):
            hit = (i, i + 1)
        elif _letters(toks[i]) and _letters(toks[i])[0] in "dy":
            best = None
            acc = ""
            for j in range(i, min(len(toks), i + MAX_NAME_TOKENS)):
                acc += _letters(toks[j])
                if len(acc) > len(SCHOOL_LETTERS) + MAX_EDITS:
                    break
                if len(acc) >= MIN_LETTERS:
                    d = _distance(acc, SCHOOL_LETTERS)
                    if d <= MAX_EDITS and (best is None or d <= best[0]):
                        best = (d, j + 1)
            if best:
                hit = (i, best[1])
        if hit:
            end = hit[1]
            while end < len(toks) and not re.search(r"\w", toks[end]):
                end += 1
            spans.append((hit[0], end))
            i = end
        else:
            i += 1
    return spans


def core(text: str) -> Tuple[str, Optional[dict]]:
    """(the poster's core text, info) or (text unchanged, None) when nothing was set aside."""
    removed: List[str] = []
    t = _WEBSITE.sub(lambda m: (removed.append(m.group(0)) or ""), text or "")
    t = re.sub(r"\s+", " ", t).strip()
    toks = t.split()
    spans = _school_spans(toks)
    start, poster_date = 0, None
    if spans and spans[0][0] < HEADER_WINDOW:
        start = spans[0][1]
        rest = " ".join(toks[start:])
        m = re.match(r"^(?:(?:DYCI|COLLEGE)\s+)?(?:" + "|".join(LABELS) + r")\b[:\s]*", rest, re.IGNORECASE)
        if m:
            start += len(m.group(0).split())
        d = _DATE_AT_START.match(" ".join(toks[start:]))
        if d:
            poster_date = d.group(0)
            start += len(d.group(0).split())
        removed.append(" ".join(toks[:start]))
        spans = spans[1:]
    end = len(toks)
    for s, e in spans:
        if s > start and len(toks) - e <= FOOTER_TAIL:
            removed.append(" ".join(toks[s:]))
            end = s
            break
    result = " ".join(toks[start:end])
    if not removed or not result:
        return text, None
    return result, {"version": VERSION, "removed": [r for r in removed if r], "poster_date": poster_date}
