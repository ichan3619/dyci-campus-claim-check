"""
Rule D: picture text, letter-spaced dates are rejoined before checking (results/round2/poster_dates_plan.md).
CAMPUS_VERIFY_POSTER_DATES=1 (default off). Picture check only; typed messages and links are unchanged.

A poster's widely spaced lettering is read in pieces: "16 N 0 VE M B E R 2 0 2 6", "NOVEMBER 3 0, 20 26". Then the
checker finds no date and cannot confirm or contradict the poster.

- D-a month: a run of 1-9 tokens, each 1-4 letters or the digit 0, whose characters joined (0 read as O) spell a month
  or its short form becomes that month (all capitals stay capitals). A single token already spelling a month is left.
- D-b numbers: after a month (optionally after a comma) a run of 1-2-digit tokens joined into a day 1-31, then (after an
  optional comma) a run of 1-4-digit tokens joined into a year 1900-2100; before a month, a run of 1-digit tokens
  joined into a day 1-31.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

ENV = "CAMPUS_VERIFY_POSTER_DATES"
VERSION = "poster-dates-v1"
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december")
SHORT = ("jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec")
MONTH_SET = frozenset(MONTHS + SHORT)
MAX_MONTH_TOKENS = 9
PIECE = re.compile(r"^[A-Za-z0]{1,9}$")


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def _tokens(text: str) -> List[str]:
    """Split on spaces; a trailing comma becomes its own token ("0," -> "0", ",")."""
    out = []
    for t in (text or "").split():
        if len(t) > 1 and t.endswith(","):
            out += [t[:-1], ","]
        else:
            out.append(t)
    return out


def _styled(month: str, first: str) -> str:
    letters = [c for c in first if c.isalpha()]
    return month.upper() if letters and all(c.isupper() for c in letters) else month.capitalize()


def _number(toks: List[str], i: int, piece_len: int, lo: int, hi: int, max_tokens: int, min_tokens: int = 1):
    """(value, tokens used): the longest run of digit pieces from i joining to a number in lo..hi."""
    best, used, acc = None, 0, ""
    for k in range(i, min(len(toks), i + max_tokens)):
        t = toks[k]
        if not t.isdigit() or len(t) > piece_len:
            break
        acc += t
        if len(acc) <= 4 and lo <= int(acc) <= hi and k - i + 1 >= min_tokens:
            best, used = acc, k - i + 1
    return best, used


def _month_at(toks: List[str], i: int):
    """(month word, tokens used, rejoined?) for a month starting at i, or (None, 0, False)."""
    for m in range(min(MAX_MONTH_TOKENS, len(toks) - i), 0, -1):
        piece = toks[i:i + m]
        if not all(PIECE.match(p) for p in piece):
            continue
        word = "".join(piece).replace("0", "o").lower()
        if word in MONTH_SET:
            return word, m, m > 1 or "0" in piece[0]
    return None, 0, False


def rejoin(text: str) -> Tuple[str, List[str]]:
    """(text with spaced dates rejoined, list of 'before -> after' changes). Text without spaced dates is returned
    exactly as it was."""
    toks = _tokens(text)
    out: List[str] = []
    changes: List[str] = []
    i = 0
    while i < len(toks):
        month, n, rejoined = _month_at(toks, i)
        if month is None:
            out.append(toks[i])
            i += 1
            continue
        old = toks[i:i + n]
        new = [_styled(month, toks[i]) if rejoined else toks[i]]
        # a day of 1-digit pieces right before the month ("1 6 NOVEMBER")
        k = len(out)
        while k > 0 and len(out) - k < 2 and out[k - 1].isdigit() and len(out[k - 1]) == 1:
            k -= 1
        if len(out) - k == 2 and 1 <= int("".join(out[k:])) <= 31:
            old = out[k:] + old
            new = ["".join(out[k:])] + new
            del out[k:]
        j = i + n
        day_found_before = len(new) == 2
        best = None                                 # (day, day tokens, comma?, year, year tokens)
        for d in ((0,) if day_found_before else (0, 1, 2)):
            day = None
            if d:
                piece = toks[j:j + d]
                if len(piece) < d or not all(t.isdigit() and len(t) <= 2 for t in piece):
                    continue
                day = "".join(piece)
                if not 1 <= int(day) <= 31 or len(day) > 2:
                    continue
            k = j + d
            comma = k < len(toks) and toks[k] == ","
            k += comma
            year, used = _number(toks, k, 4, 1900, 2100, 4)
            if year is not None and not (used > 1 or len(toks[k]) == 4):
                year, used = None, 0
            cand = (day, d, comma, year, used)
            if year is not None and (best is None or best[3] is None or d + used > best[1] + best[4]):
                best = cand
            elif year is None and day is not None and best is None:
                best = cand
        if best:
            day, d, comma, year, used = best
            if day is not None:
                old += toks[j:j + d]
                new.append(day + ("," if year is not None and comma else ""))
                j += d
            if year is not None:
                if comma:
                    old.append(",")
                    j += 1
                old += toks[j:j + used]
                new.append(year)
                j += used
        joined_old = " ".join(old).replace(" ,", ",")
        joined_new = " ".join(new)
        if joined_old != joined_new:
            changes.append(f"{joined_old} -> {joined_new}")
        out.extend(new)
        i = j
    result = " ".join(out).replace(" ,", ",")
    return (result, changes) if changes else (text, [])
