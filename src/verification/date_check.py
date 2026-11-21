"""
Date consistency between a claim and its evidence (M12.3).

WHY
    The NLI model does not compare calendar dates reliably. Two observed failures with the real models:
      * 'sept 27 shortened class' -> SUPPORTED by "DYCI Advisory 27 September 2026 ... tomorrow, September 28, 2026,
        classes ... until 1:00 PM": the 27th is the advisory's own date, the shortened day is the 28th.
      * 'online classes sept 10 dahil sa habagat' -> CONTRADICTED by an undated notice about classes suspended
        "tomorrow" because of a typhoon: an undated notice cannot say anything about 10 September.

WHAT IT DOES (it can only WITHHOLD a verdict; it never creates or upgrades one)
    Only when the claim names an explicit calendar date (month + day):
      SUPPORTED     -> INSUFFICIENT_EVIDENCE  if none of the claim's dates is among the evidence's EVENT dates
      CONTRADICTED  -> INSUFFICIENT_EVIDENCE  if the evidence contains no explicit date at all
    A CONTRADICTED verdict against DATED evidence is kept: a wrong date is exactly what a contradiction looks like
    ('midterms October 5 to 7' vs 'September 28-30').
    EVENT dates exclude the post's own header date ("DYCI Advisory 27 September 2026 ...", "COLLEGE ADVISORY
    20 September 2026"), which is when the notice was written, not when the event happens.

    prefer_dated_documents() is the retrieval-side half: among candidates that ALREADY cleared the relevance gate, only
    the ones whose text contains a claimed date are offered to passage selection. It never admits a document that
    failed the gate and changes nothing when no candidate contains the date.

WHAT IT DOES NOT DO
    Resolve relative dates ('tomorrow', 'next Monday'): the mock posts' timestamps are re-post times, not the original
    announcement dates, so resolving 'tomorrow' against them would invent dates. Numeric dates ('9/28') are not read,
    because '24/7' would be misread. Times of day are not compared.
"""
from __future__ import annotations

import re
from typing import FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

MONTHS = {
    "january": 1, "jan": 1, "enero": 1, "february": 2, "feb": 2, "pebrero": 2, "march": 3, "mar": 3, "marso": 3,
    "april": 4, "apr": 4, "abril": 4, "may": 5, "mayo": 5, "june": 6, "jun": 6, "hunyo": 6, "july": 7, "jul": 7,
    "hulyo": 7, "august": 8, "aug": 8, "agosto": 8, "september": 9, "sept": 9, "sep": 9, "setyembre": 9,
    "october": 10, "oct": 10, "oktubre": 10, "november": 11, "nov": 11, "nobyembre": 11, "december": 12, "dec": 12,
    "disyembre": 12,
}
_M = r"(?P<m>" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
_D = r"(?P<d>\d{1,2})(?:st|nd|rd|th)?"
_D2 = r"(?P<d2>\d{1,2})(?:st|nd|rd|th)?"
_RANGE = r"(?:\s*(?:-|–|—|to|until|hanggang)\s*(?:" + _M.replace("?P<m>", "?P<m2>") + r"\s+)?" + _D2 + r")?"
# 'September 28', 'sept 28-30', 'Sept. 28 to October 2'  /  '27 September', '28 to 30 September'
MONTH_FIRST = re.compile(r"\b" + _M + r"\s+" + _D + r"\b" + _RANGE + r"(?!\d)", re.IGNORECASE)
DAY_FIRST = re.compile(r"\b(?:ika-?)?" + _D + r"(?:\s*(?:-|–|to)\s*" + _D2 + r")?\s+(?:ng\s+)?" + _M + r"\b", re.IGNORECASE)
# A post's own date at its very start: '<words> ADVISORY|ANNOUNCEMENT|MEMO|NOTICE|BULLETIN <date>'.
HEADER = re.compile(r"^\W*(?:[\w#.'’-]+\s+){0,4}?(?:advisory|announcement|memo|memorandum|notice|bulletin)\s+"
                    r"(?P<date>\d{1,2}\s+[A-Za-z]+\.?(?:\s+\d{4})?|[A-Za-z]+\.?\s+\d{1,2}(?:,?\s+\d{4})?)", re.IGNORECASE)

Date = Tuple[int, int]          # (month, day)


def _valid(m: int, d: int) -> bool:
    return 1 <= d <= 31 and not (m == 2 and d > 29) and not (m in (4, 6, 9, 11) and d > 30)


def _expand(m1: int, d1: int, m2: Optional[int], d2: Optional[int]) -> Set[Date]:
    if d2 is None:
        return {(m1, d1)} if _valid(m1, d1) else set()
    m2 = m2 or m1
    if m2 == m1 and d1 <= d2 <= 31:
        return {(m1, d) for d in range(d1, d2 + 1) if _valid(m1, d)}
    return {x for x in ((m1, d1), (m2, d2)) if _valid(*x)}     # cross-month range: endpoints only


def extract_dates(text: str) -> FrozenSet[Date]:
    """Explicit calendar dates (month, day) in `text`, ranges expanded ('September 28-30' -> 28, 29, 30)."""
    found: Set[Date] = set()
    for rx in (MONTH_FIRST, DAY_FIRST):
        for mt in rx.finditer(text or ""):
            g = mt.groupdict()
            m = MONTHS[g["m"].lower()]
            if g["m"].lower() == "may" and rx is MONTH_FIRST and not g.get("d2") and _looks_like_tagalog_may(text, mt):
                continue
            d2 = int(g["d2"]) if g.get("d2") else None
            m2 = MONTHS[g["m2"].lower()] if g.get("m2") else None
            if rx is DAY_FIRST and d2 is not None:
                found |= _expand(m, int(g["d"]), m, d2)
            else:
                found |= _expand(m, int(g["d"]), m2, d2)
    return frozenset(found)


def _looks_like_tagalog_may(text: str, match: re.Match) -> bool:
    """'may 2 sections', 'may 10 slots': Tagalog existential 'may' + a count, not the month May."""
    after = text[match.end():match.end() + 20].lower()
    return bool(re.match(r"\s*[a-z]", after)) and not re.match(r"\s*(?:,|\d{4}|to\b|-)", after)


def header_dates(text: str) -> FrozenSet[Date]:
    mt = HEADER.match(text or "")
    return extract_dates(mt.group("date")) if mt else frozenset()


def event_dates(text: str) -> FrozenSet[Date]:
    """Dates in `text` other than a post's own header date. A date that is BOTH (e.g. 'Advisory 10 September ...
    suspended today, September 10') is kept, because the body states it as the event date too."""
    text = text or ""
    mt = HEADER.match(text)
    body = text[mt.end():] if mt else text
    return extract_dates(body)


def check_verdict(claim: str, verdict: str, evidence_text: str) -> Tuple[str, Optional[str]]:
    """(possibly withheld verdict, reason or None). Only SUPPORTED / CONTRADICTED are ever changed, only to
    INSUFFICIENT_EVIDENCE, and only when the claim names an explicit date."""
    claimed = extract_dates(claim)
    if not claimed or verdict not in ("SUPPORTED", "CONTRADICTED"):
        return verdict, None
    events = event_dates(evidence_text)
    if verdict == "SUPPORTED" and not (claimed & events):
        return "INSUFFICIENT_EVIDENCE", "claimed_date_not_in_evidence"
    if verdict == "CONTRADICTED" and not (events | header_dates(evidence_text)):
        return "INSUFFICIENT_EVIDENCE", "evidence_has_no_date"
    return verdict, None


def prefer_dated_documents(claim: str, candidates: Sequence, text_of=lambda d: d.text) -> List:
    """
    Among gate-passing candidates, keep only those whose text contains a claimed date -- if at least one does.
    Returns the candidates unchanged (same order) when the claim has no date or no candidate contains it.
    """
    claimed = extract_dates(claim)
    if not claimed:
        return list(candidates)
    dated = [c for c in candidates if claimed & extract_dates(text_of(c))]
    return dated or list(candidates)


MONTH_NAMES = ("January February March April May June July August September October November December").split()


def describe(dates: Iterable[Date]) -> str:
    return ", ".join(f"{MONTH_NAMES[m - 1]} {d}" for m, d in sorted(dates))
