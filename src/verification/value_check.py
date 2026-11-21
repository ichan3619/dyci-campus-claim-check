"""
Value consistency between a claim and its evidence (round 2, after M12.3). NOT wired into the API yet.

WHY
    M12.3's post-hoc diagnosis: with the right announcement retrieved, the NLI models still missed wrong-version claims
    that differ from the announcement in ONE value (a number, a date, a duration, a place) and are phrased as a doubt
    rather than an assertion. A direct comparison of typed values does not depend on phrasing or language.

WHAT IT COMPARES (closed patterns; no model)
    date      explicit month + day, via date_check (ranges expanded; a post's own header date is not an event date)
    time      clock times: '1 PM', '10:00 am', '5:30pm'
    percent   '99.5%', '12 percent', '12 porsyento'
    count_of  'N out of M' (the pair is one value)
    duration  number + unit: '3 days', '2-day', 'tatlong araw', 'two weeks', '4 oras'
    place     a closed list of countries and major cities (English and common Tagalog names)

THE RULE (applied to the verifier's final verdict)
    A value CONFLICTS when the claim states it, the PREMISE (the passage the verifier judged, i.e. the text most
    similar to the claim) states one or more values of the same kind, and the claim's value appears nowhere in the
    whole announcement. Requiring the premise to carry the competing value is the guard: the conflicting value must
    sit in the part of the announcement that is about the claim, not anywhere in the post.
      INSUFFICIENT_EVIDENCE + conflict -> CONTRADICTED            ('value_conflict')
      SUPPORTED             + conflict -> INSUFFICIENT_EVIDENCE   ('value_conflict_withheld'; the two signals disagree)
      CONTRADICTED is never changed, and nothing changes without a conflict.
    Unlike date_check this CAN create a CONTRADICTED verdict, so false CONTRADICTED must be measured before use.
"""
from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Optional, Tuple

from src.verification.date_check import event_dates, extract_dates

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "isa": 1, "isang": 1, "dalawa": 2, "dalawang": 2, "tatlo": 3, "tatlong": 3, "apat": 4, "lima": 5, "limang": 5,
    "anim": 6, "pito": 7, "pitong": 7, "walo": 8, "walong": 8, "siyam": 9, "sampu": 10, "sampung": 10,
}
UNITS = {"day": "day", "days": "day", "araw": "day", "week": "week", "weeks": "week", "linggo": "week",
         "hour": "hour", "hours": "hour", "hr": "hour", "hrs": "hour", "oras": "hour",
         "month": "month", "months": "month", "buwan": "month", "minute": "minute", "minutes": "minute",
         "mins": "minute", "minuto": "minute"}
PLACES = {
    "philippines": "philippines", "pilipinas": "philippines", "japan": "japan", "hapon": "japan", "china": "china",
    "tsina": "china", "korea": "korea", "south korea": "korea", "taiwan": "taiwan", "thailand": "thailand",
    "vietnam": "vietnam", "singapore": "singapore", "malaysia": "malaysia", "indonesia": "indonesia", "india": "india",
    "australia": "australia", "new zealand": "new zealand", "canada": "canada", "usa": "usa", "united states": "usa",
    "america": "usa", "amerika": "usa", "mexico": "mexico", "united kingdom": "uk", "uk": "uk", "england": "uk",
    "france": "france", "germany": "germany", "italy": "italy", "spain": "spain", "dubai": "uae", "uae": "uae",
    "saudi arabia": "saudi arabia", "hong kong": "hong kong",
    "tokyo": "tokyo", "osaka": "osaka", "seoul": "seoul", "beijing": "beijing", "bangkok": "bangkok",
    "sydney": "sydney", "melbourne": "melbourne", "brisbane": "brisbane", "london": "london", "paris": "paris",
    "new york": "new york", "manila": "manila", "maynila": "manila", "quezon city": "quezon city", "cebu": "cebu",
    "davao": "davao", "baguio": "baguio", "iloilo": "iloilo", "surigao": "surigao", "bulacan": "bulacan",
    "pampanga": "pampanga", "batangas": "batangas", "laguna": "laguna", "cavite": "cavite",
}

_NUM = r"(?P<n>\d+|" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")"
DURATION = re.compile(r"\b" + _NUM + r"(?:\s*-\s*|\s+)(?:na\s+)?(?P<u>" + "|".join(sorted(UNITS, key=len, reverse=True)) + r")\b",
                      re.IGNORECASE)
TIME = re.compile(r"\b(?P<h>\d{1,2})(?::(?P<mi>\d{2}))?\s*(?P<ap>a\.?\s?m\.?|p\.?\s?m\.?)(?![a-z])", re.IGNORECASE)
PERCENT = re.compile(r"(?<![\d.])(?P<p>\d+(?:\.\d+)?)\s*(?:%|percent\b|porsyento\b|porsiyento\b)", re.IGNORECASE)
COUNT_OF = re.compile(r"\b(?P<a>\d+)\s+(?:out\s+of|of\s+the)\s+(?P<b>\d+)\b", re.IGNORECASE)
PLACE = re.compile(r"\b(?P<p>" + "|".join(re.escape(p) for p in sorted(PLACES, key=len, reverse=True)) + r")\b", re.IGNORECASE)

Slots = Dict[str, FrozenSet]


def extract_values(text: str, event_only: bool = False) -> Slots:
    """{kind: frozenset(values)} for every kind found in `text`. event_only drops a post's own header date."""
    text = text or ""
    out: Dict[str, set] = {}
    dates = event_dates(text) if event_only else extract_dates(text)
    if dates:
        out["date"] = set(dates)
    for mt in TIME.finditer(text):
        h, mi = int(mt.group("h")), int(mt.group("mi") or 0)
        if 1 <= h <= 12 and mi < 60:
            pm = mt.group("ap").lower().startswith("p")
            out.setdefault("time", set()).add(((h % 12) + (12 if pm else 0), mi))
    for mt in PERCENT.finditer(text):
        out.setdefault("percent", set()).add(round(float(mt.group("p")), 2))
    for mt in COUNT_OF.finditer(text):
        out.setdefault("count_of", set()).add((int(mt.group("a")), int(mt.group("b"))))
    for mt in DURATION.finditer(text):
        n = mt.group("n").lower()
        n = int(n) if n.isdigit() else NUMBER_WORDS[n]
        out.setdefault("duration", set()).add((n, UNITS[mt.group("u").lower()]))
    for mt in PLACE.finditer(text):
        out.setdefault("place", set()).add(PLACES[mt.group("p").lower()])
    return {k: frozenset(v) for k, v in out.items()}


def find_conflicts(claim: str, premise: str, document: str) -> List[Tuple[str, object, Tuple]]:
    """[(kind, claimed value, premise values)] for each claimed value that the premise contradicts (see module doc)."""
    claimed = extract_values(claim)
    if not claimed:
        return []
    in_premise, in_document = extract_values(premise, event_only=True), extract_values(document, event_only=True)
    in_document_any = extract_values(document)              # a value anywhere in the post (header date included)
    conflicts = []
    for kind, values in claimed.items():
        competing = in_premise.get(kind)
        if not competing:
            continue
        for v in values:
            if v not in in_document.get(kind, ()) and v not in in_document_any.get(kind, ()):
                conflicts.append((kind, v, tuple(sorted(competing, key=str))))
    return conflicts


def check_values(claim: str, verdict: str, premise: Optional[str], document: Optional[str]) -> Tuple[str, Optional[str]]:
    """(possibly changed verdict, reason or None)."""
    if verdict == "CONTRADICTED" or not premise:
        return verdict, None
    if not find_conflicts(claim, premise, document or premise):
        return verdict, None
    if verdict == "INSUFFICIENT_EVIDENCE":
        return "CONTRADICTED", "value_conflict"
    return "INSUFFICIENT_EVIDENCE", "value_conflict_withheld"
