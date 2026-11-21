"""
Rule N: swapped office, level, grade or venue names (results/round2/unit_check_plan.md).
CAMPUS_VERIFY_UNIT_CHECK=1 (default off). A layer on the final answer of _answer (src/api/main.py); the frozen value
check is not changed.

A closed list of campus units in four kinds (office, level, grade, venue), no model and no thresholds. A conflict: the
message names a unit of some kind, the passage judged names one or more units of the same kind, and the message's unit
appears nowhere in the post used, unless the post covers the whole kind ("all levels", "all offices", ...). The header
label ("COLLEGE ADVISORY", "DYCI ADVISORY") is not read as a unit. SUPPORTED + conflict -> INSUFFICIENT_EVIDENCE; nothing
else changes (never creates CONTRADICTED or SUPPORTED).
"""
from __future__ import annotations

import os
import re
from typing import Dict, FrozenSet, Optional

ENV = "CAMPUS_VERIFY_UNIT_CHECK"
VERSION = "unit-check-v1"

UNITS = {
    "office": {
        "registrar": r"registrar(?:['’]?s)?",
        "accounting": r"accounting|cashier(?:['’]?s)?|finance\s+office|treasury",
        "library": r"library",
        "clinic": r"clinic|health\s+services|infirmary",
        "guidance": r"guidance",
        "student affairs": r"student\s+affairs|osa",
        "admissions": r"admissions?\s+office|admissions",
        "human resources": r"human\s+resources?|hrd|hr\s+office",
        "IT office": r"mis(?:\s+office)?|it\s+office",
        "president's office": r"president['’]?s\s+office|office\s+of\s+the\s+president",
        "dean's office": r"dean['’]?s\s+office|office\s+of\s+the\s+dean",
        "principal's office": r"principal['’]?s\s+office|office\s+of\s+the\s+principal",
    },
    "level": {
        "college": r"college",
        "senior high school": r"senior\s+high(?:\s+school)?|shs",
        "junior high school": r"junior\s+high(?:\s+school)?|jhs",
        "elementary": r"elementary|grade\s+school",
        "basic education": r"basic\s+education",
        "graduate school": r"graduate\s+school",
    },
    "venue": {
        "gymnasium": r"gym(?:nasium)?",
        "auditorium": r"auditorium",
        "open grounds": r"open\s+grounds?|quadrangle|quad",
        "AVR": r"avr|audio[\s-]?visual\s+room",
        "chapel": r"chapel",
        "canteen": r"canteen",
        "covered court": r"covered\s+court",
        "main building": r"main\s+building",
    },
}
_PATTERNS = {kind: [(name, re.compile(r"\b(?:" + pattern + r")\b", re.IGNORECASE)) for name, pattern in units.items()]
             for kind, units in UNITS.items()}
GRADE = re.compile(r"\bgrade\s+(1[0-2]|[1-9])\b", re.IGNORECASE)
HEADER_LABEL = re.compile(r"\b(?:DYCI|COLLEGE)\s+ADVISORY\b", re.IGNORECASE)
COVERS_LEVELS = re.compile(r"\ball\s+(?:year\s+)?levels\b|\ball\s+departments\b|\bwhole\s+(?:school|campus)\b|"
                           r"\bentire\s+(?:school|campus)\b|\bcampus\s+(?:is|will\s+be)\s+closed\b", re.IGNORECASE)
COVERS_OFFICES = re.compile(r"\ball\s+(?:administrative\s+)?offices\b", re.IGNORECASE)


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


# Rule W v2 V2-a (CAMPUS_VERIFY_UNIT_AGENCIES, default off; results/round2/website_source_v2_plan.md): government agencies
# as a fifth kind, compared like the others.
AGENCIES_ENV = "CAMPUS_VERIFY_UNIT_AGENCIES"
AGENCIES = {
    "DILG": r"dilg|department\s+of\s+(?:the\s+)?interior(?:\s+and\s+local\s+government)?",
    "DepEd": r"deped|department\s+of\s+education",
    "CHED": r"ched|commission\s+on\s+higher\s+education",
    "TESDA": r"tesda",
    "DOH": r"doh|department\s+of\s+health",
    "PAGASA": r"pagasa",
    "NDRRMC": r"ndrrmc",
    "provincial disaster office": r"pdrrmo|pdrrmc|provincial\s+disaster\s+risk\s+reduction\w*(?:\s+\w+)?\s+office",
    "municipal disaster office": r"mdrrmo|mdrrmc|municipal\s+disaster\s+risk\s+reduction\w*(?:\s+\w+)?\s+office",
    "barangay disaster office": r"bdrrmo|bdrrmc",
    "provincial government": r"provincial\s+government|province\s+of\s+bulacan|provincial\s+governor",
    "municipal government": r"municipal\s+government|municipality\s+of\s+bocaue|lgu|local\s+government(?:\s+unit)?",
    "Malacanang": r"malaca[nñ]ang|office\s+of\s+the\s+president",
}
_AGENCY_PATTERNS = [(name, re.compile(r"\b(?:" + p + r")\b", re.IGNORECASE)) for name, p in AGENCIES.items()]


def agencies_enabled() -> bool:
    return os.environ.get(AGENCIES_ENV, "0").strip() == "1"


def units(text: str) -> Dict[str, FrozenSet[str]]:
    """{kind: {unit names}} named in `text` (the header label is ignored)."""
    text = HEADER_LABEL.sub(" ", text or "")
    found = {}
    if agencies_enabled():
        names = frozenset(name for name, rx in _AGENCY_PATTERNS if rx.search(text))
        if names:
            found["agency"] = names
    for kind, patterns in _PATTERNS.items():
        names = frozenset(name for name, rx in patterns if rx.search(text))
        if names:
            found[kind] = names
    grades = frozenset("grade " + m.group(1) for m in GRADE.finditer(text))
    if grades:
        found["grade"] = grades
    return found


def covered(kind: str, document: str) -> bool:
    if kind in ("level", "grade"):
        return bool(COVERS_LEVELS.search(document or ""))
    if kind == "office":
        return bool(COVERS_OFFICES.search(document or ""))
    return False


def conflict(message: str, premise: str, document: str) -> Optional[dict]:
    """The first conflict between the message's units and the post's, or None (see the module doc)."""
    claimed = units(message)
    if not claimed:
        return None
    in_premise, in_document = units(premise or document), units(document)
    for kind in ("office", "level", "grade", "venue", "agency"):
        if kind not in claimed or covered(kind, document):
            continue
        competing = in_premise.get(kind)
        if not competing:
            continue
        missing = sorted(u for u in claimed[kind] if u not in in_document.get(kind, frozenset()))
        if missing:
            return {"version": VERSION, "kind": kind, "message_unit": missing[0], "post_units": sorted(competing),
                    "action": "withheld"}
    return None


def note(info: dict) -> str:
    return (f" The DYCI post is about {' / '.join(info['post_units'])}, not {info['message_unit']}, so this is not "
            f"confirmed.")
