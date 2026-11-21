"""
Tense guard (R2; results/round2/english_fixes_plan.md, committed 1015def). Setting CAMPUS_VERIFY_TENSE_GUARD, default off.

A SUPPORTED verdict on an English message becomes INSUFFICIENT_EVIDENCE when the checked statement says something HAS
happened (PAST) and the passage it rests on only says it WILL happen (FUTURE): "Has the new department name been
announced?" against "The official name ... will be formally announced tomorrow." The patterns are fixed in the plan.
It never touches CONTRADICTED or INSUFFICIENT_EVIDENCE and never creates a verdict.
"""
import os
import re
from dataclasses import replace

ENV = "CAMPUS_VERIFY_TENSE_GUARD"
VERSION = "tense-guard-v1"
PAST = re.compile(r"\b(has|have|had)\b[^.?!]{0,60}?\bbeen\b"
                  r"|\b(was|were)\s+(\w+\s+){0,2}?(\w{3,}ed|held|done|given|made|won|signed|announced|conducted|completed"
                  r"|cancelled|postponed|moved)\b"
                  r"|\balready\b", re.I)
FUTURE = re.compile(r"\b(will|shall)\b|\bto be (held|announced|signed|conducted|released)\b|\bupcoming\b"
                    r"|\b(is|are) scheduled\b", re.I)
REASON = " Tense guard: the post only says this will happen; it does not say it has happened."


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def check(statement: str, result):
    """(result, info). info is None when the rule does not apply."""
    passage = getattr(getattr(result, "selected_passage", None), "text", None) or ""
    if result.verdict != "SUPPORTED" or not PAST.search(statement or "") or not FUTURE.search(passage):
        return result, None
    info = {"version": VERSION, "past": PAST.search(statement).group(0), "future": FUTURE.search(passage).group(0),
            "withheld": True, "original_verdict": result.verdict}
    return replace(result, verdict="INSUFFICIENT_EVIDENCE", explanation=result.explanation + REASON), info
