"""
Other-school check (C2, CAMPUS_VERIFY_OTHER_SCHOOL_CHECK, default off). Version 1 was pre-registered in
results/round2/other_school_plan.json (passed, not released after the release review). Version 2 (detector v2, shared with
C1; results/round2/other_school_v2_plan_draft.md) is not yet frozen or evaluated.

When the message names a school other than DYCI (src/api/school_scope.detect, C1's detector) and the post the verdict
USED does not mention that school, a SUPPORTED/CONTRADICTED verdict becomes INSUFFICIENT_EVIDENCE. It never creates a
verdict or reverses SUPPORTED<->CONTRADICTED. Uncertain identification never changes a verdict:
    - ambiguous abbreviations (AMBIGUOUS, e.g. "BSU") do not trigger the rule;
    - a fixed short form (SHORT_FORMS) or the name's distinctive words in the post count as a mention
      (school_scope.mentions); an uncertain partial match counts as mentioned.
"""
from dataclasses import replace
from typing import List

from src.api.school_scope import AMBIGUOUS, SHORT_FORMS, detect, mentions  # noqa: F401  (shared with C1, detector v2)

VERSION = "other-school-v2"
SC = ("SUPPORTED", "CONTRADICTED")


def schools_to_check(message: str) -> List[str]:
    return [s for s in detect(message) if s not in AMBIGUOUS]


def check(message: str, result):
    """(result, info). Withholds SUPPORTED/CONTRADICTED -> INSUFFICIENT_EVIDENCE when a reliably identified school
    named in the message is not mentioned by the post used. Nothing else changes."""
    schools = schools_to_check(message)
    if not schools or result.verdict not in SC or result.used_evidence is None:
        return result, None
    missing = [s for s in schools if not mentions(result.used_evidence.text, s)]
    info = {"version": VERSION, "schools": schools, "missing_in_post": missing, "withheld": bool(missing),
            "original_verdict": result.verdict}
    if not missing:
        return result, info
    reason = (" Other-school check: the post used does not mention " + ", ".join(missing)
              + ", so the verdict was withheld.")
    return replace(result, verdict="INSUFFICIENT_EVIDENCE", explanation=result.explanation + reason), info
