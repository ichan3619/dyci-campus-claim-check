"""
School-name alignment (improvement 1a-fix; results/round2/name_align_plan.md, approved 2026-10-25).

CAMPUS_VERIFY_NAME_ALIGN (default ON since v1.4, owner option B 2026-10-25; "0" switches it off). The checking model treats "DYCI" and "Dr. Yanga's Colleges, Inc." as different
organisations (results/round2/dyci_name_probe.json). After the normal check of a claim: if the claim names the school in
one form and the post compared (the post used, or else the top retrieved post) contains only the other form, the claim is
checked again with its name written the way that post writes it, and that second result is used. No other word changes;
there are no thresholds.
"""
from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Callable, Optional, Tuple

ENV = "CAMPUS_VERIFY_NAME_ALIGN"
LONG = re.compile(r"Dr\.?\s*Yanga[’']?s?\s+Colleges?(?:,?\s*Inc\.?)?")
SHORT = re.compile(r"\bDYCI\b")                 # "DYCI" and "DYCI's"; not "DYCIans"


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"


def aligned_claim(claim: str, post_text: str) -> Optional[str]:
    """The claim with its school name written as the post writes it, or None when the rule does not apply."""
    c_long, c_short = LONG.search(claim), SHORT.search(claim)
    if bool(c_long) == bool(c_short):                     # no name, or both forms: nothing to align
        return None
    p_long, p_short = LONG.search(post_text or ""), SHORT.search(post_text or "")
    if c_short and p_long and not p_short:
        return SHORT.sub(lambda _: p_long.group(0).rstrip(" ,"), claim)
    if c_long and p_short and not p_long:
        return LONG.sub("DYCI", claim)
    return None


def apply(claim: str, result, recheck: Callable[[str], object]) -> Tuple[object, Optional[dict]]:
    """(result, info). `recheck(text)` runs the same normal check on the aligned claim."""
    post = result.used_evidence or (result.retrieved_evidence[0] if result.retrieved_evidence else None)
    if post is None:
        return result, None
    aligned = aligned_claim(claim, post.text)
    if aligned is None:
        return result, None
    second = recheck(aligned)
    form = "“DYCI”" if SHORT.search(aligned) else "the full name"
    note = (f" Name check: the post writes the school as {form}, so the claim was checked again written that way"
            f" (first result: {result.verdict}).")
    info = {"original": claim, "aligned": aligned, "first_verdict": result.verdict, "second_verdict": second.verdict}
    return replace(second, explanation=second.explanation + note), info
