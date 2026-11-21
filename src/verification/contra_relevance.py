"""
Rule K: a "Contradicted" answer needs a post about the same thing (results/round2/contra_relevance_plan.md).
CAMPUS_VERIFY_CONTRA_RELEVANCE=1 (default off).

When the answer is CONTRADICTED and has a used post, the share of the claim's content words that appear in the post's
text is computed (content words as rule O v3: common words, months, years, day numbers and times removed; "dyci"
removed; plural and 5-letter stem tolerance). Below MIN_SHARE the answer becomes INSUFFICIENT_EVIDENCE; the post stays
shown. SUPPORTED, INSUFFICIENT_EVIDENCE and withheld answers are never touched.
"""
from __future__ import annotations

import os
from typing import Optional

ENV = "CAMPUS_VERIFY_CONTRA_RELEVANCE"
VERSION = "contra-relevance-v1"
MIN_SHARE = 1 / 3
IGNORED = frozenset({"dyci"})
NOTE = (" Relevance check: the closest DYCI post is about something else, so it cannot settle this message; it is shown "
        "for reference.")


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def share(claim: str, post_text: str) -> Optional[dict]:
    """{"share", "words", "found"} for the claim against the post, or None when the claim has no content words."""
    from src.verification.org_pages import _appears, _tokens, content_words
    words = [w for w in content_words(claim or "") if w not in IGNORED]
    if not words:
        return None
    post_words = set(_tokens(post_text or ""))
    found = [w for w in words if _appears(w, post_words)]
    return {"share": round(len(found) / len(words), 3), "words": words, "found": found,
            "below": 3 * len(found) < len(words)}          # exact: below one third (MIN_SHARE), no rounding


def check(claim: str, verdict: Optional[str], post_text: Optional[str]) -> Optional[dict]:
    """The contra_relevance record when K turns this CONTRADICTED into INSUFFICIENT_EVIDENCE, else None."""
    if verdict != "CONTRADICTED" or not post_text:
        return None
    s = share(claim, post_text)
    if s is None or not s.pop("below"):
        return None
    return {"version": VERSION, **s}
