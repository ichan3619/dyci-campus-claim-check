"""
Redundancy filter for ingested evidence (M11 extension).

`append_to_corpus()` already skips an exact `document_id` match (the same
post fetched twice). This module catches the other kind of redundancy:
DIFFERENT posts (different IDs) that say the same thing, e.g. an
announcement reposted, or copied twice by different people.

Design rule -- do not delete real distinctions. Announcements that differ
only in a date, time or number ("closed on October 12" vs "closed on
October 20") are near-identical as text but are NOT redundant: they are
exactly the kind of near-miss facts the verifier must tell apart. So a
pair counts as redundant only if BOTH hold:
  1. their word overlap (Jaccard) is >= `threshold` (default 0.85), AND
  2. they contain exactly the same numbers.
Pairs with different numbers and overlap >= REVIEW_THRESHOLD (0.6) are reported as
"similar but numbers differ" and KEPT, for a human to review.

Pure Python, no extra dependencies. Documents are compared pairwise, which
is fine at thesis scale (hundreds to low thousands of documents).
"""

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from src.ingestion.evidence import NormalizedEvidence

DEFAULT_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.6   # similar-but-different-numbers pairs at/above this are flagged for review

_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d+")


def _words(text: str) -> frozenset:
    return frozenset(_WORD.findall(text.lower()))


def _numbers(text: str) -> frozenset:
    return frozenset(_NUMBER.findall(text))


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of the two texts' lowercase word sets, in [0, 1]."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


@dataclass
class DedupDecision:
    record: NormalizedEvidence
    kept: bool
    reason: str = ""                       # why skipped, or why flagged
    matched_document_id: Optional[str] = None
    score: float = 0.0
    numbers_differ: bool = False           # kept, but similar text with different numbers


def filter_redundant(
    candidates: Iterable[NormalizedEvidence],
    existing: Iterable[NormalizedEvidence] = (),
    threshold: float = DEFAULT_THRESHOLD,
) -> List[DedupDecision]:
    """
    Decide, for each candidate, whether to keep it.

    Candidates are processed oldest-first (by published_date), so when two
    candidates duplicate each other the EARLIEST post is kept. Anything
    already in `existing` always wins over a candidate. A skipped
    candidate is never used as a comparison target for later ones; a kept
    one is.
    """
    known: List[NormalizedEvidence] = list(existing)
    decisions: List[DedupDecision] = []

    ordered = sorted(candidates, key=lambda r: (r.published_date or "", r.document_id))
    for rec in ordered:
        rec_numbers = _numbers(rec.text)
        best: Tuple[float, Optional[NormalizedEvidence]] = (0.0, None)
        redundant_of: Optional[NormalizedEvidence] = None
        flagged_of: Optional[NormalizedEvidence] = None
        flagged_score = 0.0

        for other in known:
            if other.document_id == rec.document_id:
                redundant_of, best = other, (1.0, other)
                break
            score = similarity(rec.text, other.text)
            if score < REVIEW_THRESHOLD:
                continue
            if _numbers(other.text) == rec_numbers:
                if score >= threshold and score > best[0]:
                    redundant_of, best = other, (score, other)
            elif score > flagged_score:
                flagged_of, flagged_score = other, score

        if redundant_of is not None:
            decisions.append(DedupDecision(
                rec, False,
                reason="same document_id" if best[0] == 1.0 and redundant_of.document_id == rec.document_id
                else "near-identical text with the same numbers",
                matched_document_id=redundant_of.document_id, score=best[0]))
            continue

        decisions.append(DedupDecision(
            rec, True,
            reason="similar text but different numbers -- kept, please review" if flagged_of else "",
            matched_document_id=flagged_of.document_id if flagged_of else None,
            score=flagged_score, numbers_differ=flagged_of is not None))
        known.append(rec)

    return decisions
