"""
Sentence-level checking for Image Check, Part B Stage 1 (results/round2/image_sentence_check_partB_decision_record.md).

Core logic only: NOT wired into /verify/image yet (Stage 2), so the image route behaves exactly as before.
  plan_sentences(text)              split the cleaned OCR text into segments, in reading order, and mark which are checked
  check_statements(plan, check)     check each statement with the given check function (verify_text at run time), apply
                                    the existing agreement check per SUPPORTED/CONTRADICTED statement, and map outcomes
  count_outcomes / aggregate        k matches, c contradictions, i inconclusive; the approved labels for c = 0
  select_basis(text, plan)          "sentences", "whole_text_fallback" or "none" (decision B1)

Approved rules: S1 abbreviation list, S2 minimum length (4 words, 3 letters), at most 12 checked statements, the C1/C3
labels, and C2 option B (2026-10-15): a contradiction gets the strong headline only when anchored (another statement of
the image matches the same post), otherwise the soft headline. Owner condition: revert to option A if C2 changes results
too much (aggregate() records option A's headline for that check). The text checker, verdicts, the agreement rule and
the language policy are used unchanged.
"""
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from src.api.schemas import has_enough_letters
from src.image.normalize import agreement_variant

MAX_CHECKED = 12
MIN_WORDS = 4
SC = ("SUPPORTED", "CONTRADICTED")

# S1 (approved 2026-10-15): never split after these (case-insensitive, whole word including its dots).
ABBREVIATIONS = frozenset("""
mr. mrs. ms. dr. dra. prof. engr. arch. atty. hon. rev. fr. msgr. bro. sr. jr. inc. co. corp. ltd. no. nos. st. vs. etc.
e.g. i.e. a.m. p.m. sta. sto. brgy.
""".split())

# A split point: one or more of . ! ? followed by whitespace, then an uppercase letter, a digit, or an opening quote or
# bracket. Marks inside numbers and times (1.00, 1,218.50, 1:00) are never followed by whitespace, so never split.
_BOUNDARY = re.compile(r"([.!?]+)(\s+)(?=[A-Z0-9\"'“‘(\[])")
_LEADING = "\"'“‘([{"


@dataclass
class Segment:
    index: int
    text: str
    status: str          # "to_check" | "not_checked_short" | "not_checked_limit"


@dataclass
class StatementResult:
    index: int
    text: str
    result: object                    # the check function's response (a VerificationResultModel at run time)
    verdict: Optional[str]            # SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE | None, unchanged
    agreement: Optional[dict]         # {"rule": "final_full_stop_toggled", "agreed": bool}, or None when not applied
    status: str                       # "checked" | "uncertain"
    outcome: str                      # "match" | "contradiction" | "inconclusive"


@dataclass
class Counts:
    n: int = 0
    k: int = 0
    c: int = 0
    i: int = 0
    not_checked_short: int = 0
    not_checked_limit: int = 0


STRONG_CONFLICT = "A statement conflicts with an announcement"
SOFT_CONFLICT = "A statement may conflict with an announcement. Check the post."


def _ends_with_abbreviation(before: str, marks: str) -> bool:
    if marks != ".":
        return False
    words = before.split()
    if not words:
        return False
    token = words[-1].lstrip(_LEADING).lower() + "."
    return token in ABBREVIATIONS


def split_segments(text: str) -> List[str]:
    """Segments of `text` in reading order (S1). Empty pieces are dropped; nothing is reordered or rewritten."""
    text = text or ""
    pieces, start = [], 0
    for m in _BOUNDARY.finditer(text):
        if _ends_with_abbreviation(text[start:m.start()], m.group(1)):
            continue
        pieces.append(text[start:m.end(1)])
        start = m.end()
    pieces.append(text[start:])
    return [p.strip() for p in pieces if p.strip()]


def is_eligible(segment: str) -> bool:
    """S2: at least 4 whitespace-separated words and at least 3 letters."""
    return len(segment.split()) >= MIN_WORDS and has_enough_letters(segment)


def plan_sentences(text: str) -> List[Segment]:
    """Every segment with its status: the first 12 eligible ones are "to_check"; later eligible ones are
    "not_checked_limit"; ineligible ones are "not_checked_short"."""
    plan, eligible = [], 0
    for index, piece in enumerate(split_segments(text)):
        if not is_eligible(piece):
            status = "not_checked_short"
        elif eligible < MAX_CHECKED:
            status, eligible = "to_check", eligible + 1
        else:
            status = "not_checked_limit"
        plan.append(Segment(index=index, text=piece, status=status))
    return plan


def check_statements(plan: List[Segment], check: Callable[[str], object]) -> List[StatementResult]:
    """Check each "to_check" segment, one at a time, in order. `check(text)` must return an object with a `verdict`
    attribute (verify_text's response at run time). For a SUPPORTED/CONTRADICTED statement, the existing agreement check
    runs exactly as on the whole text: the statement with its final full stop toggled; a different verdict makes the
    statement "uncertain". Any exception from `check` (for example HTTP 503 when busy) propagates: no partial results."""
    results = []
    for seg in plan:
        if seg.status != "to_check":
            continue
        response = check(seg.text)
        verdict = getattr(response, "verdict", None)
        agreement = None
        if verdict in SC:
            variant = agreement_variant(seg.text)
            if variant is not None:
                agreement = {"rule": "final_full_stop_toggled",
                             "agreed": getattr(check(variant), "verdict", None) == verdict}
        uncertain = agreement is not None and not agreement["agreed"]
        outcome = ("match" if verdict == "SUPPORTED" and not uncertain else
                   "contradiction" if verdict == "CONTRADICTED" and not uncertain else "inconclusive")
        results.append(StatementResult(index=seg.index, text=seg.text, result=response, verdict=verdict,
                                       agreement=agreement, status="uncertain" if uncertain else "checked",
                                       outcome=outcome))
    return results


def count_outcomes(plan: List[Segment], results: List[StatementResult]) -> Counts:
    k = sum(r.outcome == "match" for r in results)
    c = sum(r.outcome == "contradiction" for r in results)
    i = sum(r.outcome == "inconclusive" for r in results)
    return Counts(n=len(results), k=k, c=c, i=i,
                  not_checked_short=sum(s.status == "not_checked_short" for s in plan),
                  not_checked_limit=sum(s.status == "not_checked_limit" for s in plan))


def _post_id(result: StatementResult) -> Optional[str]:
    used = getattr(result.result, "used_evidence", None)
    return getattr(used, "document_id", None) if used is not None else None


def anchored_posts(results: List[StatementResult]) -> List[str]:
    """C2 (owner decision 2026-10-15, option B): posts that a contradicted statement used AND that at least one matching
    statement of the same image also used. An edited copy of a real post typically matches that post elsewhere and
    contradicts it in the changed detail; a contradiction from an unrelated post has no such anchor."""
    matched = {_post_id(r) for r in results if r.outcome == "match"} - {None}
    return sorted({_post_id(r) for r in results if r.outcome == "contradiction"} & matched)


def aggregate(counts: Counts, results: List[StatementResult] = ()) -> dict:
    """The approved aggregate label (C1/C3, C2 option B) for sentence-based results. INSUFFICIENT_EVIDENCE, no verdict and
    uncertain statements are inconclusive: never a match, never evidence of falsity. Never an image-level "true" or
    "fake". Statement verdicts are never changed here; for c >= 1 only the headline's strength depends on anchoring.
    `option_a_kind` records what option A (every contradiction strong) would show, for the owner's revert check."""
    if counts.n == 0:
        raise ValueError("no checked statements: use the whole-text fallback (select_basis)")
    if counts.c >= 1:
        anchors = anchored_posts(list(results))
        if anchors:
            return {"kind": "conflict", "label": STRONG_CONFLICT, "anchored_posts": anchors, "option_a_kind": "conflict"}
        return {"kind": "possible_conflict", "label": SOFT_CONFLICT, "anchored_posts": [], "option_a_kind": "conflict"}
    if counts.k == 0:
        return {"kind": "not_confirmed", "label": "Not confirmed"}
    if counts.k == counts.n:
        return {"kind": "all_match", "label": f"All {counts.n} checked statements match an announcement"}
    return {"kind": "partly_confirmed", "label": f"Partly confirmed: {counts.k} of {counts.n} checked statements match"}


def select_basis(text: str, plan: List[Segment]) -> str:
    """B1: "none" when the text has fewer than 3 letters (as today: no check); "sentences" when at least one segment is
    checked; otherwise "whole_text_fallback" (the existing whole-text check is the only check)."""
    if not has_enough_letters(text):
        return "none"
    return "sentences" if any(s.status == "to_check" for s in plan) else "whole_text_fallback"
