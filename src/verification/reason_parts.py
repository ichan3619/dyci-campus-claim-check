"""
Check the parts of a message, rule P (CAMPUS_VERIFY_REASON_PARTS, default off; results/round2/reason_parts_plan.md).
Applied at the end of _answer in src/api/main.py; English messages only.

The checker judges a whole message against one passage of one post. "no classes on november 12 due to security threat"
is then not confirmed: the passage judged says "Classes ... are suspended immediately", the reason sits in the next
sentence, and the classifier does not infer "because" from two statements. Rule P splits the message at the first reason
marker into the main claim and the reason:
    - the main claim goes through the full pipeline (every adopted safeguard), with P off inside that check;
    - the reason is checked only when the main claim is SUPPORTED, against every sentence and every pair of
      neighbouring sentences of the main claim's post, as the hypothesis "There is <reason>.":
      backed (any piece SUPPORTED), different (none SUPPORTED, one CONTRADICTED), not_stated (otherwise).
Decision: a whole-message SUPPORTED is kept; otherwise main SUPPORTED + reason backed -> SUPPORTED on the main claim's
post; main SUPPORTED + reason not backed -> INSUFFICIENT_EVIDENCE; main not SUPPORTED -> unchanged. Never creates
CONTRADICTED.
"""
import os
import re
from typing import List, Optional, Tuple

ENV = "CAMPUS_VERIFY_REASON_PARTS"
VERSION = "reason-parts-v1"
MARKER = re.compile(r"\s+(?:due\s+to|because\s+of|because|owing\s+to|caused\s+by)\s+", re.IGNORECASE)
MIN_MAIN_WORDS = 3
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"          # default off (reason_parts_plan.md)


def split(message: str) -> Optional[Tuple[str, str]]:
    """(main claim, reason) at the first reason marker, or None."""
    m = MARKER.search(message or "")
    if not m:
        return None
    main = message[:m.start()].strip(" ,;:-")
    reason = message[m.end():].strip(" .!?,;:-")
    if len(main.split()) < MIN_MAIN_WORDS or not reason:
        return None
    return main, reason


def hypothesis(reason: str) -> str:
    return f"There is {reason.strip()}."


def pieces(text: str) -> List[str]:
    """Every sentence and every pair of neighbouring sentences of a post."""
    sents = [s.strip() for s in SENTENCE_END.split(text or "") if s.strip()]
    return sents + [f"{a} {b}" for a, b in zip(sents, sents[1:])]


def check_reason(classify, reason: str, post_text: str) -> Tuple[str, Optional[str]]:
    """(label, deciding piece): backed / different / not_stated."""
    hyp = hypothesis(reason)
    contradicted = None
    for p in pieces(post_text):
        label = classify(hyp, p).label
        if label == "SUPPORTED":
            return "backed", p
        if label == "CONTRADICTED" and contradicted is None:
            contradicted = p
    return ("different", contradicted) if contradicted is not None else ("not_stated", None)


def decide(whole_verdict: str, main_verdict: str, reason_label: Optional[str]) -> Tuple[str, str]:
    """(new verdict, action) by the plan's table."""
    if whole_verdict == "SUPPORTED":
        return whole_verdict, "kept_supported"
    if main_verdict != "SUPPORTED":
        return whole_verdict, "main_not_supported"
    if reason_label == "backed":
        return "SUPPORTED", "both_backed"
    return "INSUFFICIENT_EVIDENCE", "reason_" + (reason_label or "not_stated")
