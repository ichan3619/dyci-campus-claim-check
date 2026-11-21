"""
Error analysis framework (Milestone 5, Step 9).

Categorizes WHY the full pipeline (retrieval + classification, via
ClaimVerifier) got a claim wrong, so future work knows whether to
improve retrieval, improve the classifier, or fix the dataset -- rather
than only knowing THAT something is wrong.
"""

from dataclasses import dataclass
from typing import Optional

from src.verification.verifier import VerificationResult, NO_RELEVANT_EVIDENCE, RELEVANT_EVIDENCE_FOUND


# The five categories from the M5 brief, as constants (mirrors the
# pattern already used for evidence_status in verifier.py).
RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"                # relevant evidence existed but wasn't retrieved/passed threshold
RETRIEVAL_FALSE_POSITIVE = "RETRIEVAL_FALSE_POSITIVE"  # irrelevant evidence was retrieved and used
CLASSIFICATION_FAILURE = "CLASSIFICATION_FAILURE"      # correct evidence was retrieved but relationship misjudged
DATA_LABEL_PROBLEM = "DATA_LABEL_PROBLEM"              # cannot be determined automatically -- flagged for human review
GENUINE_INSUFFICIENT = "GENUINE_INSUFFICIENT"          # correctly identified as insufficient evidence (M5 category 5)
CORRECT = "CORRECT"                                    # correct SUPPORTED/CONTRADICTED prediction -- not one of the 5 error categories, but every example needs SOME category


@dataclass
class ErrorAnalysisEntry:
    claim: str
    gold_label: str
    gold_expected_group_id: Optional[str]  # None if no real evidence exists for this claim
    system_verdict: str
    category: str
    explanation: str


def categorize_result(result: VerificationResult, gold_label: str,
                       gold_expected_group_id: Optional[str]) -> ErrorAnalysisEntry:
    """
    Compare a VerificationResult against ground truth and categorize the
    outcome.

    Inputs:
        result: what ClaimVerifier actually produced.
        gold_label: the correct label, from the dataset.
        gold_expected_group_id: the group_id of the document that
            SHOULD have been used, or None if no real evidence exists
            for this claim (mirrors threshold_experiment.py's ground
            truth construction -- see that module for why
            SUPPORTED/CONTRADICTED rows have a real expected document
            and INSUFFICIENT_EVIDENCE rows do not).

    This function does NOT try to resolve genuinely ambiguous cases --
    per the M5 instruction, DATA_LABEL_PROBLEM is returned when the
    combination of facts doesn't cleanly fit one of the other four
    categories, so a human can review it rather than the code guessing.
    """
    correct = (result.verdict == gold_label)

    # Correct AND the gold label was genuinely INSUFFICIENT_EVIDENCE
    # (no real evidence exists) -- this is success, not an error.
    if correct and gold_label == "INSUFFICIENT_EVIDENCE" and gold_expected_group_id is None:
        return ErrorAnalysisEntry(
            claim=result.claim, gold_label=gold_label,
            gold_expected_group_id=gold_expected_group_id,
            system_verdict=result.verdict, category=GENUINE_INSUFFICIENT,
            explanation="Correctly identified as insufficient evidence -- no real match exists.",
        )

    if correct:
        # Correct on SUPPORTED/CONTRADICTED, or correct INSUFFICIENT_EVIDENCE
        # via Case B (evidence existed but was correctly judged unrelated) --
        # not an error, and NOT the same as GENUINE_INSUFFICIENT above (which
        # specifically means "no real evidence exists, and that was correctly
        # recognized"). This is the general "got it right" category.
        return ErrorAnalysisEntry(
            claim=result.claim, gold_label=gold_label,
            gold_expected_group_id=gold_expected_group_id,
            system_verdict=result.verdict, category=CORRECT,
            explanation="Correct prediction -- no error to categorize.",
        )

    # From here on, result.verdict != gold_label -- something went wrong.

    real_evidence_exists = gold_expected_group_id is not None

    # Case: real evidence exists, but retrieval never found it (or found
    # it but below threshold) -- a RETRIEVAL_FAILURE.
    if real_evidence_exists and result.evidence_status == NO_RELEVANT_EVIDENCE:
        return ErrorAnalysisEntry(
            claim=result.claim, gold_label=gold_label,
            gold_expected_group_id=gold_expected_group_id,
            system_verdict=result.verdict, category=RETRIEVAL_FAILURE,
            explanation=(
                f"Real evidence (group {gold_expected_group_id}) exists, but "
                f"retrieval reported no sufficiently relevant match."
            ),
        )

    # Case: real evidence exists, retrieval found SOMETHING, but it was
    # the WRONG document -- also a RETRIEVAL_FAILURE (retrieval ranked
    # the wrong thing on top).
    if real_evidence_exists and result.used_evidence is not None \
            and result.used_evidence.document_id != gold_expected_group_id:
        return ErrorAnalysisEntry(
            claim=result.claim, gold_label=gold_label,
            gold_expected_group_id=gold_expected_group_id,
            system_verdict=result.verdict, category=RETRIEVAL_FAILURE,
            explanation=(
                f"Retrieval returned '{result.used_evidence.document_id}' instead "
                f"of the correct document '{gold_expected_group_id}'."
            ),
        )

    # Case: no real evidence exists (gold says INSUFFICIENT_EVIDENCE with
    # no true match), but the system confidently used SOME evidence and
    # produced a SUPPORTED/CONTRADICTED verdict -- a RETRIEVAL_FALSE_POSITIVE
    # (retrieval should not have been confident here at all).
    if not real_evidence_exists and result.evidence_status == RELEVANT_EVIDENCE_FOUND \
            and result.verdict in ("SUPPORTED", "CONTRADICTED"):
        return ErrorAnalysisEntry(
            claim=result.claim, gold_label=gold_label,
            gold_expected_group_id=gold_expected_group_id,
            system_verdict=result.verdict, category=RETRIEVAL_FALSE_POSITIVE,
            explanation=(
                f"No real evidence exists for this claim, but retrieval "
                f"confidently returned '{result.used_evidence.document_id}' "
                f"and classification produced a definite verdict."
            ),
        )

    # Case: real evidence exists AND retrieval found the CORRECT document,
    # but the final verdict is still wrong -- the classifier misjudged the
    # relationship. This is a CLASSIFICATION_FAILURE.
    if real_evidence_exists and result.used_evidence is not None \
            and result.used_evidence.document_id == gold_expected_group_id:
        return ErrorAnalysisEntry(
            claim=result.claim, gold_label=gold_label,
            gold_expected_group_id=gold_expected_group_id,
            system_verdict=result.verdict, category=CLASSIFICATION_FAILURE,
            explanation=(
                f"Retrieval correctly found '{gold_expected_group_id}', but the "
                f"classifier predicted {result.verdict} instead of {gold_label}."
            ),
        )

    # Anything not covered above doesn't cleanly fit one category --
    # flagged for human review rather than guessed.
    return ErrorAnalysisEntry(
        claim=result.claim, gold_label=gold_label,
        gold_expected_group_id=gold_expected_group_id,
        system_verdict=result.verdict, category=DATA_LABEL_PROBLEM,
        explanation="Outcome did not cleanly match a known error category -- needs human review.",
    )
