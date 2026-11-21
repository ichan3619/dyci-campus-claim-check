"""
Retrieval relevance threshold experiment (Milestone 5, Step 8).

WHAT THIS MEASURES: whether retrieval's similarity score can reliably
distinguish "a genuinely matching campus document exists for this claim"
from "no matching document exists." This is a RETRIEVAL RELEVANCE
question -- it has nothing to do with whether the claim itself is true.
A claim can have a perfect retrieval match (relevant evidence found) and
still be FALSE (CONTRADICTED) -- that judgment belongs to classification,
not retrieval. This experiment only tunes retrieval's "found something
worth classifying" vs. "found nothing" decision.

GROUND TRUTH CONSTRUCTION: using labels already present in the dataset,
without any new annotation:
  - SUPPORTED/CONTRADICTED claims: a real matching document DOES exist
    (its own group_id's evidence). These are POSITIVE examples.
  - INSUFFICIENT_EVIDENCE claims (both the empty-evidence "Case A" kind
    AND the non-empty-but-unrelated "Case B" kind): no document in the
    corpus genuinely answers this specific claim. These are NEGATIVE
    examples for this experiment -- even Case B rows, where SOME
    document exists in the corpus, because that document is not the
    correct answer to THIS claim.

This reframes the three-class SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE
problem as a simpler binary one purely for this retrieval-only
experiment: "does genuinely relevant evidence exist for this claim, yes
or no." This simplification is deliberate and is documented here so it
is not mistaken for the actual 3-class task.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.retrieval.retriever import Retriever


@dataclass
class LabeledQuery:
    claim: str
    expected_group_id: Optional[str]  # None means "no correct document exists"


@dataclass
class ThresholdResult:
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    n_queries: int


def build_labeled_queries(records: List[dict]) -> List[LabeledQuery]:
    """
    Build the ground truth for the threshold experiment directly from
    existing dataset labels -- no new annotation required.
    """
    queries = []
    for r in records:
        if r["label"] in ("SUPPORTED", "CONTRADICTED"):
            queries.append(LabeledQuery(claim=r["claim"], expected_group_id=r["group_id"]))
        elif r["label"] == "INSUFFICIENT_EVIDENCE":
            queries.append(LabeledQuery(claim=r["claim"], expected_group_id=None))
    return queries


def evaluate_threshold(retriever: Retriever, labeled_queries: List[LabeledQuery],
                        threshold: float) -> ThresholdResult:
    """
    Test one candidate threshold against all labeled queries.

    Decision rule: retrieval "confidently matches" if its top-1 result's
    score >= threshold. This is exactly the same decision rule already
    used in src/verification/verifier.py -- this experiment tests
    candidate values for that same threshold, it does not introduce a
    different decision mechanism.
    """
    tp = fp = tn = fn = 0

    for query in labeled_queries:
        results = retriever.retrieve(query.claim, top_k=1)
        top_result = results[0] if results else None
        confidently_matched = top_result is not None and top_result.score >= threshold

        has_real_match = query.expected_group_id is not None

        if has_real_match:
            correct_match = confidently_matched and top_result.document_id == query.expected_group_id
            if correct_match:
                tp += 1
            else:
                fn += 1
        else:
            if confidently_matched:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labeled_queries) if labeled_queries else 0.0

    return ThresholdResult(
        threshold=threshold, true_positives=tp, false_positives=fp,
        true_negatives=tn, false_negatives=fn, precision=precision,
        recall=recall, f1=f1, accuracy=accuracy, n_queries=len(labeled_queries),
    )


def sweep_thresholds(retriever: Retriever, labeled_queries: List[LabeledQuery],
                      candidate_thresholds: List[float]) -> List[ThresholdResult]:
    """Run evaluate_threshold() for every candidate value."""
    return [evaluate_threshold(retriever, labeled_queries, t) for t in candidate_thresholds]
