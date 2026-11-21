"""
Side-by-side model comparison (Milestone 7).

Responsibility of this module, and ONLY this module: run TWO classifiers
over the SAME held-out test examples (using the existing, unmodified
Milestone 5 evaluate_model() for each), and align their predictions
row-by-row for reporting.

This module contains NO retrieval, NO model training/loading, and NO
metric computation of its own -- it is a thin orchestration layer over
src.evaluation.evaluate.evaluate_model() and
src.evaluation.splitting.split_by_group(), exactly the same
"orchestrate, don't reimplement" pattern already used by
src/verification/verifier.py (orchestrates retrieval+classification)
and src/evaluation/error_analysis.py (orchestrates verification+ground
truth). Per the Milestone 7 design review, this module deliberately does
NOT use error_analysis.py or ClaimVerifier -- both operate on full
retrieval+classification pipeline output, which would mix retrieval
behavior into what must be a PURE classifier-vs-classifier comparison.

============================================================================
IMPORTANT METHODOLOGICAL CAVEAT -- read before interpreting any numbers
============================================================================
As established in the Milestone 7 Step 1 design review: with the current
development dataset (25 records, 13 classification-usable groups), the
held-out test split produced by split_by_group() contains exactly THREE
examples -- one per label. This is far too small to support any claim
that one model is generally better than the other. Every function and
script built on this module labels its output a "PIPELINE VALIDATION
RUN" for exactly this reason -- see build_comparison_report() and
scripts/run_model_comparison.py.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.evaluation.evaluate import evaluate_model, EvaluationRun, ClaimEvidenceModel
from src.evaluation.metrics import EvaluationMetrics


# Threshold below which the test set is considered too small to support
# ANY comparative claim, even a tentative one. Not a scientifically
# derived number -- a conservative, clearly-labeled development
# placeholder, in the same spirit as retriever.py's
# DEVELOPMENT_ONLY_RELEVANCE_HINT_THRESHOLD from Milestone 2.
SMALL_TEST_SET_WARNING_THRESHOLD = 30


@dataclass
class PairedPrediction:
    """
    One test example's gold label alongside BOTH models' predictions,
    aligned by position -- this is what makes a genuine side-by-side
    comparison possible, rather than two separate, unaligned reports.

    Fields:
        claim, evidence, gold_label: the test example itself.
        baseline_label, baseline_confidence: TF-IDF+LR's prediction and
            its confidence for that prediction (see
            ClassificationResult.confidence -- a genuine probability
            over this model's own three classes, NOT a calibrated
            real-world probability that the claim is true).
        nli_label, nli_confidence: the pretrained NLI model's prediction
            and confidence -- same caveat applies, independently, to
            this model's own softmax output.
        models_agree: True if both models predicted the same label,
            regardless of whether that label is correct -- useful for
            quickly spotting examples where the two approaches diverge,
            which is often more informative than either model's
            accuracy alone.
    """
    claim: str
    evidence: str
    gold_label: str
    baseline_label: str
    baseline_confidence: Optional[float]
    nli_label: str
    nli_confidence: Optional[float]
    models_agree: bool


@dataclass
class ComparisonReport:
    """
    The complete output of comparing two models on the same test set.

    n_test_examples / is_small_test_set: carried at the top level
    (not buried inside metrics) specifically so callers CANNOT easily
    forget to check and display this -- see
    scripts/run_model_comparison.py, which prints this warning both
    before and after the detailed results.
    """
    baseline_metrics: EvaluationMetrics
    nli_metrics: EvaluationMetrics
    paired_predictions: List[PairedPrediction]
    n_test_examples: int
    is_small_test_set: bool


def compare_models(baseline_model: ClaimEvidenceModel, nli_model: ClaimEvidenceModel,
                    test_examples: List[dict]) -> ComparisonReport:
    """
    Evaluate both models on the SAME test_examples and return a paired
    comparison.

    Input:
        baseline_model, nli_model: any two objects implementing
            classify(claim, evidence) -> ClassificationResult -- in
            practice ClaimEvidenceClassifier and NLIClassifier, but this
            function does not import or reference either concrete class,
            exactly like evaluate_model() itself.
        test_examples: the SAME list of examples for both models (the
            caller is responsible for producing this via
            split_by_group() ONCE and passing the resulting test split
            to both -- this function does not re-split, to guarantee
            both models are compared on identical data).

    Output: a ComparisonReport with both models' aggregate metrics and
    a row-by-row paired prediction list.
    """
    if len(test_examples) == 0:
        raise ValueError("cannot compare models over zero test examples")

    baseline_run = evaluate_model(baseline_model, test_examples)
    nli_run = evaluate_model(nli_model, test_examples)

    paired = _pair_predictions(baseline_run, nli_run)

    n_test = len(test_examples)
    return ComparisonReport(
        baseline_metrics=baseline_run.metrics,
        nli_metrics=nli_run.metrics,
        paired_predictions=paired,
        n_test_examples=n_test,
        is_small_test_set=(n_test < SMALL_TEST_SET_WARNING_THRESHOLD),
    )


def _pair_predictions(baseline_run: EvaluationRun, nli_run: EvaluationRun) -> List[PairedPrediction]:
    """
    Align two EvaluationRuns' predictions by position.

    This relies on both runs having been produced from the SAME
    test_examples list, in the SAME order (which compare_models()
    guarantees by construction -- it calls evaluate_model() on the
    identical list for both models, and evaluate_model() iterates
    examples in the order given, so positions line up). A defensive
    length/claim check below catches the case where that assumption was
    somehow violated, rather than silently misaligning rows.
    """
    if len(baseline_run.predictions) != len(nli_run.predictions):
        raise ValueError(
            "baseline and NLI prediction counts differ -- were they evaluated "
            "on different test sets? compare_models() requires both models to "
            "be run on the identical example list."
        )

    paired = []
    for baseline_pred, nli_pred in zip(baseline_run.predictions, nli_run.predictions):
        if baseline_pred.claim != nli_pred.claim or baseline_pred.evidence != nli_pred.evidence:
            raise ValueError(
                "baseline and NLI predictions are misaligned (different claim/"
                "evidence at the same position) -- were they evaluated on "
                "differently-ordered test sets?"
            )

        baseline_confidence = None
        if baseline_pred.label_scores:
            baseline_confidence = baseline_pred.label_scores.get(baseline_pred.predicted_label)

        nli_confidence = None
        if nli_pred.label_scores:
            nli_confidence = nli_pred.label_scores.get(nli_pred.predicted_label)

        paired.append(PairedPrediction(
            claim=baseline_pred.claim,
            evidence=baseline_pred.evidence,
            gold_label=baseline_pred.true_label,
            baseline_label=baseline_pred.predicted_label,
            baseline_confidence=baseline_confidence,
            nli_label=nli_pred.predicted_label,
            nli_confidence=nli_confidence,
            models_agree=(baseline_pred.predicted_label == nli_pred.predicted_label),
        ))

    return paired
