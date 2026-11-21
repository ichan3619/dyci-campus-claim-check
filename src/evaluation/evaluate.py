"""
Model-agnostic evaluation harness (Milestone 5, Steps 7 and 10).

evaluate_model() works with ANY object that exposes:

    classify(claim: str, evidence: str) -> object with a `.label` attribute

This is the SAME interface src/classification/classifier.py's
ClaimEvidenceClassifier already implements. When a future pretrained NLI
model is added (a LATER milestone, NOT this one), it only needs to
implement this same method signature -- this file, the metrics module,
and the train/val/test split will not need to change AT ALL. That is the
entire point of Milestone 5 Step 10: prepare the harness now, without
building the NLI model yet.

This module contains NO model-specific code (no TF-IDF, no transformer
loading) -- exactly like src/verification/verifier.py, it only
orchestrates calling classify() and recording what happened.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from src.evaluation.metrics import EvaluationMetrics, compute_metrics


class ClaimEvidenceModel(Protocol):
    """
    The interface any model must implement to be evaluated by this
    harness. This is documentation-as-code: it doesn't enforce anything
    at runtime (Python's Protocol is structural, not a base class you
    must inherit from), but it makes explicit exactly what a future NLI
    wrapper needs to provide.
    """
    def classify(self, claim: str, evidence: str):
        ...  # must return an object with a `.label` attribute


@dataclass
class PredictionRecord:
    """
    One example's true label, predicted label, and the inputs that
    produced it -- kept together so Step 9's error analysis can look at
    exactly which claim/evidence pairs a model got wrong, not just an
    aggregate score.

    label_scores (added Milestone 7): the model's full per-label score
    dict (e.g. {"SUPPORTED": 0.6, ...}), copied from
    ClassificationResult.label_scores WHEN THE MODEL PROVIDES ONE.
    Optional and defaults to None -- this is a purely ADDITIVE change:
    any existing caller that only reads claim/evidence/true_label/
    predicted_label/is_correct (e.g. compute_metrics, which never looks
    at this field) is completely unaffected. Added specifically so
    Milestone 7's side-by-side model comparison can show each model's
    confidence next to its prediction, without which the comparison
    could only show labels, not how strongly each model held its
    prediction.
    """
    claim: str
    evidence: str
    true_label: str
    predicted_label: str
    is_correct: bool
    label_scores: Optional[dict] = None


@dataclass
class EvaluationRun:
    """
    Everything produced by one call to evaluate_model(): the aggregate
    metrics AND the individual predictions, so a caller can compute
    summary numbers (metrics) or drill into specific failures
    (predictions) from the same run, without re-running the model.
    """
    metrics: EvaluationMetrics
    predictions: List[PredictionRecord] = field(default_factory=list)


def evaluate_model(model: ClaimEvidenceModel, examples: List[dict]) -> EvaluationRun:
    """
    Run `model` over every example and compute metrics.

    Input:
        model: anything with a classify(claim, evidence) -> result.label
            method -- TF-IDF+LR today, a pretrained NLI wrapper later.
        examples: list of dicts, each with "claim", "evidence", "label"
            (the format produced by
            src.evaluation.splitting.build_classification_examples).

    Output: an EvaluationRun (metrics + per-example predictions).

    IMPORTANT: this function does not know or care whether `model` is
    TF-IDF+LR, an NLI model, or anything else. Swapping models means
    calling this function again with a different `model` argument --
    nothing else in this file changes.
    """
    if len(examples) == 0:
        raise ValueError("cannot evaluate over zero examples")

    predictions = []
    for example in examples:
        result = model.classify(example["claim"], example["evidence"])
        # label_scores is read defensively (getattr, not result.label_scores
        # directly) so this harness keeps working even with a hypothetical
        # future model whose result object doesn't provide per-label scores
        # -- the Protocol above only ever REQUIRED `.label`, so anything
        # beyond that must be treated as optional here.
        scores = getattr(result, "label_scores", None)
        predictions.append(PredictionRecord(
            claim=example["claim"],
            evidence=example["evidence"],
            true_label=example["label"],
            predicted_label=result.label,
            is_correct=(result.label == example["label"]),
            label_scores=scores,
        ))

    metrics = compute_metrics(
        true_labels=[p.true_label for p in predictions],
        predicted_labels=[p.predicted_label for p in predictions],
    )

    return EvaluationRun(metrics=metrics, predictions=predictions)
