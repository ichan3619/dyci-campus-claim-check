"""
The shared result type returned by every classifier in this project.

Why this lives in its own file (Milestone 6): both the TF-IDF+LR
baseline (classifier.py) and the pretrained NLI classifier
(nli_classifier.py) return this exact same structure. Before this move,
it was defined inside classifier.py, which meant nli_classifier.py would
have had to import a plain dataclass from a file whose actual purpose is
"TF-IDF + Logistic Regression logic" -- an odd, one-directional
dependency for something that isn't TF-IDF-specific at all. Putting it
here instead makes clear that this shape is the shared CONTRACT every
classifier must satisfy (see also src/evaluation/evaluate.py's
ClaimEvidenceModel Protocol, which depends on this same shape), not
something that belongs to any one classifier implementation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassificationResult:
    """
    The output of classifying one (claim, evidence) pair, regardless of
    which classifier produced it.

    Fields:
        label: one of SUPPORTED / CONTRADICTED / INSUFFICIENT_EVIDENCE.
        confidence: the model's own probability/score for the PREDICTED
            label, in [0, 1]. This is a genuine probability according to
            the model's internal math (softmax output), but it is NOT a
            validated, calibrated real-world probability that the claim
            itself is true -- that distinction matters for BOTH
            classifiers implemented in this project (TF-IDF+LR's
            predict_proba, and the NLI model's softmax over
            entailment/contradiction/neutral logits), and is documented
            again, specifically, in each classifier's own module.
        label_scores: the model's score for EVERY label, not just the
            predicted one -- useful for error analysis (see
            src/evaluation/error_analysis.py) and for showing "the model
            was nearly as confident in X as in Y."
        model: optional name of the model that produced this result. Set by
            LanguageRoutedClassifier (M12.3) so a response can say which NLI
            model judged the claim; None otherwise.
    """
    label: str
    confidence: float
    label_scores: dict
    model: Optional[str] = None
