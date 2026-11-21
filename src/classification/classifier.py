"""
Standalone claim/evidence classification for the campus verification system.

Responsibility of this module, and ONLY this module:

    Given a CLAIM and a piece of EVIDENCE (already provided -- from
    anywhere), determine the relationship between them:
    SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE.

This module does NOT retrieve evidence. It has no concept of a corpus,
a search index, or "relevance." It never imports anything from
src/retrieval/. It only ever answers "given these two specific texts,
what is their relationship?" -- exactly mirroring how src/retrieval/
only ever answers "how similar are these two texts?" (see
src/retrieval/retriever.py for the retrieval-side of this separation).

It knows nothing about FastAPI, the frontend, or HTTP.
"""

from typing import List

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data.schema import VALID_LABELS
from src.classification.result import ClassificationResult

# Re-exported so existing imports (`from src.classification.classifier
# import ClassificationResult`), used in src/verification/verifier.py and
# tests/test_classification.py since Milestone 3, keep working unchanged
# after this Milestone 6 move to result.py. New code should import
# ClassificationResult directly from src.classification.result instead.
__all__ = ["ClassificationResult", "ClaimEvidenceClassifier"]


class ClaimEvidenceClassifier:
    """
    A TF-IDF (separate claim/evidence vectors) + Logistic Regression
    classifier for the SUPPORTED / CONTRADICTED / INSUFFICIENT_EVIDENCE
    task.

    Usage:
        classifier = ClaimEvidenceClassifier()
        classifier.fit(claims, evidences, labels)   # lists, same length
        result = classifier.classify("claim text", "evidence text")

    Why a class, not a function? Same reason as Retriever in Milestone 2:
    fitting the TF-IDF vocabulary and training the Logistic Regression
    model are comparatively expensive, one-time steps. Once fit, calling
    classify() many times should be cheap and should reuse that learned
    state.
    """

    def __init__(self):
        # We use ONE shared TfidfVectorizer for both claims and evidence,
        # fit on the combined vocabulary of both. This ensures the word
        # "suspended" gets the SAME column index and the SAME IDF weight
        # whether it appears in a claim or in evidence -- which is what
        # makes it meaningful to later concatenate the two vectors
        # side by side (Option B from the representation discussion).
        self._vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self._model = LogisticRegression(max_iter=1000)
        self._is_fit = False

    def fit(self, claims: List[str], evidences: List[str], labels: List[str]):
        """
        Train the classifier on (claim, evidence, label) triples.

        Input: three parallel lists of equal length.
        Output: nothing -- this mutates the classifier's internal state
        (the fitted vectorizer and trained model), matching scikit-learn's
        usual fit() convention.
        """
        if not (len(claims) == len(evidences) == len(labels)):
            raise ValueError("claims, evidences, and labels must be the same length")

        if len(claims) == 0:
            raise ValueError("cannot fit a classifier on zero examples")

        invalid_labels = set(labels) - VALID_LABELS
        if invalid_labels:
            raise ValueError(f"training data contains invalid labels: {invalid_labels}")

        # Fit the shared vocabulary on ALL text (claims + evidence
        # together), so both share the same word -> column mapping.
        all_text = list(claims) + list(evidences)
        self._vectorizer.fit(all_text)

        # Now vectorize claims and evidence SEPARATELY, using that shared
        # vocabulary, and stick the two feature matrices together
        # side-by-side (hstack = "horizontal stack"). This is Option B:
        # the model sees "claim word counts" and "evidence word counts"
        # as distinct feature blocks, not merged into one bag of words.
        claim_vectors = self._vectorizer.transform(claims)
        evidence_vectors = self._vectorizer.transform(evidences)
        combined_features = hstack([claim_vectors, evidence_vectors])

        self._model.fit(combined_features, labels)
        self._is_fit = True

    def classify(self, claim: str, evidence: str) -> ClassificationResult:
        """
        Classify a single (claim, evidence) pair.

        Raises ValueError for inputs the classifier should never be
        asked to handle: an empty claim, or empty evidence. Empty
        evidence in particular is a RETRIEVAL-level signal ("nothing
        was found") that should be handled by short-circuiting to
        INSUFFICIENT_EVIDENCE *before* the classifier is ever called
        (this is enforced by whoever calls classify(), not by this
        function pretending to guess with no evidence at all).
        """
        if not self._is_fit:
            raise RuntimeError("classifier has not been trained yet -- call fit() first")

        if not claim or not claim.strip():
            raise ValueError("claim must not be empty")

        if not evidence or not evidence.strip():
            raise ValueError(
                "evidence must not be empty -- an empty-evidence case should be "
                "handled as INSUFFICIENT_EVIDENCE by the retrieval stage, before "
                "the classifier is ever called"
            )

        claim_vector = self._vectorizer.transform([claim])
        evidence_vector = self._vectorizer.transform([evidence])
        combined_features = hstack([claim_vector, evidence_vector])

        # predict_proba returns, for each class the model was trained on,
        # a number in [0, 1] that the model's internal math treats as a
        # probability (specifically: the softmax of the model's linear
        # scores -- see the module-level note above about what this
        # number does and doesn't mean).
        probabilities = self._model.predict_proba(combined_features)[0]
        class_labels = self._model.classes_

        label_scores = {
            str(label): float(prob)
            for label, prob in zip(class_labels, probabilities)
        }

        predicted_label = str(self._model.predict(combined_features)[0])
        confidence = label_scores[predicted_label]

        return ClassificationResult(
            label=predicted_label,
            confidence=confidence,
            label_scores=label_scores,
        )
