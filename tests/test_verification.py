"""
Integration tests for the verification layer (Milestone 4).

DESIGN NOTE (same principle as tests/test_classification.py): these
tests use small, deliberately hand-built, easily-separable retriever
corpora and classifier training sets -- NOT the tiny real dev CSV, and
NOT a claim of real-world accuracy. Milestone 3 already demonstrated
that our tiny CSV-trained classifier is not yet reliable; these tests
exist to prove the PIPELINE WIRING is correct (retrieval feeds
classification correctly, scores stay separate, the two "insufficient
evidence" cases are distinguished) -- not to prove the models are good.
"""

import pytest

from src.retrieval.retriever import Retriever
from src.classification.classifier import ClaimEvidenceClassifier
from src.verification.verifier import (
    ClaimVerifier,
    NO_RELEVANT_EVIDENCE,
    RELEVANT_EVIDENCE_FOUND,
)


def make_verifier():
    """
    A verifier built from a small, controlled retriever corpus and a
    classifier trained on a clean, separable set -- covering all three
    labels, engineered so each of Step 9's controlled scenarios reliably
    exercises the intended code path.
    """
    corpus = [
        {
            "document_id": "DOC_SUSPENDED",
            "text": "Official announcement: classes are suspended tomorrow due to the typhoon.",
            "source_type": "synthetic",
            "source_url": "",
        },
        {
            "document_id": "DOC_CONTINUE",
            "text": "Official announcement: classes will continue as scheduled tomorrow.",
            "source_type": "synthetic",
            "source_url": "",
        },
        {
            "document_id": "DOC_BASKETBALL",
            "text": "The university basketball team won yesterday's game.",
            "source_type": "synthetic",
            "source_url": "",
        },
    ]
    retriever = Retriever(corpus)

    classifier = ClaimEvidenceClassifier()
    classifier.fit(
        claims=[
            "Classes are suspended tomorrow.",
            "Classes are suspended tomorrow.",
            "Classes are suspended tomorrow.",
            "Enrollment starts next week.",
            "Enrollment starts next week.",
        ],
        evidences=[
            "Official announcement: classes are suspended tomorrow due to the typhoon.",
            "Official announcement: classes will continue as scheduled tomorrow.",
            "The university basketball team won yesterday's game.",
            "Registrar update: enrollment begins next week as planned.",
            "Registrar update: enrollment has been moved to next month instead.",
        ],
        labels=["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "SUPPORTED", "CONTRADICTED"],
    )

    return ClaimVerifier(retriever, classifier)


# --------------------------------------------------------------------
# 1 & 2 & 3: end-to-end supported / contradicted classification
# --------------------------------------------------------------------

def test_supported_claim_produces_supported_verdict():
    verifier = make_verifier()
    result = verifier.verify_claim("Classes are suspended tomorrow due to the typhoon.")

    assert result.verdict == "SUPPORTED"
    assert result.evidence_status == RELEVANT_EVIDENCE_FOUND
    assert result.used_evidence is not None
    assert result.used_evidence.document_id == "DOC_SUSPENDED"


def test_contradicted_claim_produces_contradicted_verdict():
    verifier = make_verifier()
    result = verifier.verify_claim("Classes will continue as scheduled tomorrow.")

    assert result.verdict == "CONTRADICTED"
    assert result.evidence_status == RELEVANT_EVIDENCE_FOUND
    assert result.used_evidence.document_id == "DOC_CONTINUE"


def test_topically_unrelated_evidence_produces_insufficient_evidence_case_b():
    """Retrieval finds SOME evidence, but it's about a different topic --
    this is Case B: RELEVANT_EVIDENCE_FOUND, verdict still
    INSUFFICIENT_EVIDENCE. Distinct from Case A (below)."""
    verifier = make_verifier()
    result = verifier.verify_claim("The university basketball team won yesterday's game.")

    assert result.verdict == "INSUFFICIENT_EVIDENCE"
    assert result.evidence_status == RELEVANT_EVIDENCE_FOUND  # evidence WAS found
    assert result.used_evidence is not None
    assert result.classification_result is not None


# --------------------------------------------------------------------
# 4 & 5: the two distinct "insufficient evidence" cases
# --------------------------------------------------------------------

def test_empty_corpus_produces_insufficient_evidence_case_a():
    """No evidence exists at all -- Case A: NO_RELEVANT_EVIDENCE.
    The classifier must NEVER be called in this case."""
    empty_retriever = Retriever([])
    classifier = ClaimEvidenceClassifier()
    # Needs at least 2 distinct labels for LogisticRegression.fit() to
    # succeed at all -- irrelevant to this test's actual purpose, since
    # Case A must short-circuit BEFORE classify() is ever called.
    classifier.fit(
        claims=["A claim.", "Another claim."],
        evidences=["Some evidence.", "Other evidence."],
        labels=["SUPPORTED", "CONTRADICTED"],
    )
    verifier = ClaimVerifier(empty_retriever, classifier)

    result = verifier.verify_claim("Classes are suspended tomorrow.")

    assert result.verdict == "INSUFFICIENT_EVIDENCE"
    assert result.evidence_status == NO_RELEVANT_EVIDENCE
    assert result.used_evidence is None
    assert result.classification_result is None  # classifier was never invoked
    assert result.retrieved_evidence == []


def test_case_a_and_case_b_are_distinguishable_despite_same_verdict():
    """The core Step 5 requirement: both cases produce verdict =
    INSUFFICIENT_EVIDENCE, but evidence_status must differ."""
    verifier = make_verifier()

    case_a_result = ClaimVerifier(Retriever([]), verifier._classifier).verify_claim(
        "Classes are suspended tomorrow."
    )
    case_b_result = verifier.verify_claim("The university basketball team won yesterday's game.")

    assert case_a_result.verdict == case_b_result.verdict == "INSUFFICIENT_EVIDENCE"
    assert case_a_result.evidence_status != case_b_result.evidence_status
    assert case_a_result.evidence_status == NO_RELEVANT_EVIDENCE
    assert case_b_result.evidence_status == RELEVANT_EVIDENCE_FOUND


def test_below_threshold_top_result_triggers_case_a_even_with_nonempty_corpus():
    """A non-empty corpus that simply has nothing relevant enough should
    ALSO trigger Case A, not attempt classification on a near-zero-score
    match."""
    corpus = [
        {"document_id": "DOC_UNRELATED", "text": "Alumni homecoming will be held next month at the gymnasium.",
         "source_type": "synthetic", "source_url": ""},
    ]
    retriever = Retriever(corpus)
    classifier = ClaimEvidenceClassifier()
    classifier.fit(
        claims=["A claim.", "Another claim."],
        evidences=["Some evidence.", "Other evidence."],
        labels=["SUPPORTED", "CONTRADICTED"],
    )

    # Force a high threshold so even a real (but weak) match is rejected.
    verifier = ClaimVerifier(retriever, classifier, relevance_threshold=0.99)
    result = verifier.verify_claim("Completely unrelated query about the weather in Tokyo.")

    assert result.evidence_status == NO_RELEVANT_EVIDENCE
    assert result.classification_result is None


# --------------------------------------------------------------------
# 6: multiple retrieved documents handling strategy
# --------------------------------------------------------------------

def test_multiple_candidates_are_retrieved_but_only_top_one_is_classified():
    """Step 2 decision (Option A): retrieval may return several
    candidates, but exactly one (the top-ranked) is passed to the
    classifier. The rest are kept in retrieved_evidence for
    transparency only."""
    verifier = make_verifier()
    result = verifier.verify_claim("Classes are suspended tomorrow due to the typhoon.")

    assert len(result.retrieved_evidence) > 1  # multiple candidates were retrieved
    assert result.used_evidence == result.retrieved_evidence[0]  # only the top one was used


# --------------------------------------------------------------------
# 7: retrieval score and classification score must remain distinct
# --------------------------------------------------------------------

def test_retrieval_score_and_classification_confidence_are_separate_fields():
    """These must never be merged into a single number anywhere in the
    result -- confirmed here by checking they are stored under distinct
    field paths and are not required to be equal or related."""
    verifier = make_verifier()
    result = verifier.verify_claim("Classes are suspended tomorrow due to the typhoon.")

    retrieval_score = result.used_evidence.score
    classification_confidence = result.classification_result.confidence

    # Both must exist independently.
    assert isinstance(retrieval_score, float)
    assert isinstance(classification_confidence, float)
    # There is no assertion that they're equal or correlated -- that's
    # the point. They measure different things (see module docstrings
    # in retriever.py and classifier.py).


# --------------------------------------------------------------------
# 8: invalid/empty claims
# --------------------------------------------------------------------

def test_empty_claim_raises_value_error():
    verifier = make_verifier()
    with pytest.raises(ValueError):
        verifier.verify_claim("")


def test_whitespace_only_claim_raises_value_error():
    verifier = make_verifier()
    with pytest.raises(ValueError):
        verifier.verify_claim("   ")


# --------------------------------------------------------------------
# 9 & 10: verification must not reimplement retrieval or classification
# --------------------------------------------------------------------

def test_verification_module_does_not_import_sklearn_directly():
    """Structural check: verifier.py should contain no direct ML library
    usage -- it should only call into retrieval/classification, never
    reimplement their internals. This is checked by inspecting the
    module's own imports rather than trusting a comment."""
    import src.verification.verifier as verifier_module
    import inspect

    source = inspect.getsource(verifier_module)
    assert "TfidfVectorizer" not in source
    assert "LogisticRegression" not in source
    assert "cosine_similarity" not in source
