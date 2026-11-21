"""
Tests for src/evaluation/ (Milestone 5).

Covers: metrics computation correctness, the model-agnostic evaluation
harness (proving it is NOT coupled to ClaimEvidenceClassifier), dataset
audit (near-duplicate detection), the threshold experiment, and error
analysis categorization.
"""

import pytest

from src.evaluation.metrics import compute_metrics, format_metrics_report
from src.evaluation.evaluate import evaluate_model
from src.evaluation.dataset_audit import build_audit_report, jaccard_similarity
from src.evaluation.threshold_experiment import (
    build_labeled_queries, evaluate_threshold, sweep_thresholds, LabeledQuery,
)
from src.evaluation.error_analysis import (
    categorize_result, RETRIEVAL_FAILURE, RETRIEVAL_FALSE_POSITIVE,
    CLASSIFICATION_FAILURE, GENUINE_INSUFFICIENT, CORRECT,
)
from src.evaluation.splitting import build_classification_examples, split_by_group
from src.retrieval.retriever import Retriever, RetrievedDocument
from src.verification.verifier import VerificationResult, NO_RELEVANT_EVIDENCE, RELEVANT_EVIDENCE_FOUND


# --------------------------------------------------------------------
# metrics.py
# --------------------------------------------------------------------

def test_compute_metrics_perfect_predictions():
    true = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    pred = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    metrics = compute_metrics(true, pred)

    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.n_examples == 3


def test_compute_metrics_all_wrong():
    true = ["SUPPORTED", "SUPPORTED"]
    pred = ["CONTRADICTED", "CONTRADICTED"]
    metrics = compute_metrics(true, pred)

    assert metrics.accuracy == 0.0
    assert metrics.macro_f1 == 0.0


def test_compute_metrics_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        compute_metrics(["SUPPORTED"], ["SUPPORTED", "CONTRADICTED"])


def test_compute_metrics_empty_raises():
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_compute_metrics_missing_class_gets_zero_not_crash():
    """A class entirely absent from a tiny split must produce a defined
    0.0, not a crash or an sklearn warning-as-error -- this matters for
    our small development dataset where a class can easily be absent
    from a given split."""
    true = ["SUPPORTED", "SUPPORTED"]
    pred = ["SUPPORTED", "SUPPORTED"]
    metrics = compute_metrics(true, pred)

    assert metrics.per_class_f1["CONTRADICTED"] == 0.0
    assert metrics.per_class_f1["INSUFFICIENT_EVIDENCE"] == 0.0


def test_format_metrics_report_produces_readable_string():
    metrics = compute_metrics(["SUPPORTED"], ["SUPPORTED"])
    report = format_metrics_report(metrics, title="TEST SPLIT")
    assert "TEST SPLIT" in report
    assert "Accuracy" in report
    assert "Macro F1" in report


# --------------------------------------------------------------------
# evaluate.py -- the model-agnostic harness
# --------------------------------------------------------------------

class _DummyModel:
    """A minimal fake model with NO relation to ClaimEvidenceClassifier
    -- used to prove evaluate_model() only depends on the classify()
    interface, not on any specific implementation (Milestone 5 Step 10's
    core requirement)."""
    class _Result:
        def __init__(self, label):
            self.label = label

    def __init__(self, fixed_label):
        self._fixed_label = fixed_label

    def classify(self, claim, evidence):
        return self._Result(self._fixed_label)


def test_evaluate_model_works_with_non_classifier_object():
    """This is the key Step 10 proof: evaluate_model() must work with
    ANY object exposing classify(claim, evidence).label -- not just the
    TF-IDF+LR classifier -- so a future NLI model needs zero changes
    here."""
    examples = [
        {"claim": "A", "evidence": "B", "label": "SUPPORTED"},
        {"claim": "C", "evidence": "D", "label": "SUPPORTED"},
        {"claim": "E", "evidence": "F", "label": "CONTRADICTED"},
    ]
    dummy = _DummyModel(fixed_label="SUPPORTED")
    run = evaluate_model(dummy, examples)

    assert run.metrics.n_examples == 3
    assert run.metrics.accuracy == pytest.approx(2 / 3)
    assert len(run.predictions) == 3
    assert run.predictions[0].is_correct is True
    assert run.predictions[2].is_correct is False


def test_evaluate_model_empty_examples_raises():
    with pytest.raises(ValueError):
        evaluate_model(_DummyModel("SUPPORTED"), [])


def test_evaluate_model_predictions_carry_original_inputs():
    """Error analysis (Step 9) needs to look at the actual claim/evidence
    text of failures, not just an aggregate score."""
    examples = [{"claim": "The sky is blue.", "evidence": "Some evidence.", "label": "CONTRADICTED"}]
    run = evaluate_model(_DummyModel("SUPPORTED"), examples)

    assert run.predictions[0].claim == "The sky is blue."
    assert run.predictions[0].true_label == "CONTRADICTED"
    assert run.predictions[0].predicted_label == "SUPPORTED"
    assert run.predictions[0].is_correct is False


# --------------------------------------------------------------------
# dataset_audit.py
# --------------------------------------------------------------------

def test_jaccard_similarity_identical_texts():
    assert jaccard_similarity("classes are suspended", "classes are suspended") == 1.0


def test_jaccard_similarity_completely_different_texts():
    assert jaccard_similarity("classes are suspended", "basketball team won") == 0.0


def test_build_audit_report_counts_empty_and_genuine_neutral_correctly():
    records = [
        {"claim": "A", "evidence": "", "label": "INSUFFICIENT_EVIDENCE",
         "source_type": "synthetic", "group_id": "g1"},
        {"claim": "B", "evidence": "Unrelated real evidence text here.", "label": "INSUFFICIENT_EVIDENCE",
         "source_type": "synthetic", "group_id": "g2"},
        {"claim": "C", "evidence": "Confirming evidence.", "label": "SUPPORTED",
         "source_type": "synthetic", "group_id": "g3"},
    ]
    report = build_audit_report(records)

    assert report.total_records == 3
    assert report.total_groups == 3
    assert len(report.empty_evidence_rows) == 1
    assert len(report.genuine_neutral_rows) == 1


def test_build_audit_report_flags_cross_group_near_duplicates():
    """Two claims from DIFFERENT groups that are near-identical should be
    flagged -- this is a real leakage risk the exact-match duplicate
    checker (Milestone 1) cannot catch."""
    records = [
        {"claim": "Classes are suspended tomorrow due to weather.", "evidence": "E1",
         "label": "SUPPORTED", "source_type": "synthetic", "group_id": "g1"},
        {"claim": "Classes are suspended tomorrow due to bad weather.", "evidence": "E2",
         "label": "SUPPORTED", "source_type": "synthetic", "group_id": "g2"},
    ]
    report = build_audit_report(records, near_duplicate_threshold=0.5)

    assert len(report.cross_group_near_duplicate_claims) == 1


def test_build_audit_report_does_not_flag_same_group_similarity():
    """A SUPPORTED and CONTRADICTED claim sharing one group_id are
    EXPECTED to be similar -- this must NOT be flagged as a problem."""
    records = [
        {"claim": "Classes are suspended tomorrow.", "evidence": "E1",
         "label": "SUPPORTED", "source_type": "synthetic", "group_id": "g1"},
        {"claim": "Classes are not suspended tomorrow.", "evidence": "E1",
         "label": "CONTRADICTED", "source_type": "synthetic", "group_id": "g1"},
    ]
    report = build_audit_report(records, near_duplicate_threshold=0.5)

    assert len(report.cross_group_near_duplicate_claims) == 0


# --------------------------------------------------------------------
# threshold_experiment.py
# --------------------------------------------------------------------

def test_build_labeled_queries_separates_positive_and_negative():
    records = [
        {"claim": "A", "label": "SUPPORTED", "group_id": "g1"},
        {"claim": "B", "label": "CONTRADICTED", "group_id": "g2"},
        {"claim": "C", "label": "INSUFFICIENT_EVIDENCE", "group_id": "g3"},
    ]
    queries = build_labeled_queries(records)

    assert len(queries) == 3
    positive = [q for q in queries if q.expected_group_id is not None]
    negative = [q for q in queries if q.expected_group_id is None]
    assert len(positive) == 2
    assert len(negative) == 1


def test_evaluate_threshold_perfect_retrieval_at_low_threshold():
    corpus = [
        {"document_id": "g1", "text": "Classes are suspended tomorrow due to the typhoon.",
         "source_type": "synthetic", "source_url": ""},
    ]
    retriever = Retriever(corpus)
    queries = [LabeledQuery(claim="Classes are suspended tomorrow.", expected_group_id="g1")]

    result = evaluate_threshold(retriever, queries, threshold=0.1)

    assert result.true_positives == 1
    assert result.false_negatives == 0


def test_evaluate_threshold_high_threshold_causes_false_negative():
    corpus = [
        {"document_id": "g1", "text": "Classes are suspended tomorrow due to the typhoon.",
         "source_type": "synthetic", "source_url": ""},
    ]
    retriever = Retriever(corpus)
    queries = [LabeledQuery(claim="Classes are suspended tomorrow.", expected_group_id="g1")]

    # A threshold above what this weak match can achieve should reject it.
    result = evaluate_threshold(retriever, queries, threshold=0.99)

    assert result.false_negatives == 1
    assert result.true_positives == 0


def test_sweep_thresholds_returns_one_result_per_candidate():
    corpus = [{"document_id": "g1", "text": "Classes suspended.", "source_type": "synthetic", "source_url": ""}]
    retriever = Retriever(corpus)
    queries = [LabeledQuery(claim="Classes suspended.", expected_group_id="g1")]

    results = sweep_thresholds(retriever, queries, [0.0, 0.5, 1.0])

    assert len(results) == 3
    assert [r.threshold for r in results] == [0.0, 0.5, 1.0]


# --------------------------------------------------------------------
# error_analysis.py
# --------------------------------------------------------------------

def make_result(verdict, evidence_status, used_evidence=None):
    return VerificationResult(
        claim="A claim.",
        verdict=verdict,
        evidence_status=evidence_status,
        used_evidence=used_evidence,
        retrieved_evidence=[],
        classification_result=None,
        explanation="",
    )


def test_categorize_correct_prediction():
    result = make_result("SUPPORTED", RELEVANT_EVIDENCE_FOUND,
                          used_evidence=RetrievedDocument("g1", 0.9, "text", "synthetic", ""))
    entry = categorize_result(result, gold_label="SUPPORTED", gold_expected_group_id="g1")
    assert entry.category == CORRECT


def test_categorize_genuine_insufficient():
    result = make_result("INSUFFICIENT_EVIDENCE", NO_RELEVANT_EVIDENCE, used_evidence=None)
    entry = categorize_result(result, gold_label="INSUFFICIENT_EVIDENCE", gold_expected_group_id=None)
    assert entry.category == GENUINE_INSUFFICIENT


def test_categorize_retrieval_failure_no_evidence_found():
    """Real evidence exists (gold_expected_group_id is set), but the
    system found nothing -- retrieval's fault, not classification's."""
    result = make_result("INSUFFICIENT_EVIDENCE", NO_RELEVANT_EVIDENCE, used_evidence=None)
    entry = categorize_result(result, gold_label="SUPPORTED", gold_expected_group_id="g1")
    assert entry.category == RETRIEVAL_FAILURE


def test_categorize_retrieval_failure_wrong_document():
    """Real evidence exists, retrieval found SOMETHING, but the wrong document."""
    wrong_doc = RetrievedDocument("g_wrong", 0.6, "unrelated text", "synthetic", "")
    result = make_result("INSUFFICIENT_EVIDENCE", RELEVANT_EVIDENCE_FOUND, used_evidence=wrong_doc)
    entry = categorize_result(result, gold_label="SUPPORTED", gold_expected_group_id="g_correct")
    assert entry.category == RETRIEVAL_FAILURE


def test_categorize_classification_failure():
    """Retrieval found the CORRECT document, but the verdict is still wrong."""
    correct_doc = RetrievedDocument("g1", 0.9, "correct text", "synthetic", "")
    result = make_result("CONTRADICTED", RELEVANT_EVIDENCE_FOUND, used_evidence=correct_doc)
    entry = categorize_result(result, gold_label="SUPPORTED", gold_expected_group_id="g1")
    assert entry.category == CLASSIFICATION_FAILURE


def test_categorize_retrieval_false_positive():
    """No real evidence exists, but the system confidently found
    something and produced a definite (wrong) verdict."""
    fake_match = RetrievedDocument("g_other", 0.5, "unrelated", "synthetic", "")
    result = make_result("SUPPORTED", RELEVANT_EVIDENCE_FOUND, used_evidence=fake_match)
    entry = categorize_result(result, gold_label="INSUFFICIENT_EVIDENCE", gold_expected_group_id=None)
    assert entry.category == RETRIEVAL_FALSE_POSITIVE


# --------------------------------------------------------------------
# splitting.py -- confirm the moved module still behaves identically
# to Milestone 3's original, AND confirm the compatibility shim works
# --------------------------------------------------------------------

def test_splitting_module_keeps_shared_groups_together():
    records = [
        {"claim": "A", "evidence": "E1", "label": "SUPPORTED", "group_id": "g1"},
        {"claim": "A-contra", "evidence": "E1", "label": "CONTRADICTED", "group_id": "g1"},
        {"claim": "B", "evidence": "E2", "label": "SUPPORTED", "group_id": "g2"},
        {"claim": "C", "evidence": "E3", "label": "SUPPORTED", "group_id": "g3"},
    ]
    train, val, test = split_by_group(records, seed=1)

    all_splits = {"train": train, "val": val, "test": test}
    g1_splits = {name for name, split in all_splits.items()
                 for e in split if e["group_id"] == "g1"}
    assert len(g1_splits) == 1


def test_classification_dataset_shim_still_importable():
    """Milestones 1-4 import from src.classification.dataset -- this
    MUST keep working after the Milestone 5 move to
    src.evaluation.splitting, or existing code silently breaks."""
    from src.classification.dataset import build_classification_examples as shimmed_fn
    from src.evaluation.splitting import build_classification_examples as real_fn

    assert shimmed_fn is real_fn  # literally the same function object, not a copy
