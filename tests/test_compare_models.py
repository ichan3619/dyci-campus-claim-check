"""
Tests for src/evaluation/compare_models.py (Milestone 7).

Uses small fake classifiers (same pattern as tests/test_nli_classifier.py's
_DummyModel and tests/test_evaluation.py's dummy models) so these tests
run without network access or real model downloads -- they verify the
COMPARISON MACHINERY (alignment, metrics-per-model, agreement flagging,
small-test-set detection), not real model accuracy.
"""

import pytest

from src.evaluation.compare_models import (
    compare_models, PairedPrediction, ComparisonReport,
    SMALL_TEST_SET_WARNING_THRESHOLD,
)


class _FixedLabelModel:
    """Returns the SAME label for every input, with a fixed confidence --
    a minimal, fully predictable model for testing alignment logic."""
    class _Result:
        def __init__(self, label, confidence):
            self.label = label
            self.label_scores = {label: confidence}

    def __init__(self, fixed_label, confidence=0.9):
        self._label = fixed_label
        self._confidence = confidence

    def classify(self, claim, evidence):
        return self._Result(self._label, self._confidence)


class _NoScoresModel:
    """A model whose result has NO label_scores attribute at all -- tests
    that compare_models() handles this gracefully (Optional confidence),
    matching evaluate_model()'s existing defensive getattr() behavior."""
    class _Result:
        def __init__(self, label):
            self.label = label
            # deliberately no label_scores attribute

    def __init__(self, fixed_label):
        self._label = fixed_label

    def classify(self, claim, evidence):
        return self._Result(self._label)


def make_examples(n=3):
    return [
        {"claim": f"Claim {i}", "evidence": f"Evidence {i}", "label": "SUPPORTED"}
        for i in range(n)
    ]


def test_compare_models_returns_comparison_report():
    baseline = _FixedLabelModel("SUPPORTED")
    nli = _FixedLabelModel("SUPPORTED")
    report = compare_models(baseline, nli, make_examples(3))

    assert isinstance(report, ComparisonReport)
    assert report.n_test_examples == 3


def test_compare_models_raises_on_empty_test_set():
    baseline = _FixedLabelModel("SUPPORTED")
    nli = _FixedLabelModel("SUPPORTED")
    with pytest.raises(ValueError):
        compare_models(baseline, nli, [])


def test_paired_predictions_align_by_position():
    baseline = _FixedLabelModel("SUPPORTED")
    nli = _FixedLabelModel("CONTRADICTED")
    examples = make_examples(3)
    report = compare_models(baseline, nli, examples)

    assert len(report.paired_predictions) == 3
    for i, pair in enumerate(report.paired_predictions):
        assert pair.claim == examples[i]["claim"]
        assert pair.evidence == examples[i]["evidence"]
        assert pair.gold_label == "SUPPORTED"


def test_models_agree_flag_true_when_labels_match():
    baseline = _FixedLabelModel("SUPPORTED")
    nli = _FixedLabelModel("SUPPORTED")
    report = compare_models(baseline, nli, make_examples(2))

    assert all(p.models_agree for p in report.paired_predictions)


def test_models_agree_flag_false_when_labels_differ():
    baseline = _FixedLabelModel("SUPPORTED")
    nli = _FixedLabelModel("CONTRADICTED")
    report = compare_models(baseline, nli, make_examples(2))

    assert all(not p.models_agree for p in report.paired_predictions)


def test_confidence_is_captured_per_model_independently():
    baseline = _FixedLabelModel("SUPPORTED", confidence=0.7)
    nli = _FixedLabelModel("SUPPORTED", confidence=0.99)
    report = compare_models(baseline, nli, make_examples(1))

    pair = report.paired_predictions[0]
    assert pair.baseline_confidence == pytest.approx(0.7)
    assert pair.nli_confidence == pytest.approx(0.99)


def test_missing_label_scores_results_in_none_confidence_not_crash():
    """A model with no label_scores at all must not crash compare_models()
    -- confidence should just be None for that model's predictions."""
    baseline = _NoScoresModel("SUPPORTED")
    nli = _FixedLabelModel("SUPPORTED", confidence=0.8)
    report = compare_models(baseline, nli, make_examples(1))

    pair = report.paired_predictions[0]
    assert pair.baseline_confidence is None
    assert pair.nli_confidence == pytest.approx(0.8)


def test_both_models_metrics_are_independently_correct():
    """baseline always right, nli always wrong -- their metrics must
    reflect that independently, proving evaluate_model() is being called
    correctly and separately for each model."""
    baseline = _FixedLabelModel("SUPPORTED")
    nli = _FixedLabelModel("CONTRADICTED")
    report = compare_models(baseline, nli, make_examples(3))  # gold label is SUPPORTED

    assert report.baseline_metrics.accuracy == 1.0
    assert report.nli_metrics.accuracy == 0.0


def test_small_test_set_flag_true_below_threshold():
    baseline = _FixedLabelModel("SUPPORTED")
    nli = _FixedLabelModel("SUPPORTED")
    report = compare_models(baseline, nli, make_examples(3))

    assert report.n_test_examples < SMALL_TEST_SET_WARNING_THRESHOLD
    assert report.is_small_test_set is True


def test_small_test_set_flag_false_above_threshold():
    baseline = _FixedLabelModel("SUPPORTED")
    nli = _FixedLabelModel("SUPPORTED")
    report = compare_models(baseline, nli, make_examples(SMALL_TEST_SET_WARNING_THRESHOLD + 5))

    assert report.is_small_test_set is False


def test_compare_models_does_not_import_retrieval_or_verification():
    """Structural check, same pattern as Milestones 4 and 6: this module
    must stay a pure classifier-vs-classifier comparison, never mixing in
    retrieval or full-pipeline verification behavior (per the M7 design
    review's explicit instruction not to use error_analysis.py/
    ClaimVerifier here)."""
    import src.evaluation.compare_models as module
    import inspect

    import_lines = [
        line for line in inspect.getsource(module).splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    source_of_imports = "\n".join(import_lines)
    assert "src.retrieval" not in source_of_imports
    assert "src.verification" not in source_of_imports
    assert "error_analysis" not in source_of_imports
