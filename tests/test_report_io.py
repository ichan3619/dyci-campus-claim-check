"""
Tests for src/evaluation/report_io.py (Milestone 7 Step 2).

Uses the same fake-classifier pattern as tests/test_compare_models.py so
these run without network access. Tests the SERIALIZATION logic only --
build_report_dict() and save_comparison_report() -- not compare_models()
itself, which already has its own test coverage and is untouched here.
"""

import json
import os
import tempfile

import pytest

from src.evaluation.compare_models import compare_models, SMALL_TEST_SET_WARNING_THRESHOLD
from src.evaluation.report_io import build_report_dict, save_comparison_report


class _FixedLabelModel:
    class _Result:
        def __init__(self, label, confidence):
            self.label = label
            self.label_scores = {label: confidence}

    def __init__(self, fixed_label, confidence=0.9):
        self._label = fixed_label
        self._confidence = confidence

    def classify(self, claim, evidence):
        return self._Result(self._label, self._confidence)


def make_examples(n=3):
    return [
        {"claim": f"Claim {i}", "evidence": f"Evidence {i}", "label": "SUPPORTED"}
        for i in range(n)
    ]


def make_small_report():
    """A report over 3 examples -- triggers is_small_test_set=True."""
    baseline = _FixedLabelModel("SUPPORTED", confidence=0.7)
    nli = _FixedLabelModel("SUPPORTED", confidence=0.95)
    return compare_models(baseline, nli, make_examples(3))


def make_large_report():
    """A report over enough examples to be ABOVE the small-test-set
    threshold -- confirms run_type differs correctly."""
    baseline = _FixedLabelModel("SUPPORTED", confidence=0.7)
    nli = _FixedLabelModel("SUPPORTED", confidence=0.95)
    return compare_models(baseline, nli, make_examples(SMALL_TEST_SET_WARNING_THRESHOLD + 5))


# --------------------------------------------------------------------
# build_report_dict() -- pure logic, no filesystem
# --------------------------------------------------------------------

def test_build_report_dict_labels_small_test_set_as_pipeline_validation():
    report = make_small_report()
    d = build_report_dict(report, dataset_path="data/dev/sample_claims.csv",
                           split_seed=42, train_size=14, val_size=4)

    assert d["run_type"] == "PIPELINE_VALIDATION_RUN"
    assert "PIPELINE VALIDATION" in d["warning"]


def test_build_report_dict_labels_large_test_set_differently():
    report = make_large_report()
    d = build_report_dict(report, dataset_path="data/dev/sample_claims.csv",
                           split_seed=42, train_size=100, val_size=20)

    assert d["run_type"] == "COMPARISON_RUN"


def test_build_report_dict_contains_reproducibility_metadata():
    report = make_small_report()
    d = build_report_dict(report, dataset_path="data/dev/sample_claims.csv",
                           split_seed=42, train_size=14, val_size=4)

    repro = d["reproducibility"]
    assert repro["dataset_path"] == "data/dev/sample_claims.csv"
    assert repro["split_seed"] == 42
    assert repro["train_size"] == 14
    assert repro["val_size"] == 4
    assert repro["test_size"] == 3
    assert repro["is_small_test_set"] is True


def test_build_report_dict_contains_both_models_metrics():
    report = make_small_report()
    d = build_report_dict(report, dataset_path="x", split_seed=42, train_size=1, val_size=1)

    assert "accuracy" in d["baseline_metrics"]
    assert "macro_f1" in d["baseline_metrics"]
    assert "confusion_matrix" in d["baseline_metrics"]
    assert "accuracy" in d["nli_metrics"]
    assert "macro_f1" in d["nli_metrics"]


def test_build_report_dict_contains_per_example_predictions():
    report = make_small_report()
    d = build_report_dict(report, dataset_path="x", split_seed=42, train_size=1, val_size=1)

    assert len(d["per_example_predictions"]) == 3
    example = d["per_example_predictions"][0]
    assert set(["claim", "evidence", "gold_label", "baseline_label",
                "baseline_confidence", "nli_label", "nli_confidence",
                "models_agree"]).issubset(example.keys())


def test_build_report_dict_states_selection_is_deferred():
    report = make_small_report()
    d = build_report_dict(report, dataset_path="x", split_seed=42, train_size=1, val_size=1)

    assert d["classifier_selection"]["status"] == "DEFERRED"
    assert "macro F1" in d["classifier_selection"]["decision_rule_for_future_runs"]


def test_build_report_dict_is_actually_json_serializable():
    """The most important structural guarantee: everything in here must
    survive a real json.dumps() call with no custom encoder."""
    report = make_small_report()
    d = build_report_dict(report, dataset_path="x", split_seed=42, train_size=1, val_size=1)

    serialized = json.dumps(d)  # raises TypeError if anything isn't JSON-safe
    assert isinstance(serialized, str)


# --------------------------------------------------------------------
# save_comparison_report() -- actual file writing
# --------------------------------------------------------------------

def test_save_comparison_report_writes_a_valid_json_file():
    report = make_small_report()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        save_comparison_report(report, path, dataset_path="data/dev/sample_claims.csv",
                                split_seed=42, train_size=14, val_size=4)

        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["run_type"] == "PIPELINE_VALIDATION_RUN"
        assert loaded["reproducibility"]["split_seed"] == 42
        assert len(loaded["per_example_predictions"]) == 3
    finally:
        os.remove(path)


def test_save_comparison_report_does_not_modify_the_report_object():
    """Confirms this module is purely a serializer -- the ComparisonReport
    passed in must be unchanged after saving."""
    report = make_small_report()
    original_accuracy = report.baseline_metrics.accuracy

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        save_comparison_report(report, path, dataset_path="x", split_seed=42,
                                train_size=1, val_size=1)
        assert report.baseline_metrics.accuracy == original_accuracy
    finally:
        os.remove(path)
