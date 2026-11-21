"""
Tests for the classification module.

IMPORTANT DESIGN NOTE: these tests do NOT assert that the model gets
predictions "right" on realistic examples. As Milestone 3's training run
showed, 14 training examples is nowhere near enough for the baseline to
be reliably accurate -- and asserting specific predictions from an
under-trained model would make these tests flaky and would test the
WRONG thing (this milestone is about proving the pipeline is correctly
built, not that the baseline is already good).

Where we DO assert a specific predicted label (test_smoke_*), we first
fit the classifier on a small set of DELIBERATELY, ARTIFICIALLY
easy-to-separate synthetic examples -- built specifically for this test,
with maximally distinct vocabulary per class -- and then check the
prediction on an example from that SAME easy distribution. This is a
"smoke test": it confirms the training/prediction code path works
end-to-end and can learn an obvious pattern, not that the model
generalizes to real campus claims.
"""

import pytest

from src.classification.classifier import ClaimEvidenceClassifier, ClassificationResult
from src.classification.dataset import build_classification_examples, split_by_group
from src.data.schema import VALID_LABELS


def make_easy_training_set():
    """
    A tiny, artificially clean-cut dataset for smoke-testing. Each class
    uses distinctly different vocabulary so a linear TF-IDF model has an
    easy time separating them -- this is intentional and is NOT meant to
    resemble the real, harder campus dataset.
    """
    claims = [
        "Classes are suspended tomorrow.",
        "Classes are suspended tomorrow.",
        "Enrollment starts next week.",
        "Enrollment starts next week.",
        "The library is open late tonight.",
        "The library is open late tonight.",
    ]
    evidences = [
        "Official notice: classes are suspended tomorrow due to weather.",
        "Official notice: classes continue as scheduled tomorrow.",
        "Registrar update: enrollment begins next week as planned.",
        "Registrar update: enrollment has been moved to next month instead.",
        "Facilities memo: the library will be open late tonight for finals.",
        "Facilities memo: the library will close early tonight for maintenance.",
    ]
    labels = [
        "SUPPORTED",
        "CONTRADICTED",
        "SUPPORTED",
        "CONTRADICTED",
        "SUPPORTED",
        "CONTRADICTED",
    ]
    # Add a couple of clearly unrelated (INSUFFICIENT_EVIDENCE) pairs too,
    # so the classifier has seen all three classes.
    claims += ["Classes are suspended tomorrow.", "Enrollment starts next week."]
    evidences += [
        "Sports update: the basketball team won their game yesterday.",
        "Alumni update: homecoming will be held at the gymnasium.",
    ]
    labels += ["INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"]

    return claims, evidences, labels


def make_trained_classifier():
    claims, evidences, labels = make_easy_training_set()
    classifier = ClaimEvidenceClassifier()
    classifier.fit(claims, evidences, labels)
    return classifier


def test_valid_pair_returns_a_valid_label():
    """Code-correctness check: classifying any valid input returns
    exactly one of the three defined labels -- never something else."""
    classifier = make_trained_classifier()
    result = classifier.classify(
        "Classes are suspended tomorrow.",
        "Official notice: classes are suspended tomorrow due to weather.",
    )
    assert isinstance(result, ClassificationResult)
    assert result.label in VALID_LABELS


def test_smoke_supported_example_on_easy_data():
    """Smoke test: on the SAME easy, artificially separable distribution
    used for training, the model should get an unambiguous SUPPORTED
    example right. This tests that the training pipeline can learn an
    obvious pattern -- not that it generalizes to real campus claims."""
    classifier = make_trained_classifier()
    result = classifier.classify(
        "Classes are suspended tomorrow.",
        "Official notice: classes are suspended tomorrow due to weather.",
    )
    assert result.label == "SUPPORTED"


def test_smoke_contradicted_example_on_easy_data():
    classifier = make_trained_classifier()
    result = classifier.classify(
        "Classes are suspended tomorrow.",
        "Official notice: classes continue as scheduled tomorrow.",
    )
    assert result.label == "CONTRADICTED"


def test_smoke_insufficient_evidence_example_on_easy_data():
    classifier = make_trained_classifier()
    result = classifier.classify(
        "Classes are suspended tomorrow.",
        "Sports update: the basketball team won their game yesterday.",
    )
    assert result.label == "INSUFFICIENT_EVIDENCE"


def test_empty_claim_raises_value_error():
    classifier = make_trained_classifier()
    with pytest.raises(ValueError):
        classifier.classify("", "Some evidence text.")


def test_empty_evidence_raises_value_error():
    classifier = make_trained_classifier()
    with pytest.raises(ValueError):
        classifier.classify("Some claim text.", "")


def test_fit_with_invalid_label_raises_value_error():
    classifier = ClaimEvidenceClassifier()
    with pytest.raises(ValueError):
        classifier.fit(
            ["A claim."], ["Some evidence."], ["MAYBE_TRUE"]  # invalid label
        )


def test_fit_with_mismatched_list_lengths_raises_value_error():
    classifier = ClaimEvidenceClassifier()
    with pytest.raises(ValueError):
        classifier.fit(
            ["Claim one.", "Claim two."], ["Evidence one."], ["SUPPORTED", "CONTRADICTED"]
        )


def test_classify_before_fit_raises_runtime_error():
    classifier = ClaimEvidenceClassifier()
    with pytest.raises(RuntimeError):
        classifier.classify("A claim.", "Some evidence.")


def test_confidence_scores_sum_to_approximately_one():
    """The label_scores should form a valid probability distribution --
    this checks the plumbing around predict_proba, not model accuracy."""
    classifier = make_trained_classifier()
    result = classifier.classify(
        "Classes are suspended tomorrow.",
        "Official notice: classes are suspended tomorrow due to weather.",
    )
    total = sum(result.label_scores.values())
    assert abs(total - 1.0) < 0.01


def test_build_classification_examples_excludes_empty_evidence():
    """Confirms the Step 3 fix: rows with empty evidence (retrieval-only
    INSUFFICIENT_EVIDENCE rows) must not appear in classifier training
    data."""
    records = [
        {"claim": "A claim.", "evidence": "Some evidence.", "label": "SUPPORTED", "group_id": "g1"},
        {"claim": "Another claim.", "evidence": "", "label": "INSUFFICIENT_EVIDENCE", "group_id": "g2"},
    ]
    examples = build_classification_examples(records)
    assert len(examples) == 1
    assert examples[0]["group_id"] == "g1"


def test_split_by_group_keeps_shared_groups_together():
    """Two rows sharing a group_id must end up on the SAME side of the
    split -- this is the core leakage-prevention guarantee."""
    records = [
        {"claim": "Claim A.", "evidence": "Evidence A.", "label": "SUPPORTED", "group_id": "evt_1"},
        {"claim": "Claim A contradicted.", "evidence": "Evidence A.", "label": "CONTRADICTED", "group_id": "evt_1"},
        {"claim": "Claim B.", "evidence": "Evidence B.", "label": "SUPPORTED", "group_id": "evt_2"},
        {"claim": "Claim C.", "evidence": "Evidence C.", "label": "SUPPORTED", "group_id": "evt_3"},
    ]
    train, val, test = split_by_group(records, train_frac=0.7, val_frac=0.15, seed=1)

    # Find which split evt_1's two rows landed in -- they must match.
    all_splits = {"train": train, "val": val, "test": test}
    evt_1_splits = set()
    for split_name, split_examples in all_splits.items():
        for e in split_examples:
            if e["group_id"] == "evt_1":
                evt_1_splits.add(split_name)

    assert len(evt_1_splits) == 1  # evt_1's rows are all in exactly one split
