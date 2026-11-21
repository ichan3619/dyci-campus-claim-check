"""ClaimVerifier(premise_mode="sliding") and CAMPUS_VERIFY_PREMISE_MODE. Fakes only: no model is loaded."""
import numpy as np
import pytest

from src.classification.result import ClassificationResult
from src.retrieval.passage_selector import PassageSelector
from src.retrieval.retriever import RetrievedDocument
from src.verification.verifier import ClaimVerifier

DOC = ("DYCI hosted the Teachers Day Celebration on October 1. "
       "The raffle and games made it extra special for everyone. "
       "The day ended with a message of appreciation for teachers.")
LAST = "The day ended with a message of appreciation for teachers."


class _Retriever:
    default_relevance_threshold = 0.5

    def retrieve(self, query, top_k=3):
        return [RetrievedDocument(document_id="n01", score=0.9, text=DOC, source_type="manual_entry", source_url="")]


class _Encoder:
    """E5 stand-in: the passage `best` gets the highest similarity, so the selector picks it."""
    def __init__(self, best=LAST):
        self.best = best

    def encode_queries(self, texts):
        return np.array([[1.0]])

    def encode_passages(self, texts):
        return np.array([[1.0 if t == self.best else 0.1] for t in texts])


class _Rules:
    """Label by substring of the premise: {substring: (label, confidence)}; otherwise neutral 0.9."""
    def __init__(self, rules):
        self.rules, self.calls = rules, 0

    def classify(self, claim, evidence):
        self.calls += 1
        for sub, (label, conf) in self.rules.items():
            if sub in evidence:
                return ClassificationResult(label=label, confidence=conf, label_scores={label: conf})
        return ClassificationResult(label="INSUFFICIENT_EVIDENCE", confidence=0.9, label_scores={"INSUFFICIENT_EVIDENCE": 0.9})


def _verifier(rules, mode="sliding", date_check=False):
    clf = _Rules(rules)
    selector = PassageSelector(_Encoder(), context_before=1)
    return ClaimVerifier(_Retriever(), clf, passage_selector=selector, date_check=date_check, premise_mode=mode), clf


def test_windows_have_todays_shape():
    w = PassageSelector(_Encoder(), context_before=1).windows_of(_Retriever().retrieve("q")[0])
    assert len(w) == 3 and w[0].endswith("October 1.") and w[2].startswith("The raffle") and w[2].endswith(LAST)


def test_window_mode_is_the_default_and_unchanged():
    v, clf = _verifier({"October 1": ("CONTRADICTED", 0.95)}, mode="window")
    r = v.verify_claim("The celebration was held on October 5.")
    assert r.verdict == "INSUFFICIENT_EVIDENCE" and clf.calls == 1 and r.sliding_windows is None
    assert ClaimVerifier(_Retriever(), _Rules({}))._premise_mode == "window"


def test_sliding_finds_the_deciding_window_and_reports_the_most_confident():
    v, clf = _verifier({"raffle": ("CONTRADICTED", 0.80), "October 1": ("CONTRADICTED", 0.95)})
    r = v.verify_claim("The celebration was held on October 5.")
    assert r.verdict == "CONTRADICTED" and clf.calls == 3
    assert r.sliding_windows["chosen_index"] == 0 and r.sliding_windows["conflict"] is False
    assert r.classification_result.confidence == 0.95 and "deciding window" in r.explanation


def test_any_supported_contradicted_conflict_is_insufficient_regardless_of_confidence():
    v, _ = _verifier({"October 1": ("CONTRADICTED", 0.99), "appreciation": ("SUPPORTED", 0.51)})
    r = v.verify_claim("The celebration was held on October 5.")
    assert r.verdict == "INSUFFICIENT_EVIDENCE" and r.sliding_windows["conflict"] is True
    assert r.sliding_windows["chosen_index"] is None and "disagreed" in r.explanation


def test_ties_go_to_the_earliest_window_and_cannot_change_the_label():
    v, _ = _verifier({"Teachers Day": ("SUPPORTED", 0.9), "raffle": ("SUPPORTED", 0.9)})
    r = v.verify_claim("There was a Teachers Day celebration.")
    assert r.verdict == "SUPPORTED" and r.sliding_windows["chosen_index"] == 0


def test_all_neutral_windows_give_insufficient_with_the_e5_window_result():
    v, _ = _verifier({})
    r = v.verify_claim("There was free food.")
    assert r.verdict == "INSUFFICIENT_EVIDENCE" and r.sliding_windows["chosen_index"] is None
    assert r.sliding_windows["e5_window_index"] == 2


def test_the_date_check_still_applies_after_the_sliding_rule():
    v, _ = _verifier({"October 1": ("SUPPORTED", 0.97)}, date_check=True)
    r = v.verify_claim("The celebration was on October 9.")
    assert r.verdict == "INSUFFICIENT_EVIDENCE" and r.date_check == "claimed_date_not_in_evidence"


def test_setting_defaults_to_window_is_validated_and_reported(api_settings):
    from fastapi.testclient import TestClient
    from src.api.pipeline import Settings
    assert Settings().premise_mode == "window"
    with pytest.raises(RuntimeError):
        Settings(premise_mode="document").validate()
    api_main = api_settings()
    with TestClient(api_main.app) as c:
        assert c.get("/health").json()["premise_mode"] == "window"
