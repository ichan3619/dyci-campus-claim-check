"""
ClaimVerifier.retrieve_information(): the M12.3 INFO_SEEKING path. Same retrieval, gate and passage selection as
verify_claim(), never a classifier call, never a verdict. Uses the fake e5 encoder from the M12.2 tests (no download).
"""
import pytest

from src.retrieval.e5_retriever import E5Encoder, E5Retriever
from src.retrieval.passage_selector import PassageSelector
from src.verification.verifier import NO_RELEVANT_EVIDENCE, RELEVANT_EVIDENCE_FOUND, ClaimVerifier, InformationResult
from tests.test_m12_2_e5_and_passages import DOCS, FakeModel, RecordingClassifier


@pytest.fixture()
def enc():
    return E5Encoder(model=FakeModel())


def _verifier(enc, threshold=0.05, selector=True):
    retr = E5Retriever(DOCS, encoder=enc, relevance_threshold=threshold)
    clf = RecordingClassifier()
    return ClaimVerifier(retr, clf, passage_selector=PassageSelector(enc) if selector else None), clf


def test_returns_evidence_and_selected_passage_without_calling_the_classifier(enc):
    v, clf = _verifier(enc)
    info = v.retrieve_information("when are classes suspended due to the typhoon")
    assert isinstance(info, InformationResult) and not hasattr(info, "verdict")
    assert info.evidence_status == RELEVANT_EVIDENCE_FOUND and info.used_evidence.document_id == "typhoon"
    assert info.selected_passage is not None and "suspended" in info.selected_passage.text
    assert clf.calls == []


def test_uses_the_same_gate_and_evidence_as_verify_claim(enc):
    v, _ = _verifier(enc)
    text = "classes suspended typhoon"
    info, result = v.retrieve_information(text), v.verify_claim(text)
    assert info.used_evidence == result.used_evidence and info.selected_passage == result.selected_passage
    assert info.retrieved_evidence == result.retrieved_evidence


def test_below_the_gate_returns_no_evidence(enc):
    v, clf = _verifier(enc, threshold=0.999)
    info = v.retrieve_information("what is the weather in paris")
    assert info.evidence_status == NO_RELEVANT_EVIDENCE and info.used_evidence is None and info.selected_passage is None
    assert clf.calls == []


def test_empty_query_is_rejected(enc):
    with pytest.raises(ValueError):
        _verifier(enc)[0].retrieve_information("  ")
