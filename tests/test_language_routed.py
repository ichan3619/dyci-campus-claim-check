"""
LanguageRoutedClassifier (M12.3 opt-in): Tagalog/Taglish claims go to the multilingual NLI model, English claims never
do. Fake classifiers only; no model downloads.
"""
import pytest

from src.classification.language_routed import LanguageRoutedClassifier
from src.classification.result import ClassificationResult
from src.routing.router import is_tagalog_or_taglish


class _Fake:
    def __init__(self, label):
        self.label, self.calls = label, []

    def classify(self, claim, evidence):
        self.calls.append(claim)
        return ClassificationResult(label=self.label, confidence=0.8, label_scores={self.label: 0.8})


def _routed():
    en, tl = _Fake("INSUFFICIENT_EVIDENCE"), _Fake("SUPPORTED")
    return LanguageRoutedClassifier(en, tl, english_name="en-model", tagalog_name="tl-model"), en, tl


@pytest.mark.parametrize("claim", ["walang pasok bukas?", "suspended ba ang klase bukas?", "may exam ba sa sept 28?"])
def test_tagalog_and_taglish_claims_use_the_multilingual_model(claim):
    clf, en, tl = _routed()
    r = clf.classify(claim, "evidence")
    assert (r.label, r.model) == ("SUPPORTED", "tl-model") and tl.calls == [claim] and en.calls == []


@pytest.mark.parametrize("claim", ["Classes are suspended tomorrow.", "enrollment august 10", "library 24/7 finals"])
def test_english_claims_never_reach_the_multilingual_model(claim):
    clf, en, tl = _routed()
    r = clf.classify(claim, "evidence")
    assert (r.label, r.model) == ("INSUFFICIENT_EVIDENCE", "en-model") and tl.calls == []


def test_detector_matches_the_router_caveat():
    from src.routing import ClaimRouter
    for text in ("walang pasok bukas?", "Classes are suspended tomorrow.", "kailan ang enrollment?", "po"):
        assert is_tagalog_or_taglish(text) == ("tagalog_or_taglish" in ClaimRouter().route(text).caveats)


def test_model_field_is_optional_and_defaults_to_none():
    assert ClassificationResult(label="SUPPORTED", confidence=1.0, label_scores={}).model is None


def test_api_rejects_the_tagalog_model_without_the_nli_backend(api_settings):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    api_main = api_settings(classifier="tfidf_lr", tagalog_nli_model="some/model")
    with pytest.raises(RuntimeError, match="CAMPUS_VERIFY_TAGALOG_NLI_MODEL"):
        with TestClient(api_main.app):
            pass


def test_api_builds_the_language_routed_classifier(api_settings, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from src.classification import nli_classifier
    monkeypatch.setattr(nli_classifier, "NLIClassifier", lambda name=None: _Fake("SUPPORTED"))
    api_main = api_settings(classifier="nli", tagalog_nli_model="C:/models/multi-nli@abc")
    with TestClient(api_main.app) as c:
        h = c.get("/health").json()
        body = c.post("/verify/text", json={"claim": "suspended ba ang klase bukas?"}).json()
    assert h["classifier"] == "LanguageRoutedClassifier" and h["tagalog_nli_model"] == "multi-nli@abc"
    if body["classification_result"]:
        assert body["classification_result"]["model"] == "multi-nli@abc"
