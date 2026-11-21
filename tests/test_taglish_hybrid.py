"""CAMPUS_VERIFY_TAGLISH_HYBRID: hybrid (E5/TF-IDF + TagalogBM25, RRF) ranking ONLY for evidence-only Tagalog/Taglish
responses. Offline (tests/conftest.py: TF-IDF, development corpus); no model is loaded."""
from fastapi.testclient import TestClient

from src.retrieval.hybrid_retriever import HybridRetriever

ENGLISH_CLAIM = "Classes are suspended tomorrow due to a typhoon."
ENGLISH_QUESTION = "When is the enrollment?"
TAGLISH_CLAIM = "walang pasok bukas dahil sa bagyo"
TAGLISH_QUESTION = "kailan ang enrollment?"


def _bodies(api_main, texts):
    with TestClient(api_main.app) as c:
        return {t: c.post("/verify/text", json={"claim": t}).json() for t in texts}, api_main.state["manager"].current()


def test_flag_defaults_on_since_v1_1_and_off_builds_nothing(api_settings):
    from src.api.pipeline import Settings
    assert Settings().taglish_hybrid is True            # adopted 2026-10-10 (pre-registered criteria met)
    _, pipeline = _bodies(api_settings(taglish_hybrid=False), [])
    assert pipeline.evidence_verifier is None


def test_flag_on_builds_a_hybrid_evidence_verifier_with_the_main_threshold(api_settings):
    api_main = api_settings(taglish_hybrid=True)
    with TestClient(api_main.app) as c:
        assert c.get("/health").json()["taglish_hybrid"] is True
        p = api_main.state["manager"].current()
    assert isinstance(p.evidence_verifier._retriever, HybridRetriever)
    assert p.evidence_verifier._relevance_threshold == p.verifier._relevance_threshold


def test_english_responses_are_identical_in_full(api_settings):
    off, _ = _bodies(api_settings(taglish_hybrid=False), [ENGLISH_CLAIM, ENGLISH_QUESTION])
    on, _ = _bodies(api_settings(taglish_hybrid=True), [ENGLISH_CLAIM, ENGLISH_QUESTION])
    assert on == off                     # whole body: verdict, route, evidence order, passage, explanation ...


def test_only_evidence_only_taglish_responses_use_the_hybrid_verifier(api_settings, monkeypatch):
    api_main = api_settings(taglish_hybrid=True, verdict_languages="english")
    calls = []
    with TestClient(api_main.app) as c:
        ev = api_main.state["manager"].current().evidence_verifier
        real = ev.retrieve_information
        monkeypatch.setattr(ev, "retrieve_information", lambda q: calls.append(q) or real(q))
        monkeypatch.setattr(ev, "verify_claim", lambda q: (_ for _ in ()).throw(AssertionError("no verdict from hybrid")))
        bodies = {t: c.post("/verify/text", json={"claim": t}).json()
                  for t in (ENGLISH_CLAIM, ENGLISH_QUESTION, TAGLISH_CLAIM, TAGLISH_QUESTION)}
    assert len(calls) == 2                                   # the Taglish claim (withheld) and the Taglish question
    assert bodies[TAGLISH_CLAIM]["verdict"] is None and bodies[TAGLISH_CLAIM]["verdict_withheld"] == "language"
    assert bodies[TAGLISH_QUESTION]["verdict"] is None and bodies[TAGLISH_QUESTION]["route"]["status"] == "INFO_SEEKING"
    assert bodies[ENGLISH_CLAIM]["verdict"] is not None


def test_with_policy_all_taglish_verdicts_still_come_from_the_unchanged_verifier(api_settings, monkeypatch):
    api_main = api_settings(taglish_hybrid=True, verdict_languages="all")
    with TestClient(api_main.app) as c:
        ev = api_main.state["manager"].current().evidence_verifier
        monkeypatch.setattr(ev, "verify_claim", lambda q: (_ for _ in ()).throw(AssertionError("no verdict from hybrid")))
        body = c.post("/verify/text", json={"claim": TAGLISH_CLAIM}).json()
    assert body["verdict"] is not None and body["verdict_withheld"] is None
