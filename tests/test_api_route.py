"""
M12.3: /verify/text and /verify/url route the RAW input first.
  CHECKABLE_CLAIM -> verified exactly as before (same verdict, evidence and scores as with the router off).
  INFO_SEEKING    -> evidence only: verdict and classification_result are null and the classifier is never called.
Fast offline configuration (tests/conftest.py). The e5 wiring test uses a fake encoder.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.normalization.question_form import CLAIM_REWRITE_KEYS
from tests.test_api import mock_html_response


def _post(api_settings, claim, path="/verify/text", url=None, **settings):
    api_main = api_settings(**{"router": True, "claim_rewrite": "none", **settings})
    with TestClient(api_main.app) as c:
        r = c.post(path, json={"claim": claim, **({"url": url} if url else {})})
    assert r.status_code == 200, r.text
    return r.json()


class _ClassifierMustNotRun:
    def classify(self, claim, evidence):
        raise AssertionError("the classifier must not be called for an INFO_SEEKING input")


# ---- INFO_SEEKING: no verdict -------------------------------------------------------------------------------------
@pytest.mark.parametrize("question", ["kailan ang enrollment?", "When is the enrollment?", "Enrollment dates?", "po ba?"])
def test_info_seeking_input_gets_evidence_and_no_verdict(question, api_settings):
    body = _post(api_settings, question)
    assert body["route"]["status"] == "INFO_SEEKING"
    assert body["verdict"] is None and body["classification_result"] is None
    assert body["claim"] == question and body["explanation"].startswith(body["route"]["guidance"])


def test_info_seeking_never_calls_the_classifier(api_settings, monkeypatch):
    api_main = api_settings(router=True)
    with TestClient(api_main.app) as c:
        monkeypatch.setattr(api_main.state["manager"].current().verifier, "_classifier", _ClassifierMustNotRun())
        r = c.post("/verify/text", json={"claim": "When does enrollment begin?"})
    assert r.status_code == 200 and r.json()["verdict"] is None


# ---- CHECKABLE_CLAIM: unchanged verification ------------------------------------------------------------------------
CLAIMS = ["Classes are suspended tomorrow due to a typhoon.", "Are classes suspended tomorrow?",
          "december 1 no class", "walang pasok bukas?", "enrollment august 10"]


@pytest.mark.parametrize("claim", CLAIMS)
def test_checkable_claims_are_verified_exactly_as_with_the_router_off(claim, api_settings):
    off, on = _post(api_settings, claim, router=False), _post(api_settings, claim, router=True)
    assert on["route"]["status"] == "CHECKABLE_CLAIM" and off["route"] is None
    for k in ("claim", "verdict", "evidence_status", "retrieved_evidence", "used_evidence", "classification_result",
              "explanation"):
        assert off[k] == on[k], k
    assert on["verdict"] in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")


def test_router_off_restores_the_old_behaviour_of_verifying_everything(api_settings):
    body = _post(api_settings, "kailan ang enrollment?", router=False)
    assert body["route"] is None and body["verdict"] in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")


# ---- /verify/url --------------------------------------------------------------------------------------------------
@patch("src.ingestion.adapters.requests.get")
def test_verify_url_routes_too(mock_get, api_settings):
    mock_get.return_value = mock_html_response(
        text="<html><body><p>Enrollment for the second semester begins August 10 and ends August 20.</p></body></html>")
    with patch("src.api.ssrf_guard.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 0))]):
        info = _post(api_settings, "kailan ang enrollment?", path="/verify/url", url="https://example.edu/a")
        claim = _post(api_settings, "Enrollment begins August 10.", path="/verify/url", url="https://example.edu/a")
    assert info["route"]["status"] == "INFO_SEEKING" and info["verdict"] is None
    assert info["used_evidence"]["source_url"] == "https://example.edu/a"
    assert claim["route"]["status"] == "CHECKABLE_CLAIM" and claim["verdict"] is not None


# ---- Option B rewrite (on by default) ---------------------------------------------------------------------------------
def test_option_b_rewrite_checks_the_statement_and_keeps_the_original(api_settings, monkeypatch):
    from src.verification.verifier import ClaimVerifier
    seen, real = [], ClaimVerifier.verify_claim
    monkeypatch.setattr(ClaimVerifier, "verify_claim", lambda self, c: seen.append(c) or real(self, c))
    body = _post(api_settings, "may exam ba sa sept 28?", claim_rewrite="optionB")
    assert body["claim"] == "may exam ba sa sept 28?"                       # what the user typed, unchanged
    assert body["claim_rewrite"]["rewritten"] == "may exam sa sept 28." and seen == ["may exam sa sept 28."]
    assert tuple(body["claim_rewrite"]) == CLAIM_REWRITE_KEYS


def test_rewrite_is_on_by_default_never_applied_to_info_seeking_and_can_be_switched_off(api_settings):
    from src.api.pipeline import Settings
    assert Settings().claim_rewrite == "optionB"
    assert _post(api_settings, "kailan ang exam?", claim_rewrite="optionB")["claim_rewrite"] is None
    assert _post(api_settings, "may exam ba sa sept 28?", claim_rewrite="none")["claim_rewrite"] is None


def test_unknown_rewrite_setting_fails_startup(api_settings):
    api_main = api_settings(claim_rewrite="translate")
    with pytest.raises(RuntimeError, match="CAMPUS_VERIFY_CLAIM_REWRITE"):
        with TestClient(api_main.app):
            pass


# ---- /health and the e5 (M12.2) configuration ------------------------------------------------------------------------
def test_health_reports_input_handling(api_settings):
    api_main = api_settings(router=True)
    with TestClient(api_main.app) as c:
        h = c.get("/health").json()
    assert h["router"].startswith("router-v") and h["retriever"] == "tfidf" and h["date_check"] is True
    assert h["corpus_loaded_at"] and h["corpus_reload_error"] is None and h["facebook_refresh"]["enabled"] is False


def test_e5_backend_builds_the_frozen_m12_2_configuration(api_settings, monkeypatch):
    from src.retrieval import e5_retriever
    from tests.test_m12_2_e5_and_passages import FakeModel
    monkeypatch.delenv("CAMPUS_VERIFY_E5_THRESHOLD", raising=False)
    monkeypatch.setattr(e5_retriever.E5Encoder, "__init__",
                        lambda self, model_name=None, device="cpu", model=None: setattr(self, "_model", FakeModel()))
    api_main = api_settings(retriever="e5", router=True)
    with TestClient(api_main.app) as c:
        p = api_main.state["manager"].current()
        assert type(p.retriever).__name__ == "E5Retriever" and p.retriever.score_mode == "passage"
        assert p.verifier._relevance_threshold == 0.833 and p.verifier._top_k == 3
        assert p.verifier._passage_selector._context_before == 1
        assert c.get("/health").json()["retriever"] == "e5"
        assert c.post("/verify/text", json={"claim": "kailan ang enrollment?"}).json()["verdict"] is None
