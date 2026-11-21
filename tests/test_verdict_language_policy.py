"""Verdict language policy (CAMPUS_VERIFY_VERDICT_LANGUAGES) and the 'closest announcement' display. Offline:
tests/conftest.py pins TF-IDF and the development corpus; no model is loaded."""
import os

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGLISH = "walang pasok bukas dahil sa bagyo"
ENGLISH = "Classes are suspended tomorrow due to a typhoon."


def _post(api_main, text):
    with TestClient(api_main.app) as c:
        return c.post("/verify/text", json={"claim": text}).json()


def test_default_policy_is_english_only_and_is_validated():
    from src.api.pipeline import Settings
    assert Settings().verdict_languages == "english"
    with pytest.raises(RuntimeError):
        Settings(verdict_languages="tagalog").validate()


def test_taglish_claim_gets_the_closest_announcements_and_no_verdict(api_settings):
    body = _post(api_settings(verdict_languages="english"), TAGLISH)
    assert body["route"]["status"] == "CHECKABLE_CLAIM"            # still read as a claim
    assert body["verdict"] is None and body["classification_result"] is None
    assert body["verdict_withheld"] == "language" and body["claim"] == TAGLISH
    assert body["retrieved_evidence"], "the closest announcements must still be shown"
    assert body["explanation"].startswith("Verdicts are given for English claims only")


def test_english_claims_and_questions_are_unchanged(api_settings):
    api_main = api_settings(verdict_languages="english")
    claim = _post(api_main, ENGLISH)
    assert claim["verdict"] in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE") and claim["verdict_withheld"] is None
    question = _post(api_main, "When is the enrollment?")
    assert question["route"]["status"] == "INFO_SEEKING" and question["verdict_withheld"] is None


def test_all_restores_verdicts_for_every_language(api_settings):
    body = _post(api_settings(verdict_languages="all"), TAGLISH)
    assert body["verdict"] is not None and body["verdict_withheld"] is None


def test_health_reports_the_policy(api_settings):
    with TestClient(api_settings(verdict_languages="english").app) as c:
        assert c.get("/health").json()["verdict_languages"] == "english"


def test_frontend_shows_evidence_only_not_confirmed_and_the_closest_post():
    app = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
    # Plain wording since 2026-10-28 (owner): the posts first, the reason second, and "not confirmed" is never "false".
    for text in ('verdict_withheld === "language"', r"Here\u2019s what DYCI posted about this",
                 "without a yes/no answer", 'headline: "Not confirmed yet"', "That does not mean it is false",
                 '"Closest announcement"',
                 '"Closest post (low match)"', "Low match, may not be related", "function renderClosest"):
        assert text in app, text
    assert "renderUsedEvidence" not in app and "renderAnswer" not in app


# ---- the verdict gate's language check (src/api/verdict_policy.py) --------------------------------------------------
from src.api.verdict_policy import withhold_for_language  # noqa: E402
from src.routing.router import is_tagalog_or_taglish  # noqa: E402


@pytest.mark.parametrize("text", ["nagka-mass nung Sept 8?", "September 8? Akala ko 9?", "may kasama silang parents?",
                                  "may novena mass ngayong october?", "saang simbahan gaganapin?", TAGLISH])
def test_taglish_the_router_check_misses_is_withheld(text):
    assert withhold_for_language(text)


@pytest.mark.parametrize("text", [ENGLISH, "The library may close early on Friday.", "Don will lead the opening prayer.",
                                  "Classes will resume at 1 PM.", "What time was the start?", "The fiesta is in San Jose."])
def test_english_claims_keep_their_verdict(text):
    assert not withhold_for_language(text)


def test_the_nli_model_routing_check_is_unchanged():
    # The wider list is used only by the verdict gate; the router's check (which picks the NLI model) is untouched.
    assert not is_tagalog_or_taglish("nagka-mass nung Sept 8?") and withhold_for_language("nagka-mass nung Sept 8?")
