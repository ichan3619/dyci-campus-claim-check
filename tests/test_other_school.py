"""Other-school check (C2, src/verification/other_school.py, CAMPUS_VERIFY_OTHER_SCHOOL_CHECK default off). Covers the
pre-registered criterion 8: aliases, capitals-only abbreviations, ambiguous names, several schools, partial mentions,
posts that correctly name the school, messages without a reliably detected school; uncertainty never changes a verdict."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.verification import other_school as osc
from src.verification.verifier import VerificationResult

BOARD = NS(document_id="n06", text="Congratulations to our DYCIan Registered Nurses for a 99.35% passing rate.")
TEACHERS = NS(document_id="n01", text="DYCI hosted the Teachers' Day of the Bocaue Private Schools Association with a "
                                      "speaker from Ateneo de Manila University.")
SPUS_FULL = NS(document_id="n13", text="DYCI officials visited St. Paul University Surigao (SPUS) for benchmarking.")
SPUS_ABBR = NS(document_id="n13b", text="The benchmarking visit to SPUS ended on Friday.")
SPUS_SHORT = NS(document_id="n13c", text="Our team visited St. Paul Surigao this week.")
EXCHANGE = NS(document_id="n12", text="Students from the University of Electro-Communications joined our classes.")
SEOUL = NS(document_id="seoul", text="Guests from Seoul National came to Elida campus.")


def _result(verdict, post):
    return VerificationResult(claim="c", verdict=verdict, evidence_status="RELEVANT_EVIDENCE_FOUND", retrieved_evidence=[post],
                              used_evidence=post, classification_result=None, explanation="E.")


# ---- the targets -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("message, post, verdict", [
    ("The passing rate of STI College in the nursing board exam was 99.35%.", BOARD, "SUPPORTED"),     # M04
    ("Did La Salle hold a Teachers' Day celebration?", TEACHERS, "CONTRADICTED"),                       # M05
    ("Did the DYCI delegation visit St. Paul University Surigao?", EXCHANGE, "CONTRADICTED"),           # EQ05
])
def test_school_missing_from_the_post_used_withholds(message, post, verdict):
    out, info = osc.check(message, _result(verdict, post))
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["withheld"] and info["original_verdict"] == verdict
    assert info["missing_in_post"] and "Other-school check" in out.explanation and out.used_evidence is post


# ---- aliases, capitals-only, posts that name the school ------------------------------------------------------------------
@pytest.mark.parametrize("message, post", [
    ("The benchmarking visit to St. Paul University Surigao was from September 27 to 29.", SPUS_FULL),   # C08-like
    ("Has the Memorandum of Agreement with SPUS been signed?", SPUS_FULL),                               # EQ08-like (alias)
    ("Is the benchmarking visit to SPUS from September 2 to 4?", SPUS_ABBR),                              # abbr in post
    ("Was the guest speaker from Ateneo?", TEACHERS),
])
def test_post_that_names_the_school_keeps_the_verdict(message, post):
    original = _result("CONTRADICTED", post)
    out, info = osc.check(message, original)
    assert out is original and info["withheld"] is False and info["missing_in_post"] == []


def test_capitals_only_abbreviations():
    # "UP" in capitals is a school; "up" is a word and never triggers.
    assert osc.check("UP has no classes on Monday", _result("SUPPORTED", BOARD))[1]["withheld"] is True
    r = _result("SUPPORTED", BOARD)
    assert osc.check("classes are up on Monday", r) == (r, None)
    assert osc.check("sabi ng ama ko walang pasok", r) == (r, None)


# ---- uncertainty never changes a verdict -----------------------------------------------------------------------------------
def test_an_ambiguous_abbreviation_alone_never_triggers():
    r = _result("SUPPORTED", BOARD)
    assert osc.schools_to_check("BSU suspended classes") == []
    assert osc.check("BSU suspended classes on September 10", r) == (r, None)


@pytest.mark.parametrize("message, post", [
    ("Did the delegation visit St. Paul University Surigao?", SPUS_SHORT),     # fixed short form "st. paul"
    ("Seoul National University students visited", SEOUL),                     # pattern name without "University"
])
def test_partial_mention_counts_as_a_mention(message, post):
    original = _result("SUPPORTED", post)
    out, info = osc.check(message, original)
    assert out is original and info["withheld"] is False


def test_no_reliably_detected_school_never_acts():
    for message in ("No classes on September 27.", "Dr. Yanga's Colleges suspended classes", "The College of Health Sciences topped",
                    "hogwarts got a 99.35% passing rate"):           # invented name without a keyword: documented false negative
        r = _result("SUPPORTED", BOARD)
        assert osc.check(message, r) == (r, None), message


# ---- several schools -------------------------------------------------------------------------------------------------------
def test_several_schools_one_missing_withholds_and_lists_it():
    out, info = osc.check("SPUS and BulSU signed an agreement", _result("SUPPORTED", SPUS_FULL))
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["missing_in_post"] == ["Bulacan State University"]
    assert info["schools"] == ["St. Paul University", "Bulacan State University"]


def test_ambiguous_plus_reliable_school_checks_only_the_reliable_one():
    original = _result("SUPPORTED", SPUS_FULL)
    out, info = osc.check("BSU and SPUS signed an agreement", original)
    assert out is original and info["schools"] == ["St. Paul University"] and info["withheld"] is False


# ---- safety --------------------------------------------------------------------------------------------------------------
def test_never_creates_or_reverses_a_verdict():
    for v in ("INSUFFICIENT_EVIDENCE",):
        r = _result(v, BOARD)
        assert osc.check("STI passing rate was 99.35%", r) == (r, None)
    no_used = VerificationResult(claim="c", verdict="SUPPORTED", evidence_status="RELEVANT_EVIDENCE_FOUND",
                                 retrieved_evidence=[], used_evidence=None, classification_result=None, explanation="E.")
    assert osc.check("STI passing rate", no_used) == (no_used, None)
    for v in ("SUPPORTED", "CONTRADICTED"):
        assert osc.check("STI passing rate", _result(v, BOARD))[0].verdict in (v, "INSUFFICIENT_EVIDENCE")


TEXTS = ["Enrollment at BulSU for the second semester starts August 10.", "Enrollment for the second semester starts August 10.",
         "Classes are suspended tomorrow due to a typhoon.", "walang pasok sa BulSU sa March 3", "kailan ang enrollment sa UP?"]


def _bodies(api_main):
    with TestClient(api_main.app) as c:
        return c.get("/health").json(), {t: c.post("/verify/text", json={"claim": t}).json() for t in TEXTS}


def test_default_on_since_v1_2_and_on_only_withholds_school_verdicts(api_settings):
    from src.api.pipeline import Settings
    assert Settings().other_school_check is True            # adopted 2026-10-12 (C2 v2, all criteria passed)
    h_off, off = _bodies(api_settings(other_school_check=False, verdict_languages="english"))
    h_on, on = _bodies(api_settings(other_school_check=True, verdict_languages="english"))
    assert h_off["other_school_check"] is False and h_on["other_school_check"] is True
    assert all(b["other_school_check"] is None for b in off.values())
    school = "Enrollment at BulSU for the second semester starts August 10."
    assert off[school]["verdict"] in osc.SC
    assert on[school]["verdict"] == "INSUFFICIENT_EVIDENCE" and on[school]["other_school_check"]["withheld"] is True
    strip = lambda b: {k: v for k, v in b.items() if k not in ("verdict", "explanation", "other_school_check")}
    assert strip(on[school]) == strip(off[school])
    for t in TEXTS[1:]:
        assert on[t] == off[t], t                    # no school, Taglish evidence-only and questions: identical


@patch("src.ingestion.adapters.requests.get")
def test_url_route_is_untouched(mock_get, api_settings):
    from tests.test_api import mock_html_response
    mock_get.return_value = mock_html_response(text="<html><body><p>Classes are suspended tomorrow due to a typhoon.</p></body></html>")
    with TestClient(api_settings(other_school_check=True).app) as c:
        r = c.post("/verify/url", json={"claim": "At BulSU classes are suspended tomorrow due to a typhoon.",
                                        "url": "https://example.edu/announcement"})
    assert r.status_code == 200 and r.json()["other_school_check"] is None


def test_frontend_explains_a_withheld_verdict():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = open(os.path.join(root, "frontend", "js", "app.js"), encoding="utf-8").read()
    assert "data.other_school_check && data.other_school_check.withheld" in app
    assert "The post used does not mention " in app
