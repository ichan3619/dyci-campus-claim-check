"""Scope notice (C1, display only, src/api/school_scope.py): "This checker only covers DYCI announcements. The posts shown
are DYCI's, not BulSU's." for messages that name another school no shown post mentions. Never changes a decision."""
import os
from types import SimpleNamespace as Post
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import school_scope
from src.api.school_scope import detect, notice

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- detection ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, schools", [
    ("The passing rate of STI College in the nursing board exam was 99.35%.", ["STI"]),
    ("Did La Salle hold a Teachers' Day celebration?", ["De La Salle"]),
    ("BulSU has no classes on August 21.", ["Bulacan State University"]),
    ("walang pasok sa bulsu bukas", ["Bulacan State University"]),
    ("Bulacan State University suspended classes", ["Bulacan State University"]),
    ("UP Diliman midterms are on September 28", ["University of the Philippines"]),
    ("PUP and UST have no classes", ["Polytechnic University of the Philippines", "University of Santo Tomas"]),
    ("AMA has no classes tomorrow", ["AMA"]),
    ("Ateneo won the debate", ["Ateneo"]),
    ("Is there a guest speaker from Ateneo de Manila University?", ["Ateneo"]),     # one school, not two
    ("Seoul National University students visited", ["Seoul National University"]),   # name pattern
    ("Students from the University of Electro-Communications came", ["University of Electro-Communications"]),
])
def test_known_names_abbreviations_and_patterns(text, schools):
    assert detect(text) == schools


def test_ambiguous_abbreviation_is_shown_as_typed():
    assert detect("BSU suspended classes") == ["BSU"]


@pytest.mark.parametrize("text", ["classes are up again", "sabi ng ama ko walang pasok", "the pup was cute",
                                  "must be ust a typo", "sti ll no classes"])
def test_ordinary_words_are_not_schools(text):
    assert detect(text) == []


@pytest.mark.parametrize("text", ["Dr. Yanga's Colleges suspended classes", "Dr. Yanga’s Colleges, Inc.", "DYCI Engineering College",
                                  "The College of Health Sciences topped the exam", "School of Psychology Day 19",
                                  "Francisco Balagtas Academy was founded in 1950", "No classes tomorrow."])
def test_dyci_names_and_units_are_never_reported(text):
    assert detect(text) == []


def test_several_schools_in_order_without_repeats():
    assert detect("BulSU and UP suspended classes, BulSU again") == ["Bulacan State University", "University of the Philippines"]


# ---- notice -------------------------------------------------------------------------------------------------------------
BOARD = Post(document_id="n06", text="Congratulations to our DYCIan Registered Nurses for a 99.35% passing rate.")
SPUS = Post(document_id="n13", text="DYCI officials visited St. Paul University Surigao (SPUS) for benchmarking.")


def test_notice_wording_is_informational():
    n = notice("The passing rate of STI College in the nursing board exam was 99.35%.", [BOARD])
    assert n == {"schools": ["STI"],
                 "text": "This checker only covers DYCI announcements. The posts shown are DYCI’s, not STI's."}
    for word in ("false", "wrong", "contradict", "not true", "fake"):
        assert word not in n["text"].lower()


def test_multiple_schools_in_one_sentence():
    assert notice("BulSU and UP suspended classes", [BOARD])["text"].endswith(
        "not Bulacan State University's or University of the Philippines's.")


def test_a_school_a_shown_post_mentions_gets_no_notice():
    assert notice("Did the delegation visit St. Paul University Surigao?", [SPUS]) is None
    # The post mentions one school but not the other: only the other is listed.
    n = notice("Did St. Paul University and BulSU sign an agreement?", [SPUS, BOARD])
    assert n["schools"] == ["Bulacan State University"]


def test_no_school_no_notice():
    assert notice("No classes on September 27.", [BOARD]) is None
    assert notice("walang pasok bukas", []) is None


# ---- API ----------------------------------------------------------------------------------------------------------------
CASES = {
    "verdict": "Enrollment at BulSU for the second semester starts August 10.",
    "evidence_only": "walang pasok sa BulSU sa March 3",
    "question": "kailan ang enrollment sa UP?",
    "low_match": "The AMA moon festival is on December 25.",
    "none": "Enrollment for the second semester starts August 10.",
}


def _run(api_main):
    with TestClient(api_main.app) as c:
        return {k: c.post("/verify/text", json={"claim": t}).json() for k, t in CASES.items()}


def test_every_response_type_gets_the_notice_and_no_decision_changes(api_settings, monkeypatch):
    api_main = api_settings(verdict_languages="english")
    on = _run(api_main)
    assert on["verdict"]["verdict"] is not None
    assert on["evidence_only"]["verdict_withheld"] == "language"
    assert on["question"]["route"]["status"] == "INFO_SEEKING"
    assert on["low_match"]["evidence_status"] == "NO_RELEVANT_EVIDENCE"
    for key in ("verdict", "evidence_only", "question", "low_match"):
        assert on[key]["scope_notice"]["text"].startswith("This checker only covers DYCI announcements."), key
    assert on["none"]["scope_notice"] is None
    monkeypatch.setattr(school_scope, "notice", lambda *a, **k: None)
    off = _run(api_main)
    for key in CASES:
        a, b = dict(on[key]), dict(off[key])
        a.pop("scope_notice"); b.pop("scope_notice")
        assert a == b, key


@patch("src.ingestion.adapters.requests.get")
def test_url_route_never_carries_the_notice(mock_get, api_settings):
    from tests.test_api import mock_html_response
    mock_get.return_value = mock_html_response(text="<html><body><p>BulSU: classes are suspended tomorrow.</p></body></html>")
    with TestClient(api_settings().app) as c:
        r = c.post("/verify/url", json={"claim": "BulSU classes are suspended tomorrow.", "url": "https://example.edu/announcement"})
    assert r.status_code == 200 and r.json()["scope_notice"] is None


def test_frontend_shows_the_notice_under_the_message():
    app = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
    assert "data.scope_notice && data.scope_notice.text" in app
