"""Display-only date notice (src/api/date_notice.py): "This post does not mention September 27." when the raw message
names an explicit date a shown post does not contain. It never changes a decision."""
import os
from types import SimpleNamespace as Post

import pytest
from fastapi.testclient import TestClient

from src.api.date_notice import date_notice

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEPT10 = Post(document_id="sept10", text="DYCI ADVISORY 10 September 2026 IN-PERSON CLASSES are SUSPENDED today, "
                                         "September 10, 2026, due to the Southwest Monsoon.")
SEPT27 = Post(document_id="sept27", text="DYCI Advisory 27 September 2026 SHORTENED CLASS HOURS Please be informed that "
                                         "tomorrow, September 28, 2026, classes will be until 1:00 PM only.")
EXAMS = Post(document_id="exams", text="The midterm examinations are scheduled on September 28 to 30, 2026.")
NODATE = Post(document_id="nodate", text="The library is open to all students.")


# ---- unit -----------------------------------------------------------------------------------------------------------
def test_single_date_missing_from_a_post():
    n = date_notice("september 27 walang pasok", [SEPT10])
    assert n == {"message_dates": ["September 27"],
                 "posts": {"sept10": {"missing": ["September 27"], "text": "This post does not mention September 27."}}}


def test_date_present_in_the_post_gives_no_notice():
    assert date_notice("walang pasok sa sept 10?", [SEPT10]) is None


def test_range_with_every_date_present_gives_no_notice():
    assert date_notice("Are the midterms on September 28-30?", [EXAMS]) is None


def test_range_with_some_dates_present_lists_only_the_missing_ones():
    n = date_notice("Classes are suspended Sept 27-29.", [SEPT27])
    assert n["posts"]["sept27"] == {"missing": ["September 29"], "text": "This post does not mention September 29."}
    n = date_notice("Exams on September 27 to 29", [EXAMS])
    assert n["posts"]["exams"]["text"] == "This post does not mention September 27."


@pytest.mark.parametrize("message", ["walang pasok bukas", "Is there class on Monday?", "9/27 walang pasok",
                                     "may 2 sections daw", ""])
def test_no_explicit_supported_date_gives_no_notice(message):
    # Unsupported formats are documented, not parsed: numeric dates, 'bukas', weekdays, Tagalog existential 'may'.
    assert date_notice(message, [SEPT10, NODATE]) is None


def test_several_posts_with_different_coverage():
    n = date_notice("No classes on September 28.", [SEPT10, SEPT27, EXAMS, NODATE, None, SEPT10])
    assert set(n["posts"]) == {"sept10", "nodate"}                  # sept27 and exams both contain September 28
    assert n["posts"]["nodate"]["text"] == "This post does not mention September 28."


def test_header_date_counts_as_mentioned_and_no_event_date_is_inferred():
    # The Sept 27 advisory is ABOUT September 28; its header date (27 September) is still text in the post, so neither
    # date gets a notice. The notice reports textual absence only.
    assert date_notice("walang pasok sa September 27", [SEPT27]) is None
    assert date_notice("walang pasok sa September 28", [SEPT27]) is None
    assert date_notice("walang pasok sa September 26", [SEPT27])["posts"]["sept27"]["missing"] == ["September 26"]


def test_wording_is_neutral():
    text = date_notice("Classes are suspended Sept 1-3.", [NODATE])["posts"]["nodate"]["text"]
    assert text == "This post does not mention September 1, September 2 or September 3."
    for word in ("false", "wrong", "irrelevant", "contradict", "not related"):
        assert word not in text.lower()


# ---- API: every response type, and no decision changes --------------------------------------------------------------
CASES = {
    "verdict": "Enrollment for the second semester starts September 1.",        # RELEVANT_EVIDENCE_FOUND, a verdict
    "verdict_supported": "Enrollment for the second semester starts August 10.",
    "verdict_partial_range": "Second semester enrollment is from August 10 to 12.",
    "evidence_only_taglish": "walang pasok sa March 3",                          # withheld by the language policy
    "question": "kailan ang exam sa March 3?",                                  # INFO_SEEKING
    "low_match": "The moon festival is on December 25.",                       # NO_RELEVANT_EVIDENCE closest post
    "no_date": "kailan ang enrollment?",
}


def _run(api_main, monkeypatch=None, off=False):
    if off:
        monkeypatch.setattr(api_main, "date_notice", lambda *a, **k: None)
    with TestClient(api_main.app) as c:
        return {k: c.post("/verify/text", json={"claim": t}).json() for k, t in CASES.items()}


def _shown(body):
    docs = list(body["retrieved_evidence"]) + ([body["used_evidence"]] if body.get("used_evidence") else [])
    return [Post(document_id=d["document_id"], text=d["text"]) for d in docs]


def test_each_response_type_carries_the_notice_computed_from_its_own_posts(api_settings):
    bodies = _run(api_settings(verdict_languages="english"))
    assert bodies["verdict"]["verdict"] is not None and bodies["verdict"]["evidence_status"] == "RELEVANT_EVIDENCE_FOUND"
    assert bodies["evidence_only_taglish"]["verdict_withheld"] == "language"
    assert bodies["question"]["route"]["status"] == "INFO_SEEKING"
    assert bodies["low_match"]["evidence_status"] == "NO_RELEVANT_EVIDENCE" and bodies["low_match"]["retrieved_evidence"]
    for key, body in bodies.items():
        assert body["date_notice"] == date_notice(CASES[key], _shown(body)), key
    for key in ("verdict", "evidence_only_taglish", "question", "low_match"):
        assert bodies[key]["date_notice"] is not None, key
    assert bodies["no_date"]["date_notice"] is None
    supported = bodies["verdict_supported"]
    assert supported["used_evidence"]["document_id"] not in (supported["date_notice"] or {}).get("posts", {})
    partial = bodies["verdict_partial_range"]
    assert partial["date_notice"]["posts"][partial["used_evidence"]["document_id"]]["missing"] == ["August 11", "August 12"]


@pytest.mark.parametrize("languages", ["english", "all"])
def test_verdicts_ranking_routing_and_language_policy_are_unchanged(api_settings, monkeypatch, languages):
    api_main = api_settings(verdict_languages=languages)
    on = _run(api_main)
    off = _run(api_main, monkeypatch, off=True)
    for key in CASES:
        a, b = dict(on[key]), dict(off[key])
        a.pop("date_notice"); b.pop("date_notice")
        assert a == b, key                          # verdict, route, withheld, evidence order, used post, explanation


def test_frontend_shows_the_notice_under_the_closest_post_and_in_the_list():
    app = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
    assert "function dateNoteFor(data, doc)" in app and "data.date_notice.posts[doc.document_id]" in app
    assert app.count("dateNoteFor(data, doc)") >= 2 and "dateNoteFor(data, top)" in app   # closest (used and low match)
    assert "dateNoteFor(data || {}, doc)" in app                                         # each post in the details list
    assert app.count('"Shown above", data)') == 2 and '"Used by the classifier", data)' in app
