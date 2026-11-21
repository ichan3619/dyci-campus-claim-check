"""Relative dates (src/verification/relative_date.py): the display-only notice (A) and CAMPUS_VERIFY_RELATIVE_DATE_CHECK
(B, default off), which withholds SUPPORTED/CONTRADICTED on 'tomorrow'/'bukas' messages unless the post is from today or
yesterday."""
import datetime as dt
import os
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from src.verification import relative_date as rd
from src.verification.verifier import VerificationResult

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = dt.date(2026, 10, 12)
AUG20 = NS(document_id="aug20", text="DYCI ADVISORY 20 August 2026 To all DYCIans, there will be NO CLASSES AT ALL "
                                    "LEVELS tomorrow, August 21, 2026.")
UNDATED = NS(document_id="undated", text="Classes are suspended tomorrow due to the typhoon.")


# ---- words and post dates ---------------------------------------------------------------------------------------------
def test_relative_words_in_order_without_repeats():
    assert rd.relative_words("No classes tomorrow? walang pasok bukas, tomorrow talaga") == ["tomorrow", "bukas"]
    assert rd.relative_words("No classes on September 27.") == []
    assert rd.relative_words("Tom said classes resume on Monday") == []          # names and weekdays are not included


def test_post_date_priority():
    assert rd.post_date(AUG20.text, NS(original_date="2026-08-19", provenance="test_page_mirror")) == ("full", dt.date(2026, 8, 19))
    assert rd.post_date("x", NS(original_date="", published_date="2026-08-20", provenance="manual_entry")) == ("full", dt.date(2026, 8, 20))
    # A test-Page copy's re-post date is not DYCI's date: fall back to the header date.
    assert rd.post_date(AUG20.text, NS(original_date="", published_date="2026-10-04T10:00:00+0000",
                                       provenance="test_page_mirror")) == ("monthday", (8, 20))
    assert rd.post_date(UNDATED.text, NS(original_date="", published_date="2026-10-04", provenance="test_page_mirror")) is None
    assert rd.post_date(UNDATED.text) is None


@pytest.mark.parametrize("pd, recent", [(("full", TODAY), True), (("full", dt.date(2026, 10, 11)), True),
                                        (("full", dt.date(2026, 10, 10)), False), (("monthday", (10, 11)), True),
                                        (("monthday", (8, 20)), False), (None, False)])
def test_recent_means_today_or_yesterday(pd, recent):
    assert rd.is_recent(pd, TODAY) is recent


# ---- A: display notice ------------------------------------------------------------------------------------------------
def test_notice_names_the_word_and_the_post_date():
    n = rd.notice("is there no classes tomorrow", [AUG20, UNDATED, None, AUG20])
    assert n["words"] == ["tomorrow"]
    assert n["posts"]["aug20"] == {"post_date": "20 August", "text": "Your message says “tomorrow”. This post is "
                                   "from 20 August, so check that it is about the day you mean."}
    assert n["posts"]["undated"]["text"].startswith("Your message says “tomorrow”. This post’s date is not recorded")
    sources = {"aug20": NS(original_date="", published_date="2026-08-20", provenance="manual_entry")}
    assert rd.notice("walang pasok bukas?", [AUG20], sources)["posts"]["aug20"]["post_date"] == "20 August 2026"


def test_no_relative_word_no_notice():
    assert rd.notice("No classes on August 21.", [AUG20]) is None
    assert rd.notice("tomorrow", []) is None


# ---- B: the check -----------------------------------------------------------------------------------------------------
def _result(verdict, doc=AUG20):
    return VerificationResult(claim="c", verdict=verdict, evidence_status="RELEVANT_EVIDENCE_FOUND", retrieved_evidence=[doc],
                              used_evidence=doc, classification_result=None, explanation="E.")


@pytest.mark.parametrize("verdict", ["SUPPORTED", "CONTRADICTED"])
def test_old_or_undated_post_withholds_to_insufficient(verdict):
    for doc in (AUG20, UNDATED):
        out, info = rd.check("is there no classes tomorrow", _result(verdict, doc), today=TODAY)
        assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["withheld"] and info["original_verdict"] == verdict
        assert out.used_evidence is doc and "Relative date check" in out.explanation


@pytest.mark.parametrize("verdict", ["SUPPORTED", "CONTRADICTED"])
def test_post_from_today_or_yesterday_keeps_the_verdict(verdict):
    original = _result(verdict)
    out, info = rd.check("no classes tomorrow", original, today=dt.date(2026, 8, 21))     # the post is from yesterday
    assert out is original and info["withheld"] is False


def test_never_creates_or_flips_and_ignores_messages_without_relative_words():
    for v in ("INSUFFICIENT_EVIDENCE",):
        r = _result(v)
        assert rd.check("no classes tomorrow", r, today=TODAY) == (r, None)
    r = _result("SUPPORTED")
    assert rd.check("No classes on August 21.", r, today=TODAY) == (r, None)
    no_used = VerificationResult(claim="c", verdict="SUPPORTED", evidence_status="RELEVANT_EVIDENCE_FOUND",
                                 retrieved_evidence=[], used_evidence=None, classification_result=None, explanation="E.")
    assert rd.check("tomorrow", no_used, today=TODAY) == (no_used, None)
    for v in ("SUPPORTED", "CONTRADICTED"):
        assert rd.check("tomorrow", _result(v), today=TODAY)[0].verdict in (v, "INSUFFICIENT_EVIDENCE")


# ---- API ----------------------------------------------------------------------------------------------------------------
TEXTS = ["Classes are suspended tomorrow due to a typhoon.", "Classes will continue as normal tomorrow.",
         "Enrollment for the second semester starts August 10.", "walang pasok bukas", "kailan ang enrollment bukas?",
         "The moon festival is tomorrow."]


def _bodies(api_main):
    with TestClient(api_main.app) as c:
        return c.get("/health").json(), {t: c.post("/verify/text", json={"claim": t}).json() for t in TEXTS}


def test_check_defaults_on_since_v1_3_and_the_notice_changes_no_decision(api_settings, monkeypatch):
    monkeypatch.setenv("CAMPUS_VERIFY_RESOLVE_RELATIVE", "0")   # the v1.3 rule on its own (v2.0 resolves "tomorrow" first)
    from src.api.pipeline import Settings
    assert Settings().relative_date_check is True            # adopted 2026-10-13 (v1.3)
    api_main = api_settings(verdict_languages="english", relative_date_check=False)   # the notice alone, B off
    health, on = _bodies(api_main)
    assert health["relative_date_check"] is False
    assert on["Classes are suspended tomorrow due to a typhoon."]["relative_date_notice"]["words"] == ["tomorrow"]
    assert on["walang pasok bukas"]["relative_date_notice"]["words"] == ["bukas"]                  # evidence only
    assert on["kailan ang enrollment bukas?"]["relative_date_notice"] is not None                  # question
    assert on["Enrollment for the second semester starts August 10."]["relative_date_notice"] is None
    assert all(b["relative_date_check"] is None for b in on.values())
    monkeypatch.setattr(rd, "notice", lambda *a, **k: None)
    _, off = _bodies(api_main)
    for t in TEXTS:
        a, b = dict(on[t]), dict(off[t])
        a.pop("relative_date_notice"); b.pop("relative_date_notice")
        assert a == b, t


def test_check_on_withholds_only_relative_verdicts(api_settings, monkeypatch):
    monkeypatch.setenv("CAMPUS_VERIFY_RESOLVE_RELATIVE", "0")   # the v1.3 rule on its own (v2.0 resolves "tomorrow" first)
    _, off = _bodies(api_settings(relative_date_check=False))
    health, on = _bodies(api_settings(relative_date_check=True))
    assert health["relative_date_check"] is True
    changed = 0
    for t in TEXTS:
        a, b = off[t], on[t]
        if a["verdict"] in rd.SC and rd.relative_words(t):
            changed += 1                                     # the development notices are undated: never recent
            assert b["verdict"] == "INSUFFICIENT_EVIDENCE" and b["relative_date_check"]["withheld"] is True
            assert {k: v for k, v in b.items() if k not in ("verdict", "explanation", "relative_date_check")} == \
                   {k: v for k, v in a.items() if k not in ("verdict", "explanation", "relative_date_check")}
        else:
            assert b == a, t
    assert changed > 0


def test_frontend_shows_the_relative_notice_the_withheld_reason_and_the_low_match_message():
    app = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
    assert "function relativeNoteFor(data, doc)" in app and "relativeNoteFor(data, top)" in app
    assert "relativeNoteFor(data || {}, doc)" in app
    assert "data.relative_date_check && data.relative_date_check.withheld" in app
    assert "No campus announcement matches this. This checker answers claims and questions about DYCI announcements" in app
