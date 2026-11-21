"""Relative dates as real dates (src/verification/relative_resolve.py, CAMPUS_VERIFY_RESOLVE_RELATIVE default off;
results/round2/relative_resolve_plan.md). The plan's fresh test set is used once, by its run; these are synthetic."""
import datetime as dt
import os
import re

import pytest
from fastapi.testclient import TestClient

from src.verification import relative_resolve as rr

DAY = dt.date(2026, 10, 27)                                         # a Tuesday


@pytest.mark.parametrize("text, out", [
    ("semi-finals exam tomorrow", "semi-finals exam on October 28, 2026"),
    ("Tomorrow, classes are suspended.", "On October 28, 2026, classes are suspended."),
    ("Are classes suspended until tomorrow?", "Are classes suspended until October 28, 2026?"),
    ("No classes today and tomorrow.", "No classes on October 27, 2026 and on October 28, 2026."),
    ("The fiesta is the day after tomorrow.", "The fiesta is on October 29, 2026."),
    ("DYCI held a seminar yesterday.", "DYCI held a seminar on October 26, 2026."),
    ("Exam tmrw at 8 AM", "Exam on October 28, 2026 at 8 AM"),
    ("Classes end early tonight.", "Classes end early on October 27, 2026."),
])
def test_relative_words_become_calendar_dates(text, out):
    resolved, info = rr.resolve(text, DAY)
    assert resolved == out and info["checked_as"] == out and info["today"] == "2026-10-27"


def test_the_spoken_form_names_the_weekday():
    assert rr.resolve("exam tomorrow", DAY)[1]["read_as"] == ["Wednesday, 28 October 2026"]


@pytest.mark.parametrize("text", [
    "Tomorrow's exam is cancelled.",                                   # possessive: left to the v1.3 rule
    "No classes on October 28, even though they said tomorrow.",       # an explicit date wins
    "Walang pasok bukas",                                              # not an English relative word
    "No classes on Friday.", "", "Classes resume next week.",
])
def test_anything_else_is_left_unchanged(text):
    assert rr.resolve(text, DAY) == (text, None)


def test_today_is_the_philippine_date_whatever_the_server_zone(monkeypatch):
    class Fixed(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(2026, 10, 27, 17, 30, tzinfo=dt.timezone.utc).astimezone(tz)   # 01:30 on the 28th in Manila
    monkeypatch.setattr(rr._dt, "datetime", Fixed)
    assert rr.today() == dt.date(2026, 10, 28)


def test_the_setting_is_on_by_default_since_v2_0(monkeypatch):
    monkeypatch.delenv(rr.ENV, raising=False)
    assert rr.enabled() is True
    monkeypatch.setenv(rr.ENV, "0")
    assert rr.enabled() is False


def test_the_route_checks_the_resolved_text_and_keeps_the_claim_as_typed(api_settings, monkeypatch):
    monkeypatch.setattr(rr, "today", lambda: DAY)
    seen, bodies, original = {}, {}, {}
    for on in ("0", "1"):
        monkeypatch.setenv(rr.ENV, on)
        app = api_settings()
        with TestClient(app.app) as c:
            verifier_cls = type(app.state["manager"].current().verifier)
            original.setdefault("f", verifier_cls.verify_claim)          # the real one, captured once

            def spy(self, text, _on=on):
                seen.setdefault(_on, text)                              # the first text this pass verified
                return original["f"](self, text)
            monkeypatch.setattr(verifier_cls, "verify_claim", spy)
            bodies[on] = c.post("/verify/text", json={"claim": "Classes are suspended tomorrow."}).json()
    assert "tomorrow" in seen["0"] and seen["1"] == "Classes are suspended on October 28, 2026."
    assert "relative_resolution" not in bodies["0"]
    assert bodies["1"]["claim"] == "Classes are suspended tomorrow."
    assert bodies["1"]["relative_resolution"]["read_as"] == ["Wednesday, 28 October 2026"]
    assert not bodies["1"].get("relative_date_check")                 # the v1.3 rule no longer sees "tomorrow"


def test_the_page_shows_how_the_word_was_read_and_never_sets_a_verdict():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = open(os.path.join(root, "frontend", "js", "app.js"), encoding="utf-8").read()
    block = app[app.index("function resolutionNote(data)"):app.index("// Display-only relative-date notice")]
    assert "Philippine time" in block and "read_as" in block and not re.search(r"verdict\s*=(?!=)", block)


# ---- v2: the nearby-date guard (results/round2/relative_resolve_v2_plan.md) ------------------------------------------
from types import SimpleNamespace as NS                                                    # noqa: E402

from src.verification.verifier import VerificationResult                                  # noqa: E402

AUG_HOLIDAY = "DYCI ADVISORY 20 August 2026. No classes at all levels tomorrow, August 21, 2026."
OCT_EXAMS = "Semi-Final Examinations are scheduled on October 28-30, 2026."


def _contradicted(post_text):
    post = NS(document_id="p", text=post_text)
    return VerificationResult(claim="c", verdict="CONTRADICTED", evidence_status="RELEVANT_EVIDENCE_FOUND",
                              retrieved_evidence=[post], used_evidence=post, classification_result=None, explanation="E.")


def test_a_post_about_a_far_off_day_cannot_contradict_a_resolved_message():
    _, info = rr.resolve("No classes tomorrow.", DAY)                       # 28 October
    out, near = rr.guard(_contradicted(AUG_HOLIDAY), info)
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and near["withheld"] and near["nearest_post_date_days"] == 68
    assert "Date guard" in out.explanation


@pytest.mark.parametrize("asked, post, gap", [
    (dt.date(2026, 8, 21), AUG_HOLIDAY, 1),                                # tomorrow = 22 Aug; the post names 21 Aug
    (dt.date(2026, 10, 30), OCT_EXAMS, 1),                                 # tomorrow = 31 Oct; the post ends 30 Oct
])
def test_a_post_naming_a_nearby_day_still_contradicts(asked, post, gap):
    _, info = rr.resolve("There are no classes tomorrow.", asked)
    out, near = rr.guard(_contradicted(post), info)
    assert out.verdict == "CONTRADICTED" and near["withheld"] is False and near["nearest_post_date_days"] == gap


def test_a_post_with_no_date_cannot_contradict_and_nothing_else_is_touched():
    _, info = rr.resolve("No classes tomorrow.", DAY)
    assert rr.guard(_contradicted("Classes will push through as scheduled."), info)[0].verdict == "INSUFFICIENT_EVIDENCE"
    supported = _contradicted(AUG_HOLIDAY)
    supported = supported.__class__(**{**supported.__dict__, "verdict": "SUPPORTED"})
    assert rr.guard(supported, info) == (supported, None)
    assert rr.guard(_contradicted(AUG_HOLIDAY), None)[1] is None              # not a resolved message


def test_the_year_boundary_is_handled():
    assert rr.nearest_days("Classes resume on January 2.", dt.date(2026, 12, 31)) == 2
