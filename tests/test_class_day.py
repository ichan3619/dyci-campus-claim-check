"""Class-day date guard (src/verification/class_day.py, CAMPUS_VERIFY_CLASS_DAY_GUARD default off) and amendment H-A2
(Undas, CAMPUS_VERIFY_UNDAS_NO_CLASSES default off); results/round2/class_day_guard_plan.md. Synthetic cases only."""
import datetime as dt
from types import SimpleNamespace as NS

import pytest

from src.verification import class_day as cd
from src.verification import holidays as h
from src.verification.verifier import VerificationResult

TODAY = dt.date(2026, 10, 27)
AUG = "DYCI ADVISORY 20 August 2026. There will be NO CLASSES AT ALL LEVELS tomorrow, August 21, 2026."
PINNING = "DYCI held its Capping and Pinning Ceremony on September 4, 2026, at the Elida Convention Center."


def _result(verdict, post_text):
    post = NS(document_id="p", text=post_text)
    return VerificationResult(claim="c", verdict=verdict, evidence_status="RELEVANT_EVIDENCE_FOUND",
                              retrieved_evidence=[post], used_evidence=post, classification_result=None, explanation="E.")


@pytest.mark.parametrize("verdict", ["SUPPORTED", "CONTRADICTED"])
def test_a_post_about_another_day_cannot_settle_classes_on_this_day(verdict):
    out, info = cd.check("There are no classes on March 16.", _result(verdict, AUG), TODAY)
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["withheld"] and info["original_verdict"] == verdict
    assert "Class-day guard" in out.explanation and "March 16" in out.explanation


def test_a_post_about_that_day_still_decides():
    r = _result("SUPPORTED", AUG)
    out, info = cd.check("There are no classes on August 21.", r, TODAY)
    assert out is r and info["nearest_post_date_days"] == 0 and not info["withheld"]


def test_events_and_messages_without_a_date_are_untouched():
    r = _result("CONTRADICTED", PINNING)
    assert cd.check("The capping and pinning ceremony was held on September 14.", r, TODAY) == (r, None)
    r2 = _result("CONTRADICTED", AUG)
    assert cd.check("There are no classes.", r2, TODAY) == (r2, None)


def test_insufficient_or_a_holiday_verdict_is_left_alone():
    r = _result("INSUFFICIENT_EVIDENCE", AUG)
    assert cd.check("No classes on March 16.", r, TODAY) == (r, None)
    holiday = r.__class__(**{**r.__dict__, "verdict": "SUPPORTED", "used_evidence": None})
    assert cd.check("No classes on December 25.", holiday, TODAY) == (holiday, None)


def test_the_settings_are_on_by_default_since_v2_2(monkeypatch):
    monkeypatch.delenv(cd.ENV, raising=False)
    monkeypatch.delenv(h.UNDAS_ENV, raising=False)
    assert cd.enabled() is True and h.undas_no_classes() is True
    monkeypatch.setenv(cd.ENV, "0")
    monkeypatch.setenv(h.UNDAS_ENV, "0")
    assert cd.enabled() is False and h.undas_no_classes() is False


def test_undas_settles_classes_only_with_amendment_h_a2(monkeypatch):
    monkeypatch.setenv(h.UNDAS_ENV, "0")
    assert "no classes" not in h.statement(dt.date(2026, 11, 1))
    monkeypatch.setenv(h.UNDAS_ENV, "1")
    assert "no classes" in h.statement(dt.date(2026, 11, 1)) and "no classes" in h.statement(dt.date(2026, 11, 2))
    assert "no classes" not in h.statement(dt.date(2026, 12, 24))                 # other special days unchanged


def test_v2_a_post_six_days_away_cannot_settle_a_class_day():                 # v1's D09
    suspended = "DYCI ADVISORY 19 August 2026. IN-PERSON classes are SUSPENDED tomorrow, THURSDAY, August 20, 2026."
    out, info = cd.check("Classes are suspended on August 26 because of rain.", _result("CONTRADICTED", suspended), TODAY)
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["nearest_post_date_days"] == 6 and info["window_days"] == 1


def test_v2_the_next_or_previous_day_still_counts():
    r = _result("SUPPORTED", AUG)                                              # the post names 20 and 21 August
    assert cd.check("There are no classes on August 22.", r, TODAY)[0] is r
