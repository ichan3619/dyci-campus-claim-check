"""Philippine holiday calendar (src/verification/holidays.py, CAMPUS_VERIFY_HOLIDAYS default off;
results/round2/holidays_plan.md). The plan's set H is used once, by its run; these are synthetic."""
import datetime as dt
import os
import re
from types import SimpleNamespace as NS

import pytest

from src.classification.result import ClassificationResult
from src.verification import holidays as h
from src.verification.verifier import VerificationResult

TODAY = dt.date(2026, 10, 27)


def _insufficient():
    post = NS(document_id="p", text="Some DYCI post.")
    return VerificationResult(claim="c", verdict="INSUFFICIENT_EVIDENCE", evidence_status="RELEVANT_EVIDENCE_FOUND",
                              retrieved_evidence=[post], used_evidence=post, classification_result=None,
                              explanation="E.")


def _classifier(label, confidence, seen=None):
    def classify(claim, evidence):
        if seen is not None:
            seen.append(evidence)
        return ClassificationResult(label=label, confidence=confidence, label_scores={label: confidence})
    return classify


def test_the_list_matches_proclamation_1006():
    regular = sorted(k for k, (_, kind) in h.HOLIDAYS.items() if kind == h.REGULAR)
    special = sorted(k for k, (_, kind) in h.HOLIDAYS.items() if kind == h.SPECIAL)
    assert regular == [(1, 1), (4, 2), (4, 3), (4, 9), (5, 1), (6, 12), (8, 31), (11, 30), (12, 25), (12, 30)]
    assert special == [(2, 17), (4, 4), (8, 21), (11, 1), (11, 2), (12, 8), (12, 24), (12, 31)]
    assert [k for k, (_, kind) in h.HOLIDAYS.items() if kind == h.WORKING] == [(2, 25)]


def test_a_regular_holiday_statement_settles_classes_a_special_day_does_not():
    assert "no classes" in h.statement(dt.date(2026, 12, 25)) and "Friday" in h.statement(dt.date(2026, 12, 25))
    assert "classes" not in h.statement(dt.date(2026, 12, 24))                 # Undas is the exception since v2.2
    assert "not a holiday" in h.statement(dt.date(2026, 2, 25))


def test_an_insufficient_answer_on_a_holiday_gets_the_verdict_with_the_warning():
    seen = []
    out, info = h.check("There are classes on December 25.", "There are classes on December 25.", _insufficient(),
                        None, TODAY, _classifier("CONTRADICTED", 0.98, seen))
    assert out.verdict == "CONTRADICTED" and out.used_evidence is None and info["applied"]
    assert info["holiday"] == "Christmas Day" and "Christmas Day" in seen[0] and h.WARNING in out.explanation


def test_low_confidence_or_a_dyci_verdict_is_left_alone():
    r = _insufficient()
    out, info = h.check("No classes on December 25.", "No classes on December 25.", r, None, TODAY,
                        _classifier("SUPPORTED", 0.6))
    assert out is r and info["applied"] is False
    decided = r.__class__(**{**r.__dict__, "verdict": "SUPPORTED"})
    assert h.check("No classes on December 25.", "No classes on December 25.", decided, None, TODAY,
                   _classifier("CONTRADICTED", 0.99)) == (decided, None)


@pytest.mark.parametrize("text", ["No classes on December 25 and December 30.", "No classes on October 28.",
                                  "No classes on December 25, 2027.", "No classes."])
def test_two_dates_a_normal_day_another_year_or_no_date_do_nothing(text):
    r = _insufficient()
    assert h.check(text, text, r, None, TODAY, _classifier("SUPPORTED", 0.99)) == (r, None)


def test_tomorrow_and_bukas_count_for_the_note():
    eve = dt.date(2026, 11, 29)
    _, info = __import__("src.verification.relative_resolve", fromlist=["resolve"]).resolve("No classes tomorrow.", eve)
    assert "Bonifacio Day" in h.note("No classes on November 30, 2026.", info, eve)["text"]
    assert "All Souls' Day" in h.note("may pasok ba bukas?", None, dt.date(2026, 11, 1))["text"]
    assert h.note("may pasok ba bukas?", None, TODAY) is None


def test_the_setting_is_on_by_default_since_v2_1(monkeypatch):
    monkeypatch.delenv(h.ENV, raising=False)
    assert h.enabled() is True
    monkeypatch.setenv(h.ENV, "0")
    assert h.enabled() is False


def test_the_page_shows_the_warning_and_never_sets_a_verdict():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = open(os.path.join(root, "frontend", "js", "app.js"), encoding="utf-8").read()
    block = app[app.index("function holidayNote(data)"):app.index("function hearsayNote(data)")]
    assert "h.warning" in block and "h.applied" in block and not re.search(r"verdict\s*=(?!=)", block)


@pytest.mark.parametrize("text, dates", [
    ("Christmas Day is a regular holiday.", ["2026-12-25"]),
    ("may pasok ba sa christmas eve?", ["2026-12-24"]),
    ("The Christmas caroling continues.", []),
    ("may pasok ba sa undas?", ["2026-11-01", "2026-11-02"]),
    ("No classes on New Year's Eve", ["2026-12-31"]),
    ("Chinese New Year is a holiday", ["2026-02-17"]),
])
def test_holiday_names_count_as_their_dates(text, dates):                     # amendment H-A1
    assert [d.isoformat() for d in h.message_dates(text, None, TODAY)] == dates


def test_a_dyci_post_about_another_time_cannot_settle_a_holiday_date():       # amendment H-A1
    post = NS(document_id="p", text="Classes resume on September 1, 2026.")
    r = VerificationResult(claim="c", verdict="CONTRADICTED", evidence_status="RELEVANT_EVIDENCE_FOUND",
                           retrieved_evidence=[post], used_evidence=post, classification_result=None, explanation="E.")
    out, info = h.far_post_guard("There are no classes on December 25.", r, None, TODAY)
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["withheld"] and info["nearest_post_date_days"] > 7
    near = r.__class__(**{**r.__dict__, "used_evidence": NS(document_id="q", text="Classes resume on December 28.")})
    assert h.far_post_guard("There are no classes on December 25.", near, None, TODAY)[0] is near
