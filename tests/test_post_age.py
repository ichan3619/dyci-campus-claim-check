"""
Post-age rule (src/verification/post_age.py), on development examples only; the frozen test claims are never used here.
"""
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from src.verification import post_age as pa

TODAY = dt.date(2026, 10, 25)
SC_VERDICTS = ("SUPPORTED", "CONTRADICTED")


@dataclass
class _Doc:
    text: str
    document_id: str = "ingested:X"


@dataclass
class _Info:
    original_date: str = ""
    provenance: str = "manual_entry"
    published_date: str = ""


@dataclass
class _Result:
    verdict: str
    explanation: str = "Checked."
    used_evidence: Optional[_Doc] = None
    retrieved_evidence: List[_Doc] = field(default_factory=list)


def _check(message, verdict="SUPPORTED", published="2026-08-20", text="Library hours change for the semester."):
    sources = {"ingested:X": _Info(published_date=published)}
    return pa.check(message, _Result(verdict, used_evidence=_Doc(text)), sources, on=TODAY)


def test_an_old_post_cannot_confirm_a_message_about_now():
    result, info = _check("The library is closed on weekends.")
    assert result.verdict == "INSUFFICIENT_EVIDENCE" and info["post_age_days"] == 66
    assert "from 20 August 2026 (66 days ago)" in result.explanation and "can't confirm what is true now" in result.explanation


@pytest.mark.parametrize("published, withheld", [("2026-09-25", False), ("2026-09-24", True), ("2026-10-20", False)])
def test_thirty_days_is_still_fresh_thirty_one_is_not(published, withheld):
    result, _ = _check("The library is closed on weekends.", published=published)
    assert (result.verdict == "INSUFFICIENT_EVIDENCE") is withheld


@pytest.mark.parametrize("message", [
    "The library closed on weekends last semester.",          # past tense: no marker
    "The library is closed on September 30, 2026.",           # explicit date
    "The library is closed tomorrow.",                        # relative day word (rule B's job)
])
def test_past_tense_dated_and_relative_messages_are_not_touched(message):
    result, info = _check(message)
    assert result.verdict == "SUPPORTED" and info is None


def test_it_never_creates_a_verdict_and_needs_a_post():
    for verdict in ("INSUFFICIENT_EVIDENCE", None):
        result, info = _check("The library is closed.", verdict=verdict)
        assert result.verdict == verdict and info is None
    no_post = _Result("SUPPORTED")
    assert pa.check("The library is closed.", no_post, {}, on=TODAY) == (no_post, None)


def test_contradictions_are_withheld_too_and_unknown_dates_count_as_old():
    result, _ = _check("The library is open on Sundays.", verdict="CONTRADICTED")
    assert result.verdict == "INSUFFICIENT_EVIDENCE"
    undated, info = pa.check("The library is open.", _Result("SUPPORTED", used_evidence=_Doc("No date here.")),
                             {"ingested:X": _Info()}, on=TODAY)
    assert undated.verdict == "INSUFFICIENT_EVIDENCE" and info["post_age_days"] is None and "undated" in undated.explanation


def test_day_and_month_dates_count_from_their_last_occurrence():
    assert pa.age_in_days(("monthday", (10, 20)), TODAY) == 5
    assert pa.age_in_days(("monthday", (12, 1)), TODAY) == 328          # last December
    assert pa.age_in_days(("full", dt.date(2026, 9, 25)), TODAY) == 30
    assert pa.age_in_days(None, TODAY) is None


def test_markers_and_default_setting(monkeypatch):
    assert pa.speaks_about_now("Classes aren’t held now.") and not pa.speaks_about_now("Classes were held.")
    monkeypatch.delenv(pa.ENV, raising=False)
    assert pa.enabled() is True                                         # on by default since v1.4
    monkeypatch.setenv(pa.ENV, "0")
    assert pa.enabled() is False


def test_the_app_warns_and_keeps_the_verdict():
    for verdict in SC_VERDICTS:
        result = _Result(verdict, used_evidence=_Doc("Library hours change for the semester."))
        n = pa.notice("The library is closed on weekends.", result, {"ingested:X": _Info(published_date="2026-08-20")}, on=TODAY)
        assert result.verdict == verdict                                # never changed
        assert n["post_age_days"] == 66 and n["document_id"] == "ingested:X" and n["max_days"] == 30
        assert n["text"] == ("Old post: the post this is based on is from 20 August 2026 (66 days ago). It may be outdated "
                             "or about a different event, so it may not reflect what is true now.")


def test_the_warning_appears_exactly_where_the_withholding_rule_would_act():
    def both(message, verdict="SUPPORTED", published="2026-08-20", text="No date here."):
        result = _Result(verdict, used_evidence=_Doc(text))
        sources = {"ingested:X": _Info(published_date=published)}
        return pa.notice(message, result, sources, on=TODAY), pa.check(message, result, sources, on=TODAY)[1]
    cases = [("The library is closed.",), ("The library is closed.", "SUPPORTED", "2026-10-20"),
             ("The library closed last semester.",), ("The library is closed tomorrow.",),
             ("The library is closed.", "INSUFFICIENT_EVIDENCE"), ("The library is closed.", "CONTRADICTED", "")]
    for case in cases:
        n, withheld = both(*case)
        assert (n is None) == (withheld is None), case
    undated, _ = both("The library is closed.", "SUPPORTED", "")
    assert undated["post_age_days"] is None and "has no date" in undated["text"]


# ---- v2 (results/round2/post_age_v2_plan.md; development examples only, never the frozen test claims) ----------------
def _v2(message, published="2026-08-20", verdict="SUPPORTED", v2=True):
    result = _Result(verdict, used_evidence=_Doc("No date here."))
    return pa.notice(message, result, {"ingested:X": _Info(published_date=published)}, on=TODAY, v2=v2)


def test_v2_warns_on_messages_without_a_now_word_with_the_softer_sentence():
    for message in ("library closed", "The library closed last semester."):
        n = _v2(message)
        assert n["text"] == ("Old post: the post this is based on is from 20 August 2026 (66 days ago). If your message is "
                             "about something recent, this post may be outdated or about a different event.")
        assert _v2(message, v2=False) is None                        # v1.5 behaviour without v2


def test_v2_keeps_the_v15_sentence_for_messages_with_a_now_word():
    assert _v2("The library is closed.")["text"] == _v2("The library is closed.", v2=False)["text"]
    assert "so it may not reflect what is true now" in _v2("The library is closed.")["text"]


def test_v2_never_warns_on_dated_or_recent_or_unverdicted_messages():
    assert _v2("library closed on September 30, 2026") is None      # explicit date
    assert _v2("library closed tomorrow") is None                   # relative day word (rule B)
    assert _v2("library closed", published="2026-10-20") is None    # recent post
    assert _v2("library closed", verdict="INSUFFICIENT_EVIDENCE") is None
    assert _v2("library closed", published="")["post_age_days"] is None   # undated counts as old


def test_v2_setting_is_on_by_default_since_v1_6(monkeypatch):
    monkeypatch.delenv(pa.V2_ENV, raising=False)
    assert pa.v2_enabled() is True
    monkeypatch.setenv(pa.V2_ENV, "0")
    assert pa.v2_enabled() is False

