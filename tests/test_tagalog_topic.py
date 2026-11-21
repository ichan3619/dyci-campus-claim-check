"""Topic guard for Tagalog/Taglish verdicts (src/verification/tagalog_topic.py, CAMPUS_VERIFY_TAGALOG_TOPIC_GUARD default
off; results/round2/tagalog_verdicts_plan.md). The plan's test sets are used once, by its run; these are synthetic."""
from types import SimpleNamespace as NS

import pytest

from src.verification import tagalog_topic as tt
from src.verification.verifier import VerificationResult

N20 = "DYCI ADVISORY 20 August 2026. There will be NO CLASSES AT ALL LEVELS tomorrow, August 21, 2026."
N09 = "DYCI held its Capping, Pinning, and Candle Lighting Ceremony for the Nursing students on September 4, 2026."


def _result(verdict, post_text):
    post = NS(document_id="p", text=post_text)
    return VerificationResult(claim="c", verdict=verdict, evidence_status="RELEVANT_EVIDENCE_FOUND",
                              retrieved_evidence=[post], used_evidence=post, classification_result=None, explanation="E.")


@pytest.mark.parametrize("verdict", ["SUPPORTED", "CONTRADICTED"])
def test_a_post_on_another_topic_cannot_settle_a_tagalog_message(verdict):
    out, info = tt.check("may mga pasalubong ba?", _result(verdict, N20))
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["withheld"] and info["original_verdict"] == verdict
    assert "Topic check" in out.explanation


def test_a_shared_topic_word_keeps_the_verdict():
    r = _result("SUPPORTED", N09)
    out, info = tt.check("may pinning ba ang nursing?", r)
    assert out is r and info["shared_topic_words"] == ["nursing", "pinning"] and not info["withheld"]


def test_tagalog_and_campus_words_do_not_count_as_shared():
    assert tt.message_topic_words("walang pasok bukas sa DYCI") == set()


def test_insufficient_or_no_post_is_left_alone():
    r = _result("INSUFFICIENT_EVIDENCE", N20)
    assert tt.check("may mga pasalubong ba?", r) == (r, None)


def test_the_setting_is_off_by_default(monkeypatch):
    monkeypatch.delenv(tt.ENV, raising=False)
    assert tt.enabled() is False
    monkeypatch.setenv(tt.ENV, "1")
    assert tt.enabled() is True
