"""English answer fixes (results/round2/english_fixes_plan.md): R1 "been" questions (src/normalization/english_been.py,
CAMPUS_VERIFY_EN_BEEN_RULE) and R2 the tense guard (src/verification/tense_guard.py, CAMPUS_VERIFY_TENSE_GUARD). Both
default off. Synthetic wording only; the plan's fresh test set is used once, by its run."""
import os
import re
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from src.normalization import english_been as eb
from src.normalization.english_question import rewrite_english_question
from src.verification import tense_guard as tg
from src.verification.verifier import VerificationResult


@pytest.mark.parametrize("question, statement", [
    ("Has the agreement between the two partner schools of the region been renewed?",
     "The agreement between the two partner schools of the region has been renewed."),
    ("Have the uniforms of the varsity players of the college been delivered?",
     "The uniforms of the varsity players of the college have been delivered."),
])
def test_long_subject_been_questions_become_statements(question, statement):
    assert rewrite_english_question(question)[1] == "no_subject_boundary"       # the case R1 exists for
    assert eb.rewrite(question) == statement


@pytest.mark.parametrize("question", [
    "Has the library been closed?",                          # (en-question-v1 handles short subjects itself)
    "Why has the agreement of the partner schools been renewed?",
    "Has the agreement of the partner schools not been renewed?",
    "Has the agreement of the partner schools of the region of the north of the country been renewed?",  # > 12 words
    "The agreement has been renewed.", "Is the library of the main campus open?",
])
def test_anything_else_is_left_alone_or_handled_elsewhere(question):
    out = eb.rewrite(question)
    assert out is None or rewrite_english_question(question)[1] is None


def _result(verdict, passage):
    post = NS(document_id="p", text=passage, score=0.9, source_type="facebook_post", source_url="https://www.facebook.com/x")
    return VerificationResult(claim="c", verdict=verdict, evidence_status="RELEVANT_EVIDENCE_FOUND", retrieved_evidence=[post],
                              used_evidence=post, classification_result=None, explanation="E.",
                              selected_passage=NS(document_id="p", text=passage, context_before_text="", score=0.9,
                                                  passage_index=0))


@pytest.mark.parametrize("statement, passage", [
    ("The new name has been announced.", "The new name will be announced tomorrow during the assembly."),
    ("The examinations were held on May 5.", "The examinations are scheduled on May 5."),
    ("The library already reopened.", "The library is to be held open during the upcoming week."),
])
def test_a_past_claim_on_a_future_only_post_is_withheld(statement, passage):
    out, info = tg.check(statement, _result("SUPPORTED", passage))
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["withheld"] and "Tense guard" in out.explanation


@pytest.mark.parametrize("statement, passage, verdict", [
    ("The seminar was held on May 5.", "The seminar was held on May 5 at the gym.", "SUPPORTED"),        # past post
    ("The fiesta mass will be held on May 5.", "The fiesta mass will be held on May 5.", "SUPPORTED"),  # future claim
    ("The new name has been announced.", "The new name will be announced tomorrow.", "CONTRADICTED"),   # not SUPPORTED
])
def test_everything_else_is_untouched(statement, passage, verdict):
    out, info = tg.check(statement, _result(verdict, passage))
    assert out.verdict == verdict and info is None


def test_r1_is_on_by_default_since_v1_8_and_r2_stays_off(monkeypatch):
    monkeypatch.delenv(eb.ENV, raising=False)
    monkeypatch.delenv(tg.ENV, raising=False)
    assert eb.enabled() is True and tg.enabled() is False
    monkeypatch.setenv(eb.ENV, "0")
    assert eb.enabled() is False


def test_the_route_uses_both_only_when_on_and_adds_no_field_when_off(api_settings, monkeypatch):
    seen = {}
    r = _result("SUPPORTED", "The new name will be announced tomorrow during the assembly.")
    bodies = {}
    for on in ("0", "1"):
        monkeypatch.setenv(eb.ENV, on)
        monkeypatch.setenv(tg.ENV, on)
        app = api_settings()
        with TestClient(app.app) as c:
            def fake(self, text, _on=on):
                seen[_on] = text
                return r
            monkeypatch.setattr(type(app.state["manager"].current().verifier), "verify_claim", fake)
            bodies[on] = c.post("/verify/text", json={
                "claim": "Has the new name of the merged engineering department been announced?"}).json()
    assert seen["1"] == "The new name of the merged engineering department has been announced."
    assert seen["0"] != seen["1"]
    assert "tense_guard_check" not in bodies["0"] and bodies["0"]["verdict"] == "SUPPORTED"
    assert bodies["1"]["verdict"] == "INSUFFICIENT_EVIDENCE" and bodies["1"]["tense_guard_check"]["withheld"] is True


def test_the_page_explains_the_withheld_verdict_and_never_sets_one():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = open(os.path.join(root, "frontend", "js", "app.js"), encoding="utf-8").read()
    assert app.count("data.tense_guard_check && data.tense_guard_check.withheld") == 2
    assert not re.search(r"tense_guard_check[^\n]*verdict\s*=(?!=)", app)
