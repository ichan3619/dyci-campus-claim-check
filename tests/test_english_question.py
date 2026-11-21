"""src/normalization/english_question.py and CAMPUS_VERIFY_EN_QUESTION_REWRITE. Invented examples only: the evaluation
questions (data/acceptance/english_questions_v1.json) are never used to build or test the rule."""
import pytest
from fastapi.testclient import TestClient

from src.normalization.english_question import rewrite_english_question


@pytest.mark.parametrize("question, statement", [
    ("Are the new ID cards available at the Registrar?", "The new ID cards are available at the Registrar."),
    ("Will the enrollment period end on Friday?", "The enrollment period will end on Friday."),
    ("Have the final grades been released?", "The final grades have been released."),
    ("Was the orientation held on June 3?", "The orientation was held on June 3."),
])
def test_rewrites_by_reordering_only(question, statement):
    out, refused = rewrite_english_question(question)
    assert (out, refused) == (statement, None)
    assert sorted(w.lower().strip("?.") for w in out.split()) == sorted(w.lower().strip("?.") for w in question.split())


@pytest.mark.parametrize("question, reason", [
    ("Does the library open at 7?", "auxiliary_not_handled"),          # do-support: would need verb inflection
    ("Should students bring their IDs?", "auxiliary_not_handled"),
    ("When is the enrollment?", "auxiliary_not_handled"),
    ("Is the canteen not open today?", "wh_or_negative_question"),
    ("Is the parking lot behind the gym closed tomorrow?", "no_subject_boundary"),     # 6-word subject: over the limit
    ("Will the students of the college department and the faculty attend?", "no_subject_boundary"),
    ("The exam is on Monday.", "not_a_single_question"),
    ("Is it?", "too_short"),
])
def test_refuses_what_it_cannot_reorder_safely(question, reason):
    assert rewrite_english_question(question) == (question, reason)


def test_known_limitation_a_preposition_inside_the_subject_is_taken_as_the_boundary():
    # Documented in the plan: the boundary is the FIRST predicate-start word, so a prepositional phrase inside the
    # subject splits it. Reordering only, so no word changes, but the statement can read differently; the evaluation's
    # zero-new-harmful criterion is the safeguard.
    out, refused = rewrite_english_question("Is the enrollment for transferees extended?")
    assert (out, refused) == ("The enrollment is for transferees extended.", None)


def _post(api_settings, text, **kw):
    with TestClient(api_settings(**kw).app) as c:
        return c.post("/verify/text", json={"claim": text}).json()


def test_setting_defaults_on_since_v1_0_and_is_reported(api_settings):
    from src.api.pipeline import Settings
    assert Settings().en_question_rewrite is True          # adopted 2026-10-10 (pre-registered criteria met)
    with TestClient(api_settings(en_question_rewrite=True).app) as c:
        assert c.get("/health").json()["en_question_rewrite"] is True


def test_api_rewrites_an_english_question_option_b_left_unchanged(api_settings):
    q = "Are the new ID cards available at the Registrar?"
    off = _post(api_settings, q, en_question_rewrite=False)
    assert not (off.get("claim_rewrite") or {}).get("changed")
    on = _post(api_settings, q, en_question_rewrite=True)
    assert on["claim_rewrite"]["rewritten"] == "The new ID cards are available at the Registrar."
    assert on["claim_rewrite"]["rules_applied"] == ["EN_Q_aux_inversion"] and on["claim"] == q


def test_api_never_touches_statements_taglish_or_option_b_rewrites(api_settings):
    statement = _post(api_settings, "The new ID cards are available at the Registrar.", en_question_rewrite=True)
    assert not (statement.get("claim_rewrite") or {}).get("rules_applied") or \
        "EN_Q_aux_inversion" not in statement["claim_rewrite"]["rules_applied"]
    taglish = _post(api_settings, "may pasok ba bukas sa college?", en_question_rewrite=True, verdict_languages="english")
    assert taglish["verdict_withheld"] == "language"
    there = _post(api_settings, "Is there a class on Friday?", en_question_rewrite=True)
    assert "EN_Q_aux_inversion" not in there["claim_rewrite"]["rules_applied"]       # Option B's R4 already handled it
