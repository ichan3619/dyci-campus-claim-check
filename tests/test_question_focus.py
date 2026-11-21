"""Question focus (src/normalization/question_focus.py, CAMPUS_VERIFY_QUESTION_FOCUS default on since v2.3;
results/round2/question_focus_plan.md). Synthetic cases; the plan's set Q is used once, by its run."""
import pytest
from fastapi.testclient import TestClient

from src.normalization import question_focus as qf


@pytest.mark.parametrize("text, question", [
    ("Will there be classes tomorrow? It's Friday.", "Will there be classes tomorrow?"),
    ("My friend said this. Are the midterms on October 5? I hope not!", "Are the midterms on October 5?"),
])
def test_one_question_among_several_sentences_is_picked(text, question):
    assert qf.pick(text)["question"] == question and qf.pick(text)["used"] == "question"


@pytest.mark.parametrize("text, checked, used", [
    ("was the st. paul surigao benchmarking only one day? my friend said so.",
     "was the st. paul surigao benchmarking only one day?", "question"),
    ("No classes tomorrow. Is that true?", "No classes tomorrow.", "statement"),
    ("capping and pinning is on sept 14 daw. legit?", "capping and pinning is on sept 14 daw.", "statement"),
    ("Is the fiesta mass on oct 12? my lola wants to attend.", "Is the fiesta mass on oct 12?", "question"),
])
def test_v2_abbreviations_and_tag_questions(text, checked, used):
    got = qf.pick(text)
    assert got["checked"] == checked and got["used"] == used


@pytest.mark.parametrize("text", [
    "was the st. paul surigao benchmarking only one day?",   # v2: "st." does not end a sentence
    "Did Dr. Yanga's Colleges, Inc. host the event?",
    "Will there be classes tomorrow?",                       # one sentence
    "Classes are suspended tomorrow. It is raining.",        # no question
    "Is it true? Are classes suspended?",                    # two questions
    "", "Okay.",
])
def test_anything_else_is_left_alone(text):
    assert qf.pick(text) is None


def test_the_setting_is_on_by_default_since_v2_3(monkeypatch):
    monkeypatch.delenv(qf.ENV, raising=False)
    assert qf.enabled() is True
    monkeypatch.setenv(qf.ENV, "0")
    assert qf.enabled() is False


def test_the_route_judges_the_question_and_keeps_the_message_as_typed(api_settings, monkeypatch):
    seen = {}
    monkeypatch.setenv(qf.ENV, "1")
    app = api_settings()
    with TestClient(app.app) as c:
        verifier_cls = type(app.state["manager"].current().verifier)
        original = verifier_cls.verify_claim

        def spy(self, text):
            seen.setdefault("text", text)
            return original(self, text)
        monkeypatch.setattr(verifier_cls, "verify_claim", spy)
        body = c.post("/verify/text", json={"claim": "Are the midterms on September 28? My friend said so."}).json()
    assert "friend" not in seen["text"] and "midterms" in seen["text"]
    assert body["claim"] == "Are the midterms on September 28? My friend said so."
    assert body["question_focus"]["question"] == "Are the midterms on September 28?"
