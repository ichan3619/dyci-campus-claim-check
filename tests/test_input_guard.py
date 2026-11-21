"""Input guard (2026-10-13): a message with fewer than 3 letters ("a", emoji only, digits) is refused with a friendly
message before any retrieval, on /verify/text and /verify/url and in the page. Real messages are never affected (the
shortest development message has 10 letters)."""
import csv
import json
import os

import pytest
from fastapi.testclient import TestClient

from src.api.schemas import MIN_CLAIM_LETTERS, TOO_SHORT_MESSAGE, has_enough_letters

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("text", ["a", "ok", "po","\U0001F64F\U0001F64F\U0001F64F", "12", "??", "!!! 28", "a 1 b"])
def test_too_few_letters(text):
    assert not has_enough_letters(text)


@pytest.mark.parametrize("text", ["san ginanap?", "UST?", "may pasok?", "No classes tomorrow."])
def test_enough_letters(text):
    assert has_enough_letters(text)


def test_every_development_message_passes_the_guard():
    msgs = [r["wording_exactly_as_typed"] for r in csv.DictReader(open(os.path.join(ROOT, "data", "dev2", "dev2_labels.csv"), encoding="utf-8"))]
    for f in ("english_v5.json", "english_questions_v1.json"):
        msgs += [c["input"] for c in json.load(open(os.path.join(ROOT, "data", "acceptance", f), encoding="utf-8"))["cases"]]
    assert all(has_enough_letters(m) for m in msgs)


@pytest.mark.parametrize("text", ["a", "\U0001F64F\U0001F64F\U0001F64F", "12"])
def test_api_refuses_with_the_friendly_message(api_settings, text):
    with TestClient(api_settings().app) as c:
        r = c.post("/verify/text", json={"claim": text})
        u = c.post("/verify/url", json={"claim": text, "url": "https://example.edu/a"})
    assert r.status_code == 400 and r.json()["detail"] == TOO_SHORT_MESSAGE
    assert u.status_code == 400 and u.json()["detail"] == TOO_SHORT_MESSAGE


def test_api_still_answers_short_real_questions(api_settings):
    with TestClient(api_settings().app) as c:
        assert c.post("/verify/text", json={"claim": "san ginanap?"}).status_code == 200


def test_frontend_applies_the_same_rule():
    app = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
    assert "(claim.match(/\\p{L}/gu) || []).length < 3" in app and MIN_CLAIM_LETTERS == 3
    assert "Please type a claim or question about a DYCI announcement" in app
