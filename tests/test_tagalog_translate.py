"""Local Tagalog translation (src/normalization/tagalog_translate.py, CAMPUS_VERIFY_TAGALOG_TRANSLATE default on since v2.3;
results/round2/tagalog_translate_plan.md). The server is mocked; the plan's test set G is used once, by its run."""
import os
import re

import pytest
from fastapi.testclient import TestClient

from src.api.verdict_policy import withhold_for_language
from src.normalization import tagalog_translate as tt


@pytest.mark.parametrize("message, english, failed", [
    ("walang pasok bukas", "There are no classes tomorrow.", None),
    ("may pasok ba sa 21?", "Are there classes on the 21st?", None),
    ("walang pasok bukas", "There are classes tomorrow.", "C3_negation"),        # a flipped meaning
    ("may misa sa sept 8?", "Is there a mass on September 9?", "C2_numbers"),    # a garbled date
    ("may misa ba?", "UNCLEAR", "C1_form"),
    ("may misa ba?", "May misa ba?", "C1_form"),                                # still Tagalog
    ("may misa ba?", "Is there a mass?\nYes.", "C1_form"),
    ("suspended ba f2f ngayon?", "Are face-to-face classes suspended today?", None),     # B1: "f2f" is no number
    ("midterms na sa 28 to 30", "The midterms are from October 28 to 30.", "C4_calendar"),   # B1: an invented month
    ("may pasok ba sa biyernes?", "Are there classes on Friday?", None),
    ("may misa sa mayo 3", "There is a mass on May 3.", None),
])
def test_the_checks(message, english, failed):
    got = tt.checks(message, english, withhold_for_language)
    assert [k for k, ok in got.items() if not ok] == ([failed] if failed else [])


def test_a_server_that_is_down_keeps_todays_response(monkeypatch):
    monkeypatch.setenv(tt.URL_ENV, "http://127.0.0.1:9")                       # nothing listens there
    assert tt.translate("walang pasok bukas", withhold_for_language) is None


def test_the_setting_is_on_by_default_since_v2_3(monkeypatch):
    monkeypatch.delenv(tt.ENV, raising=False)
    assert tt.enabled() is True
    monkeypatch.setenv(tt.ENV, "0")
    assert tt.enabled() is False


def test_the_route_checks_the_accepted_translation_and_keeps_the_message_as_typed(api_settings, monkeypatch):
    seen = []
    monkeypatch.setattr(tt, "_ask", lambda m: seen.append(m) or "There are no classes tomorrow.")
    monkeypatch.setenv(tt.ENV, "1")
    app = api_settings()
    with TestClient(app.app) as c:
        body = c.post("/verify/text", json={"claim": "walang pasok daw bukas"}).json()
    assert seen == ["walang pasok daw bukas"]
    assert body["claim"] == "walang pasok daw bukas" and body["tagalog_translation"]["accepted"]
    assert body["relative_resolution"]["words"] == ["tomorrow"] and body.get("hearsay_notice")


def test_a_rejected_translation_is_recorded_but_not_used(api_settings, monkeypatch):
    monkeypatch.setattr(tt, "_ask", lambda m: "There are classes tomorrow.")     # negation lost
    monkeypatch.setenv(tt.ENV, "1")
    app = api_settings()
    with TestClient(app.app) as c:
        body = c.post("/verify/text", json={"claim": "walang pasok bukas"}).json()
    assert body["tagalog_translation"]["accepted"] is False and "relative_resolution" not in body


def test_the_page_shows_only_an_accepted_translation_and_never_sets_a_verdict():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = open(os.path.join(root, "frontend", "js", "app.js"), encoding="utf-8").read()
    block = app[app.index("function tagalogNote(data)"):app.index("// Display-only relative-date notice")]
    assert "m.accepted" in block and "machine translation" in block and not re.search(r"verdict\s*=(?!=)", block)


def test_every_request_turns_the_prompt_cache_off_so_the_same_message_gives_the_same_english(monkeypatch):
    sent = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices": [{"message": {"content": "There are no classes tomorrow."}}]}'

    monkeypatch.setattr(tt.urllib.request, "urlopen", lambda req, timeout: sent.append(req.data) or Reply())
    tt._ask("walang pasok bukas")
    import json
    body = json.loads(sent[0])
    assert body["cache_prompt"] is False and body["temperature"] == 0 and body["seed"] == 42
    assert tt.VERSION == "tagalog-translate-v1.1"
