"""CAMPUS_VERIFY_UI_EXTRAS (default off; display only): example messages, copy/share and next steps on the page."""
import os
import re

from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
BLOCK = APP[APP.index("// ---- CAMPUS_VERIFY_UI_EXTRAS"):APP.index("// ---- events ----")]


def test_health_reports_the_setting_only_when_it_is_on(api_settings, monkeypatch):
    for on, expected in (("0", None), ("1", True)):
        monkeypatch.setenv("CAMPUS_VERIFY_UI_EXTRAS", on)
        app = api_settings()
        with TestClient(app.app) as c:
            assert c.get("/health").json().get("ui_extras") is expected


def test_the_extras_never_set_a_verdict_and_say_it_is_not_official():
    assert not re.search(r"verdict\s*=(?!=)", BLOCK)
    assert "Not an official DYCI statement" in BLOCK and "national holiday list" in BLOCK


def test_the_extras_are_inserted_as_text_never_as_html():
    assert "innerHTML" not in BLOCK and "insertAdjacentHTML" not in BLOCK


def test_the_block_has_balanced_brackets():
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', BLOCK)               # drop string contents
    code = code.replace("/\\s+/g", "")                                # the one regex literal
    for a, b in ("()", "{}", "[]"):
        assert code.count(a) == code.count(b), a + b
