"""Scenario fixes from the probes (results/round2/scenario_probes_report.md): the login-wall check for /verify/url
(src/api/login_wall.py, CAMPUS_VERIFY_LOGIN_WALL) and the hearsay notice (src/verification/hearsay.py,
CAMPUS_VERIFY_HEARSAY_NOTE). Both default off. Synthetic pages and messages only; no network."""
import os
import re
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import login_wall as lw
from src.verification import hearsay as hs

LOGIN = ("Log into Facebook. You must log in to continue. Log in to Facebook to see this post. Email or phone number. "
         "Password. Forgot password? Create new account.")
UNAVAILABLE = "This content isn't available right now. When this happens, it's usually because the owner only shared it."
POST = ("DYCI ADVISORY 20 August 2026. To all DYCIans, please be advised that there will be no classes at all levels "
        "tomorrow, August 21, 2026 (Friday), in observance of Ninoy Aquino Day, a Special Non-Working Holiday, pursuant "
        "to Republic Act No. 9256. All students, faculty members and personnel are advised to take note of the holiday "
        "and plan accordingly. Regular classes and school operations may resume on the next scheduled school day. "
        "Please be guided. ")


@pytest.mark.parametrize("text", [LOGIN, UNAVAILABLE, "Log in or sign up to view. Create new account."])
def test_login_and_unavailable_pages_are_detected(text):
    assert lw.detect(text) is True


@pytest.mark.parametrize("text", [
    POST,
    POST + "See more of Dr. Yanga's Colleges, Inc. on Facebook. Log In. Create new account.",   # banner around a post
    "Password reset dates for the student portal are posted on the registrar's page.",       # one phrase only
    "",
])
def test_a_real_post_is_read_as_before(text):
    assert lw.detect(text) is False


def _page(text):
    r = Mock()
    r.status_code, r.text, r.headers = 200, f"<html><body><p>{text}</p></body></html>", {}
    r.iter_content = Mock(return_value=[r.text.encode("utf-8")])
    r.apparent_encoding, r.encoding, r.is_redirect, r.close = "utf-8", None, False, Mock()
    return r


def _url_check(api_settings, monkeypatch, text, on):
    monkeypatch.setenv(lw.ENV, "1" if on else "0")
    app = api_settings()
    with TestClient(app.app) as c, \
            patch("src.api.ssrf_guard._resolve_hostname_ips", return_value=["157.240.22.35"]), \
            patch("src.ingestion.adapters.requests.get", return_value=_page(text)):
        return c.post("/verify/url", json={"claim": "No classes on August 21, 2026.",
                                           "url": "https://www.facebook.com/YangaCollege/posts/x"})


def test_the_url_route_reports_a_login_page_as_unreadable_only_with_the_setting_on(api_settings, monkeypatch):
    off, on = _url_check(api_settings, monkeypatch, LOGIN, False), _url_check(api_settings, monkeypatch, LOGIN, True)
    assert off.status_code == 200
    assert on.status_code == 502 and on.json()["detail"] == lw.MESSAGE


def test_a_real_post_link_still_gets_a_verdict_with_the_setting_on(api_settings, monkeypatch):
    r = _url_check(api_settings, monkeypatch, POST, True)
    assert r.status_code == 200 and "verdict" in r.json()


@pytest.mark.parametrize("message, word", [
    ("Classes will probably be suspended tomorrow because of heavy rain.", "probably"),
    ("I heard classes at DYCI are suspended on August 21.", "i heard"),
    ("Baka walang pasok bukas dahil sa ulan.", "baka"),
    ("Wala daw pasok sa DYCI tom, legit ba?", "daw"),
    ("Sabi nila suspended ang klase bukas.", "sabi nila"),
    ("Rumors say the midterms moved to October.", "rumors"),
])
def test_hearsay_and_predictions_get_a_notice(message, word):
    n = hs.notice(message)
    assert n is not None and n["words"][0] == word and f"“{word}”" in n["text"]


@pytest.mark.parametrize("message", [
    "Classes are suspended at all levels on September 10, 2026.", "The art club will draw murals.",
    "Students may bring raw materials", "Is there no class on August 21?"])
def test_plain_statements_get_no_notice(message):
    assert hs.notice(message) is None


@pytest.mark.parametrize("env", [lw.ENV, hs.ENV])
def test_both_settings_are_on_by_default_since_v1_9(env, monkeypatch):
    monkeypatch.delenv(env, raising=False)
    assert (lw.enabled() if env == lw.ENV else hs.enabled()) is True
    monkeypatch.setenv(env, "0")
    assert (lw.enabled() if env == lw.ENV else hs.enabled()) is False


def test_the_notice_is_added_only_with_the_setting_on_and_changes_nothing_else(api_settings, monkeypatch):
    bodies = {}
    for on in ("0", "1"):
        monkeypatch.setenv(hs.ENV, on)
        app = api_settings()
        with TestClient(app.app) as c:
            bodies[on] = c.post("/verify/text", json={"claim": "Classes will probably be suspended tomorrow."}).json()
    assert "hearsay_notice" not in bodies["0"]                      # left out of the response, not null
    assert bodies["1"]["hearsay_notice"]["words"] == ["probably"]
    assert {k: v for k, v in bodies["1"].items() if k != "hearsay_notice"} == bodies["0"]


def test_the_page_shows_the_note_and_never_sets_a_verdict():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = open(os.path.join(root, "frontend", "js", "app.js"), encoding="utf-8").read()
    block = app[app.index("function hearsayNote"):app.index("// Display-only relative-date notice")]
    assert "n.text" in block and not re.search(r"verdict\s*=(?!=)", block)
