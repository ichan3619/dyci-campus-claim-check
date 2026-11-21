"""Guided first-visit tour (CAMPUS_VERIFY_TOUR, default off; display only) and the account's "tour seen" flag."""
import os
import re

from fastapi.testclient import TestClient

from test_auth import _client, signin, token          # noqa: F401  (the signed-in app fixture)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
TOUR = APP[APP.index("// ---- guided tour"):APP.index("// ---- startup: GET /health")]


def test_health_reports_the_tour_only_when_it_is_on(api_settings, monkeypatch):
    for on, expected in (("0", None), ("1", True)):
        monkeypatch.setenv("CAMPUS_VERIFY_TOUR", on)
        for public in ("0", "1"):
            monkeypatch.setenv("CAMPUS_VERIFY_PUBLIC", public)
            app = api_settings()
            with TestClient(app.app) as c:
                assert c.get("/health").json().get("tour") is expected


def test_a_signed_in_account_remembers_that_the_tour_was_seen(signin):
    with _client(signin) as c:
        assert c.post("/auth/google", json={"credential": token()}).json()["user"]["tour_seen"] is False
        assert c.post("/account/tour-seen").json()["user"]["tour_seen"] is True
        assert c.get("/auth/me").json()["user"]["tour_seen"] is True


def test_the_tour_can_always_be_left_and_stores_nothing_in_the_browser():
    assert "Skip tour" in open(os.path.join(ROOT, "frontend", "index.html"), encoding="utf-8").read()
    assert '"Escape"' in TOUR and "endTour" in TOUR and "aria-modal" in open(
        os.path.join(ROOT, "frontend", "index.html"), encoding="utf-8").read()
    for forbidden in ("localStorage", "sessionStorage", "document.cookie", "innerHTML"):
        assert forbidden not in TOUR
    assert not re.search(r"verdict\s*=(?!=)", TOUR)


def test_the_menu_only_switches_pages_for_items_that_name_one():
    assert '.closest(".nav-item[data-view]")' in APP
