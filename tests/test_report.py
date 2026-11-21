"""The dashboard's report (panel request 2026-10-29): GET /stats.csv (the same counts as /stats as a spreadsheet, same
access rule) and the printable page (Print / Save as PDF)."""
import csv
import datetime as dt
import io
import os

from fastapi.testclient import TestClient

from src.api import stats as st
from tests.test_auth import _client as _https_client, signin, token  # noqa: F401  (signin is a fixture)

DAY = dt.date(2026, 10, 26)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_one_row_per_day_and_kind_with_the_dashboards_counts():
    data = {"2026-10-25": {"text": {"outcomes": {"supported": 2, "contradicted": 1}, "matched": 3},
                           "picture": {"outcomes": {"level_4": 1}, "matched": 1}}}
    rows = list(csv.DictReader(io.StringIO(st.to_csv(data, 3, DAY))))
    assert [(r["date"], r["input_type"]) for r in rows] == [(d, k) for d in ("2026-10-24", "2026-10-25", "2026-10-26")
                                                      for k in ("text", "link", "picture")]
    text = rows[3]
    assert text["total_verifications"] == "3" and text["supported"] == "2" and text["contradicted"] == "1"
    assert text["matched_a_dyci_post"] == "3" and text["level_4"] == ""          # a picture level is not a text outcome
    assert rows[5]["level_4"] == "1" and rows[5]["supported"] == ""
    assert all(r["total_verifications"] == "0" for r in rows[:3])                            # empty days are rows too
    summary = st.summarise(data, 3, DAY)
    assert sum(int(r["total_verifications"]) for r in rows) == summary["total"]


def test_the_spreadsheet_holds_counts_only_and_is_off_with_the_dashboard(api_settings, monkeypatch, tmp_path):
    monkeypatch.setenv(st.ENV, "0")
    app = api_settings()
    app.state.pop("stats", None)
    with TestClient(app.app) as c:
        assert c.get("/stats.csv").status_code == 404
    monkeypatch.setenv(st.ENV, "1")
    monkeypatch.setenv(st.FILE_ENV, str(tmp_path / "counts.json"))
    app.state.pop("stats", None)
    secret = "Zyxwv classes are suspended at the Qwerty campus tomorrow."
    with TestClient(app.app) as c:
        c.post("/verify/text", json={"claim": secret})
        r = c.get("/stats.csv?days=7")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="dyci-claim-check-report-' in r.headers["content-disposition"]
    assert "Zyxwv" not in r.text and "Qwerty" not in r.text
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) == 7 * 3 and sum(int(x["total_verifications"]) for x in rows if x["input_type"] == "text") == 1


def test_with_sign_in_on_the_spreadsheet_is_for_administrators(signin, monkeypatch):  # noqa: F811
    monkeypatch.setenv(st.ENV, "1")
    with _https_client(signin) as student, _https_client(signin) as admin:
        assert student.get("/stats.csv").status_code == 401
        student.post("/auth/google", json={"credential": token()})
        assert student.get("/stats.csv").status_code == 403
        admin.post("/auth/google", json={"credential": token(email="admin.user@dyci.edu.ph")})
        assert admin.get("/stats.csv").status_code == 200


def test_the_printed_report_has_a_header_hides_the_app_chrome_and_builds_text_only():
    page = open(os.path.join(ROOT, "frontend", "index.html"), encoding="utf-8").read()
    css = open(os.path.join(ROOT, "frontend", "css", "styles.css"), encoding="utf-8").read()
    app = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
    assert 'id="dash-print"' in page and 'id="dash-csv"' in page and 'id="report-period"' in page
    printing = css[css.index("@media print"):]
    assert ".sidebar" in printing and ".dash-controls" in printing and ".print-only { display: block; }" in printing
    block = app[app.index("function reportTable"):app.index("// ---- Sign in with Google")]
    assert "window.print()" in block and "innerHTML" not in block and "fake" not in block.lower()


def test_the_pages_own_files_are_rechecked_on_each_visit(api_settings):
    app = api_settings()
    with TestClient(app.app) as c:
        for path in ("/css/styles.css", "/js/app.js", "/index.html", "/"):
            assert c.get(path).headers.get("cache-control") == "no-cache", path
        assert c.get("/health").headers.get("cache-control") != "no-cache"
