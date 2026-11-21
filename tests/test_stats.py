"""Privacy-safe usage statistics (src/api/stats.py, CAMPUS_VERIFY_STATS default off) and the dashboard page."""
import datetime as dt
import json
import os
import re

import pytest
from fastapi.testclient import TestClient

from src.api import stats as st

DAY = dt.date(2026, 10, 26)


@pytest.mark.parametrize("kind, status, body, expected", [
    ("text", 200, {"verdict": "SUPPORTED", "evidence_status": "RELEVANT_EVIDENCE_FOUND"}, ("supported", True)),
    ("text", 200, {"verdict": "CONTRADICTED", "evidence_status": "RELEVANT_EVIDENCE_FOUND"}, ("contradicted", True)),
    ("text", 200, {"verdict": "INSUFFICIENT_EVIDENCE", "evidence_status": "NO_RELEVANT_EVIDENCE"}, ("not_confirmed", False)),
    ("text", 200, {"verdict": None, "verdict_withheld": "language", "evidence_status": "RELEVANT_EVIDENCE_FOUND"},
     ("withheld_language", True)),
    ("text", 200, {"verdict": None, "route": {"status": "INFO_SEEKING"}}, ("question", False)),
    ("link", 502, None, ("could_not_read", False)),
    ("picture", 200, {"assessment": {"level": 4}, "sentence_summary": {"matches": 2}}, ("level_4", True)),
    ("picture", 200, {"status": "no_text"}, ("no_text", False)),
    ("text", 400, {"detail": "too short"}, None),
    ("text", 429, None, None),
])
def test_each_response_maps_to_one_outcome(kind, status, body, expected):
    assert st.outcome_for(kind, status, body) == expected


def test_counts_and_coverage_over_the_last_days():
    c = st.Counter()
    c.record("text", "supported", True, DAY)
    c.record("text", "contradicted", True, DAY)
    c.record("link", "could_not_read", False, DAY)
    c.record("picture", "level_3", False, DAY - dt.timedelta(days=1))
    c.record("text", "made_up", True, DAY)                               # unknown outcomes are ignored
    s = c.snapshot(days=7, day=DAY)
    assert s["total"] == 4 and s["checks"] == {"text": 2, "link": 1, "picture": 1}
    assert s["coverage"] == round(2 / 3, 3)                              # could_not_read is not an answered check
    assert [d["date"] for d in s["daily"]][-1] == "2026-10-26" and len(s["daily"]) == 7
    assert s["daily"][-1]["text"] == 2 and s["daily"][-1]["contradicted"] == 1 and s["daily"][-2]["picture"] == 1


def test_the_saved_file_holds_counts_only(tmp_path):
    path = tmp_path / "counts.json"
    c = st.Counter(str(path))
    c.record("text", "contradicted", True, DAY)
    saved = path.read_text(encoding="utf-8")
    assert json.loads(saved)["days"]["2026-10-26"]["text"]["outcomes"] == {"contradicted": 1}
    assert st.Counter(str(path)).snapshot(day=DAY)["total"] == 1          # survives a restart


def test_the_setting_is_off_by_default(monkeypatch):
    monkeypatch.delenv(st.ENV, raising=False)
    assert st.enabled() is False


# ---- the routes -------------------------------------------------------------------------------------------------------
def _client(api_settings, monkeypatch, on, tmp_path=None):
    monkeypatch.setenv(st.ENV, "1" if on else "0")
    if tmp_path is not None:
        monkeypatch.setenv(st.FILE_ENV, str(tmp_path / "counts.json"))
    app = api_settings()
    app.state.pop("stats", None)
    return app


def test_off_means_no_endpoint_no_health_flag_and_no_counting(api_settings, monkeypatch):
    app = _client(api_settings, monkeypatch, False)
    with TestClient(app.app) as c:
        assert c.get("/stats").status_code == 404
        assert "stats" not in c.get("/health").json()
        c.post("/verify/text", json={"claim": "Classes are suspended tomorrow."})
    assert app.state.get("stats") is None


def test_on_counts_checks_and_never_stores_the_message(api_settings, monkeypatch, tmp_path):
    app = _client(api_settings, monkeypatch, True, tmp_path)
    secret = "Zyxwv classes are suspended at the Qwerty campus tomorrow."
    with TestClient(app.app) as c:
        assert c.get("/health").json()["stats"] is True
        body = c.post("/verify/text", json={"claim": secret}).json()
        c.post("/verify/text", json={"claim": "x"})                         # refused input: not counted
        s = c.get("/stats").json()
    assert s["checks"]["text"] == 1 and "verdict" in body                  # the response itself is passed on unchanged
    saved = (tmp_path / "counts.json").read_text(encoding="utf-8")
    assert "Zyxwv" not in saved and "Qwerty" not in saved and "Zyxwv" not in json.dumps(s)


def test_the_public_health_shows_the_flag_only_when_on(api_settings, monkeypatch):
    monkeypatch.setenv("CAMPUS_VERIFY_PUBLIC", "1")
    for on in (False, True):
        app = _client(api_settings, monkeypatch, on)
        with TestClient(app.app) as c:
            h = c.get("/health").json()
        assert ("stats" in h) is on


# ---- the page ---------------------------------------------------------------------------------------------------------
def _read(*parts):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return open(os.path.join(root, "frontend", *parts), encoding="utf-8").read()


def test_the_dashboard_never_calls_anything_fake_news_and_is_shown_only_when_counting_is_on():
    app, html = _read("js", "app.js"), _read("index.html")
    block = app[app.index("// ---- dashboard (GET /stats)"):app.index("// ---- startup: GET /health")]
    assert "not a count of fake news" in block and "Not confirmed" in block
    assert not re.search(r"(?i)fake news detected|% fake|likely fake|ai[- ]generated", block)
    assert re.search(r'<li id="nav-dashboard" hidden>', html)
    assert 'if (h.stats === true) { $("nav-dashboard").hidden = false; }' in app


def test_the_level_gauge_is_a_fixed_scale_not_a_percentage():
    app = _read("js", "app.js")
    gauge = app[app.index("// A half-dial with five segments"):app.index("function renderLevelCard(data)")]
    assert "%" not in gauge and "not a probability" in gauge


def test_the_totals_follow_the_chosen_range_and_all_time_is_kept(tmp_path):
    import datetime as _d
    c = st.Counter()
    end = _d.date(2026, 10, 28)
    c.record("text", "supported", True, end)
    c.record("text", "contradicted", True, end - _d.timedelta(days=20))
    week, month = c.snapshot(days=7, day=end), c.snapshot(days=30, day=end)
    assert week["total"] == 1 and month["total"] == 2 and week["all_time"]["total"] == 2
    assert week["days"] == 7 and len(month["daily"]) == 30


def test_retention_merges_days_older_than_400_calendar_days_into_months(tmp_path, monkeypatch):
    from src.api import db
    monkeypatch.setenv(db.RETENTION_ENV, "1")
    database = db.Database(str(tmp_path / "r.db"))
    counter = st.SqliteCounter(database)
    old, recent, now = dt.date(2025, 9, 1), dt.date(2025, 9, 21), dt.date(2026, 10, 26)
    for d, outcome in ((old, "supported"), (old, "supported"), (recent, "contradicted")):
        counter.record("text", outcome, True, day=d)
    counter.record("text", "question", False, day=dt.date(2025, 9, 22))  # day 400 counting today: kept
    counter.record("text", "not_confirmed", False, day=now)
    rows = database.conn.execute("SELECT DISTINCT day FROM daily_counts ORDER BY day").fetchall()
    assert rows == [("2025-09-22",), ("2026-10-26",)]
    assert sorted(database.conn.execute("SELECT * FROM monthly_counts").fetchall()) == [
        ("2025-09", "text", "contradicted", 1), ("2025-09", "text", "supported", 2)]
    assert database.conn.execute("SELECT n FROM monthly_matched").fetchone()[0] == 3
    s = counter.snapshot(days=7, day=now)
    assert s["all_time"] == {"total": 5, "since": "2025-09"} and s["earlier_months"] == [{"month": "2025-09", "total": 3}]
    assert database.conn.execute("PRAGMA secure_delete").fetchone()[0] == 1


def test_without_retention_the_old_rule_stays_and_secure_delete_is_off(tmp_path, monkeypatch):
    from src.api import db
    monkeypatch.delenv(db.RETENTION_ENV, raising=False)
    database = db.Database(str(tmp_path / "r.db"))
    counter = st.SqliteCounter(database)
    counter.record("text", "supported", True, day=dt.date(2020, 1, 1))
    counter.record("text", "supported", True, day=dt.date(2026, 10, 26))
    assert len(database.conn.execute("SELECT day FROM daily_counts").fetchall()) == 2   # only 2 days with data
    assert database.conn.execute("SELECT COUNT(*) FROM monthly_counts").fetchone()[0] == 0
    assert database.conn.execute("PRAGMA secure_delete").fetchone()[0] == 0
