"""The SQLite database (src/api/db.py, CAMPUS_VERIFY_DB_FILE unset by default) and the dashboard counts kept in it
(stats.SqliteCounter): the same numbers as the JSON counter, an existing JSON file imported once, KEEP_DAYS kept."""
import datetime as dt
import json
import threading

from fastapi.testclient import TestClient

from src.api import db, stats

DAY = dt.date(2026, 10, 28)
EVENTS = [("text", "supported", True, DAY), ("text", "not_confirmed", False, DAY), ("link", "could_not_read", False, DAY),
          ("picture", "level_4", True, DAY - dt.timedelta(days=1)), ("text", "contradicted", True, DAY - dt.timedelta(days=3)),
          ("text", "nonsense", True, DAY), ("other", "supported", True, DAY)]                 # the last two are ignored


def test_the_schema_is_created_once_and_versioned(tmp_path):
    path = str(tmp_path / "app.db")
    latest = db.MIGRATIONS[-1][0]
    first = db.Database(path)
    assert first.version() == latest
    first.close()
    again = db.Database(path)                                                                  # reopening adds nothing
    assert again.version() == latest
    assert again.conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == len(db.MIGRATIONS)


def test_the_database_gives_the_same_dashboard_as_the_json_file(tmp_path):
    plain, sql = stats.Counter(str(tmp_path / "counts.json")), stats.SqliteCounter(db.Database(str(tmp_path / "app.db")))
    for e in EVENTS:
        plain.record(*e)
        sql.record(*e)
    for days in (1, 7, 31):
        assert sql.snapshot(days=days, day=DAY) == plain.snapshot(days=days, day=DAY)


def test_an_existing_json_file_is_imported_once(tmp_path):
    json_file = str(tmp_path / "counts.json")
    old = stats.Counter(json_file)
    for e in EVENTS:
        old.record(*e)
    database = db.Database(str(tmp_path / "app.db"))
    sql = stats.SqliteCounter(database, import_json=json_file)
    assert sql.snapshot(days=7, day=DAY) == old.snapshot(days=7, day=DAY)
    sql.record("text", "supported", True, DAY)
    again = stats.SqliteCounter(database, import_json=json_file)                              # a restart: no second import
    assert again.snapshot(days=7, day=DAY)["checks"]["text"] == old.snapshot(days=7, day=DAY)["checks"]["text"] + 1


def test_only_keep_days_are_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, "KEEP_DAYS", 3)
    sql = stats.SqliteCounter(db.Database(str(tmp_path / "app.db")))
    for i in range(5):
        sql.record("text", "supported", True, DAY - dt.timedelta(days=4 - i))
    assert sql.snapshot(days=7, day=DAY)["since"] == (DAY - dt.timedelta(days=2)).isoformat()


def test_records_from_many_threads_are_all_counted(tmp_path):
    sql = stats.SqliteCounter(db.Database(str(tmp_path / "app.db")))
    threads = [threading.Thread(target=lambda: [sql.record("text", "supported", False, DAY) for _ in range(25)])
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sql.snapshot(days=1, day=DAY)["checks"]["text"] == 200


def test_the_app_keeps_its_counts_in_the_database_when_set(api_settings, monkeypatch, tmp_path):
    path = str(tmp_path / "app.db")
    monkeypatch.setenv("CAMPUS_VERIFY_STATS", "1")
    monkeypatch.setenv(db.ENV, path)
    app = api_settings()
    app.state.pop("stats", None)
    app.state.pop("db", None)
    with TestClient(app.app) as c:
        c.post("/verify/text", json={"claim": "The midterm exams are on September 28 to 30."})
        total = c.get("/stats").json()["total"]
    assert total == 1
    rows = db.Database(path).conn.execute("SELECT SUM(n) FROM daily_counts").fetchone()[0]
    assert rows == 1
    assert "midterm" not in json.dumps(db.Database(path).conn.execute("SELECT * FROM daily_counts").fetchall())


def test_an_older_database_is_upgraded_and_keeps_its_counts(tmp_path, monkeypatch):
    path = str(tmp_path / "app.db")
    monkeypatch.setattr(db, "MIGRATIONS", db.MIGRATIONS[:1])                        # a database made before accounts
    old = db.Database(path)
    stats.SqliteCounter(old).record("text", "supported", True, DAY)
    old.close()
    monkeypatch.undo()
    new = db.Database(path)
    assert new.version() == db.MIGRATIONS[-1][0]
    assert new.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    assert stats.SqliteCounter(new).snapshot(days=1, day=DAY)["checks"]["text"] == 1
