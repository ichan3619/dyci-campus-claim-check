"""'My checks' (src/api/history.py; CAMPUS_VERIFY_HISTORY, default off). Saved only for an account that turned history on;
only the owner sees or deletes it; at most 50 entries and 90 days; erased with the account."""
import datetime as dt

import pytest

from src.api import db, history
from tests.test_auth import _client, signin, token  # noqa: F401  (signin is a fixture)

MESSAGE = "The midterm examinations are on September 28 to 30."


@pytest.fixture
def on(signin, monkeypatch):  # noqa: F811
    monkeypatch.setenv(history.ENV, "1")
    return signin


def _rows(tmp_path):
    return db.Database(str(tmp_path / "app.db")).conn.execute("SELECT user_id, kind, message FROM checks").fetchall()


def test_the_setting_is_off_by_default(signin, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.delenv(history.ENV, raising=False)
    assert history.enabled() is False
    with _client(signin) as c:
        c.post("/auth/google", json={"credential": token()})
        c.post("/account/history", json={"history_enabled": True})
        assert c.post("/verify/text", json={"claim": MESSAGE}).status_code == 200
        assert c.get("/account/checks").status_code == 404
        assert "history" not in c.get("/health").json()
    assert _rows(tmp_path) == []


def test_nothing_is_saved_until_the_student_turns_history_on(on, tmp_path):
    with _client(on) as c:
        c.post("/auth/google", json={"credential": token()})
        c.post("/verify/text", json={"claim": MESSAGE})
        assert c.get("/account/checks").json()["checks"] == []
        c.post("/account/history", json={"history_enabled": True})
        r = c.post("/verify/text", json={"claim": MESSAGE})
        assert r.status_code == 200 and r.json()["verdict"]                # the route still read the message
        checks = c.get("/account/checks").json()["checks"]
    assert len(checks) == 1 and checks[0]["kind"] == "text" and checks[0]["message"] == MESSAGE
    assert checks[0]["outcome"] in ("supported", "contradicted", "not_confirmed", "question")
    assert len(_rows(tmp_path)) == 1


def test_checks_without_an_account_are_never_saved(on, tmp_path):
    with _client(on) as c:
        assert c.post("/verify/text", json={"claim": MESSAGE}).status_code == 200
        assert c.get("/account/checks").status_code == 401
    assert _rows(tmp_path) == []


def test_only_the_owner_can_see_or_delete_an_entry(on):
    with _client(on) as juan, _client(on) as maria, _client(on) as admin:
        juan.post("/auth/google", json={"credential": token()})
        juan.post("/account/history", json={"history_enabled": True})
        juan.post("/verify/text", json={"claim": MESSAGE})
        entry = juan.get("/account/checks").json()["checks"][0]
        maria.post("/auth/google", json={"credential": token(email="maria.santos@dyci.edu.ph", name="Maria")})
        admin.post("/auth/google", json={"credential": token(email="admin.user@dyci.edu.ph", name="Admin")})
        for other in (maria, admin):                                       # no admin view either
            assert other.get("/account/checks").json()["checks"] == []
            assert other.post(f"/account/checks/{entry['id']}/delete").status_code == 404
        assert "message" not in str(admin.get("/admin/users").json())
        assert juan.post(f"/account/checks/{entry['id']}/delete").json() == {"deleted": True}
        assert juan.get("/account/checks").json()["checks"] == []


def test_turning_history_off_can_erase_what_was_saved(on):
    with _client(on) as c:
        c.post("/auth/google", json={"credential": token()})
        c.post("/account/history", json={"history_enabled": True})
        c.post("/verify/text", json={"claim": MESSAGE})
        c.post("/account/history", json={"history_enabled": False})       # kept unless asked
        assert len(c.get("/account/checks").json()["checks"]) == 1
        c.post("/verify/text", json={"claim": MESSAGE})                     # off: not saved
        assert len(c.get("/account/checks").json()["checks"]) == 1
        c.post("/account/history", json={"history_enabled": False, "delete_saved": True})
        assert c.get("/account/checks").json()["checks"] == []
        c.post("/account/history", json={"history_enabled": True})
        c.post("/verify/text", json={"claim": MESSAGE})
        assert c.post("/account/checks/clear").json() == {"deleted": 1}


def test_deleting_the_account_erases_its_checks(on, tmp_path):
    with _client(on) as c:
        c.post("/auth/google", json={"credential": token()})
        c.post("/account/history", json={"history_enabled": True})
        c.post("/verify/text", json={"claim": MESSAGE})
        assert len(_rows(tmp_path)) == 1
        assert c.post("/account/delete", json={"credential": token(), "confirm": "DELETE"}).status_code == 200
    assert _rows(tmp_path) == []


def _user(database, email="a@dyci.edu.ph"):
    with database.lock:
        cur = database.conn.execute("INSERT INTO users (email, created_at, last_login_at) VALUES (?, '', '')", (email,))
    return cur.lastrowid


ENTRY = {"kind": "text", "message": "m", "url": None, "outcome": "supported", "post_id": None, "post_date": None,
         "post_url": None, "post_excerpt": None}


def test_only_the_newest_50_are_kept(tmp_path):
    database = db.Database(str(tmp_path / "h.db"))
    uid, other = _user(database), _user(database, "b@dyci.edu.ph")
    start = dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)
    history.save(database, other, ENTRY, now=start)
    for i in range(history.KEEP_LAST + 5):
        history.save(database, uid, {**ENTRY, "message": f"m{i}"}, now=start + dt.timedelta(minutes=i))
    kept = history.list_for(database, uid, now=start + dt.timedelta(days=1))
    assert len(kept) == history.KEEP_LAST and kept[0]["message"] == f"m{history.KEEP_LAST + 4}" and kept[-1]["message"] == "m5"
    assert len(history.list_for(database, other, now=start + dt.timedelta(days=1))) == 1     # others untouched


def test_entries_older_than_90_days_are_removed(tmp_path):
    database = db.Database(str(tmp_path / "h.db"))
    uid = _user(database)
    start = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)
    history.save(database, uid, ENTRY, now=start)
    assert len(history.list_for(database, uid, now=start + dt.timedelta(days=history.KEEP_DAYS - 1))) == 1
    assert history.list_for(database, uid, now=start + dt.timedelta(days=history.KEEP_DAYS + 1)) == []


def test_what_an_entry_keeps():
    body = {"used_evidence": {"document_id": "ingested:N20", "text": "  Classes   are suspended " + "x" * 300,
                              "published_date": "2026-08-21", "source_url": "https://facebook.com/p/1"}}
    e = history.entry_for("text", "contradicted", {"claim": " No classes today "}, body)
    assert e["message"] == "No classes today" and e["url"] is None and e["post_id"] == "ingested:N20"
    assert e["post_date"] == "2026-08-21" and len(e["post_excerpt"]) == history.EXCERPT_CHARS
    assert e["post_excerpt"].startswith("Classes are suspended")
    link = history.entry_for("link", "could_not_read", {"claim": "c", "url": "https://example.com/a"}, None)
    assert link["url"] == "https://example.com/a" and link["post_id"] is None
    picture = history.entry_for("picture", "level_3", {"claim": "never kept"}, {"used_evidence": {"text": "t"}})
    assert picture["message"] is None and picture["post_id"] is None and picture["url"] is None
    assert history.entry_for("text", "supported", {}, body) is None


def test_the_page_shows_saved_checks_as_text_and_offers_check_again():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = open(os.path.join(root, "frontend", "js", "app.js"), encoding="utf-8").read()
    block = app[app.index("// ---- My checks"):app.index("function initHistory")]
    assert "innerHTML" not in block and "Check again" in block and "requestSubmit" in block
    page = open(os.path.join(root, "frontend", "index.html"), encoding="utf-8").read()
    assert 'id="checks-card"' in page and "coming soon" not in page
