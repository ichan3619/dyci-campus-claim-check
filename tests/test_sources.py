"""The Sources page (src/api/sources.py, src/api/secret_box.py; CAMPUS_VERIFY_SOURCES, default off). Meta is SIMULATED:
synthetic page IDs, posts and tokens; no network. Tokens are encrypted with a test Fernet key."""
import datetime as dt
import json
import os
from unittest.mock import Mock, patch

import pytest
from cryptography.fernet import Fernet

from src.api import db, secret_box, sources
from src.ingestion import pages as pg
from tests.test_auth import _client, signin, token  # noqa: F401  (signin is a fixture)

GOOD = "EAAsyntheticGoodToken1234567890abcdef"
OTHER = "EAAsyntheticOtherToken1234567890abcde"
EXPIRED = "EAAsyntheticExpiredToken123456789abcd"
PAGE, NAME = "11111111111", "Synthetic DYCI Page"
ADMIN = "admin.user@dyci.edu.ph"


class Meta:
    """GET /{page}?fields=id,name, GET /{page}/posts and GET /debug_token, answered from dicts."""

    def __init__(self, posts=None, expires_in_days=None):
        self.posts = posts or {}
        self.expires_in_days = expires_in_days
        self.dead = {EXPIRED}

    def __call__(self, url, params=None, timeout=None):
        params, r = dict(params or {}), Mock(status_code=200)
        path = url.split("/v21.0/")[1]
        tok = params.get("access_token", "")
        if path == "debug_token":
            data = {"is_valid": True, "expires_at": 0}
            if self.expires_in_days is not None:
                data["expires_at"] = int((dt.datetime.now(dt.timezone.utc)
                                          + dt.timedelta(days=self.expires_in_days)).timestamp())
            r.json = Mock(return_value={"data": data})
        elif tok in self.dead:
            r.status_code = 400
            r.json = Mock(return_value={"error": {"code": 190, "message": f"Session has expired ({tok})"}})
        elif path.endswith("/posts"):
            r.json = Mock(return_value={"data": self.posts.get(path.split("/")[0], [])})
        else:
            r.json = Mock(return_value={"id": path, "name": NAME})
        return r


def post(n, text, day="2026-10-20"):
    return {"id": f"{PAGE}_{n}", "message": text, "created_time": f"{day}T09:00:00+0000",
            "permalink_url": f"https://www.facebook.com/synthetic/posts/{n}"}


@pytest.fixture
def on(signin, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setenv(sources.ENV, "1")
    monkeypatch.setenv(secret_box.KEY_ENV, Fernet.generate_key().decode())
    monkeypatch.setenv(sources.ARCHIVE_ENV, str(tmp_path / "archive"))
    monkeypatch.setattr(sources, "RESERVATION", str(tmp_path / "reservation.json"))
    monkeypatch.delenv(sources.APP_ID_ENV, raising=False)
    monkeypatch.delenv(sources.APP_SECRET_ENV, raising=False)
    signin.state.pop("secret_box", None)
    return signin


def _admin(c):
    c.post("/auth/google", json={"credential": token(email=ADMIN, name="Admin User")})
    return c


def _add(c, meta, tok=GOOD, page=PAGE, tier="official"):
    with patch("src.api.sources.requests.get", meta):
        return c.post("/admin/sources", json={"page_id": page, "tier": tier, "token": tok})


def test_off_by_default_and_for_administrators_only(signin, on, monkeypatch):  # noqa: F811
    with _client(on) as student, _client(on) as admin:
        assert student.get("/admin/sources").status_code == 401
        student.post("/auth/google", json={"credential": token()})
        assert student.get("/admin/sources").status_code == 403
        assert _admin(admin).get("/admin/sources").status_code == 200
        assert admin.get("/health").json().get("sources") is True
        monkeypatch.delenv(sources.ENV)
        assert admin.get("/admin/sources").status_code == 404
        assert "sources" not in admin.get("/health").json()


def test_a_page_is_added_only_with_a_token_meta_accepts_and_the_token_is_never_readable(on, tmp_path):
    meta = Meta()
    with _client(on) as c:
        _admin(c)
        bad = _add(c, meta, tok=EXPIRED)
        assert bad.status_code == 400 and "expired" in bad.json()["detail"] and EXPIRED not in bad.text
        assert _add(c, meta, page="12ab").status_code == 400                          # not a page ID
        r = _add(c, meta)
        assert r.status_code == 200 and GOOD not in r.text
        s = r.json()["source"]
        assert s["name"] == NAME and s["key"] == "synthetic_dyci_page" and s["token_hint"] == GOOD[-4:]
        assert s["token_status"] == "ok" and "token" not in s
        assert _add(c, meta).status_code == 400                                       # the same page twice
        listing = c.get("/admin/sources")
        assert GOOD not in listing.text and listing.json()["activity"][0]["by"] == "Admin User"
    raw = db.Database(str(tmp_path / "app.db")).conn.execute("SELECT token FROM sources").fetchone()[0]
    assert GOOD.encode() not in raw and len(raw) > len(GOOD)


def test_an_expired_token_is_flagged_and_can_be_replaced(on):
    meta = Meta()
    with _client(on) as c:
        _admin(c)
        _add(c, meta)
        meta.dead.add(GOOD)                                                           # the page admin's token dies
        with patch("src.api.sources.requests.get", meta):
            s = c.post("/admin/sources/synthetic_dyci_page/check").json()["source"]
            assert s["token_status"] == "invalid" and "replace it" in s["token_detail"] and "[REDACTED]" in s["token_detail"]
            assert c.get("/admin/sources").json()["needs_attention"] == ["synthetic_dyci_page"]
            assert c.post("/admin/sources/synthetic_dyci_page/token", json={"token": EXPIRED}).status_code == 400
            s = c.post("/admin/sources/synthetic_dyci_page/token", json={"token": OTHER}).json()["source"]
        assert s["token_status"] == "ok" and s["token_hint"] == OTHER[-4:]
        assert c.get("/admin/sources").json()["needs_attention"] == []


def test_with_the_apps_credentials_an_expiry_two_weeks_out_is_warned(on, monkeypatch):
    monkeypatch.setenv(sources.APP_ID_ENV, "123")
    monkeypatch.setenv(sources.APP_SECRET_ENV, "synthetic-app-secret")
    with _client(on) as c:
        _admin(c)
        s = _add(c, Meta(expires_in_days=5)).json()["source"]
    assert s["token_status"] == "expiring" and "day" in s["token_detail"] and s["token_expires_at"]
    assert sources.check_token(PAGE, GOOD, Meta(), environ={"META_APP_ID": "1", "META_APP_SECRET": "s"})["expires_at"] == "never"


def test_fetching_is_locked_until_the_d5_cutoff_is_recorded(on):
    meta = Meta(posts={PAGE: [post(1, "Enrollment for the second semester opens on November 3.")]})
    with _client(on) as c:
        _admin(c)
        _add(c, meta)
        r = c.post("/admin/sources/synthetic_dyci_page/fetch")
        assert r.status_code == 400 and "cutoff" in r.json()["detail"]
        assert c.get("/admin/sources").json()["cutoff"] is None


def test_a_fetch_fills_the_archive_reserves_new_posts_and_never_writes_the_token(on, tmp_path):
    pg.set_cutoff(sources.RESERVATION, "2026-11-01")
    meta = Meta(posts={PAGE: [post(1, "Enrollment for the second semester opens on November 3."),
                              post(2, "Midterm examinations start on November 10.", day="2026-11-02")]})
    with _client(on) as c:
        _admin(c)
        _add(c, meta)
        with patch("src.api.sources.requests.get", meta), patch("src.ingestion.adapters.requests.get", meta):
            r = c.post("/admin/sources/synthetic_dyci_page/fetch")
        assert r.status_code == 200
        result = r.json()["result"]["synthetic_dyci_page"]
        assert result["status"] == "ok" and result["added"] == 1 and result["reserved_for_final_evaluation"] == 1
        s = c.get("/admin/sources").json()
        assert s["cutoff"] == "2026-11-01" and s["sources"][0]["archived_posts"] == 1
        assert s["sources"][0]["last_fetch"]["added"] == 1
        c.post("/admin/sources/synthetic_dyci_page", json={"include": False})
        assert c.post("/admin/sources/synthetic_dyci_page/fetch").status_code == 400  # turned off
    archive = tmp_path / "archive"
    written = "".join(p.read_text(encoding="utf-8") for p in archive.rglob("*") if p.is_file())
    assert "November 3" in written and "Midterm" not in written and GOOD not in written


def test_a_token_that_dies_during_a_fetch_is_marked_and_skipped_after(on, tmp_path):
    pg.set_cutoff(sources.RESERVATION, "2026-11-01")
    meta = Meta(posts={PAGE: [post(1, "Enrollment for the second semester opens on November 3.")]})
    with _client(on) as c:
        _admin(c)
        _add(c, meta)
        meta.dead.add(GOOD)
        with patch("src.ingestion.adapters.requests.get", meta):
            r = c.post("/admin/sources/synthetic_dyci_page/fetch").json()["result"]["synthetic_dyci_page"]
        assert r["status"] == "failed" and GOOD not in json.dumps(r)
        assert c.get("/admin/sources").json()["sources"][0]["token_status"] == "invalid"
        assert c.post("/admin/sources/synthetic_dyci_page/fetch").status_code == 400  # nothing left to fetch


def test_removing_a_page_drops_its_token_and_keeps_its_archive(on, tmp_path):
    pg.set_cutoff(sources.RESERVATION, "2026-11-01")
    meta = Meta(posts={PAGE: [post(1, "Enrollment for the second semester opens on November 3.")]})
    with _client(on) as c:
        _admin(c)
        _add(c, meta)
        with patch("src.ingestion.adapters.requests.get", meta):
            c.post("/admin/sources/synthetic_dyci_page/fetch")
        assert c.post("/admin/sources/synthetic_dyci_page/remove").json() == {"removed": True}
        listing = c.get("/admin/sources").json()
        assert listing["sources"] == [] and listing["activity"][0]["action"] == "removed"
    assert (tmp_path / "archive" / "synthetic_dyci_page.jsonl").exists()
    assert db.Database(str(tmp_path / "app.db")).conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 0


def test_the_schedule_checks_stale_tokens_then_fetches(on, tmp_path):
    pg.set_cutoff(sources.RESERVATION, "2026-11-01")
    meta = Meta(posts={PAGE: [post(1, "Enrollment for the second semester opens on November 3.")]})
    with _client(on) as c:
        _admin(c)
        _add(c, meta)
    database = db.Database(str(tmp_path / "app.db"))
    database.conn.execute("UPDATE sources SET token_checked_at = '2026-01-01T00:00:00+00:00'")
    box = secret_box.SecretBox()
    with patch("src.api.sources.requests.get", meta), patch("src.ingestion.adapters.requests.get", meta):
        sources.Scheduler(lambda: database, lambda: box, 60).run_once()
    s = sources.list_sources(database)[0]
    assert s["token_checked_at"] > "2026-01-01" and s["archived_posts"] == 1
    assert [a["by"] for a in sources.activity(database)][:2] == ["Automatic", "Automatic"]


def test_the_secret_box_round_trips_and_refuses_the_wrong_key():
    box = secret_box.SecretBox(key=Fernet.generate_key().decode())
    blob = box.protect(GOOD)
    assert GOOD.encode() not in blob and box.unprotect(blob) == GOOD
    with pytest.raises(secret_box.SecretBoxError):
        secret_box.SecretBox(key=Fernet.generate_key().decode()).unprotect(blob)
    with pytest.raises(secret_box.SecretBoxError):
        secret_box.SecretBox(key="not-a-key")
    if os.name == "nt":
        windows = secret_box.SecretBox(key="")
        assert windows.method == "windows-dpapi" and windows.unprotect(windows.protect(GOOD)) == GOOD


def test_the_page_takes_tokens_in_password_fields_and_clears_them():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page = open(os.path.join(root, "frontend", "index.html"), encoding="utf-8").read()
    app = open(os.path.join(root, "frontend", "js", "app.js"), encoding="utf-8").read()
    assert 'id="source-token" type="password"' in page and 'id="token-input" type="password"' in page
    block = app[app.index("// ---- Sources (administrators"):app.index("// ---- guided tour")]
    assert 'tokenBox.value = ""' in block and '$("token-input").value = ""' in block
    assert "token_hint" in block and "localStorage" not in block and "console." not in block


# ---- record the cutoff and freeze the checker (docs/plans/past_and_new_posts_plan.md, 2026-11-16) ----------------------
class Git:
    """git answered from a dict; records every call."""

    def __init__(self, dirty="", tags=""):
        self.calls, self.dirty, self.tags = [], dirty, tags

    def __call__(self, args):
        self.calls.append(args)
        out = {"status": self.dirty, "rev-parse": "0123456789abcdef0123456789abcdef01234567"}.get(args[0], "")
        if args[:2] == ["tag", "--list"]:
            out = self.tags
        return Mock(returncode=0, stdout=out, stderr="")


def test_recording_the_cutoff_tags_the_code_writes_the_file_once_and_unlocks_fetching(on, monkeypatch):
    git = Git()
    monkeypatch.setattr(sources, "GIT_RUN", git)
    monkeypatch.setattr(sources, "RUNNING_COMMIT", "0123456789abcdef0123456789abcdef01234567")
    with _client(on) as c:
        _admin(c)
        r = c.post("/admin/sources/cutoff", json={"day": "2026-11-10"})
        assert r.status_code == 200, r.text
        frozen = r.json()["frozen"]
        assert frozen["cutoff_date"] == "2026-11-10" and frozen["frozen_tag"] == "final-eval-frozen"
        assert frozen["frozen_commit"].startswith("0123456")
        assert ["tag", "-a", "final-eval-frozen"] == git.calls[-1][:3]
        record = json.load(open(sources.RESERVATION, encoding="utf-8"))
        assert "replaces D5" in record["decision"] and record["cutoff_starts"] == "2026-11-10T00:00:00+08:00"
        s = c.get("/admin/sources").json()
        assert s["cutoff"] == "2026-11-10" and s["frozen"]["frozen_tag"] == "final-eval-frozen"
        assert s["activity"][0]["action"] == "cutoff recorded"
        again = c.post("/admin/sources/cutoff", json={"day": "2026-11-11"})
        assert again.status_code == 400 and "already recorded" in again.json()["detail"]


def test_the_cutoff_is_refused_with_uncommitted_code_a_future_date_or_an_existing_tag(on, tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "RUNNING_COMMIT", "0123456789abcdef0123456789abcdef01234567")

    def record(day, git):
        return sources.record_cutoff(db.Database(str(tmp_path / "t.db")), None, day, today=dt.date(2026, 11, 16),
                                     git_run=git)
    with pytest.raises(sources.SourcesError, match="not committed"):
        record("2026-11-10", Git(dirty=" M src/api/main.py"))
    with pytest.raises(sources.SourcesError, match="future"):
        record("2026-11-17", Git())
    with pytest.raises(sources.SourcesError, match="YYYY-MM-DD"):
        record("10/11/2026", Git())
    with pytest.raises(sources.SourcesError, match="already exists"):
        record("2026-11-10", Git(tags="final-eval-frozen"))
    monkeypatch.setattr(sources, "RUNNING_COMMIT", "an older commit")
    with pytest.raises(sources.SourcesError, match="Restart the checker"):
        record("2026-11-10", Git())
    monkeypatch.setattr(sources, "RUNNING_COMMIT", "0123456789abcdef0123456789abcdef01234567")
    assert not os.path.exists(sources.RESERVATION)
    git = Git()
    with patch.object(sources, "set_cutoff", side_effect=OSError("disk full")), pytest.raises(OSError):
        record("2026-11-10", git)
    assert git.calls[-1] == ["tag", "-d", "final-eval-frozen"]          # the tag is removed again


def test_with_import_after_cutoff_on_new_posts_are_imported_not_held_back(on, monkeypatch):
    pg.set_cutoff(sources.RESERVATION, "2026-11-01")
    monkeypatch.setenv(pg.IMPORT_ENV, "1")
    meta = Meta(posts={PAGE: [post(1, "Enrollment for the second semester opens on November 3."),
                              post(2, "Midterm examinations start on November 10.", day="2026-11-02")]})
    with _client(on) as c:
        _admin(c)
        _add(c, meta)
        with patch("src.api.sources.requests.get", meta), patch("src.ingestion.adapters.requests.get", meta):
            result = c.post("/admin/sources/synthetic_dyci_page/fetch").json()["result"]["synthetic_dyci_page"]
        assert result["status"] == "ok" and result["added"] == 2 and result["reserved_for_final_evaluation"] == 0


# ---- past posts (backfill), docs/plans/past_and_new_posts_plan.md section 2 -----------------------------------------
class PagedMeta(Meta):
    """Meta with a page history: /posts honours `until` and `limit` and splits into pages with paging.next."""

    def __init__(self, n_posts, newest=dt.datetime(2026, 11, 1, 9, tzinfo=dt.timezone.utc), **kw):
        super().__init__(**kw)
        self.history = [{"id": f"{PAGE}_{i}", "message": f"Announcement number {i}: the office opens at {i % 12 + 1} AM.",
                         "created_time": (newest - dt.timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%S+0000"),
                         "permalink_url": f"https://www.facebook.com/synthetic/posts/{i}"} for i in range(n_posts)]
        self.post_requests, self.rate_limited = [], False

    def __call__(self, url, params=None, timeout=None):
        from urllib.parse import parse_qs, urlparse
        if "/posts" not in url:
            return super().__call__(url, params, timeout)
        q = dict(params or {k: v[0] for k, v in parse_qs(urlparse(url).query).items()})
        self.post_requests.append(q)
        r = Mock(status_code=200)
        if self.rate_limited:
            r.status_code = 400
            r.json = Mock(return_value={"error": {"code": 4, "message": "Application request limit reached"}})
            return r
        until = int(q["until"]) if q.get("until") else None
        rows = [p for p in self.history if until is None
                or dt.datetime.strptime(p["created_time"], "%Y-%m-%dT%H:%M:%S%z").timestamp() <= until]
        offset, limit = int(q.get("offset", 0)), int(q["limit"])
        page = {"data": rows[offset:offset + limit]}
        if offset + limit < len(rows):
            nxt = {k: v for k, v in q.items() if k != "access_token"}
            nxt.update(offset=offset + limit, access_token=q["access_token"])
            page["paging"] = {"next": f"https://graph.facebook.com/v21.0/{PAGE}/posts?" +
                              "&".join(f"{k}={v}" for k, v in nxt.items())}
        r.json = Mock(return_value=page)
        return r


@pytest.fixture
def paged(on, monkeypatch):
    pg.set_cutoff(sources.RESERVATION, "2026-11-01")
    monkeypatch.setenv(pg.IMPORT_ENV, "1")
    monkeypatch.setenv(sources.BACKFILL_FROM_ENV, "2024-08-01")
    return on


def _step(meta, now=None):
    from src.api.main import _database, _secret_box
    with patch("src.api.sources.requests.get", meta), patch("src.ingestion.adapters.requests.get", meta):
        return sources.backfill(_database(), _secret_box(), now=now)


def _row(key="synthetic_dyci_page"):
    from src.api.main import _database
    return next(s for s in sources.list_sources(_database()) if s["key"] == key)


def test_past_posts_are_read_backwards_in_steps_until_the_first_post_then_cost_nothing(paged):
    meta = PagedMeta(250)
    with _client(paged) as c:
        _admin(c)
        _add(c, meta)
        first = _step(meta)["synthetic_dyci_page"]
        assert first["status"] == "ok" and first["added"] == 200 and len(meta.post_requests) == 2
        assert meta.post_requests[0]["limit"] == 100 and "until" not in meta.post_requests[0]   # empty archive: newest
        assert not _row()["backfill_done"] and _row()["backfill_note"] == "going"
        second = _step(meta)["synthetic_dyci_page"]
        assert second["added"] == 50
        assert int(meta.post_requests[2]["until"]) == int(dt.datetime.strptime(
            meta.history[199]["created_time"], "%Y-%m-%dT%H:%M:%S%z").timestamp())        # continues where it stopped
        row = _row()
        assert row["backfill_done"] and row["backfill_added"] == 250 and "first post" in row["backfill_note"]
        assert row["archived_posts"] == 250
        calls = len(meta.post_requests)
        assert _step(meta) == {} and len(meta.post_requests) == calls                     # finished: no request
        s = c.get("/admin/sources").json()
        assert s["activity"][0]["action"] == "past posts finished" and s["backfill_from"] == "2024-08-01"


def test_past_posts_start_below_what_the_hourly_fetch_already_has_and_stop_at_the_floor(paged, monkeypatch):
    monkeypatch.setenv(sources.BACKFILL_FROM_ENV, "2026-06-01")
    meta = PagedMeta(400)
    with _client(paged) as c:
        _admin(c)
        _add(c, meta)
        with patch("src.api.sources.requests.get", meta), patch("src.ingestion.adapters.requests.get", meta):
            assert c.post("/admin/sources/synthetic_dyci_page/fetch").status_code == 200   # newest 125
        meta.post_requests.clear()
        _step(meta)
        oldest_archived = dt.datetime.strptime(meta.history[124]["created_time"], "%Y-%m-%dT%H:%M:%S%z")
        assert int(meta.post_requests[0]["until"]) == int(oldest_archived.timestamp())
        row = _row()
        assert row["backfill_done"] and "2026-06-01" in row["backfill_note"]               # 2026-11-01 minus 199 days
        assert row["archived_posts"] == 125 + row["backfill_added"]


def test_a_rate_limit_pauses_that_page_for_an_hour(paged):
    meta = PagedMeta(50)
    meta.rate_limited = True
    now = dt.datetime(2026, 11, 16, 10, tzinfo=dt.timezone.utc)
    with _client(paged) as c:
        _admin(c)
        _add(c, meta)
        assert _step(meta, now=now)["synthetic_dyci_page"]["status"] == "rate_limited"
        assert _row()["backfill_note"].startswith("paused")
        meta.rate_limited, calls = False, len(meta.post_requests)
        assert _step(meta, now=now + dt.timedelta(minutes=30)) == {} and len(meta.post_requests) == calls
        assert _step(meta, now=now + dt.timedelta(minutes=61))["synthetic_dyci_page"]["added"] == 50
        assert _row()["backfill_done"]


def test_no_step_while_the_page_is_being_fetched_and_none_before_the_cutoff(on):
    meta = PagedMeta(10)
    with _client(on) as c:
        _admin(c)
        _add(c, meta)
        with pytest.raises(sources.SourcesError, match="cutoff"):
            _step(meta)
        pg.set_cutoff(sources.RESERVATION, "2026-11-01")
        lock = sources._page_lock("synthetic_dyci_page")
        with lock:
            assert _step(meta) == {}
        assert meta.post_requests == []


def test_the_scheduler_runs_past_posts_on_their_own_interval():
    s = sources.Scheduler(lambda: None, lambda: None, minutes=0, backfill_minutes=10)
    s.start()
    assert s._thread is None and s._backfill_thread is not None
    s.stop()


def _archived_days(key="synthetic_dyci_page"):
    from src.ingestion.corpus_storage import load_corpus
    from src.ingestion.pages import archive_path
    return sorted(r.published_date[:10] for r in load_corpus(archive_path(sources.archive_dir(), key)))


def test_posts_older_than_the_floor_are_never_kept(paged, monkeypatch):
    monkeypatch.setenv(sources.BACKFILL_FROM_ENV, "2026-06-01")
    meta = PagedMeta(400)                                # one post a day back from 2026-11-01
    with _client(paged) as c:
        _admin(c)
        _add(c, meta)
        _step(meta)
        days = _archived_days()
        assert days[0] >= "2026-06-01" and _row()["backfill_done"]
        assert len(days) == _row()["backfill_added"] == 154                                   # 2026-06-01 .. 2026-11-01


def test_archived_posts_older_than_the_floor_are_removed_and_logged(paged, monkeypatch):
    meta = PagedMeta(120)                                # back to 2026-07-04
    with _client(paged) as c:
        _admin(c)
        _add(c, meta)
        _step(meta)
        assert len(_archived_days()) == 120 and _row()["backfill_done"]
        monkeypatch.setenv(sources.BACKFILL_FROM_ENV, "2026-10-01")
        calls = len(meta.post_requests)
        _step(meta)
        assert _archived_days()[0] >= "2026-10-01" and len(_archived_days()) == 32
        assert len(meta.post_requests) == calls                                                # a local file only
        log = c.get("/admin/sources").json()["activity"][0]
        assert log["action"] == "older posts removed" and log["detail"].startswith("88 post(s) from before 2026-10-01")
        _step(meta)
        assert c.get("/admin/sources").json()["activity"][0]["action"] == "older posts removed"  # logged once, not again
        assert len([a for a in c.get("/admin/sources").json()["activity"] if a["action"] == "older posts removed"]) == 1
