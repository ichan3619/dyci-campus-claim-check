"""Fix B: live fetch status for administrators (src/api/fetch_status.py, src/ingestion/facebook_page.py fetch log)."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.api import fetch_status as fs
from src.ingestion import facebook_page as fp

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)


def refresh(**kw):
    base = dict(enabled=True, interval_minutes=10, last_run="2026-11-17T11:55:00Z", last_added=0, last_error=None)
    return SimpleNamespace(**{**base, **kw})


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(fs.ENV, raising=False)
    assert fs.enabled() is False


def test_healthy():
    log = [{"at": "2026-11-17T11:55:00Z", "ok": True, "skipped_no_text": 0}]
    out = fs.build(refresh(), log, {"posts": 19, "newest_post": "2026-11-17", "file_readable": True}, NOW)
    assert out["warnings"] == [] and out["auto_fetch"]["last_success"] == "2026-11-17T11:55:00Z"


def test_failed_and_stale():
    log = [{"at": "2026-11-17T11:55:00Z", "ok": False, "error": "token expired"},
           {"at": "2026-11-17T10:00:00Z", "ok": True}]
    out = fs.build(refresh(last_error="(code=190) -> the access token is expired"), log, {"file_readable": True}, NOW)
    assert any("last fetch failed" in w for w in out["warnings"])
    assert any("No successful fetch in the last 30 minutes" in w for w in out["warnings"])


def test_picture_only_posts_are_reported():
    log = [{"at": "2026-11-17T11:55:00Z", "ok": True, "skipped_no_text": 2}]
    out = fs.build(refresh(), log, {"file_readable": True}, NOW)
    assert any("2 post(s) were skipped because they have no text" in w for w in out["warnings"])


def test_auto_fetch_off():
    out = fs.build(refresh(enabled=False), [], {"file_readable": True}, NOW)
    assert any("Auto-fetch is off" in w for w in out["warnings"])


def test_served(tmp_path):
    p = tmp_path / "live.jsonl"
    p.write_text(json.dumps({"published_date": "2026-11-10T08:00:00+0000"}) + "\n\n"
                 + json.dumps({"published_date": "2026-11-17T08:05:53+0000"}) + "\n", encoding="utf-8")
    assert fs.served(str(p)) == {"posts": 2, "newest_post": "2026-11-17", "file_readable": True}
    assert fs.served(str(tmp_path / "missing.jsonl"))["file_readable"] is False


def test_fetch_log_records_counts_ids_and_scrubbed_errors(monkeypatch, tmp_path):
    fp._FETCH_LOG.clear()
    result = SimpleNamespace(added=1, edited=0, retired=0, unusable_skipped=1, redundant_skipped=0, flagged_for_review=0,
                             retire_refused=None)

    def ok(adapter, path, **kw):
        adapter.skipped_posts = [{"post_id": "1_2", "reason": "missing or empty 'message' field", "text": "secret"}]
        return result
    monkeypatch.setattr(fp, "refresh_corpus_from_facebook", ok)
    fp.refresh_from_mock_page(str(tmp_path / "c.jsonl"), "123", "TOKEN123")

    def boom(adapter, path, **kw):
        raise RuntimeError("Graph API error (code=190) access_token=TOKEN123 failed")
    monkeypatch.setattr(fp, "refresh_corpus_from_facebook", boom)
    with pytest.raises(RuntimeError):
        fp.refresh_from_mock_page(str(tmp_path / "c.jsonl"), "123", "TOKEN123")
    newest, older = fp.fetch_log()
    assert older["ok"] and older["added"] == 1 and older["skipped_no_text"] == 1
    assert older["skipped_posts"] == [{"post_id": "1_2", "reason": "missing or empty 'message' field"}]
    assert not newest["ok"] and "TOKEN123" not in newest["error"] and "expired" in newest["error"]
    fp._FETCH_LOG.clear()


def test_log_is_bounded():
    fp._FETCH_LOG.clear()
    for i in range(fp.FETCH_LOG_SIZE + 5):
        fp._record_fetch(None, SimpleNamespace(skipped_posts=[]), f"e{i}")
    assert len(fp.fetch_log()) == fp.FETCH_LOG_SIZE and fp.fetch_log()[0]["error"] == f"e{fp.FETCH_LOG_SIZE + 4}"
    fp._FETCH_LOG.clear()


def test_endpoint_is_404_when_off(monkeypatch):
    from fastapi import HTTPException
    import src.api.main as api
    monkeypatch.delenv(fs.ENV, raising=False)
    with pytest.raises(HTTPException) as e:
        api.admin_fetch_status(SimpleNamespace())
    assert e.value.status_code == 404
