"""
Live fetch status for administrators (fix B, owner 2026-11-17). CAMPUS_VERIFY_FETCH_STATUS=1 (default off).

The checker's evidence from the test page arrives through the auto-fetch every CAMPUS_VERIFY_FB_REFRESH_MINUTES. When it
fails (an expired token) or skips posts (a post that is only a picture has no text), nothing showed it: the public
/health is trimmed and the Sources page shows the archive. GET /admin/fetch-status gives admins the last fetch, the
recent fetches, what the checker is serving from the page, and plain warnings. Display only; nothing is changed.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

ENV = "CAMPUS_VERIFY_FETCH_STATUS"
STALE_INTERVALS = 3          # no successful fetch in this many intervals -> warning


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def served(path: str) -> dict:
    """How many posts the checker serves from the test-page file, and the newest post date (no post text)."""
    count, newest = 0, None
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                count += 1
                day = (json.loads(line).get("published_date") or "")[:10]
                if day and (newest is None or day > newest):
                    newest = day
    except (OSError, ValueError):
        return {"posts": None, "newest_post": None, "file_readable": False}
    return {"posts": count, "newest_post": newest, "file_readable": True}


def build(refresh, log: list, served_info: dict, now: Optional[datetime] = None) -> dict:
    """The response: the auto-fetch status, recent fetches (newest first), what is served, and warnings."""
    now = now or datetime.now(timezone.utc)
    website = [e for e in log if e.get("source") == "dyci_website"]      # rule W reads, shown but kept apart
    log_fb = [e for e in log if e.get("source") != "dyci_website"]
    warnings = []
    if website and not website[0].get("ok"):
        warnings.append("The last read of DYCI's website failed: " + (website[0].get("error") or "unknown error")
                        + " The website posts already stored are still used.")
    log, full_log = log_fb, log
    last_ok = next((e["at"] for e in log if e.get("ok")), None)
    if not refresh.enabled:
        warnings.append("Auto-fetch is off: new test-page posts are not added (META_PAGE_ID / META_ACCESS_TOKEN not set "
                        "when the checker started).")
    else:
        if refresh.last_error:
            warnings.append("The last fetch failed: " + refresh.last_error)
        limit = timedelta(minutes=max(refresh.interval_minutes, 1) * STALE_INTERVALS)
        ok_at = _parse(last_ok)
        if log and (ok_at is None or now - ok_at > limit):
            warnings.append(f"No successful fetch in the last {int(limit.total_seconds() // 60)} minutes.")
    no_text = sum(e.get("skipped_no_text") or 0 for e in log if e.get("ok"))
    if no_text:
        warnings.append(f"{no_text} post(s) were skipped because they have no text (for example a post that is only "
                        "a picture). The checker cannot use them as evidence.")
    if served_info.get("file_readable") is False:
        warnings.append("The served test-page file could not be read.")
    return {"auto_fetch": {"enabled": refresh.enabled, "interval_minutes": refresh.interval_minutes,
                           "last_run": refresh.last_run, "last_added": refresh.last_added,
                           "last_error": refresh.last_error, "last_success": last_ok},
            "recent": full_log, "website_last": website[0] if website else None, "served": served_info,
            "warnings": warnings,
            "note": "Recent fetches are kept in memory since the checker last started."}
