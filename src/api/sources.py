"""
Facebook pages as sources, managed by administrators on the Sources page (CAMPUS_VERIFY_SOURCES, default off; needs
CAMPUS_VERIFY_AUTH=google and CAMPUS_VERIFY_DB_FILE; panel request for an admin part, owner approval 2026-11-04).

WHAT IT KEEPS (database schema v5)
- sources: each page's numeric ID, the name Meta reports for it, its tier (official / recognized / unverified),
  whether it is fetched, its token ENCRYPTED (src/api/secret_box.py) with only the last 4 characters kept readable,
  the token's last check and the last fetch's summary.
- source_actions: who added, changed, checked, fetched or removed a page, and when (no token, ever).

WHAT IT DOES
- check(): asks Meta whether the token still reads the page (GET /{page-id}?fields=id,name). With META_APP_ID and
  META_APP_SECRET set (the app's own credentials, environment only) it also reads the token's expiry from
  /debug_token. Status: ok, expiring (within WARN_DAYS), invalid, no_access, rate_limited or unreachable.
- fetch(): runs the existing multi-page refresh (src/ingestion/pages.py refresh_pages, unchanged) for the included
  pages, with their tokens decrypted in memory only. Posts go to the ARCHIVE (data/corpus/archive/<key>.jsonl), which
  the checker does not search: serving them is the separate served-collection step (scale plan S3). The final-
  evaluation reservation (D5) applies: refresh_pages refuses to run before the cutoff is recorded, and reserved posts
  are never written.
- Scheduler (CAMPUS_VERIFY_SOURCES_FETCH_MINUTES, default 0 = off): fetches every N minutes and checks every token
  once a day, in a background thread; a page whose token is invalid is skipped until an administrator replaces it.
- backfill() (CAMPUS_VERIFY_SOURCES_BACKFILL_MINUTES, default 0 = off; docs/plans/past_and_new_posts_plan.md): every N
  minutes, one small step backwards through each page's PAST posts (2 requests of 100 posts), starting below the
  oldest post already archived, until CAMPUS_VERIFY_SOURCES_BACKFILL_FROM (default 2024-08-01) or the page's first
  post. Posts older than that date are never kept: a step leaves them out, and any already archived are removed. Add-only; the checkpoint is kept in the database (schema v7). A finished page costs no request; a rate limit
  pauses that page for an hour. A step never runs while that page is being fetched (one lock per page).

A token can be renewed only by a person who manages the page (Meta issues no refresh token): the check warns early,
and replacing a token is one form on the Sources page.
"""
import datetime as _dt
import json
import logging
import os
import re
import subprocess
import threading
from typing import Callable, Dict, List, Optional

import requests

from src.ingestion.adapters import DEFAULT_GRAPH_API_VERSION
from src.ingestion.facebook_page import scrub
from src.ingestion.pages import (DEFAULT_ARCHIVE, DEFAULT_RESERVATION, MANILA, PAGE_ID_RE, PageConfig, PagesConfigError,
                                 archive_path, load_cutoff, refresh_pages, set_cutoff, write_run_record)

ENV = "CAMPUS_VERIFY_SOURCES"
FETCH_ENV = "CAMPUS_VERIFY_SOURCES_FETCH_MINUTES"
ARCHIVE_ENV = "CAMPUS_VERIFY_SOURCES_ARCHIVE"
BACKFILL_ENV = "CAMPUS_VERIFY_SOURCES_BACKFILL_MINUTES"
BACKFILL_FROM_ENV = "CAMPUS_VERIFY_SOURCES_BACKFILL_FROM"
BACKFILL_FROM = "2024-08-01"                             # plan B2: the current window plus one school year
BACKFILL_PAUSE_MINUTES = 60                              # after a rate limit
APP_ID_ENV, APP_SECRET_ENV = "META_APP_ID", "META_APP_SECRET"
TIERS = ("official", "recognized", "unverified")
WARN_DAYS = 14
CHECK_EVERY_HOURS = 24
GRAPH = "https://graph.facebook.com"
TIMEOUT = 15
INVALID_CODES = {"190", "102"}
NO_ACCESS_CODES = {"10", "100", "200", "803"}
RATE_CODES = {"4", "17", "32", "613"}
_COLUMNS = ("id, key, name, page_id, tier, include, token_hint, token_set_at, token_status, token_detail, "
            "token_checked_at, token_expires_at, last_fetch_at, last_fetch, created_at, backfill_until, backfill_done, "
            "backfill_added, backfill_at, backfill_note, backfill_retry_after")
logger = logging.getLogger(__name__)


class SourcesError(ValueError):
    """A request the Sources page cannot carry out; the message is safe to show (never holds a token)."""


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def fetch_minutes() -> float:
    try:
        return max(0.0, float(os.environ.get(FETCH_ENV, "0")))
    except ValueError:
        return 0.0


def archive_dir() -> str:
    return os.environ.get(ARCHIVE_ENV, "").strip() or DEFAULT_ARCHIVE


def backfill_minutes() -> float:
    try:
        return max(0.0, float(os.environ.get(BACKFILL_ENV, "0")))
    except ValueError:
        return 0.0


def backfill_from() -> _dt.datetime:
    day = os.environ.get(BACKFILL_FROM_ENV, "").strip() or BACKFILL_FROM
    return _dt.datetime.combine(_dt.date.fromisoformat(day), _dt.time(), MANILA)


_page_locks: Dict[str, threading.Lock] = {}
_page_locks_guard = threading.Lock()


def _page_lock(key: str) -> threading.Lock:
    """One lock per page: a fetch and a past-posts step never write the same archive file at once."""
    with _page_locks_guard:
        return _page_locks.setdefault(key, threading.Lock())


RESERVATION = DEFAULT_RESERVATION                        # the D5 cutoff file (src/ingestion/pages.py)


def cutoff_day() -> Optional[str]:
    cutoff = load_cutoff(RESERVATION)
    return cutoff.date().isoformat() if cutoff else None


FROZEN_TAG = "final-eval-frozen"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FROZEN_PATHS = ("src", "frontend", "scripts", "requirements.txt", "requirements-lock.txt")
GIT_RUN = None                                           # tests replace how git is run


def _started_commit() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


# The commit of the code this process loaded (src/api/main.py imports this module at start-up). Record cutoff refuses
# when HEAD has moved since, so the tag is always the code that is running.
RUNNING_COMMIT = _started_commit()


def cutoff_info() -> Optional[Dict]:
    """The recorded cutoff and the frozen checker version, or None (shown on the Sources page)."""
    if load_cutoff(RESERVATION) is None:
        return None
    with open(RESERVATION, encoding="utf-8") as f:
        r = json.load(f)
    return {k: r.get(k) for k in ("cutoff_date", "recorded_at", "frozen_tag", "frozen_commit")}


def _git(*args, git_run=None) -> str:
    run = git_run or GIT_RUN or (lambda a: subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                               timeout=30))
    r = run(list(args))
    if r.returncode != 0:
        raise SourcesError(f"git {args[0]} failed: {(r.stderr or r.stdout or '').strip()[:200]}")
    return (r.stdout or "").strip()


def record_cutoff(database, admin: Optional[Dict], day: str, today: Optional[_dt.date] = None, git_run=None) -> Dict:
    """Record the final-evaluation cutoff once and freeze the checker: Git tag FROZEN_TAG on the current commit.
    Refused while the code has uncommitted changes or was committed after this process started (the tag must be the
    code that is running), when the date is in the future, or when a cutoff or the tag already exists. The tag is created first and removed again if the cutoff cannot
    be written."""
    try:
        cutoff = _dt.date.fromisoformat((day or "").strip())
    except ValueError:
        raise SourcesError("Enter the date as YYYY-MM-DD.") from None
    if cutoff > (today or _dt.datetime.now(MANILA).date()):
        raise SourcesError("The cutoff cannot be in the future.")
    if load_cutoff(RESERVATION) is not None:
        raise SourcesError(f"A cutoff is already recorded ({cutoff_day()}). It cannot be changed.")
    if _git("tag", "--list", FROZEN_TAG, git_run=git_run):
        raise SourcesError(f"The Git tag {FROZEN_TAG} already exists. Remove it or ask the project owner.")
    if _git("status", "--porcelain", "--", *FROZEN_PATHS, git_run=git_run):
        raise SourcesError("The code has changes that are not committed. Commit them first, so the frozen version is "
                           "exactly the code that is running.")
    commit = _git("rev-parse", "HEAD", git_run=git_run)
    if commit != RUNNING_COMMIT:
        raise SourcesError("The code was committed after the checker started. Restart the checker first, so the "
                           "frozen version is exactly the code that is running.")
    _git("tag", "-a", FROZEN_TAG, "-m", f"Checker frozen for the final evaluation; cutoff {cutoff.isoformat()} "
         "(docs/plans/past_and_new_posts_plan.md section 1)", commit, git_run=git_run)
    try:
        set_cutoff(RESERVATION, cutoff.isoformat(),
                   decision="owner 2026-11-16 (replaces D5 = b): the checker is frozen as Git tag "
                            f"{FROZEN_TAG} when the cutoff is recorded; the final evaluation runs that version. "
                            "Posts from the cutoff on are imported when CAMPUS_VERIFY_IMPORT_AFTER_CUTOFF=1",
                   extra={"frozen_tag": FROZEN_TAG, "frozen_commit": commit})
    except Exception:
        _git("tag", "-d", FROZEN_TAG, git_run=git_run)
        raise
    with database.lock:
        _log(database, admin["id"] if admin else None, "cutoff recorded", "-",
             f"{cutoff.isoformat()}; checker frozen as {FROZEN_TAG} ({commit[:7]})")
    return cutoff_info()


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso(t: _dt.datetime) -> str:
    return t.replace(microsecond=0).isoformat()


# ---- asking Meta -----------------------------------------------------------------------------------------------------
def _graph_get(path: str, params: dict, http_get: Callable) -> dict:
    """The JSON of one Graph API GET; a Graph error comes back as {"error": {...}}; a network failure raises."""
    r = http_get(f"{GRAPH}/{DEFAULT_GRAPH_API_VERSION}/{path}", params=params, timeout=TIMEOUT)
    try:
        return r.json()
    except ValueError:
        return {"error": {"message": f"HTTP {r.status_code}, not JSON", "code": None}}


def check_token(page_id: str, token: str, http_get: Optional[Callable] = None, now: Optional[_dt.datetime] = None,
                environ=os.environ) -> Dict:
    """{'status', 'detail', 'name', 'expires_at'} for this token on this page. Never raises; never returns the token."""
    now, http_get = now or _now(), http_get or requests.get
    try:
        page = _graph_get(page_id, {"fields": "id,name", "access_token": token}, http_get)
    except requests.exceptions.RequestException as e:
        return {"status": "unreachable", "detail": scrub(f"Meta could not be reached: {type(e).__name__}", token),
                "name": None, "expires_at": None}
    err = page.get("error")
    if err:
        code = str(err.get("code"))
        status = ("invalid" if code in INVALID_CODES else "rate_limited" if code in RATE_CODES
                  else "no_access" if code in NO_ACCESS_CODES else "invalid")
        advice = {"invalid": "The token is expired or was revoked. Ask a page admin for a new token and replace it.",
                  "no_access": "The token works but cannot read this page. Check the page ID and the token's "
                               "permissions (pages_read_engagement).",
                  "rate_limited": "Meta is limiting requests. Try again later."}[status]
        return {"status": status, "detail": scrub(f"{advice} (Meta: {err.get('message', '')}, code {code})", token)[:400],
                "name": None, "expires_at": None}
    if str(page.get("id")) != str(page_id):
        return {"status": "no_access", "detail": "Meta answered for a different page. Check the page ID.",
                "name": None, "expires_at": None}
    result = {"status": "ok", "detail": "The token reads this page.", "name": page.get("name"), "expires_at": None}
    app_id, secret = environ.get(APP_ID_ENV, "").strip(), environ.get(APP_SECRET_ENV, "").strip()
    if app_id and secret:
        try:
            info = _graph_get("debug_token", {"input_token": token, "access_token": f"{app_id}|{secret}"},
                              http_get).get("data") or {}
        except requests.exceptions.RequestException:
            info = {}
        ends = [int(info[k]) for k in ("expires_at", "data_access_expires_at") if int(info.get(k) or 0) > 0]
        if info.get("is_valid") is False:
            result.update(status="invalid", detail="Meta reports this token as no longer valid. Replace it.")
        elif ends:
            when = _dt.datetime.fromtimestamp(min(ends), _dt.timezone.utc)
            result["expires_at"] = _iso(when)
            days = (when - now).total_seconds() / 86400
            if days <= WARN_DAYS:
                result.update(status="expiring", detail=f"The token stops working in {max(0, int(days))} day(s). "
                                                        "Ask a page admin for a new token and replace it.")
        elif info:
            result["expires_at"] = "never"
    return result


# ---- the database --------------------------------------------------------------------------------------------------
def _row(r) -> Dict:
    d = dict(zip([c.strip() for c in _COLUMNS.split(",")], r))
    d["include"] = bool(d["include"])
    d["backfill_done"] = bool(d["backfill_done"])
    d["last_fetch"] = json.loads(d["last_fetch"]) if d["last_fetch"] else None
    return d


def _log(database, admin_id: Optional[int], action: str, key: str, detail: str = "") -> None:
    database.conn.execute("INSERT INTO source_actions (at, admin_user_id, action, source_key, detail) "
                          "VALUES (?, ?, ?, ?, ?)", (_iso(_now()), admin_id, action, key, detail[:300]))


def get(database, key: str) -> Dict:
    with database.lock:
        r = database.conn.execute(f"SELECT {_COLUMNS} FROM sources WHERE key = ?", (key,)).fetchone()
    if r is None:
        raise SourcesError("No such page.")
    return _row(r)


def list_sources(database, archive: Optional[str] = None) -> List[Dict]:
    archive = archive or archive_dir()
    with database.lock:
        rows = [_row(r) for r in database.conn.execute(f"SELECT {_COLUMNS} FROM sources ORDER BY name")]
    for s in rows:
        path = archive_path(archive, s["key"])
        s["archived_posts"] = sum(1 for line in open(path, encoding="utf-8") if line.strip()) if os.path.exists(path) else 0
    return rows


def activity(database, limit: int = 30) -> List[Dict]:
    with database.lock:
        rows = database.conn.execute(
            "SELECT a.at, a.action, a.source_key, a.detail, u.name, u.email FROM source_actions a "
            "LEFT JOIN users u ON u.id = a.admin_user_id ORDER BY a.id DESC LIMIT ?", (limit,)).fetchall()
    return [{"at": r[0], "action": r[1], "source": r[2], "detail": r[3],
             "by": (r[4] or r[5]) if (r[4] or r[5]) else "Automatic"} for r in rows]


def _clean_token(token: str) -> str:
    token = (token or "").strip()
    if not 20 <= len(token) <= 1000 or re.search(r"\s", token):
        raise SourcesError("That does not look like a page access token. Paste the whole token, without spaces.")
    return token


def _key_for(database, name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")[:30] or "page"
    if not base[0].isalpha():
        base = "p_" + base
    key, n = base, 2
    with database.lock:
        while database.conn.execute("SELECT 1 FROM sources WHERE key = ?", (key,)).fetchone():
            key, n = f"{base[:27]}_{n}", n + 1
    return key


def add(database, box, admin: Dict, page_id: str, tier: str, token: str, http_get: Optional[Callable] = None) -> Dict:
    """Add a page after Meta confirms the token reads it. The page's name comes from Meta, not from the form."""
    page_id, tier, token = (page_id or "").strip(), (tier or "").strip(), _clean_token(token)
    if not PAGE_ID_RE.match(page_id):
        raise SourcesError("The page ID is the page's number (5 to 25 digits), found under the page's About > "
                           "Page transparency, or in Meta Business Suite.")
    if tier not in TIERS:
        raise SourcesError(f"Choose a tier: {', '.join(TIERS)}.")
    with database.lock:
        if database.conn.execute("SELECT 1 FROM sources WHERE page_id = ?", (page_id,)).fetchone():
            raise SourcesError("This page is already a source. Replace its token instead.")
    result = check_token(page_id, token, http_get)
    if result["status"] not in ("ok", "expiring"):
        raise SourcesError(f"Not added: {result['detail']}")
    name = (result["name"] or f"Page {page_id}")[:120]
    key, now = _key_for(database, name), _iso(_now())
    with database.lock:
        database.conn.execute(
            "INSERT INTO sources (key, name, page_id, tier, include, token, token_hint, token_set_at, token_status, "
            "token_detail, token_checked_at, token_expires_at, created_at) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, name, page_id, tier, box.protect(token), token[-4:], now, result["status"], result["detail"], now,
             result["expires_at"], now))
        _log(database, admin["id"], "added", key, f"{name} ({page_id}), tier {tier}")
    return get(database, key)


def replace_token(database, box, admin: Dict, key: str, token: str, http_get: Optional[Callable] = None) -> Dict:
    s, token = get(database, key), _clean_token(token)
    result = check_token(s["page_id"], token, http_get)
    if result["status"] not in ("ok", "expiring"):
        raise SourcesError(f"Token not replaced: {result['detail']}")
    now = _iso(_now())
    with database.lock:
        database.conn.execute(
            "UPDATE sources SET token = ?, token_hint = ?, token_set_at = ?, token_status = ?, token_detail = ?, "
            "token_checked_at = ?, token_expires_at = ? WHERE key = ?",
            (box.protect(token), token[-4:], now, result["status"], result["detail"], now, result["expires_at"], key))
        _log(database, admin["id"], "token replaced", key, f"ending in {token[-4:]}")
    return get(database, key)


def update(database, admin: Dict, key: str, include: Optional[bool] = None, tier: Optional[str] = None) -> Dict:
    s = get(database, key)
    if tier is not None and tier not in TIERS:
        raise SourcesError(f"Choose a tier: {', '.join(TIERS)}.")
    with database.lock:
        if include is not None and include != s["include"]:
            database.conn.execute("UPDATE sources SET include = ? WHERE key = ?", (1 if include else 0, key))
            _log(database, admin["id"], "enabled" if include else "disabled", key)
        if tier is not None and tier != s["tier"]:
            database.conn.execute("UPDATE sources SET tier = ? WHERE key = ?", (tier, key))
            _log(database, admin["id"], "tier changed", key, f"{s['tier']} -> {tier}")
    return get(database, key)


def remove(database, admin: Dict, key: str) -> None:
    """The page and its token go; its archive file stays (the posts already collected are not deleted here)."""
    s = get(database, key)
    with database.lock:
        database.conn.execute("DELETE FROM sources WHERE key = ?", (key,))
        _log(database, admin["id"], "removed", key, f"{s['name']} ({s['page_id']}); archive kept")


def check(database, box, admin: Optional[Dict], key: str, http_get: Optional[Callable] = None) -> Dict:
    s = get(database, key)
    with database.lock:
        blob = database.conn.execute("SELECT token FROM sources WHERE key = ?", (key,)).fetchone()[0]
    result = check_token(s["page_id"], box.unprotect(blob), http_get)
    with database.lock:
        database.conn.execute("UPDATE sources SET token_status = ?, token_detail = ?, token_checked_at = ?, "
                              "token_expires_at = COALESCE(?, token_expires_at) WHERE key = ?",
                              (result["status"], result["detail"], _iso(_now()), result["expires_at"], key))
        _log(database, admin["id"] if admin else None, "checked", key, result["status"])
    return get(database, key)


def fetch(database, box, admin: Optional[Dict], only: Optional[List[str]] = None, archive: Optional[str] = None,
          reservation: Optional[str] = None, adapter_factory=None) -> Dict:
    """One fetch of the included pages (or `only` those keys) into the archive. Returns a short summary per page."""
    archive, reservation = archive or archive_dir(), reservation or RESERVATION
    if load_cutoff(reservation) is None:
        raise SourcesError("Fetching is locked until the final-evaluation cutoff is recorded. Record it once with "
                           "Record cutoff at the top of this page.")
    rows = [s for s in list_sources(database, archive) if (s["key"] in only if only else s["include"])]
    rows = [s for s in rows if s["token_status"] != "invalid"]
    if not rows:
        raise SourcesError("No page to fetch: add a page, enable it, or replace its invalid token.")
    environ, pages = {}, []
    for s in rows:
        with database.lock:
            blob = database.conn.execute("SELECT token FROM sources WHERE key = ?", (s["key"],)).fetchone()[0]
        env = f"META_TOKEN_{s['key'].upper()}"[:71]
        environ[env] = box.unprotect(blob)
        pages.append(PageConfig(s["key"], s["name"], s["page_id"], s["tier"], env, True))
    kwargs = {"adapter_factory": adapter_factory} if adapter_factory else {}
    locks = [_page_lock(k) for k in sorted(p.key for p in pages)]
    for lock in locks:
        lock.acquire()
    try:
        record = refresh_pages(pages, archive, only=[p.key for p in pages], environ=environ, reservation=reservation,
                               **kwargs)
    except PagesConfigError as e:
        raise SourcesError(str(e)) from None
    finally:
        environ.clear()
        for lock in reversed(locks):
            lock.release()
    write_run_record(archive, record)
    now, summary = _iso(_now()), {}
    with database.lock:
        for p in record["pages"]:
            short = {k: p[k] for k in ("status", "added", "edited", "retired", "retire_refused",
                                       "reserved_for_final_evaluation", "error", "reason") if p.get(k) not in (None, [])}
            for k in ("edited", "retired"):
                if isinstance(short.get(k), list):
                    short[k] = len(short[k])
            summary[p["key"]] = short
            database.conn.execute("UPDATE sources SET last_fetch_at = ?, last_fetch = ? WHERE key = ?",
                                  (now, json.dumps(short), p["key"]))
            if p["status"] == "failed" and re.search(r"code[= ](190|102)\b", p.get("error", "")):
                database.conn.execute("UPDATE sources SET token_status = 'invalid', token_detail = ?, "
                                      "token_checked_at = ? WHERE key = ?",
                                      ("The token stopped working during a fetch. Replace it.", now, p["key"]))
            _log(database, admin["id"] if admin else None, "fetched", p["key"],
                 f"{p['status']}: {short.get('added', 0)} added" if p["status"] == "ok" else p["status"])
    return summary


def _archive_oldest(path: str) -> Optional[_dt.datetime]:
    """The oldest post time already archived for a page: past-posts fetching starts below it."""
    oldest = None
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                t = _created_time(json.loads(line).get("published_date") or "")
                if t and (oldest is None or t < oldest):
                    oldest = t
    return oldest


def _created_time(value: str) -> Optional[_dt.datetime]:
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", None):
        try:
            return _dt.datetime.strptime(value, fmt) if fmt else _dt.datetime.fromisoformat(value)
        except (TypeError, ValueError):
            continue
    return None


def backfill(database, box, archive: Optional[str] = None, reservation: Optional[str] = None, adapter_factory=None,
             now: Optional[_dt.datetime] = None) -> Dict:
    """One past-posts step for each included page that is not finished, not paused and not being fetched right now.
    Returns a short summary per page that took a step."""
    archive, reservation = archive or archive_dir(), reservation or RESERVATION
    if load_cutoff(reservation) is None:
        raise SourcesError("Past posts are fetched only after the final-evaluation cutoff is recorded.")
    now, floor, summary = now or _now(), backfill_from(), {}
    for s in list_sources(database, archive):
        _trim(database, s["key"], archive, floor)
        if not s["include"] or s["backfill_done"] or s["token_status"] == "invalid":
            continue                                     # finished pages cost nothing: no request at all
        if s["backfill_retry_after"] and s["backfill_retry_after"] > _iso(now):
            continue
        lock = _page_lock(s["key"])
        if not lock.acquire(blocking=False):
            continue                                     # being fetched: the next step comes soon enough
        try:
            summary[s["key"]] = _backfill_one(database, box, s, archive, reservation, adapter_factory, now, floor)
        finally:
            lock.release()
    return summary


def _trim(database, key: str, archive: str, floor: _dt.datetime) -> None:
    """Remove the page's archived posts older than the floor (a local file only: no request). Skipped this round while
    the page is being fetched."""
    from src.ingestion.pages import trim_archive
    lock = _page_lock(key)
    if not lock.acquire(blocking=False):
        return
    try:
        removed = trim_archive(archive_path(archive, key), floor)
    finally:
        lock.release()
    if removed:
        with database.lock:
            _log(database, None, "older posts removed", key,
                 f"{removed} post(s) from before {floor.date().isoformat()} (not kept, plan decision B2)")


def _backfill_one(database, box, s: Dict, archive: str, reservation: str, adapter_factory, now: _dt.datetime,
                  floor: _dt.datetime) -> Dict:
    from src.ingestion.pages import backfill_page
    if s["backfill_until"]:
        until = int(s["backfill_until"])
    else:
        oldest = _archive_oldest(archive_path(archive, s["key"]))
        until = int(oldest.timestamp()) if oldest else None
    with database.lock:
        blob = database.conn.execute("SELECT token FROM sources WHERE key = ?", (s["key"],)).fetchone()[0]
    env = f"META_TOKEN_{s['key'].upper()}"[:71]
    environ = {env: box.unprotect(blob)}
    kwargs = {"adapter_factory": adapter_factory} if adapter_factory else {}
    try:
        r = backfill_page(PageConfig(s["key"], s["name"], s["page_id"], s["tier"], env, True), archive,
                          until=str(until) if until else None, environ=environ, reservation=reservation, floor=floor,
                          **kwargs)
    finally:
        environ.clear()
    at, fields = _iso(now), {}
    if r["status"] == "ok":
        oldest = _dt.datetime.fromisoformat(r["oldest"]) if r["oldest"] else None
        if oldest is None:
            fields = {"backfill_done": 1, "backfill_note": "done: no older posts on the page"}
        else:
            nxt = int(oldest.timestamp())
            if until is not None and nxt >= until:
                nxt = until - 1                          # posts sharing one second: step past them, never loop
            done = oldest < floor or not r["has_more"]
            fields = {"backfill_until": str(nxt), "backfill_done": int(done),
                      "backfill_note": ("done: reached " + floor.date().isoformat() if oldest < floor else
                                        "done: reached the page's first post" if done else "going")}
        fields["backfill_retry_after"] = None
    elif r["status"] == "rate_limited":
        fields = {"backfill_note": "paused: Meta's rate limit, retrying in an hour",
                  "backfill_retry_after": _iso(now + _dt.timedelta(minutes=BACKFILL_PAUSE_MINUTES))}
    else:
        fields = {"backfill_note": (r.get("error") or r.get("reason") or r["status"])[:200]}
    with database.lock:
        sets = ", ".join(f"{k} = ?" for k in fields)
        database.conn.execute(f"UPDATE sources SET {sets}, backfill_at = ?, backfill_added = backfill_added + ? "
                              "WHERE key = ?", (*fields.values(), at, r.get("added", 0), s["key"]))
        if fields.get("backfill_done"):
            total = s["backfill_added"] + r.get("added", 0)
            _log(database, None, "past posts finished", s["key"], f"{total} added; {fields['backfill_note']}")
        if r["status"] == "failed" and re.search(r"code[= ](190|102)\b", r.get("error", "")):
            database.conn.execute("UPDATE sources SET token_status = 'invalid', token_detail = ?, "
                                  "token_checked_at = ? WHERE key = ?",
                                  ("The token stopped working while fetching past posts. Replace it.", at, s["key"]))
    return {k: r[k] for k in ("status", "added", "fetched") if k in r}


def needs_attention(sources: List[Dict]) -> List[Dict]:
    return [s for s in sources if s["include"] and s["token_status"] not in ("ok", "unchecked")]


# ---- background schedule -------------------------------------------------------------------------------------------
class Scheduler:
    """Fetch every `minutes` and check each token once a day (CAMPUS_VERIFY_SOURCES_FETCH_MINUTES > 0)."""

    def __init__(self, database_getter: Callable, box_getter: Callable, minutes: float, backfill_minutes: float = 0):
        self._db, self._box, self.minutes, self.backfill_minutes = database_getter, box_getter, minutes, backfill_minutes
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._backfill_thread: Optional[threading.Thread] = None

    def backfill_once(self) -> None:
        database = self._db()
        if database is None:
            return
        try:
            backfill(database, self._box())
        except SourcesError as e:
            logger.info("Past-posts step skipped: %s", e)
        except Exception as e:
            logger.warning("Past-posts step failed: %s", type(e).__name__)

    def run_once(self) -> None:
        database, box = self._db(), self._box()
        if database is None:
            return
        stale = _iso(_now() - _dt.timedelta(hours=CHECK_EVERY_HOURS))
        for s in list_sources(database):
            if s["include"] and (s["token_checked_at"] or "") < stale:
                try:
                    check(database, box, None, s["key"])
                except Exception as e:                       # one page never stops the others
                    logger.warning("Token check for %s failed: %s", s["key"], type(e).__name__)
        try:
            fetch(database, box, None)
        except SourcesError as e:
            logger.info("Scheduled fetch skipped: %s", e)
        except Exception as e:
            logger.warning("Scheduled fetch failed: %s", type(e).__name__)

    def start(self) -> None:
        def every(minutes, job):
            def loop():
                while not self._stop.wait(minutes * 60):     # first run after one interval, not during startup
                    job()
            return loop

        if self.minutes > 0 and self._thread is None:
            self._thread = threading.Thread(target=every(self.minutes, self.run_once), name="sources-fetch",
                                            daemon=True)
            self._thread.start()
        if self.backfill_minutes > 0 and self._backfill_thread is None:
            self._backfill_thread = threading.Thread(target=every(self.backfill_minutes, self.backfill_once),
                                                     name="sources-backfill", daemon=True)
            self._backfill_thread.start()

    def stop(self) -> None:
        self._stop.set()
        for t in (self._thread, self._backfill_thread):
            if t is not None:
                t.join(timeout=5)
