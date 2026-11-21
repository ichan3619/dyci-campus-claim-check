"""
"My checks": a signed-in student's own history (CAMPUS_VERIFY_HISTORY, default off; needs CAMPUS_VERIFY_AUTH=google and
CAMPUS_VERIFY_DB_FILE).

Saved only for an account whose owner turned "Keep a history of my checks" on (users.history_enabled). Each entry
holds what the student typed (the message, and the link for a link check; for a picture only that it was a picture
check), when, the answer, and which DYCI post it rested on. Never the picture, never an address. Only the owner can
list or delete their entries; there is no admin view. At most KEEP_LAST entries per account and none older than
KEEP_DAYS; deleting the account deletes its entries (ON DELETE CASCADE).

An entry is the answer as it was on that day. The page offers "Check again" against today's posts instead of reusing
it, because "tomorrow" and the posts change.
"""
import datetime as _dt
import os
from typing import Optional

ENV = "CAMPUS_VERIFY_HISTORY"
KEEP_LAST = 50
KEEP_DAYS = 90
EXCERPT_CHARS = 160
_COLUMNS = "id, at, kind, message, url, outcome, post_id, post_date, post_url, post_excerpt"


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso(t: _dt.datetime) -> str:
    return t.replace(microsecond=0).isoformat()


def entry_for(kind: str, outcome: str, sent: Optional[dict], body: Optional[dict]) -> Optional[dict]:
    """The fields to keep for one check, or None when there is nothing to keep (a text or link check with no message)."""
    sent = sent or {}
    message = None if kind == "picture" else (sent.get("claim") or "").strip() or None
    if kind != "picture" and message is None:
        return None
    post = (body or {}).get("used_evidence") if kind != "picture" else None
    if not post and kind != "picture":
        post = ((body or {}).get("retrieved_evidence") or [None])[0] if outcome == "question" else None
    post = post if isinstance(post, dict) else {}
    text = " ".join(str(post.get("text") or "").split())
    url = (sent.get("url") or "").strip() or None if kind == "link" else None
    return {"kind": kind, "message": message, "url": url,
            "outcome": outcome, "post_id": post.get("document_id"),
            "post_date": post.get("original_date") or post.get("published_date"),
            "post_url": post.get("original_url") or post.get("source_url"),
            "post_excerpt": (text[:EXCERPT_CHARS - 1] + "…") if len(text) > EXCERPT_CHARS else (text or None)}


def purge_old(database, now: Optional[_dt.datetime] = None) -> None:
    cutoff = _iso((now or _now()) - _dt.timedelta(days=KEEP_DAYS))
    with database.lock:
        database.conn.execute("DELETE FROM checks WHERE at < ?", (cutoff,))


def save(database, user_id: int, entry: dict, now: Optional[_dt.datetime] = None) -> None:
    """Add one entry, then keep only the account's newest KEEP_LAST and drop anything older than KEEP_DAYS."""
    with database.lock:
        c = database.conn
        c.execute("INSERT INTO checks (user_id, at, kind, message, url, outcome, post_id, post_date, post_url, "
                  "post_excerpt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (user_id, _iso(now or _now()), entry["kind"], entry["message"], entry["url"], entry["outcome"],
                   entry["post_id"], entry["post_date"], entry["post_url"], entry["post_excerpt"]))
        c.execute("DELETE FROM checks WHERE user_id = ? AND id NOT IN "
                  "(SELECT id FROM checks WHERE user_id = ? ORDER BY at DESC, id DESC LIMIT ?)",
                  (user_id, user_id, KEEP_LAST))
    purge_old(database, now)


def list_for(database, user_id: int, now: Optional[_dt.datetime] = None) -> list:
    purge_old(database, now)
    with database.lock:
        rows = database.conn.execute(f"SELECT {_COLUMNS} FROM checks WHERE user_id = ? ORDER BY at DESC, id DESC",
                                     (user_id,)).fetchall()
    return [dict(zip(_COLUMNS.split(", "), r)) for r in rows]


def delete_one(database, user_id: int, check_id: int) -> bool:
    """Only the owner's own entry: another account's id deletes nothing."""
    with database.lock:
        return database.conn.execute("DELETE FROM checks WHERE id = ? AND user_id = ?", (check_id, user_id)).rowcount > 0


def clear(database, user_id: int) -> int:
    with database.lock:
        return database.conn.execute("DELETE FROM checks WHERE user_id = ?", (user_id,)).rowcount
