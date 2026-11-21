"""
The app's SQLite database (CAMPUS_VERIFY_DB_FILE, unset by default: nothing is stored in a database).

One file, opened with Python's built-in sqlite3, so there is no database server and no new dependency. The schema is
created and upgraded here, in numbered steps recorded in `schema_version`, so a later step (accounts, history) can be
added without touching the earlier ones.

    v1  daily_counts (day, kind, outcome, n), daily_matched (day, kind, n), meta (key, value): the dashboard's counts,
        no message text, picture, link or address (src/api/stats.py)
    v2  users, sessions, admin_actions: DYCI Google accounts (src/api/auth.py). A user row holds the DYCI email, the
        name Google gives, sign-in times, the admin and active flags and the history preference; never a password,
        a photo or anything the student checked. Sessions store only a SHA-256 of the cookie token.
    v3  users.tour_seen: whether a signed-in student has finished or skipped the guided tour (CAMPUS_VERIFY_TOUR)
    v4  checks: "My checks" (CAMPUS_VERIFY_HISTORY, src/api/history.py), only for accounts that turned history on: the
        message (and link) as typed, when, the answer and the DYCI post used. Never a picture. Removed with the account.
    v5  sources, source_actions: Facebook pages managed on the Sources page (CAMPUS_VERIFY_SOURCES, src/api/sources.py).
        Each page's token is stored encrypted (src/api/secret_box.py), with only its last 4 characters readable.
    v6  monthly_counts, monthly_matched: daily counts older than KEEP_DAYS, merged into one row per month before the
        daily rows are deleted (CAMPUS_VERIFY_RETENTION, src/api/stats.py). Counts only, like v1.
    v7  sources.backfill_*: where each page's past-posts fetch has reached (src/api/sources.py backfill). No posts and
        no token: a date, counts and a short status.
    v8  sessions.last_seen_at: when the session was last used, for the inactivity time-out
        (CAMPUS_VERIFY_SESSION_IDLE_MINUTES, src/api/auth.py). A time only.

CAMPUS_VERIFY_RETENTION=1 also turns on SQLite's secure_delete, so deleted rows (an erased account, a deleted saved
check, an expired session or count) are overwritten in the file instead of only being marked free.
"""
import os
import sqlite3
import threading
from typing import Optional

ENV = "CAMPUS_VERIFY_DB_FILE"

MIGRATIONS = (
    (1, """
        CREATE TABLE daily_counts (day TEXT NOT NULL, kind TEXT NOT NULL, outcome TEXT NOT NULL, n INTEGER NOT NULL,
                                   PRIMARY KEY (day, kind, outcome));
        CREATE TABLE daily_matched (day TEXT NOT NULL, kind TEXT NOT NULL, n INTEGER NOT NULL,
                                    PRIMARY KEY (day, kind));
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """),
    (2, """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE, name TEXT NOT NULL DEFAULT '',
                            created_at TEXT NOT NULL, last_login_at TEXT NOT NULL,
                            is_admin INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,
                            history_enabled INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE sessions (token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                               created_at TEXT NOT NULL, expires_at TEXT NOT NULL);
        CREATE INDEX sessions_user ON sessions (user_id);
        CREATE TABLE admin_actions (id INTEGER PRIMARY KEY, at TEXT NOT NULL, admin_user_id INTEGER NOT NULL,
                                    action TEXT NOT NULL, target_user_id INTEGER NOT NULL);
    """),
    (3, """
        ALTER TABLE users ADD COLUMN tour_seen INTEGER NOT NULL DEFAULT 0;
    """),
    (4, """
        CREATE TABLE checks (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                             at TEXT NOT NULL, kind TEXT NOT NULL, message TEXT, url TEXT, outcome TEXT NOT NULL,
                             post_id TEXT, post_date TEXT, post_url TEXT, post_excerpt TEXT);
        CREATE INDEX checks_user_at ON checks (user_id, at);
    """),
    (5, """
        CREATE TABLE sources (id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
                              page_id TEXT NOT NULL UNIQUE, tier TEXT NOT NULL, include INTEGER NOT NULL DEFAULT 1,
                              token BLOB, token_hint TEXT, token_set_at TEXT,
                              token_status TEXT NOT NULL DEFAULT 'unchecked', token_detail TEXT, token_checked_at TEXT,
                              token_expires_at TEXT, last_fetch_at TEXT, last_fetch TEXT, created_at TEXT NOT NULL);
        CREATE TABLE source_actions (id INTEGER PRIMARY KEY, at TEXT NOT NULL, admin_user_id INTEGER,
                                     action TEXT NOT NULL, source_key TEXT NOT NULL, detail TEXT);
    """),
    (6, """
        CREATE TABLE monthly_counts (month TEXT NOT NULL, kind TEXT NOT NULL, outcome TEXT NOT NULL, n INTEGER NOT NULL,
                                     PRIMARY KEY (month, kind, outcome));
        CREATE TABLE monthly_matched (month TEXT NOT NULL, kind TEXT NOT NULL, n INTEGER NOT NULL,
                                      PRIMARY KEY (month, kind));
    """),
    (7, """
        ALTER TABLE sources ADD COLUMN backfill_until TEXT;
        ALTER TABLE sources ADD COLUMN backfill_done INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE sources ADD COLUMN backfill_added INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE sources ADD COLUMN backfill_at TEXT;
        ALTER TABLE sources ADD COLUMN backfill_note TEXT;
        ALTER TABLE sources ADD COLUMN backfill_retry_after TEXT
    """),
    (8, """
        ALTER TABLE sessions ADD COLUMN last_seen_at TEXT
    """),
)
RETENTION_ENV = "CAMPUS_VERIFY_RETENTION"


def retention() -> bool:
    """CAMPUS_VERIFY_RETENTION (default off): the retention schedule (privacy notice) -- counts by calendar date with a
    monthly roll-up (src/api/stats.py), and secure_delete for every deletion in this database."""
    return os.environ.get(RETENTION_ENV, "0").strip() == "1"


def path_from_env() -> Optional[str]:
    return os.environ.get(ENV, "").strip() or None


class Database:
    """A single connection shared by the app's threads, serialised by a lock (the load is a few writes a minute)."""

    def __init__(self, path: str):
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        if retention():
            self.conn.execute("PRAGMA secure_delete=ON")       # deleted rows are overwritten, not only marked free
        self._migrate()

    def _migrate(self) -> None:
        with self.lock:
            self.conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = row[0] or 0
            for version, sql in MIGRATIONS:
                if version <= current:
                    continue
                self.conn.execute("BEGIN")
                try:
                    for statement in [s for s in sql.split(";") if s.strip()]:
                        self.conn.execute(statement)
                    self.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                    self.conn.execute("COMMIT")
                except Exception:
                    self.conn.execute("ROLLBACK")
                    raise

    def version(self) -> int:
        with self.lock:
            return self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0

    def close(self) -> None:
        with self.lock:
            self.conn.close()
