"""
Privacy-safe usage statistics for the dashboard (owner request 2026-10-26: the prototype's dashboard, kept honest).
Setting CAMPUS_VERIFY_STATS, default off; CAMPUS_VERIFY_STATS_FILE (optional) keeps the counts across restarts in a
JSON file. With CAMPUS_VERIFY_DB_FILE set (src/api/db.py), the counts are kept in that SQLite database instead
(SqliteCounter); on its first start, an existing CAMPUS_VERIFY_STATS_FILE is imported once. Both keep KEEP_DAYS days.

What is counted, per day (server local date): the KIND of check (text, link, picture), its OUTCOME, and whether a
DYCI post matched (coverage). Nothing else: no message text, no picture, no link, no IP address, no time of day.

    text / link  supported | contradicted | not_confirmed | withheld_language | question | could_not_read
    picture      level_1 ... level_5 | no_text | other_page | could_not_read

"Contradicted" is NOT reported as "fake news": "not confirmed" does not mean false, and Tagalog/Taglish verdicts are
withheld by policy. The dashboard shows the outcomes under their own names.
"""
import datetime as _dt
import json
import os
import tempfile
import threading
from typing import Optional

ENV = "CAMPUS_VERIFY_STATS"
FILE_ENV = "CAMPUS_VERIFY_STATS_FILE"
KINDS = ("text", "link", "picture")
OUTCOMES = {
    "text": ("supported", "contradicted", "not_confirmed", "withheld_language", "question", "could_not_read"),
    "picture": ("level_1", "level_2", "level_3", "level_4", "level_5", "no_text", "other_page", "could_not_read"),
}
OUTCOMES["link"] = OUTCOMES["text"]
KEEP_DAYS = 400


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def today() -> _dt.date:
    return _dt.date.today()


def outcome_for(kind: str, status: int, body: Optional[dict]) -> Optional[tuple]:
    """(outcome, covered) for one response, or None when it is not counted (bad input: 4xx other than 404/413/415)."""
    if status == 502:
        return "could_not_read", False
    if status != 200 or not isinstance(body, dict):
        return None
    if kind == "picture":
        level = (body.get("assessment") or {}).get("level")
        if level in (1, 2, 3, 4, 5):
            outcome = f"level_{level}"
        else:
            outcome = "other_page" if body.get("status") == "other_page" else "no_text"
        matched = bool((body.get("image_signals") or {}).get("photo_match")) or \
            (body.get("sentence_summary") or {}).get("matches", 0) > 0 or \
            (body.get("text_check") or {}).get("evidence_status") == "RELEVANT_EVIDENCE_FOUND"
        return outcome, matched
    route = (body.get("route") or {}).get("status")
    if route == "INFO_SEEKING":
        outcome = "question"
    elif body.get("verdict_withheld") == "language":
        outcome = "withheld_language"
    else:
        outcome = {"SUPPORTED": "supported", "CONTRADICTED": "contradicted",
                   "INSUFFICIENT_EVIDENCE": "not_confirmed"}.get(body.get("verdict"))
        if outcome is None:
            return None
    return outcome, body.get("evidence_status") == "RELEVANT_EVIDENCE_FOUND"


class Counter:
    """Counts by day -> kind -> outcome, plus matched counts. Thread-safe; optionally saved to a JSON file."""

    def __init__(self, path: Optional[str] = None):
        self._lock = threading.Lock()
        self._path = path
        self._days = {}
        if path and os.path.exists(path):
            try:
                self._days = json.load(open(path, encoding="utf-8")).get("days", {})
            except (OSError, ValueError):
                self._days = {}

    def record(self, kind: str, outcome: str, matched: bool, day: Optional[_dt.date] = None) -> None:
        if kind not in KINDS or outcome not in OUTCOMES[kind]:
            return
        key = (day or today()).isoformat()
        with self._lock:
            d = self._days.setdefault(key, {})
            k = d.setdefault(kind, {"outcomes": {}, "matched": 0})
            k["outcomes"][outcome] = k["outcomes"].get(outcome, 0) + 1
            k["matched"] += 1 if matched else 0
            for old in sorted(self._days)[:-KEEP_DAYS]:
                del self._days[old]
            self._save()

    def _save(self) -> None:
        if not self._path:
            return
        folder = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"what": "campus claim check counts (no message text)", "days": self._days}, f)
        os.replace(tmp, self._path)

    def raw(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._days))

    def snapshot(self, days: int = 7, day: Optional[_dt.date] = None) -> dict:
        return summarise(self.raw(), days, day or today())

    def csv(self, days: int = 7, day: Optional[_dt.date] = None) -> str:
        return to_csv(self.raw(), days, day or today())


def summarise(data: dict, days: int, end: _dt.date) -> dict:
    """The dashboard's view of {day: {kind: {"outcomes": {...}, "matched": n}}}, whatever stores it. The totals, outcomes
    and coverage cover the last `days` days (the range the dashboard shows); `all_time` counts every kept day."""
    span = [(end - _dt.timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    totals = {kind: {o: 0 for o in OUTCOMES[kind]} for kind in KINDS}
    matched = {kind: 0 for kind in KINDS}
    all_time = sum(n for d in data.values() for v in d.values() for n in v["outcomes"].values())
    for key, d in data.items():
        if not span[0] <= key <= span[-1]:
            continue
        for kind, v in d.items():
            for o, n in v["outcomes"].items():
                totals[kind][o] += n
            matched[kind] += v["matched"]
    daily = []
    for key in span:
        d = data.get(key, {})
        row = {"date": key}
        for kind in KINDS:
            row[kind] = sum((d.get(kind) or {}).get("outcomes", {}).values())
        row["contradicted"] = sum(((d.get(k) or {}).get("outcomes", {})).get("contradicted", 0) for k in ("text", "link"))
        daily.append(row)
    checks = {kind: sum(totals[kind].values()) for kind in KINDS}
    answered = {kind: checks[kind] - totals[kind].get("could_not_read", 0) for kind in KINDS}
    all_answered = sum(answered.values())
    return {"version": "stats-v2", "days": days, "since": min(data) if data else None,
            "all_time": {"total": all_time, "since": min(data) if data else None}, "checks": checks,
            "total": sum(checks.values()), "outcomes": totals, "matched": matched,
            "coverage": round(sum(matched.values()) / all_answered, 3) if all_answered else None,
            "daily": daily, "note": "Counts only: no message text, picture, link or address is stored."}


CSV_OUTCOMES = tuple(dict.fromkeys(OUTCOMES["text"] + OUTCOMES["picture"]))


def to_csv(data: dict, days: int, end: _dt.date) -> str:
    """The report's spreadsheet: one row per day and kind of check over the last `days` days (empty days included, so
    a chart made from it has no gaps). The same counts as the dashboard; nothing but dates, fixed names and numbers."""
    span = [(end - _dt.timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    lines = [",".join(("date", "kind", "checks") + CSV_OUTCOMES + ("matched_a_dyci_post",))]
    for key in span:
        d = data.get(key, {})
        for kind in KINDS:
            v = d.get(kind) or {}
            outcomes = v.get("outcomes", {})
            cells = [str(outcomes.get(o, 0)) if o in OUTCOMES[kind] else "" for o in CSV_OUTCOMES]
            lines.append(",".join([key, kind, str(sum(outcomes.values()))] + cells + [str(v.get("matched", 0))]))
    return "\r\n".join(lines) + "\r\n"


class SqliteCounter:
    """The same counts as Counter, kept in the app database (src/api/db.py, CAMPUS_VERIFY_DB_FILE)."""

    def __init__(self, database, import_json: Optional[str] = None):
        self._db = database
        if import_json and os.path.exists(import_json):
            self._import_once(import_json)

    def _import_once(self, path: str) -> None:
        with self._db.lock:
            done = self._db.conn.execute("SELECT value FROM meta WHERE key = 'stats_imported_from'").fetchone()
            empty = self._db.conn.execute("SELECT COUNT(*) FROM daily_counts").fetchone()[0] == 0
        if done or not empty:
            return
        try:
            days = json.load(open(path, encoding="utf-8")).get("days", {})
        except (OSError, ValueError):
            return
        with self._db.lock:
            c = self._db.conn
            c.execute("BEGIN")
            try:
                for day, kinds in days.items():
                    for kind, v in kinds.items():
                        if kind not in KINDS:
                            continue
                        for outcome, n in v.get("outcomes", {}).items():
                            if outcome in OUTCOMES[kind]:
                                c.execute("INSERT INTO daily_counts VALUES (?, ?, ?, ?)", (day, kind, outcome, int(n)))
                        c.execute("INSERT INTO daily_matched VALUES (?, ?, ?)", (day, kind, int(v.get("matched", 0))))
                c.execute("INSERT INTO meta VALUES ('stats_imported_from', ?)", (os.path.abspath(path),))
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise

    def _roll_up(self, c, today_: _dt.date) -> None:
        """CAMPUS_VERIFY_RETENTION: days older than KEEP_DAYS calendar days are added to their month's totals, then the
        daily rows are deleted. Runs inside record()'s transaction."""
        cutoff = (today_ - _dt.timedelta(days=KEEP_DAYS - 1)).isoformat()
        for d, kind, outcome, n in c.execute("SELECT day, kind, outcome, n FROM daily_counts WHERE day < ?",
                                             (cutoff,)).fetchall():
            c.execute("INSERT INTO monthly_counts VALUES (?, ?, ?, ?) ON CONFLICT (month, kind, outcome) "
                      "DO UPDATE SET n = n + excluded.n", (d[:7], kind, outcome, n))
        for d, kind, n in c.execute("SELECT day, kind, n FROM daily_matched WHERE day < ?", (cutoff,)).fetchall():
            c.execute("INSERT INTO monthly_matched VALUES (?, ?, ?) ON CONFLICT (month, kind) "
                      "DO UPDATE SET n = n + excluded.n", (d[:7], kind, n))
        c.execute("DELETE FROM daily_counts WHERE day < ?", (cutoff,))
        c.execute("DELETE FROM daily_matched WHERE day < ?", (cutoff,))

    def record(self, kind: str, outcome: str, matched: bool, day: Optional[_dt.date] = None) -> None:
        from src.api.db import retention
        if kind not in KINDS or outcome not in OUTCOMES[kind]:
            return
        key = (day or today()).isoformat()
        with self._db.lock:
            c = self._db.conn
            c.execute("BEGIN")
            try:
                c.execute("INSERT INTO daily_counts VALUES (?, ?, ?, 1) "
                          "ON CONFLICT (day, kind, outcome) DO UPDATE SET n = n + 1", (key, kind, outcome))
                c.execute("INSERT INTO daily_matched VALUES (?, ?, ?) "
                          "ON CONFLICT (day, kind) DO UPDATE SET n = n + excluded.n", (key, kind, 1 if matched else 0))
                if retention():
                    self._roll_up(c, day or today())
                else:
                    keep = [r[0] for r in c.execute("SELECT DISTINCT day FROM daily_counts ORDER BY day DESC LIMIT ?",
                                                    (KEEP_DAYS,))]
                    if len(keep) == KEEP_DAYS:
                        c.execute("DELETE FROM daily_counts WHERE day < ?", (keep[-1],))
                        c.execute("DELETE FROM daily_matched WHERE day < ?", (keep[-1],))
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise

    def raw(self) -> dict:
        data = {}
        with self._db.lock:
            for d, kind, outcome, n in self._db.conn.execute("SELECT day, kind, outcome, n FROM daily_counts"):
                data.setdefault(d, {}).setdefault(kind, {"outcomes": {}, "matched": 0})["outcomes"][outcome] = n
            for d, kind, n in self._db.conn.execute("SELECT day, kind, n FROM daily_matched"):
                data.setdefault(d, {}).setdefault(kind, {"outcomes": {}, "matched": 0})["matched"] = n
        return data

    def snapshot(self, days: int = 7, day: Optional[_dt.date] = None) -> dict:
        s = summarise(self.raw(), days, day or today())
        with self._db.lock:
            months = self._db.conn.execute("SELECT month, SUM(n) FROM monthly_counts GROUP BY month ORDER BY month").fetchall()
        if months:                                       # CAMPUS_VERIFY_RETENTION: older days kept as monthly totals
            s["earlier_months"] = [{"month": m, "total": n} for m, n in months]
            s["all_time"] = {"total": s["all_time"]["total"] + sum(n for _, n in months), "since": months[0][0]}
        return s

    def csv(self, days: int = 7, day: Optional[_dt.date] = None) -> str:
        return to_csv(self.raw(), days, day or today())
