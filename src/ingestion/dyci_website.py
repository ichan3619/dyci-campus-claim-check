"""
Rule W: DYCI's website as an official source (results/round2/website_source_plan.md, Amendment 1).
CAMPUS_VERIFY_WEBSITE=1 (default off).

dyci.edu.ph is a JavaScript app; its content comes from public read feeds that every visitor's browser calls
(robots.txt: Allow: /). This module reads three of them (announcements, DYCIan News articles, events) into one record
per published item. Never stored: who created, updated or wrote an item (createdBy, updatedBy, articleAuthor).
"""
from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

ENV = "CAMPUS_VERIFY_WEBSITE"
VERSION = "dyci-website-v1"
BASE = "https://dyci.edu.ph"
API = BASE + "/api"
USER_AGENT = "DYCI-Campus-Claim-Check/1.0 (student thesis project; reads public announcements)"
TIMEOUT = 20
ARTICLE_PAGE_SIZE = 50
MAX_ARTICLE_PAGES = 30
NEVER_STORED = ("createdBy", "updatedBy", "articleAuthor")


def mode() -> str:
    """"off", "1" (rule W v1: served with the posts; FAIL) or "2" (rule W v2: kept out of the served pool, used only
    by the rule-based confirmer and the link lookup; results/round2/website_source_v2_plan.md, Amendment 1)."""
    value = os.environ.get(ENV, "0").strip()
    return value if value in ("1", "2") else "off"


def enabled() -> bool:
    return mode() != "off"


def stored_items(project_root: str) -> List[dict]:
    """The stored website records (the website file), or [] when there is none."""
    p = file_path(project_root)
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def http_json(url: str):
    """GET a feed as JSON. Raises on a network error, a non-200 answer or a body that is not JSON."""
    import requests
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    if r.status_code != 200:
        raise ValueError(f"{url.split('?')[0]} answered HTTP {r.status_code}")
    return r.json()


def text_of(value) -> str:
    """HTML removed, entities decoded, whitespace collapsed."""
    t = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", str(value or ""))
    t = re.sub(r"(?s)<br\s*/?>|</p>|</li>|</div>", ". ", t)
    t = html.unescape(re.sub(r"(?s)<[^>]+>", " ", t))
    t = re.sub(r"\s*\.\s*(\.\s*)+", ". ", t)
    return re.sub(r"\s+", " ", t).strip(" .") + ("" if not t.strip() else "")


# The site's own slug rules (its script: announcements yM, articles gM).
def slug_announcement(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", str(title or "").lower())
    return re.sub(r"-{2,}", "-", re.sub(r"\s+", "-", s)).strip()


def slug_article(title: str) -> str:
    return re.sub(r"[^\w-]+", "", re.sub(r"\s+", "-", str(title or "").lower().strip()))


def iso_date(value) -> str:
    """YYYY-MM-DD from the feeds' date forms: ISO ("2026-11-12", "2026-11-12 08:00:00") or MMDDYYYY ("07062025", some
    articles); "" when neither, so an unreadable date can never pass the cutoff by accident."""
    s = str(value or "").strip()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.fullmatch(r"(\d{2})(\d{2})(\d{4})", s)
    if m and 1 <= int(m.group(1)) <= 12 and 1 <= int(m.group(2)) <= 31:
        return f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
    return ""


def _published(item: dict, field: str) -> bool:
    return str(item.get(field) or "").strip().lower() == "published"


def _record(kind: str, item_id, title: str, body: str, date: str, path: str, now: str) -> Optional[dict]:
    title, body = text_of(title), text_of(body)
    text = (f"{title}. {body}" if body and not body.lower().startswith(title.lower()) else body or title).strip()
    if not text:
        return None
    return {"document_id": f"website:{kind}:{item_id}", "text": text, "source_type": "official_site",
            "source_url": BASE + path, "published_date": iso_date(date), "fetched_at": now,
            "credibility_tier": "official", "source_identifier": f"{kind}:{item_id}", "origin_url": "", "origin_date": "",
            "acquisition_note": f"DYCI website feed ({kind}), read by {VERSION}"}


def records_from(announcements: list, articles: list, events: list, now: Optional[str] = None) -> List[dict]:
    """One record per published item, in a stable order (announcements, articles, events; by id)."""
    now = now or datetime.now(timezone.utc).isoformat()
    out = []
    for a in sorted(announcements or [], key=lambda x: str(x.get("announcementID"))):
        if _published(a, "announcementStatus"):
            r = _record("announcement", a.get("announcementID"), a.get("announcementTitle"),
                        a.get("announcementDescription"), str(a.get("announcementDate") or ""),
                        "/announcements/" + slug_announcement(text_of(a.get("announcementTitle"))), now)
            if r:
                out.append(r)
    for a in sorted(articles or [], key=lambda x: str(x.get("articleID"))):
        if _published(a, "articleStatus"):
            r = _record("article", a.get("articleID"), a.get("articleTitle"), a.get("articleExcerpt"),
                        str(a.get("articlePublishedDate") or ""), "/dycian-news/" + slug_article(text_of(a.get("articleTitle"))), now)
            if r:
                out.append(r)
    for e in sorted(events or [], key=lambda x: str(x.get("eventID"))):
        if _published(e, "eventStatus"):
            when = str(e.get("eventStart") or "")
            extra = " ".join(p for p in (f"Starts {when}." if when else "",
                                         f"Ends {e.get('eventEnd')}." if e.get("eventEnd") else "",
                                         f"Venue: {text_of(e.get('eventLocation'))}." if e.get("eventLocation") else "",
                                         text_of(e.get("eventDescription"))) if p)
            r = _record("event", e.get("eventID"), e.get("eventTitle"), extra, when,
                        "/events/" + slug_announcement(text_of(e.get("eventTitle"))), now)
            if r:
                out.append(r)
    return out


def read_feeds(get: Callable[[str], object] = http_json) -> Tuple[list, list, list]:
    """(announcements, articles, events) as the feeds give them. Raises if any feed cannot be read."""
    ann = get(f"{API}/get-announcements.php")
    ann = ann if isinstance(ann, list) else (ann or {}).get("data", [])
    articles = []
    for page in range(1, MAX_ARTICLE_PAGES + 1):
        d = get(f"{API}/get-articles.php?mode=paginate&page={page}&limit={ARTICLE_PAGE_SIZE}&status=Published")
        batch = d.get("data", []) if isinstance(d, dict) else (d or [])
        if not batch:
            break
        articles.extend(batch)
        total = d.get("total") if isinstance(d, dict) else None
        if isinstance(total, int) and len(articles) >= total:
            break
    ev = get(f"{API}/get-events.php?mode=all&status=Published")
    ev = ev.get("data", []) if isinstance(ev, dict) else (ev or [])
    return ann, articles, ev


def before_cutoff(records: List[dict], cutoff_day: Optional[str], import_after: bool) -> List[dict]:
    """W-b: items dated on or after the cutoff are kept only when imports after the cutoff are on. An item without a
    readable date is dropped whenever the cutoff applies (it could be after it)."""
    if not cutoff_day or import_after:
        return list(records)
    return [r for r in records if r.get("published_date") and r["published_date"] < cutoff_day]


def write(path: str, records: List[dict]) -> Dict[str, int]:
    """Replace the file with these records (atomic); {added, edited, removed} against the previous file."""
    old = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    old[r["document_id"]] = r["text"]
    new = {r["document_id"]: r["text"] for r in records}
    counts = {"added": sum(1 for k in new if k not in old), "removed": sum(1 for k in old if k not in new),
              "edited": sum(1 for k in new if k in old and old[k] != new[k])}
    if counts["added"] or counts["removed"] or counts["edited"] or not os.path.isfile(path):
        with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
    return counts


def refresh(path: str, get: Callable[[str], object] = http_json, cutoff_day: Optional[str] = None,
            import_after: bool = False) -> Dict[str, int]:
    """Read the feeds and update the file; a failed read raises and leaves the file as it was."""
    ann, articles, events = read_feeds(get)
    recs = before_cutoff(records_from(ann, articles, events), cutoff_day, import_after)
    counts = write(path, recs)
    counts["items"] = len(recs)
    return counts


# ---- serving (W-c) and links (W-d) --------------------------------------------------------------------------------------
FILE_ENV, MINUTES_ENV = "CAMPUS_VERIFY_WEBSITE_FILE", "CAMPUS_VERIFY_WEBSITE_MINUTES"
DEFAULT_FILE = os.path.join("data", "corpus", "dyci_website.jsonl")
LINK_PATHS = ("/announcements/", "/dycian-news/", "/events/")


def file_path(project_root: str) -> str:
    value = os.environ.get(FILE_ENV, "").strip() or DEFAULT_FILE
    return value if os.path.isabs(value) else os.path.join(project_root, value)


def minutes() -> float:
    try:
        return float(os.environ.get(MINUTES_ENV, "60"))
    except ValueError:
        return 60.0


def cutoff_settings() -> Tuple[Optional[str], bool]:
    """(cutoff day YYYY-MM-DD or None, imports after the cutoff on) from the recorded final-evaluation reservation."""
    from src.ingestion import pages
    cutoff = pages.load_cutoff(pages.DEFAULT_RESERVATION) if os.path.isfile(pages.DEFAULT_RESERVATION) else None
    return (cutoff.date().isoformat() if cutoff else None), pages.import_after_cutoff()


def item_for_link(url: str, project_root: str) -> Optional[dict]:
    """W-d: the stored record whose link is this dyci.edu.ph item link (same path, any trailing slash), or None."""
    from urllib.parse import urlsplit
    try:
        parts = urlsplit(url or "")
    except ValueError:
        return None
    host = (parts.hostname or "").lower().rstrip(".")
    path = parts.path.rstrip("/")
    if host not in ("dyci.edu.ph", "www.dyci.edu.ph") or not path.startswith(LINK_PATHS):
        return None
    target = (BASE + path).lower()
    p = file_path(project_root)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if (r.get("source_url") or "").rstrip("/").lower() == target:
                    return r
    return None


class Scheduler:
    """W-a: read the feeds now and every `minutes`; each result goes to `report(counts, error)`. Never raises."""

    def __init__(self, path: str, every_minutes: float, report: Callable[[Optional[dict], Optional[str]], None]):
        import threading
        self.path, self.every, self.report = path, every_minutes, report
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="dyci-website", daemon=True)

    def run_once(self) -> None:
        try:
            cutoff_day, import_after = cutoff_settings()
            self.report(refresh(self.path, cutoff_day=cutoff_day, import_after=import_after), None)
        except Exception as e:                     # a failed read keeps the file; the error is shown, never raised
            self.report(None, f"{type(e).__name__}: {str(e)[:200]}")

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.every * 60)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
