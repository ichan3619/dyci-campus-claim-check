"""
Several Facebook pages (scale plan S2, docs/plans/scale_plan.md; approved 2026-10-26).

Each page in a config file (data/pages.json; data/pages.example.json shows the shape) is refreshed into its own
archive file, data/corpus/archive/<key>.jsonl, with the existing, unchanged refresh (src/ingestion/refresh.py): new
posts are added, edits and deletions synced, and old text kept in <key>.history.jsonl. The archive is NOT served:
nothing reads it until the served-collection build (scale plan S3).

WHAT THIS ADDS
- One page never stops the others. A failed fetch raises before any write (refresh.py), so a page whose token
  expired, lacks permission or hit a rate limit keeps its archive exactly as it was; the run record says why.
- Rate limits (Graph error codes 4, 17, 32, 613) are recorded as "rate_limited, retry later": no waiting in a run.
- Each record carries its page's tier (src/data/schema.py VALID_CREDIBILITY_TIERS) and a note naming the page.
- Posts shared on two pages are reported, not removed: the archives stay faithful per page, and S3's build decides
  which copy is served.
- A run record (archive/runs/<UTC time>.json): per page what was added, edited, retired or refused, and any error.

FINAL-EVALUATION RESERVATION (owner decision D5 = (b), 2026-10-26)
The final independent evaluation needs real DYCI posts that development has never seen (data/final_eval/README.md).
Posts created on or after a cutoff (00:00 Philippine time on the cutoff day) are reserved for it: they are never
written and only counted, and the run record holds neither their text nor their IDs. The cutoff is recorded once in
data/final_eval/reservation.json (set_cutoff; the owner's choice: the day the page keys arrive) and cannot be
changed by these tools. refresh_pages() refuses to run while no cutoff is recorded.

FREEZE INSTEAD OF HOLD-BACK (owner 2026-11-16, docs/plans/past_and_new_posts_plan.md section 1)
With CAMPUS_VERIFY_IMPORT_AFTER_CUTOFF=1 (default off), posts from the cutoff on are imported like any other: the final
evaluation is protected by running the checker as it was frozen (a Git tag) when the cutoff was recorded
(src/api/sources.py record_cutoff), not by keeping the posts out. The cutoff is still required before any fetch, so
the freeze always comes first.

SECRETS
The config holds only the NAME of each page's token variable (token_env), never a token; a value that looks like a
token is refused. Every error message is scrubbed of the page's token, and the finished run record is checked
against every token used.
"""
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from src.data.schema import VALID_CREDIBILITY_TIERS
from src.ingestion.adapters import GraphAPIFetchError, MetaGraphAPIAdapter
from src.ingestion.corpus_storage import load_corpus, rewrite_corpus
from src.ingestion.dedup import DEFAULT_THRESHOLD, _numbers, similarity
from src.ingestion.facebook_page import explain_fetch_error, scrub
from src.ingestion.refresh import refresh_corpus_from_facebook

DEFAULT_CONFIG = os.path.join("data", "pages.json")
DEFAULT_ARCHIVE = os.path.join("data", "corpus", "archive")
FIELDS = ("key", "name", "page_id", "tier", "token_env", "include")
KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
PAGE_ID_RE = re.compile(r"^\d{5,25}$")
TOKEN_ENV_RE = re.compile(r"^META_TOKEN_[A-Z0-9_]{1,60}$")
LOOKS_LIKE_TOKEN = re.compile(r"EAA[A-Za-z0-9]{20,}|[A-Za-z0-9_-]{80,}")
RATE_LIMIT_CODES = {"4", "17", "32", "613"}
DEFAULT_RESERVATION = os.path.join("data", "final_eval", "reservation.json")
IMPORT_ENV = "CAMPUS_VERIFY_IMPORT_AFTER_CUTOFF"
MANILA = timezone(timedelta(hours=8))


class PagesConfigError(ValueError):
    """The pages config is malformed, or no reservation cutoff is recorded; nothing is fetched."""


def load_cutoff(path: str) -> Optional[datetime]:
    """The reservation cutoff (start of that day, Philippine time), or None when none is recorded."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        day = (json.load(f) or {}).get("cutoff_date")
    if not day:
        return None
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=MANILA)


def import_after_cutoff() -> bool:
    """CAMPUS_VERIFY_IMPORT_AFTER_CUTOFF (default off): import posts from the cutoff on instead of holding them back."""
    return os.environ.get(IMPORT_ENV, "0").strip() == "1"


def set_cutoff(path: str, day: str, now: Optional[str] = None, decision: Optional[str] = None,
               extra: Optional[Dict] = None) -> datetime:
    """Record the cutoff once. Refuses to change one that is already recorded."""
    cutoff = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=MANILA)
    if load_cutoff(path) is not None:
        raise PagesConfigError(f"a reservation cutoff is already recorded in {path}; it cannot be changed here")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {"decision": decision or ("scale plan D5 = (b), owner 2026-10-26: posts created on or after the cutoff are "
                                       "reserved for the final independent evaluation and are never imported for "
                                       "development"),
              "cutoff_date": day, "cutoff_starts": cutoff.isoformat(), "timezone": "Asia/Manila (UTC+08:00)",
              "recorded_at": now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **(extra or {})}
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=1)
    os.replace(tmp, path)
    return cutoff


def _created(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")
    except (TypeError, ValueError):
        return None


@dataclass
class ReservingGraphAdapter(MetaGraphAPIAdapter):
    """The Graph adapter, with posts created on or after `cutoff` left out entirely (counted in self.reserved).
    A post with no readable created_time is also left out: it cannot be shown to be before the cutoff.
    `floor` (past-posts fetching): posts created before it are left out too (counted in self.too_old); an undated post
    is kept. fetched_created_times still lists them, so the caller sees that the floor was reached."""
    cutoff: Optional[datetime] = None
    floor: Optional[datetime] = None

    def fetch_posts(self, *args, **kwargs):
        posts = self._fetch_within_cutoff(*args, **kwargs)
        self.too_old = 0
        if self.floor is not None:
            kept = [r for r in posts if (_created(r.published_date) or self.floor) >= self.floor]
            self.too_old = len(posts) - len(kept)
            posts = kept
        return posts

    def _fetch_within_cutoff(self, *args, **kwargs):
        posts = super().fetch_posts(*args, **kwargs)
        if self.cutoff is None:
            raise PagesConfigError("no reservation cutoff")
        if import_after_cutoff():                        # freeze instead of hold-back: nothing is left out
            self.reserved = 0
            return posts
        # Decided per record from its own date. (The parent's fetched_ids and fetched_created_times are not aligned by
        # position: a post without created_time has an ID but no time.)
        before = lambda value: (_created(value) or self.cutoff) < self.cutoff
        kept = [r for r in posts if before(r.published_date)]
        reserved_ids = {r.source_identifier for r in posts if not before(r.published_date)}
        self.reserved = len(posts) - len(kept)
        # Deletion detection (refresh.plan_sync) reads the oldest fetched time and the fetched IDs: reserved posts are
        # newer than anything kept, so dropping them changes neither the oldest time nor any archived post's status.
        self.fetched_created_times = [t for t in self.fetched_created_times if before(t)]
        self.fetched_ids = [i for i in self.fetched_ids if i not in reserved_ids]
        return kept


@dataclass(frozen=True)
class PageConfig:
    key: str
    name: str
    page_id: str
    tier: str
    token_env: str
    include: bool = True

    @property
    def note(self) -> str:
        return (f"{self.name} (Facebook page '{self.key}', collected through the Graph API with access granted by the "
                f"page's admin)")


def load_pages(path: str) -> List[PageConfig]:
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise PagesConfigError(f"{path} is not valid JSON: {e}") from None
    entries = data.get("pages") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise PagesConfigError(f'{path} needs a non-empty "pages" list')
    pages, keys, ids = [], set(), set()
    for n, entry in enumerate(entries, 1):
        where = f"{path}, page {n}"
        if not isinstance(entry, dict):
            raise PagesConfigError(f"{where}: must be an object")
        for field, value in entry.items():
            if isinstance(value, str) and LOOKS_LIKE_TOKEN.search(value):
                raise PagesConfigError(f"{where}: {field!r} looks like an access token. Put the token in the "
                                       f"environment variable named by token_env, never in this file")
        unknown = sorted(set(entry) - set(FIELDS))
        if unknown:
            raise PagesConfigError(f"{where}: unknown field(s) {unknown}; allowed: {list(FIELDS)}")
        missing = [k for k in FIELDS[:5] if not isinstance(entry.get(k), str) or not entry[k].strip()]
        if missing:
            raise PagesConfigError(f"{where}: missing or empty {missing}")
        include = entry.get("include", True)
        if not isinstance(include, bool):
            raise PagesConfigError(f"{where}: include must be true or false")
        page = PageConfig(entry["key"].strip(), entry["name"].strip(), entry["page_id"].strip(), entry["tier"].strip(),
                          entry["token_env"].strip(), include)
        if not KEY_RE.match(page.key):
            raise PagesConfigError(f"{where}: key {page.key!r} must be lowercase letters, digits or _ (2-41 chars)")
        if not PAGE_ID_RE.match(page.page_id):
            raise PagesConfigError(f"{where}: page_id must be the page's numeric ID")
        if page.tier not in VALID_CREDIBILITY_TIERS:
            raise PagesConfigError(f"{where}: tier {page.tier!r} must be one of {sorted(VALID_CREDIBILITY_TIERS)}")
        if not TOKEN_ENV_RE.match(page.token_env):
            raise PagesConfigError(f"{where}: token_env must be a variable name like META_TOKEN_DYCI_MAIN")
        if page.key in keys or page.page_id in ids:
            raise PagesConfigError(f"{where}: key or page_id used twice")
        keys.add(page.key)
        ids.add(page.page_id)
        pages.append(page)
    return pages


def archive_path(archive_dir: str, key: str) -> str:
    return os.path.join(archive_dir, f"{key}.jsonl")


def _code(message: str) -> Optional[str]:
    m = re.search(r"\(code=(\d+)\)", message)
    return m.group(1) if m else None


def refresh_pages(pages: Sequence[PageConfig], archive_dir: str, *, only: Optional[Sequence[str]] = None,
                  dry_run: bool = False, max_pages: int = 5, since: Optional[str] = None,
                  environ: Mapping[str, str] = os.environ, adapter_factory: Callable = ReservingGraphAdapter,
                  now: Optional[str] = None, reservation: str = DEFAULT_RESERVATION) -> Dict:
    """Refresh every included page (or only the keys in `only`) into its archive file. Returns the run record.
    Refuses to run while no reservation cutoff is recorded (D5)."""
    cutoff = load_cutoff(reservation)
    if cutoff is None:
        raise PagesConfigError(f"no final-evaluation cutoff is recorded ({reservation}). Record it first, once: "
                               f"python scripts/ingest_pages.py --reserve-from YYYY-MM-DD (the owner's choice: the "
                               f"day the page keys arrive)")
    known = {p.key for p in pages}
    if only:
        unknown = sorted(set(only) - known)
        if unknown:
            raise PagesConfigError(f"unknown page key(s) {unknown}; configured: {sorted(known)}")
    selected = [p for p in pages if (p.key in only if only else p.include)]
    os.makedirs(archive_dir, exist_ok=True)
    tokens, results, added = [], [], {}
    for page in selected:
        entry = {"key": page.key, "name": page.name, "page_id": page.page_id, "tier": page.tier}
        token = (environ.get(page.token_env) or "").strip()
        if not token:
            results.append({**entry, "status": "skipped", "reason": f"{page.token_env} is not set"})
            continue
        tokens.append(token)
        adapter = adapter_factory(page_id=page.page_id, access_token=token, credibility_tier=page.tier,
                                  acquisition_note=page.note, cutoff=cutoff)
        try:
            r = refresh_corpus_from_facebook(adapter, archive_path(archive_dir, page.key), max_pages=max_pages,
                                             since=since, dry_run=dry_run)
        except GraphAPIFetchError as e:
            message = explain_fetch_error(str(e), token)
            status = "rate_limited" if _code(message) in RATE_LIMIT_CODES else "failed"
            results.append({**entry, "status": status, "error": message + (" (retry later)" if status == "rate_limited" else ""),
                            "archive_unchanged": True})
            continue
        except Exception as e:                           # never let one page stop the others
            results.append({**entry, "status": "failed", "error": scrub(f"{type(e).__name__}: {e}", token),
                            "archive_unchanged": True})
            continue
        added[page.key] = r.added_document_ids
        results.append({**entry, "status": "ok", "added": r.added, "edited": r.edited, "retired": r.retired,
                        "retire_refused": r.retire_refused, "known_skipped": r.known_skipped,
                        "redundant_skipped": r.redundant_skipped, "unusable_skipped": r.unusable_skipped,
                        "flagged_for_review": r.flagged_for_review, "has_more_pages": r.has_more_pages,
                        "reserved_for_final_evaluation": getattr(adapter, "reserved", 0),
                        "added_document_ids": r.added_document_ids})
    record = {"run_at": now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "dry_run": dry_run,
              "reservation_cutoff": cutoff.isoformat(),
              "max_pages": max_pages, "since": since, "pages": results,
              "cross_page_duplicates": ("not checked in a dry run" if dry_run
                                        else cross_page_duplicates(pages, archive_dir, added))}
    text = json.dumps(record, ensure_ascii=False)
    for token in tokens:                                 # defence in depth: no token may survive into the record
        if token in text:
            record = json.loads(text.replace(token, "[REDACTED]"))
            text = json.dumps(record, ensure_ascii=False)
    return record


def backfill_page(page: PageConfig, archive_dir: str, *, until: Optional[str], environ: Mapping[str, str],
                  adapter_factory: Callable = ReservingGraphAdapter, reservation: str = DEFAULT_RESERVATION,
                  limit: int = 100, max_pages: int = 2, floor: Optional[datetime] = None) -> Dict:
    """One step of fetching a page's PAST posts (docs/plans/past_and_new_posts_plan.md section 2): up to `max_pages`
    pages of `limit` posts created up to `until` (a Unix time; None = from the newest), added to the page's archive.
    Add-only: edits and deletions are never inferred from a slice of the page. Posts created before `floor` are not
    added. Returns status, counts and the oldest post time seen (None when the page returned nothing)."""
    cutoff = load_cutoff(reservation)
    if cutoff is None:
        raise PagesConfigError(f"no final-evaluation cutoff is recorded ({reservation})")
    token = (environ.get(page.token_env) or "").strip()
    if not token:
        return {"status": "skipped", "reason": f"{page.token_env} is not set"}
    adapter = adapter_factory(page_id=page.page_id, access_token=token, credibility_tier=page.tier,
                              acquisition_note=page.note, cutoff=cutoff, **({"floor": floor} if floor else {}))
    os.makedirs(archive_dir, exist_ok=True)
    try:
        r = refresh_corpus_from_facebook(adapter, archive_path(archive_dir, page.key), limit=limit, max_pages=max_pages,
                                         until=until, sync_existing=False)
    except GraphAPIFetchError as e:
        message = explain_fetch_error(str(e), token)
        return {"status": "rate_limited" if _code(message) in RATE_LIMIT_CODES else "failed", "error": message}
    except Exception as e:                               # never let one page stop the others
        return {"status": "failed", "error": scrub(f"{type(e).__name__}: {e}", token)}
    times = [t for t in (_created(x) for x in getattr(adapter, "fetched_created_times", [])) if t]
    return {"status": "ok", "added": r.added, "fetched": len(getattr(adapter, "fetched_ids", [])),
            "oldest": min(times).isoformat() if times else None, "has_more": r.has_more_pages}


def trim_archive(path: str, floor: datetime) -> int:
    """Remove archived posts created before `floor` (undated posts stay). Returns how many were removed. Their text is
    not copied anywhere: keeping only what has a purpose is the point (plan decision B2, owner 2026-11-16)."""
    if not os.path.exists(path):
        return 0
    old = [r.document_id for r in load_corpus(path) if (_created(r.published_date) or floor) < floor]
    if old:
        rewrite_corpus(path, {i: None for i in old})
    return len(old)


def cross_page_duplicates(pages: Sequence[PageConfig], archive_dir: str, added: Mapping[str, Sequence[str]],
                          threshold: float = DEFAULT_THRESHOLD) -> List[Dict]:
    """Posts added in this run that are near-identical (same numbers, the dedup rule) to a post on another page."""
    archives = {p.key: load_corpus(archive_path(archive_dir, p.key)) for p in pages
                if os.path.exists(archive_path(archive_dir, p.key))}
    out = []
    for key, ids in added.items():
        mine = [r for r in archives.get(key, []) if r.document_id in set(ids)]
        for rec in mine:
            for other_key, others in archives.items():
                if other_key == key:
                    continue
                for other in others:
                    score = similarity(rec.text, other.text)
                    if score >= threshold and _numbers(rec.text) == _numbers(other.text):
                        out.append({"document_id": rec.document_id, "page": key, "duplicate_of": other.document_id,
                                    "other_page": other_key, "score": round(score, 3)})
    return out


def write_run_record(archive_dir: str, record: Dict) -> str:
    runs = os.path.join(archive_dir, "runs")
    os.makedirs(runs, exist_ok=True)
    stamp = record["run_at"].replace("-", "").replace(":", "")
    path = os.path.join(runs, f"{stamp}{'-dry-run' if record['dry_run'] else ''}.json")
    if os.path.exists(path):
        path = path[:-5] + f"-{uuid.uuid4().hex[:6]}.json"
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path
