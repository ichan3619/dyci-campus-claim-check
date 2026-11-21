"""
The served collection: which archived posts the checker searches (scale plan S3, docs/plans/scale_plan.md; build
approved 2026-10-26). Building it changes nothing for students: the checker keeps searching its configured files
until CAMPUS_VERIFY_CORPUS is pointed at the output, and that switch waits for S3's own pre-registered run and the
owner's decision.

WHAT IS KEPT (from data/corpus/archive/*.jsonl, the page archives of scale plan S2)
1. Posts whose date is within `window_days` days before the build day (Philippine time), the build day included.
   The window length is the owner's decision (D2), so it has no default here.
   Date: the original announcement date (origin_date) if known, else the post's own date (published_date).
2. Plus posts listed in the "always use" file (D3), whatever their date: long-lasting facts such as a department
   name or a policy. Columns: document_id, reason, added_by, date. A listed post must be in the archive; one that is
   not is reported.
Left out: posts outside the window, undated posts (unless listed), posts dated after the build day, and, as a
second guard, any post on or after the final-evaluation cutoff (D5), even if listed.

POSTS SHARED ON SEVERAL PAGES
Near-identical posts with the same numbers (the existing dedup rule) are served once: the earliest copy is kept and
the others are listed in the build record with the copy they duplicate. The rule only ever matches posts whose sets
of numbers are equal, so posts are compared only within groups that share the same numbers. That keeps and drops
exactly the same posts as comparing every pair (tested), without the cost growing with the square of the collection
(measured on 3,000 synthetic posts: 2 min comparing every pair, 0.5 s grouped, byte-identical output). What it does
not produce is the dedup module's "similar text but different numbers, please review" flag on kept posts, which needs
comparisons across groups; this build does not use that flag.

REPRODUCIBLE
Same archive, list, window and build day give a byte-identical output (posts sorted by date, then document_id). The
build record (<output>.build.json) holds the inputs' SHA-256, the settings, the counts per page and the output's
SHA-256; it has no timestamp, so it is reproducible too.
"""
import csv
import hashlib
import json
import os
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from src.ingestion.corpus_storage import load_corpus
from src.ingestion.dedup import DEFAULT_THRESHOLD, _numbers, filter_redundant
from src.ingestion.evidence import NormalizedEvidence

MANILA = timezone(timedelta(hours=8))
ALWAYS_USE_COLUMNS = ["document_id", "reason", "added_by", "date"]


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def post_day(record: NormalizedEvidence) -> Optional[date]:
    """The post's day in Philippine time: origin_date if known, else published_date; None when neither parses."""
    for value in (record.origin_date, record.published_date):
        if not value:
            continue
        for parse in (lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S%z"), datetime.fromisoformat):
            try:
                when = parse(value)
            except (TypeError, ValueError):
                continue
            if isinstance(when, datetime) and when.tzinfo is None:
                return when.date()                       # a plain date (e.g. "2026-09-10"): already a day
            return when.astimezone(MANILA).date()
    return None


def load_always_use(path: Optional[str]) -> Dict[str, dict]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ALWAYS_USE_COLUMNS:
            raise ValueError(f"{path}: columns must be exactly {ALWAYS_USE_COLUMNS}")
        rows = {}
        for n, row in enumerate(reader, 2):
            if not all((row.get(c) or "").strip() for c in ALWAYS_USE_COLUMNS):
                raise ValueError(f"{path}, line {n}: every column needs a value")
            rows[row["document_id"].strip()] = {k: row[k].strip() for k in ALWAYS_USE_COLUMNS}
        return rows


def archive_files(archive_dir: str) -> List[str]:
    return sorted(os.path.join(archive_dir, n) for n in os.listdir(archive_dir)
                  if n.endswith(".jsonl") and not n.endswith(".history.jsonl"))


def served_once(records: List[NormalizedEvidence], threshold: float = DEFAULT_THRESHOLD):
    """filter_redundant() keep/drop decisions for `records`, computed per group of equal number sets (module notes).
    Kept posts carry no "please review" flag here (that flag needs comparisons across groups)."""
    groups: Dict[frozenset, List[NormalizedEvidence]] = {}
    for r in records:
        groups.setdefault(_numbers(r.text), []).append(r)
    decisions = []
    for group in groups.values():
        decisions.extend(filter_redundant(group, (), threshold=threshold))
    return decisions


def build_served(archive_dir: str, out_path: str, *, window_days: int, build_day: date,
                 always_use_path: Optional[str] = None, cutoff: Optional[datetime] = None,
                 threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Write the served collection and its build record; return the record."""
    if not isinstance(window_days, int) or window_days < 1:
        raise ValueError("window_days must be a whole number of days, at least 1 (owner decision D2)")
    files = archive_files(archive_dir)
    if not files:
        raise ValueError(f"no archive files in {archive_dir}")
    listed = load_always_use(always_use_path)
    cutoff_day = cutoff.astimezone(MANILA).date() if cutoff else None
    first_day = build_day - timedelta(days=window_days)
    counts: Dict[str, Dict[str, int]] = {}
    candidates: List[Tuple[NormalizedEvidence, Optional[date], str]] = []
    seen_ids = set()
    for path in files:
        page = os.path.splitext(os.path.basename(path))[0]
        c = counts.setdefault(page, {"in_archive": 0, "in_window": 0, "always_use": 0, "outside_window": 0,
                                     "undated": 0, "after_build_day": 0, "reserved": 0})
        for rec in load_corpus(path):
            c["in_archive"] += 1
            seen_ids.add(rec.document_id)
            day = post_day(rec)
            if cutoff_day is not None and day is not None and day >= cutoff_day:
                c["reserved"] += 1                       # second guard (S2 never archives these): never served, even if listed
                continue
            if rec.document_id in listed:
                c["always_use"] += 1
                candidates.append((rec, day, page))
            elif day is None:
                c["undated"] += 1
            elif day > build_day:
                c["after_build_day"] += 1
            elif day < first_day:
                c["outside_window"] += 1
            else:
                c["in_window"] += 1
                candidates.append((rec, day, page))
    page_of = {rec.document_id: page for rec, _, page in candidates}
    decisions = served_once([rec for rec, _, _ in candidates], threshold)
    kept_ids = {d.record.document_id for d in decisions if d.kept}
    duplicates = [{"document_id": d.record.document_id, "page": page_of[d.record.document_id],
                   "duplicate_of": d.matched_document_id, "other_page": page_of.get(d.matched_document_id),
                   "score": round(d.score, 3)} for d in decisions if not d.kept]
    served = sorted(((rec, day) for rec, day, _ in candidates if rec.document_id in kept_ids),
                    key=lambda x: ((x[1] or date.min).isoformat(), x[0].document_id))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for rec, _ in served:
            f.write(json.dumps(asdict(rec)) + "\n")
    os.replace(tmp, out_path)
    record = {
        "build_day": build_day.isoformat(), "window_days": window_days, "window_first_day": first_day.isoformat(),
        "timezone": "Asia/Manila (UTC+08:00)", "reservation_cutoff": cutoff.isoformat() if cutoff else None,
        "inputs": {os.path.basename(p): _sha256(p) for p in files},
        "always_use": {"file_sha256": _sha256(always_use_path) if listed else None, "listed": len(listed),
                       "not_in_archive": sorted(set(listed) - seen_ids)},
        "per_page": counts, "shared_posts_served_once": duplicates, "served": len(served),
        "output_sha256": _sha256(out_path),
    }
    with open(out_path + ".build.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, ensure_ascii=False, indent=1, sort_keys=True)
    return record
