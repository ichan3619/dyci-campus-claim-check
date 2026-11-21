"""
Facebook source refresh (M8.5-C).

Pipeline position:

    MetaGraphAPIAdapter.fetch_posts() -> [this module] -> corpus_storage -> Retriever

Responsibility of this module, and ONLY this module: one idempotent
"refresh" of a persistent corpus file from an already-configured
MetaGraphAPIAdapter -- work out which fetched posts are new, add only
those, and report exactly what happened. It fetches nothing itself
(the adapter does), normalizes nothing itself (evidence.normalize() via
the adapter does), and does no retrieval or classification.

IDENTITY vs REDUNDANCY (two separate concepts, kept separate)
-------------------------------------------------------------
1. Identity -- the Graph API post ID. A corpus record's `source_identifier`
   IS that ID and its document_id is derived from it, so "already seen"
   is decided by ID, never by text. Two posts with identical text but
   different IDs are two different posts.
2. Redundancy -- src/ingestion/dedup.py, unchanged. A DIFFERENT post whose
   text is near-identical AND carries the same numbers is skipped (oldest
   kept); similar text with different numbers is kept and flagged. This
   is the project's existing, documented redundancy design, applied here
   as-is.

INCREMENTAL STRATEGY (what this actually does)
----------------------------------------------
Default: read up to `max_pages` pages, newest first. Posts whose ID is
already in the corpus are dropped inside the adapter BEFORE normalization
(so they are not re-processed), then anything new is checked for
redundancy and appended. Correctness first: a new post cannot be missed
as long as it falls inside the pages read (`has_more_pages` says when
older pages were not read).
Optional, opt-in: `since` (Graph API filter) and `stop_at_known_page`
(stop paging at the first page made up entirely of known posts). Both can
miss a new post with an old created_time, so they are never on by default.

IDEMPOTENCE
-----------
Running a refresh twice against an unchanged Page adds 0 records and
leaves the corpus file byte-for-byte identical.
"""

import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.ingestion.adapters import MetaGraphAPIAdapter
from src.ingestion.corpus_storage import append_to_corpus, load_corpus, rewrite_corpus
from src.ingestion.evidence import NormalizedEvidence, normalize
from src.ingestion.dedup import DEFAULT_THRESHOLD, DedupDecision, filter_redundant
from src.ingestion.source_map import annotate_origin


@dataclass
class RefreshResult:
    corpus_size_before: int
    corpus_size_after: int
    added: int                       # records written (or that WOULD be, if dry_run)
    known_skipped: int               # post ID already in the corpus (not re-processed)
    redundant_skipped: int           # different post, near-identical text + same numbers
    unusable_skipped: int            # no usable text (adapter.skipped_posts)
    flagged_for_review: int          # kept, but similar text with different numbers
    added_without_origin: int        # added records with no source-map entry
    has_more_pages: bool             # older pages existed and were not read
    stopped_at_known_page: bool
    dry_run: bool
    added_document_ids: List[str] = field(default_factory=list)
    # Edits and deletions of posts already in the corpus (sync_existing=True). See plan_sync().
    edited: int = 0
    retired: int = 0
    edited_document_ids: List[str] = field(default_factory=list)
    retired_document_ids: List[str] = field(default_factory=list)
    retire_refused: Optional[str] = None          # why missing posts were NOT retired (a likely glitch), if so


# ---- edits and deletions ----------------------------------------------------------------------------------------------
# A post already in the corpus can be EDITED on the Page (its text changes) or DELETED / HIDDEN. Left alone, the old
# text keeps being used as evidence. plan_sync() decides, from what the last fetch returned:
#   edited   the post came back with different text          -> the record's text is replaced (same document_id)
#   emptied  the post came back with no text                  -> the record is retired (no longer evidence)
#   missing  the post did not come back although the fetch covered its date (created_time >= the oldest post fetched)
#            AND it belongs to the fetched Page (Graph post IDs are "<page id>_<post id>")   -> retired
# Safety: an empty fetch retires nothing; if more than `max_retire_fraction` of the Page's records (and more than 2) would
# be retired as missing, none are (likely an API glitch or a wrong page/token) and the result says so.
# Old text is never lost: every change is appended to <corpus>.history.jsonl.
DEFAULT_MAX_RETIRE_FRACTION = 0.5


@dataclass
class SyncPlan:
    edited: List[Tuple[NormalizedEvidence, NormalizedEvidence]] = field(default_factory=list)   # (old, new)
    retired: List[Tuple[NormalizedEvidence, str]] = field(default_factory=list)                 # (record, reason)
    retire_refused: Optional[str] = None


def history_path(corpus_path: str) -> str:
    return os.path.splitext(corpus_path)[0] + ".history.jsonl"


def _when(value: str) -> Optional[datetime]:
    for parse in (lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S%z"), datetime.fromisoformat):
        try:
            return parse(value)
        except (TypeError, ValueError):
            continue
    return None


def plan_sync(existing: List[NormalizedEvidence], adapter, max_retire_fraction: float = DEFAULT_MAX_RETIRE_FRACTION,
              now: Optional[str] = None) -> SyncPlan:
    now = now or datetime.now(timezone.utc).isoformat()
    plan = SyncPlan()
    by_id = {r.source_identifier: r for r in existing if r.source_identifier}
    for post_id, current in getattr(adapter, "known_current", {}).items():
        old = by_id.get(post_id)
        if old is None:
            continue
        if current.get("message") is None:
            plan.retired.append((old, "the post has no text any more"))
            continue
        fresh = normalize(text=current["message"], source_type=old.source_type, source_url=old.source_url,
                          published_date=old.published_date, fetched_at=now, credibility_tier=old.credibility_tier,
                          source_identifier=old.source_identifier)
        if fresh.text != old.text:
            plan.edited.append((old, replace(old, text=fresh.text, fetched_at=now)))

    times = [t for t in (_when(x) for x in getattr(adapter, "fetched_created_times", [])) if t]
    if not times:
        return plan                                       # nothing was fetched: never infer deletions from that
    oldest, seen = min(times), set(getattr(adapter, "fetched_ids", []))
    prefix = f"{adapter.page_id}_"
    page_records = [r for r in existing if (r.source_identifier or "").startswith(prefix)]
    missing = []
    for r in page_records:
        published = _when(r.published_date)
        if r.source_identifier not in seen and published is not None and published >= oldest:
            missing.append((r, "no longer on the Page (deleted or hidden)"))
    if len(missing) > 2 and len(missing) > max_retire_fraction * len(page_records):
        plan.retire_refused = (f"{len(missing)} of {len(page_records)} posts from this Page did not come back; not "
                               f"retiring any (likely an API glitch, a wrong Page ID or a token without access)")
    else:
        plan.retired.extend(missing)
    return plan


def _write_history(corpus_path: str, plan: SyncPlan, now: str) -> None:
    with open(history_path(corpus_path), "a", encoding="utf-8") as f:
        for old, new in plan.edited:
            f.write(json.dumps({"time": now, "action": "edited", "document_id": old.document_id,
                                "source_identifier": old.source_identifier, "old_text": old.text, "new_text": new.text}) + "\n")
        for old, reason in plan.retired:
            f.write(json.dumps({"time": now, "action": "retired", "document_id": old.document_id,
                                "source_identifier": old.source_identifier, "reason": reason, "old_text": old.text}) + "\n")


def refresh_corpus_from_facebook(
    adapter: MetaGraphAPIAdapter,
    corpus_path: str,
    *,
    limit: int = 25,
    max_pages: int = 5,
    since: Optional[str] = None,
    until: Optional[str] = None,
    stop_at_known_page: bool = False,
    source_map: Optional[Dict[str, Tuple[str, str]]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    dry_run: bool = False,
    sync_existing: bool = True,
    max_retire_fraction: float = DEFAULT_MAX_RETIRE_FRACTION,
) -> RefreshResult:
    """Fetch the Page, add NEW posts to `corpus_path`, apply edits/deletions of known posts (sync_existing), and
    report counts. With no edits or deletions the file is only appended to, exactly as before."""
    existing = load_corpus(corpus_path) if os.path.exists(corpus_path) else []
    known_post_ids = {r.source_identifier for r in existing if r.source_identifier}

    fetch_kwargs = {"until": until} if until else {}  # only when asked: older adapters and test doubles lack it
    fetched = adapter.fetch_posts(
        limit=limit, max_pages=max_pages, since=since,
        known_post_ids=known_post_ids, stop_at_known_page=stop_at_known_page, **fetch_kwargs,
    )

    if source_map:
        fetched, unmapped = annotate_origin(fetched, source_map)
    else:
        unmapped = list(fetched)
    unmapped_ids = {r.document_id for r in unmapped}

    now = datetime.now(timezone.utc).isoformat()
    plan = plan_sync(existing, adapter, max_retire_fraction, now) if sync_existing else SyncPlan()
    retired_ids = {r.document_id for r, _ in plan.retired}
    edited_new = {old.document_id: new for old, new in plan.edited}
    # Redundancy is judged against the corpus as it will be: a re-post of a deleted post is not "redundant".
    remaining = [edited_new.get(r.document_id, r) for r in existing if r.document_id not in retired_ids]
    decisions: List[DedupDecision] = filter_redundant(fetched, remaining, threshold=threshold)
    kept = [d.record for d in decisions if d.kept]

    added_ids: List[str] = []
    if dry_run:
        added_ids = [r.document_id for r in kept]
    elif plan.edited or plan.retired:                # one atomic rewrite; untouched lines keep their bytes
        rewrite_corpus(corpus_path, {**edited_new, **{i: None for i in retired_ids}}, kept)
        _write_history(corpus_path, plan, now)
        added_ids = [r.document_id for r in kept]
    else:                                            # nothing changed: append only, exactly as before
        added_ids = [r.document_id for r in kept if append_to_corpus(r, corpus_path)]

    return RefreshResult(
        corpus_size_before=len(existing),
        corpus_size_after=len(existing) + (0 if dry_run else len(added_ids) - len(retired_ids)),
        added=len(added_ids),
        known_skipped=len(adapter.known_skipped),
        redundant_skipped=sum(1 for d in decisions if not d.kept),
        unusable_skipped=len(adapter.skipped_posts),
        flagged_for_review=sum(1 for d in decisions if d.kept and d.numbers_differ),
        added_without_origin=sum(1 for i in added_ids if i in unmapped_ids),
        has_more_pages=adapter.has_more_pages,
        stopped_at_known_page=adapter.stopped_at_known_page,
        dry_run=dry_run,
        added_document_ids=added_ids,
        edited=len(plan.edited),
        retired=len(plan.retired),
        edited_document_ids=[old.document_id for old, _ in plan.edited],
        retired_document_ids=sorted(retired_ids),
        retire_refused=plan.retire_refused,
    )
