"""
Refreshing the corpus from the MOCK/TEST Facebook Page -- shared by scripts/ingest_facebook_page.py (manual) and the
API's optional background refresh (src/api/pipeline.py), so both write records the same way.

THIS IS NOT DYCI INGESTION. Every record is stamped credibility_tier="unverified" with an acquisition note saying it
came from a MOCK/TEST mirror Page. Credentials come from environment variables ONLY (META_PAGE_ID, META_ACCESS_TOKEN),
and every error message passes through scrub() so the token is never printed, logged or returned by /health.
"""
import os
import re
from typing import Dict, Optional, Tuple

from src.ingestion.adapters import MetaGraphAPIAdapter
from src.ingestion.refresh import RefreshResult, refresh_corpus_from_facebook
from src.ingestion.source_map import load_source_map

DEFAULT_CORPUS = os.path.join("data", "corpus", "mock_facebook_corpus.jsonl")
DEFAULT_SOURCE_MAP = os.path.join("data", "source_map.csv")
MOCK_NOTE = "MOCK/TEST mirror Page - not the official DYCI Facebook Page"
PAGE_ID_ENV, TOKEN_ENV = "META_PAGE_ID", "META_ACCESS_TOKEN"


def scrub(text: str, token: str = "") -> str:
    """Remove the access token from any text before it is shown anywhere."""
    if token:
        text = text.replace(token, "[REDACTED]")
    return re.sub(r"(access_token=)[^&\s'\"]+", r"\1[REDACTED]", text)


# Graph API error codes that mean "fix the credentials", with what to do about them.
_CREDENTIAL_HINTS = {
    "190": "the access token is expired or invalid -- generate a new Page access token in the Meta developer tools "
           "and set META_ACCESS_TOKEN again",
    "10": "the token lacks permission to read this Page's posts (pages_read_engagement / Page Public Content Access)",
    "200": "the token lacks permission to read this Page's posts (pages_read_engagement / Page Public Content Access)",
    "100": "META_PAGE_ID is wrong, or the token cannot see that Page",
}


def explain_fetch_error(message: str, token: str = "") -> str:
    """The scrubbed error message, plus what to do when the Graph API rejected the credentials."""
    text = scrub(message, token)
    m = re.search(r"\(code=(\d+)\)", text)
    hint = _CREDENTIAL_HINTS.get(m.group(1)) if m else None
    return f"{text} -> {hint}" if hint else text


def credentials_from_env() -> Optional[Tuple[str, str]]:
    """(page_id, token) from the environment, or None if either is missing."""
    page_id, token = os.environ.get(PAGE_ID_ENV, "").strip(), os.environ.get(TOKEN_ENV, "").strip()
    return (page_id, token) if page_id and token else None


def load_optional_source_map(path: str) -> Optional[Dict[str, Tuple[str, str]]]:
    """The mock-post -> original URL/date map, or None when the file is absent. Raises ValueError if it is malformed."""
    return load_source_map(path) if os.path.exists(path) else None


def make_mock_page_adapter(page_id: str, token: str) -> MetaGraphAPIAdapter:
    return MetaGraphAPIAdapter(page_id=page_id, access_token=token, credibility_tier="unverified", acquisition_note=MOCK_NOTE)


def refresh_from_mock_page(corpus_path: str, page_id: str, token: str, source_map_path: str = DEFAULT_SOURCE_MAP,
                           **options) -> Tuple[RefreshResult, MetaGraphAPIAdapter]:
    """Fetch the mock Page and append only NEW posts to `corpus_path` (see src/ingestion/refresh.py for what 'new' means)."""
    adapter = make_mock_page_adapter(page_id, token)
    from src.ingestion import post_images
    if post_images.enabled():           # fix A: caption + the text read from the post's picture (cached by post ID)
        adapter.post_text = post_images.PostImageReader(corpus_path + ".image_text.json")
    from src.image import auto_index
    if auto_index.enabled():            # rule I: store each post's picture and its reading for the photo check
        adapter.on_post = auto_index.PictureCollector(auto_index.folder_for(corpus_path))
    try:
        result = refresh_corpus_from_facebook(adapter, corpus_path, source_map=load_optional_source_map(source_map_path),
                                              **options)
    except Exception as e:
        _record_fetch(None, adapter, explain_fetch_error(str(e), token))
        raise
    _record_fetch(result, adapter, None)
    return result, adapter


# ---- fetch log (Sources page "Live fetch" box; CAMPUS_VERIFY_FETCH_STATUS shows it) ------------------------------------
# The last FETCH_LOG_SIZE fetches, newest last, in memory only (gone after a restart). Counts and post IDs with the reason
# a post was skipped; never post text, never the token (errors are scrubbed by explain_fetch_error).
FETCH_LOG_SIZE = 30
_FETCH_LOG: list = []


def _record_fetch(result: Optional[RefreshResult], adapter: MetaGraphAPIAdapter, error: Optional[str]) -> None:
    from datetime import datetime, timezone
    entry = {"at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "ok": error is None, "error": error}
    if result is not None:
        entry.update({"added": result.added, "edited": result.edited, "retired": result.retired,
                      "skipped_no_text": result.unusable_skipped, "skipped_near_duplicate": result.redundant_skipped,
                      "flagged_for_review": result.flagged_for_review, "retire_refused": result.retire_refused})
    problems = [*(getattr(getattr(adapter, "post_text", None), "problems", None) or []),   # fix A: pictures not read
                *(getattr(getattr(adapter, "on_post", None), "problems", None) or [])]     # rule I: pictures not stored
    entry["skipped_posts"] = [{"post_id": str(s.get("post_id", "")), "reason": str(s.get("reason", ""))}
                              for s in [*(getattr(adapter, "skipped_posts", None) or []), *problems]][:20]
    reader = getattr(adapter, "post_text", None)
    if reader is not None and hasattr(reader, "read_now"):
        entry["pictures_read"] = len(reader.read_now)
    collector = getattr(adapter, "on_post", None)
    if collector is not None and hasattr(collector, "saved"):
        entry["pictures_stored"] = len(collector.saved)
    _FETCH_LOG.append(entry)
    del _FETCH_LOG[:-FETCH_LOG_SIZE]


def fetch_log() -> list:
    """A copy of the recent fetches, newest first."""
    return [dict(e) for e in reversed(_FETCH_LOG)]


def record_website_fetch(counts: Optional[dict], error: Optional[str]) -> None:
    """Rule W: a read of DYCI's website feeds, in the same log as the test-page fetches (source "dyci_website")."""
    from datetime import datetime, timezone
    entry = {"at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "ok": error is None, "error": error,
             "source": "dyci_website", "skipped_posts": []}
    if counts:
        entry.update({"added": counts.get("added"), "edited": counts.get("edited"), "retired": counts.get("removed"),
                      "items": counts.get("items")})
    _FETCH_LOG.append(entry)
    del _FETCH_LOG[:-FETCH_LOG_SIZE]
