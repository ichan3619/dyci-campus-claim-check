"""
Tests for M8.5-C -- Facebook evidence ingestion and source refresh
(src/ingestion/refresh.py, plus the adapter/evidence/source-map additions
it relies on).

PROVENANCE OF TEST DATA: everything here is SYNTHETIC. A small in-memory
FakePage stands in for the Graph API; `requests.get` is mocked in every
test, so nothing touches the network and no access token is needed. This
is an ingestion/integration test only -- it says nothing about model
accuracy, and it is NOT validation against the real DYCI Page
(M8.5-B = BLOCKED_ON_AUTHORIZATION).
"""

import hashlib
import os
import re
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from src.api.corpus_config import CORPUS_ENV_VAR, DEV_CORPUS_LABEL, load_configured_corpus
from src.ingestion.adapters import GraphAPIFetchError, MetaGraphAPIAdapter
from src.ingestion.corpus_storage import (
    load_corpus, load_corpus_for_retrieval, save_corpus,
)
from src.ingestion.evidence import NormalizedEvidence, normalize
from src.ingestion.refresh import refresh_corpus_from_facebook
from src.ingestion.source_map import annotate_origin, load_source_map
from src.retrieval.retriever import Retriever

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_CSV = os.path.join(ROOT, "data", "dev", "sample_claims.csv")
TOKEN = "SUPERSECRETTOKEN123"
MOCK_NOTE = "MOCK/TEST mirror Page - not the official DYCI Facebook Page"


# ---------------------------------------------------------------------------
# A tiny fake Graph API Page
# ---------------------------------------------------------------------------

class FakePage:
    """Newest-first list of posts, served page by page like /{page}/posts."""

    def __init__(self):
        self.posts = []      # newest first
        self.calls = []      # (url, params) for every GET

    def add(self, post_id, message, created):
        self.posts.insert(0, {
            "id": post_id, "message": message, "created_time": created,
            "permalink_url": f"https://www.facebook.com/mockpage/posts/{post_id}",
        })
        self.posts.sort(key=lambda p: p["created_time"], reverse=True)

    def _page(self, offset, limit):
        chunk = self.posts[offset:offset + limit]
        payload = {"data": chunk}
        if offset + limit < len(self.posts):
            # Like the real API, the next URL embeds the access token.
            payload["paging"] = {"next": (
                f"https://graph.facebook.com/v21.0/p/posts?offset={offset + limit}"
                f"&limit={limit}&access_token={TOKEN}")}
        return payload

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if params is not None:                       # first request
            offset, limit = 0, int(params["limit"])
        else:                                        # following paging.next
            q = parse_qs(urlparse(url).query)
            offset, limit = int(q["offset"][0]), int(q["limit"][0])
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = self._page(offset, limit)
        return resp


@pytest.fixture
def page():
    fp = FakePage()
    with patch("src.ingestion.adapters.requests.get", side_effect=fp.get):
        yield fp


def make_adapter(**kw):
    kw.setdefault("credibility_tier", "unverified")
    kw.setdefault("acquisition_note", MOCK_NOTE)
    return MetaGraphAPIAdapter(page_id="123", access_token=TOKEN, **kw)


def refresh(corpus, **kw):
    kw.setdefault("limit", 2)
    kw.setdefault("max_pages", 10)
    adapter = kw.pop("adapter", None) or make_adapter()
    return refresh_corpus_from_facebook(adapter, corpus, **kw), adapter


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def populate(page, n=3):
    texts = [
        "All classes are suspended on October 5 due to severe weather.",
        "The library will be closed on October 12 for inventory.",
        "Enrollment for the second semester ends on November 30 at the Registrar.",
        "Basketball tryouts start next Monday at the gymnasium.",
        "The scholarship application deadline is on January 15.",
    ]
    for i in range(n):
        page.add(f"P_{i + 1}", texts[i], f"2026-10-0{i + 1}T08:00:00+0000")


# ---------------------------------------------------------------------------
# Acceptance 1-6: first / second / third refresh, idempotence, retrieval
# ---------------------------------------------------------------------------

def test_first_refresh_adds_all_new_posts_and_persists_them(page, tmp_path):
    populate(page, 3)
    corpus = str(tmp_path / "mock.jsonl")

    r, _ = refresh(corpus)

    assert (r.added, r.known_skipped, r.corpus_size_before, r.corpus_size_after) == (3, 0, 0, 3)
    stored = load_corpus(corpus)
    assert len(stored) == 3
    assert all(isinstance(e, NormalizedEvidence) for e in stored)


def test_second_refresh_with_no_changes_is_idempotent(page, tmp_path):
    populate(page, 3)
    corpus = str(tmp_path / "mock.jsonl")
    refresh(corpus)
    before = file_hash(corpus)

    r, adapter = refresh(corpus)

    assert (r.added, r.redundant_skipped) == (0, 0)
    assert r.known_skipped == 3
    assert (r.corpus_size_before, r.corpus_size_after) == (3, 3)
    assert file_hash(corpus) == before                      # byte-for-byte unchanged
    assert len(load_corpus(corpus)) == 3


def test_third_refresh_adds_only_the_new_post_and_leaves_existing_untouched(page, tmp_path):
    populate(page, 3)
    corpus = str(tmp_path / "mock.jsonl")
    refresh(corpus)
    original_lines = open(corpus, encoding="utf-8").read().splitlines()

    page.add("P_4", "Basketball tryouts start next Monday at the gymnasium.", "2026-10-09T08:00:00+0000")
    r, _ = refresh(corpus)

    assert (r.added, r.known_skipped, r.corpus_size_after) == (1, 3, 4)
    assert r.added_document_ids == ["ingested:P_4"]
    new_lines = open(corpus, encoding="utf-8").read().splitlines()
    assert new_lines[:3] == original_lines                  # existing records byte-identical
    assert len(new_lines) == 4


def test_newly_ingested_evidence_is_retrievable_for_a_relevant_claim(page, tmp_path):
    page.add("P_1", "All classes are suspended on October 5 due to severe weather.", "2026-10-04T08:00:00+0000")
    page.add("P_2", "Basketball tryouts start next Monday at the gymnasium.", "2026-10-03T08:00:00+0000")
    corpus = str(tmp_path / "mock.jsonl")
    refresh(corpus)

    retriever = Retriever(load_corpus_for_retrieval(corpus))
    hits = retriever.retrieve("Classes are suspended on October 5 because of severe weather.", top_k=1)

    assert hits[0].document_id == "ingested:P_1"
    assert hits[0].source_type == "facebook"
    assert hits[0].score > 0.15                              # above the relevance threshold


def test_api_corpus_config_loads_the_refreshed_corpus(page, tmp_path, monkeypatch):
    populate(page, 2)
    corpus = str(tmp_path / "mock.jsonl")
    refresh(corpus)

    monkeypatch.setenv(CORPUS_ENV_VAR, corpus)
    docs, label = load_configured_corpus(ROOT, DEV_CSV)

    assert label == "mock.jsonl"
    assert {d["document_id"] for d in docs} == {"ingested:P_1", "ingested:P_2"}


def test_dry_run_reports_but_writes_nothing(page, tmp_path):
    populate(page, 2)
    corpus = str(tmp_path / "mock.jsonl")

    r, _ = refresh(corpus, dry_run=True)

    assert (r.added, r.dry_run, r.corpus_size_after) == (2, True, 0)
    assert not os.path.exists(corpus)


# ---------------------------------------------------------------------------
# Identity: post ID, never text
# ---------------------------------------------------------------------------

def test_known_posts_are_not_reprocessed(page, tmp_path):
    populate(page, 3)
    corpus = str(tmp_path / "mock.jsonl")
    refresh(corpus)

    with patch("src.ingestion.adapters.normalize", wraps=normalize) as spy:
        r, _ = refresh(corpus)

    assert spy.call_count == 0                               # nothing re-normalized
    assert r.known_skipped == 3


def test_identity_is_the_post_id_not_the_text(page, tmp_path):
    # An edited post is the SAME post: it never becomes a second record. Since the edit/deletion sync (M12.3 follow-up)
    # its stored text is updated; with sync_existing=False the M8.5-C behaviour (keep the old text) is unchanged.
    page.add("P_1", "The library will be closed on October 12 for inventory.", "2026-10-01T08:00:00+0000")
    corpus = str(tmp_path / "mock.jsonl")
    refresh(corpus)

    page.posts[0]["message"] = "Something completely different now."
    r, _ = refresh(corpus, sync_existing=False)
    assert (r.added, r.known_skipped, r.edited) == (0, 1, 0)
    assert "library" in load_corpus(corpus)[0].text

    r, _ = refresh(corpus)
    assert (r.added, r.known_skipped, r.edited) == (0, 1, 1)
    records = load_corpus(corpus)
    assert len(records) == 1 and records[0].document_id == "ingested:P_1"
    assert records[0].text == "Something completely different now."


def test_distinct_numbers_keep_distinct_posts_but_true_repost_is_redundant(page, tmp_path):
    page.add("P_1", "The library will be closed on October 12 for inventory.", "2026-10-01T08:00:00+0000")
    page.add("P_2", "The library will be closed on October 20 for inventory.", "2026-10-02T08:00:00+0000")   # different date
    page.add("P_3", "The library will be closed on October 12 for inventory.", "2026-10-03T08:00:00+0000")   # true repost
    corpus = str(tmp_path / "mock.jsonl")

    r, _ = refresh(corpus)

    ids = {e.document_id for e in load_corpus(corpus)}
    assert ids == {"ingested:P_1", "ingested:P_2"}           # differing date kept, exact repost skipped
    assert (r.added, r.redundant_skipped, r.flagged_for_review) == (2, 1, 1)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def test_refresh_follows_next_cursors_across_pages(page, tmp_path):
    populate(page, 5)
    corpus = str(tmp_path / "mock.jsonl")

    r, adapter = refresh(corpus, limit=2)

    assert r.added == 5
    assert len(page.calls) == 3                              # 2 + 2 + 1
    assert page.calls[1][1] is None and "access_token" in page.calls[1][0]   # next URL used as-is
    assert r.has_more_pages is False


def test_max_pages_limit_is_reported_not_silent(page, tmp_path):
    populate(page, 5)
    corpus = str(tmp_path / "mock.jsonl")

    r, _ = refresh(corpus, limit=2, max_pages=2)

    assert r.added == 4
    assert r.has_more_pages is True                          # one older post was not read
    r2, _ = refresh(corpus, limit=2, max_pages=10)           # a later, wider refresh picks it up
    assert (r2.added, r2.corpus_size_after) == (1, 5)


def test_same_post_on_two_pages_is_stored_once(page, tmp_path):
    populate(page, 3)
    page.posts.insert(2, dict(page.posts[0]))                # same post reappears on page 2
    corpus = str(tmp_path / "mock.jsonl")

    r, _ = refresh(corpus, limit=2)

    ids = [e.document_id for e in load_corpus(corpus)]
    assert len(ids) == len(set(ids)) == 3
    assert r.added == 3


def test_page_two_failure_raises_scrubbed_error_and_writes_nothing(tmp_path):
    corpus = str(tmp_path / "mock.jsonl")
    good = Mock(status_code=200)
    good.json.return_value = {
        "data": [{"id": "P_1", "message": "A notice.", "created_time": "2026-10-01T00:00:00+0000",
                  "permalink_url": "https://fb/1"}],
        "paging": {"next": f"https://graph.facebook.com/v21.0/p/posts?offset=1&access_token={TOKEN}"},
    }
    boom = requests.exceptions.ConnectionError(f"failed: https://graph.facebook.com/x?access_token={TOKEN}")
    with patch("src.ingestion.adapters.requests.get", side_effect=[good, boom]):
        with pytest.raises(GraphAPIFetchError) as exc:
            refresh(corpus, limit=1)

    assert TOKEN not in str(exc.value)
    assert not os.path.exists(corpus)                        # a failed refresh leaves no partial corpus


def test_since_is_opt_in_and_passed_to_the_graph_api(page, tmp_path):
    populate(page, 1)
    corpus = str(tmp_path / "mock.jsonl")
    refresh(corpus)
    assert "since" not in page.calls[0][1]                   # off by default

    page.calls.clear()
    refresh(corpus, since="2026-10-01")
    assert page.calls[0][1]["since"] == "2026-10-01"


def test_stop_at_known_page_is_opt_in_and_saves_requests(page, tmp_path):
    populate(page, 5)
    corpus = str(tmp_path / "mock.jsonl")
    refresh(corpus, limit=2)
    page.add("P_NEW", "Foundation Day is on December 8.", "2026-10-20T08:00:00+0000")

    page.calls.clear()
    r_default, _ = refresh(corpus, limit=2)                  # default: reads every page
    assert len(page.calls) == 3 and r_default.added == 1

    page.add("P_NEW2", "Intramurals begin on February 3.", "2026-10-21T08:00:00+0000")
    page.calls.clear()
    r_fast, _ = refresh(corpus, limit=2, stop_at_known_page=True)
    assert len(page.calls) == 2                              # new+known page, then an all-known page -> stop
    assert (r_fast.added, r_fast.stopped_at_known_page) == (1, True)


def test_stop_at_known_page_can_miss_a_backdated_post_default_does_not(page, tmp_path):
    # Documents WHY the optimisation is off by default.
    populate(page, 5)
    corpus = str(tmp_path / "mock.jsonl")
    refresh(corpus, limit=2)
    page.add("P_BACKDATED", "Graduation rehearsal is on March 3.", "2026-09-01T08:00:00+0000")   # older than all known

    fast, _ = refresh(corpus, limit=2, stop_at_known_page=True)
    assert fast.added == 0                                   # missed: it sits behind an all-known page

    full, _ = refresh(corpus, limit=2)
    assert full.added == 1                                   # default correctness-first mode finds it


# ---------------------------------------------------------------------------
# Provenance: acquisition source vs content origin; mock never masquerades
# ---------------------------------------------------------------------------

def test_records_are_marked_as_mock_mirror_not_official(page, tmp_path):
    populate(page, 1)
    corpus = str(tmp_path / "mock.jsonl")
    refresh(corpus)

    rec = load_corpus(corpus)[0]

    assert rec.source_type == "facebook"                     # existing allowed type, unchanged
    assert rec.credibility_tier == "unverified"              # existing tier: not official/recognized
    assert "MOCK" in rec.acquisition_note and "not the official DYCI" in rec.acquisition_note
    assert rec.source_identifier == "P_1"                    # the mock post's Graph ID
    assert rec.source_url == "https://www.facebook.com/mockpage/posts/P_1"   # the mock post's URL
    assert rec.credibility_tier not in ("official", "recognized")


def test_source_map_records_origin_without_losing_the_mock_url_or_date(page, tmp_path):
    populate(page, 2)
    map_csv = tmp_path / "map.csv"
    map_csv.write_text(
        "mock_post,original_url,original_date\n"
        "P_1,https://www.dyci.edu.ph/announcement/1,2026-10-01\n", encoding="utf-8")
    corpus = str(tmp_path / "mock.jsonl")

    r, _ = refresh(corpus, source_map=load_source_map(str(map_csv)))

    by_id = {e.source_identifier: e for e in load_corpus(corpus)}
    mapped, unmapped = by_id["P_1"], by_id["P_2"]
    assert mapped.origin_url == "https://www.dyci.edu.ph/announcement/1"
    assert mapped.origin_date == "2026-10-01"
    assert mapped.source_url.endswith("/posts/P_1")          # mock permalink preserved
    assert mapped.published_date == "2026-10-01T08:00:00+0000"   # mirror timestamp preserved
    assert (unmapped.origin_url, unmapped.origin_date) == ("", "")
    assert r.added_without_origin == 1


def test_annotate_origin_does_not_alter_identity_or_text():
    r = normalize(text="Notice.", source_type="facebook", source_url="https://mock/1",
                  published_date="2026-10-05T00:00:00+0000", source_identifier="1_1")
    out, unmapped = annotate_origin([r], {"1_1": ("https://dyci/orig", "2026-09-15")})
    assert (out[0].document_id, out[0].text, out[0].source_identifier) == (r.document_id, r.text, "1_1")
    assert unmapped == []


def test_provenance_fields_never_reach_retrieval_documents(page, tmp_path):
    populate(page, 1)
    corpus = str(tmp_path / "mock.jsonl")
    refresh(corpus)
    doc = load_corpus_for_retrieval(corpus)[0]
    assert set(doc.keys()) == {"document_id", "text", "source_type", "source_url"}


# ---------------------------------------------------------------------------
# Persistence / backward compatibility
# ---------------------------------------------------------------------------

def test_corpus_written_before_m85c_still_loads_with_empty_provenance(tmp_path):
    old_line = ('{"document_id": "ingested:1_1", "text": "Old record.", "source_type": "facebook", '
                '"source_url": "https://fb/1", "published_date": "2026-09-27T04:26:04+0000", '
                '"fetched_at": "2026-09-29T11:07:30+00:00", "credibility_tier": "", '
                '"source_identifier": "1_1"}\n')
    path = tmp_path / "old.jsonl"
    path.write_text(old_line, encoding="utf-8")

    rec = load_corpus(str(path))[0]

    assert (rec.origin_url, rec.origin_date, rec.acquisition_note) == ("", "", "")


def test_new_fields_round_trip_through_save_and_load(tmp_path):
    r = normalize(text="Notice.", source_type="facebook", source_identifier="1_1",
                  origin_url="https://dyci/orig", origin_date="2026-09-15", acquisition_note=MOCK_NOTE)
    path = str(tmp_path / "c.jsonl")
    save_corpus([r], path)
    assert load_corpus(path)[0] == r


def test_refresh_never_persists_the_access_token(page, tmp_path):
    populate(page, 5)
    corpus = str(tmp_path / "mock.jsonl")
    refresh(corpus, limit=2)
    with open(corpus, encoding="utf-8") as f:
        assert TOKEN not in f.read()


# ---------------------------------------------------------------------------
# Corpus isolation
# ---------------------------------------------------------------------------

def test_refresh_touches_only_the_corpus_path_it_is_given(page, tmp_path):
    populate(page, 2)
    dev_before = file_hash(DEV_CSV)
    corpus = str(tmp_path / "mock.jsonl")

    refresh(corpus)

    assert file_hash(DEV_CSV) == dev_before                  # dev dataset untouched
    assert os.listdir(tmp_path) == ["mock.jsonl"]            # no stray files


def test_default_api_corpus_is_still_the_dev_dataset(page, tmp_path, monkeypatch):
    populate(page, 2)
    refresh(str(tmp_path / "mock.jsonl"))
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)

    docs, label = load_configured_corpus(ROOT, DEV_CSV)      # mock corpus exists but is opt-in only

    assert label == DEV_CORPUS_LABEL
    assert not any(d["document_id"].startswith("ingested:") for d in docs)


# ---------------------------------------------------------------------------
# Documentation guard: real DYCI stays blocked
# ---------------------------------------------------------------------------

def test_methodology_keeps_real_dyci_facebook_blocked():
    with open(os.path.join(ROOT, "docs", "methodology.md"), encoding="utf-8") as f:
        text = f.read()
    assert "M8.5-C" in text
    assert re.search(r"M8\.5-B[^\n]*BLOCKED_ON_AUTHORIZATION", text)
