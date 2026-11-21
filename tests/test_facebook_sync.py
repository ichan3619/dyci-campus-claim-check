"""
Edited and deleted Facebook posts (src/ingestion/refresh.py plan_sync + corpus_storage.rewrite_corpus). Uses the fake
Graph API Page from tests/test_m85c_refresh.py; post IDs follow the real "<page id>_<post id>" form (page id "123").
"""
import json
import os

from src.ingestion.corpus_storage import append_to_corpus, load_corpus
from src.ingestion.refresh import history_path
from tests.test_m11_2_corpus import rec
from tests.test_m85c_refresh import page, refresh  # noqa: F401  (page is a fixture)


def _seed(page, corpus, n=4):
    for i in range(1, n + 1):
        page.add(f"123_{i}", f"Notice {i}: office hours change on October {10 + i}.", f"2026-10-0{i}T08:00:00+0000")
    refresh(corpus)
    return {r.source_identifier: r for r in load_corpus(corpus)}


def _history(corpus):
    with open(history_path(corpus), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_an_edited_post_replaces_its_text_and_keeps_the_old_text_in_history(page, tmp_path):
    corpus = str(tmp_path / "fb.jsonl")
    before = _seed(page, corpus)
    raw_before = open(corpus, "rb").read().splitlines(keepends=True)
    next(p for p in page.posts if p["id"] == "123_2")["message"] = "Notice 2: office hours change on October 20."
    r, _ = refresh(corpus)
    after = {x.source_identifier: x for x in load_corpus(corpus)}
    assert (r.edited, r.retired, r.added) == (1, 0, 0) and r.edited_document_ids == [before["123_2"].document_id]
    assert after["123_2"].text.endswith("October 20.") and after["123_2"].document_id == before["123_2"].document_id
    raw_after = open(corpus, "rb").read().splitlines(keepends=True)
    changed = [i for i, (a, b) in enumerate(zip(raw_before, raw_after)) if a != b]
    assert len(changed) == 1                                            # every other line byte-identical
    h = _history(corpus)
    assert h[0]["action"] == "edited" and "October 12" in h[0]["old_text"] and "October 20" in h[0]["new_text"]


def test_a_deleted_post_inside_the_fetched_range_is_retired(page, tmp_path):
    corpus = str(tmp_path / "fb.jsonl")
    _seed(page, corpus)
    page.posts = [p for p in page.posts if p["id"] != "123_3"]
    r, _ = refresh(corpus)
    ids = {x.source_identifier for x in load_corpus(corpus)}
    assert r.retired == 1 and "123_3" not in ids and len(ids) == 3 and r.corpus_size_after == 3
    assert _history(corpus)[0]["reason"].startswith("no longer on the Page")


def test_posts_older_than_what_the_fetch_covered_are_not_judged(page, tmp_path):
    corpus = str(tmp_path / "fb.jsonl")
    _seed(page, corpus)
    page.posts = [p for p in page.posts if p["id"] != "123_1"]           # the OLDEST post is gone ...
    r, _ = refresh(corpus, limit=1, max_pages=1)                        # ... but only the newest post was fetched
    assert r.retired == 0 and len(load_corpus(corpus)) == 4


def test_posts_from_another_page_are_never_retired(page, tmp_path):
    corpus = str(tmp_path / "fb.jsonl")
    _seed(page, corpus)
    other = rec("Posted on a different Page.", "999_7", date="2026-10-03T09:00:00+0000")
    append_to_corpus(other, corpus)
    r, _ = refresh(corpus)
    assert r.retired == 0 and "999_7" in {x.source_identifier for x in load_corpus(corpus)}


def test_a_post_whose_text_was_removed_is_retired(page, tmp_path):
    corpus = str(tmp_path / "fb.jsonl")
    _seed(page, corpus)
    next(p for p in page.posts if p["id"] == "123_4")["message"] = ""
    r, _ = refresh(corpus)
    assert r.retired == 1 and "123_4" not in {x.source_identifier for x in load_corpus(corpus)}
    assert _history(corpus)[0]["reason"] == "the post has no text any more"


def test_an_empty_fetch_retires_nothing(page, tmp_path):
    corpus = str(tmp_path / "fb.jsonl")
    _seed(page, corpus)
    page.posts = []
    r, _ = refresh(corpus)
    assert r.retired == 0 and len(load_corpus(corpus)) == 4


def test_mass_disappearance_is_refused_as_a_likely_glitch(page, tmp_path):
    corpus = str(tmp_path / "fb.jsonl")
    _seed(page, corpus, n=5)
    page.posts = [p for p in page.posts if p["id"] == "123_1"]          # only the oldest post still comes back
    before = open(corpus, "rb").read()
    r, _ = refresh(corpus)
    assert r.retired == 0 and r.retire_refused and "4 of 5" in r.retire_refused
    assert open(corpus, "rb").read() == before


def test_dry_run_reports_but_writes_nothing(page, tmp_path):
    corpus = str(tmp_path / "fb.jsonl")
    _seed(page, corpus)
    page.posts = [p for p in page.posts if p["id"] != "123_2"]
    next(p for p in page.posts if p["id"] == "123_3")["message"] = "Notice 3: changed."
    before = open(corpus, "rb").read()
    r, _ = refresh(corpus, dry_run=True)
    assert (r.edited, r.retired) == (1, 1) and open(corpus, "rb").read() == before
    assert not os.path.exists(history_path(corpus))


def test_a_repost_of_a_deleted_notice_comes_back(page, tmp_path):
    corpus = str(tmp_path / "fb.jsonl")
    _seed(page, corpus)
    old = next(p for p in page.posts if p["id"] == "123_2")
    page.posts = [p for p in page.posts if p["id"] != "123_2"]
    page.add("123_9", old["message"], "2026-10-09T08:00:00+0000")       # deleted, then posted again
    r, _ = refresh(corpus)
    ids = {x.source_identifier for x in load_corpus(corpus)}
    assert r.retired == 1 and r.added == 1 and "123_9" in ids and "123_2" not in ids


def test_unchanged_page_leaves_the_file_and_history_untouched(page, tmp_path):
    corpus = str(tmp_path / "fb.jsonl")
    _seed(page, corpus)
    before = open(corpus, "rb").read()
    r, _ = refresh(corpus)
    assert (r.edited, r.retired, r.added) == (0, 0, 0) and open(corpus, "rb").read() == before
    assert not os.path.exists(history_path(corpus))
