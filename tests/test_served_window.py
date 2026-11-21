"""
The served-collection build (src/ingestion/served_window.py, scripts/build_served_corpus.py; scale plan S3).
Synthetic archives only.
"""
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from src.ingestion.corpus_storage import load_corpus, save_corpus
from src.ingestion.evidence import normalize
from src.ingestion import served_window as sw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY = date(2026, 11, 3)


def rec(ident, text, when, origin=""):
    r = normalize(text=text, source_type="facebook", source_url=f"https://synthetic.invalid/{ident}",
                  published_date=when, fetched_at="2026-11-03T00:00:00+00:00", credibility_tier="official",
                  source_identifier=ident)
    return replace(r, origin_date=origin) if origin else r


def archive(tmp_path, pages):
    d = tmp_path / "archive"
    d.mkdir(exist_ok=True)
    for key, records in pages.items():
        save_corpus(records, str(d / f"{key}.jsonl"))
    return d


def build(tmp_path, d, window=30, day=DAY, **kw):
    out = tmp_path / "served" / "window.jsonl"
    return sw.build_served(str(d), str(out), window_days=window, build_day=day, **kw), out


def served_ids(out):
    return [r.document_id for r in load_corpus(str(out))]


def test_the_window_keeps_recent_posts_and_leaves_out_old_undated_and_future_ones(tmp_path):
    d = archive(tmp_path, {"main": [
        rec("p_new", "Enrollment opens on November 10 for second semester.", "2026-11-01T02:00:00+0000"),
        rec("p_edge", "Library inventory on October 4 closes the reading room.", "2026-10-04T02:00:00+0000"),
        rec("p_old", "Typhoon suspension of classes on October 2.", "2026-10-02T02:00:00+0000"),
        rec("p_future", "A post dated after the build day, room 5.", "2026-11-05T02:00:00+0000"),
        rec("p_undated", "A post with no usable date, room 9.", "")]})
    record, out = build(tmp_path, d)
    assert served_ids(out) == ["ingested:p_edge", "ingested:p_new"]           # 30 days back, edge day included
    assert record["per_page"]["main"] == {"in_archive": 5, "in_window": 2, "always_use": 0, "outside_window": 1,
                                          "undated": 1, "after_build_day": 1, "reserved": 0}
    assert record["window_first_day"] == "2026-10-04"


def test_days_are_philippine_days_and_the_original_date_wins(tmp_path):
    d = archive(tmp_path, {"main": [
        rec("late", "Posted 00:30 on October 4 in Manila, room 1.", "2026-10-03T16:30:00+0000"),
        rec("orig", "Mirror of an old announcement, room 2.", "2026-11-01T00:00:00+0000", origin="2026-09-01")]})
    _, out = build(tmp_path, d)
    assert served_ids(out) == ["ingested:late"]                                # Oct 3 16:30 UTC = Oct 4 Manila


def test_always_use_posts_are_kept_whatever_their_date_and_missing_ones_reported(tmp_path):
    d = archive(tmp_path, {"main": [
        rec("dept", "The new department name is the College of Technology and Engineering.", "2025-07-16T02:00:00+0000"),
        rec("nodate", "The dress code policy applies to all students.", "")]})
    listed = tmp_path / "always_use.csv"
    listed.write_text("document_id,reason,added_by,date\n"
                      "ingested:dept,long-lasting name,owner,2026-11-03\n"
                      "ingested:nodate,standing policy,owner,2026-11-03\n"
                      "ingested:gone,removed post,owner,2026-11-03\n", encoding="utf-8")
    record, out = build(tmp_path, d, always_use_path=str(listed))
    assert set(served_ids(out)) == {"ingested:dept", "ingested:nodate"}
    assert record["always_use"]["listed"] == 3 and record["always_use"]["not_in_archive"] == ["ingested:gone"]


@pytest.mark.parametrize("content", ["id,reason\nx,y\n", "document_id,reason,added_by,date\ningested:x,,owner,2026-11-03\n"])
def test_a_malformed_always_use_file_is_refused(tmp_path, content):
    d = archive(tmp_path, {"main": [rec("a", "Text, room 1.", "2026-11-01T02:00:00+0000")]})
    listed = tmp_path / "always_use.csv"
    listed.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        build(tmp_path, d, always_use_path=str(listed))


def test_a_post_shared_on_two_pages_is_served_once_the_earliest_copy(tmp_path):
    text = "Classes are suspended on November 4 because of the typhoon."
    d = archive(tmp_path, {"college": [rec("c1", text, "2026-11-02T05:00:00+0000")],
                           "main": [rec("m1", text, "2026-11-02T01:00:00+0000")]})
    record, out = build(tmp_path, d)
    assert served_ids(out) == ["ingested:m1"]
    assert record["shared_posts_served_once"] == [{"document_id": "ingested:c1", "page": "college",
                                                   "duplicate_of": "ingested:m1", "other_page": "main", "score": 1.0}]


def test_posts_on_or_after_the_final_evaluation_cutoff_are_never_served_even_if_listed(tmp_path):
    d = archive(tmp_path, {"main": [rec("before", "Before the cutoff, room 1.", "2026-10-31T15:00:00+0000"),
                                    rec("after", "After the cutoff, room 2.", "2026-10-31T16:00:00+0000")]})
    listed = tmp_path / "always_use.csv"
    listed.write_text("document_id,reason,added_by,date\ningested:after,test,owner,2026-11-03\n", encoding="utf-8")
    cutoff = datetime(2026, 11, 1, tzinfo=timezone(timedelta(hours=8)))
    record, out = build(tmp_path, d, cutoff=cutoff, always_use_path=str(listed))
    assert served_ids(out) == ["ingested:before"] and record["per_page"]["main"]["reserved"] == 1


def test_the_same_inputs_give_byte_identical_files(tmp_path):
    d = archive(tmp_path, {"main": [rec(f"p{i}", f"Notice {i}: room {i} schedule.", f"2026-10-{10 + i}T02:00:00+0000")
                                    for i in range(8)]})
    _, out = build(tmp_path, d)
    first = (out.read_bytes(), (tmp_path / "served" / "window.jsonl.build.json").read_bytes())
    _, out = build(tmp_path, d)
    assert (out.read_bytes(), (tmp_path / "served" / "window.jsonl.build.json").read_bytes()) == first


def test_history_files_are_not_inputs_and_the_window_is_required(tmp_path):
    d = archive(tmp_path, {"main": [rec("a", "Text, room 1.", "2026-11-01T02:00:00+0000")]})
    (d / "main.history.jsonl").write_text('{"action": "edited"}\n', encoding="utf-8")
    record, _ = build(tmp_path, d)
    assert list(record["inputs"]) == ["main.jsonl"]
    for bad in (0, -5, 7.5):
        with pytest.raises(ValueError, match="D2"):
            build(tmp_path, d, window=bad)
    cli = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_served_corpus.py"), "--archive", str(d),
                          "--out", str(tmp_path / "o.jsonl")], capture_output=True, text=True, cwd=ROOT)
    assert cli.returncode == 2 and "--window-days" in cli.stderr


def test_the_command_builds_a_file_the_checker_can_load(tmp_path):
    d = archive(tmp_path, {"main": [rec("a", "Enrollment opens on November 10, room 3.", "2026-11-01T02:00:00+0000")]})
    out = tmp_path / "served.jsonl"
    cli = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_served_corpus.py"), "--window-days",
                          "30", "--on", "2026-11-03", "--archive", str(d), "--out", str(out),
                          "--reservation", str(tmp_path / "none.json"), "--always-use", str(tmp_path / "none.csv")],
                         capture_output=True, text=True, cwd=ROOT)
    assert cli.returncode == 0, cli.stderr
    from src.ingestion.corpus_storage import load_corpus_for_retrieval
    docs = load_corpus_for_retrieval(str(out))
    assert [doc["document_id"] for doc in docs] == ["ingested:a"]


def test_grouping_by_numbers_keeps_and_drops_exactly_the_posts_comparing_every_pair_does():
    from src.ingestion.dedup import filter_redundant
    base = ["Classes are suspended on November 4 because of the typhoon.",
            "Enrollment for the second semester opens on November 10.",
            "The library is closed for inventory.", "The library is closed for inventory today.",
            "Exam permits are released at the Finance Office."]
    records = []
    for i in range(60):
        text = base[i % 5] + ("" if i % 3 else f" Room {i % 4}.")
        records.append(rec(f"g{i}", text, f"2026-10-{1 + i % 28:02d}T0{i % 9}:00:00+0000"))
    # Same posts kept, and every dropped post dropped as a duplicate of the same post. (Kept posts' review flags,
    # from comparisons across number groups, are not produced by the grouped version and not used by the build.)
    key = lambda ds: sorted((d.record.document_id, d.kept, None if d.kept else d.matched_document_id) for d in ds)
    assert key(sw.served_once(records)) == key(filter_redundant(records, ()))
    assert any(not d.kept for d in sw.served_once(records))            # the sample really has duplicates

