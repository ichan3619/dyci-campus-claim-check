"""
Several Facebook pages (src/ingestion/pages.py, scripts/ingest_pages.py; scale plan S2). The real Graph adapter runs
against SIMULATED HTTP responses: synthetic page IDs, posts and tokens only; no network.
"""
import json
import os
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

from src.ingestion import pages as pg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENS = {"META_TOKEN_MAIN": "EAAsyntheticTokenMain1234567890abcdef",
          "META_TOKEN_COLLEGE": "EAAsyntheticTokenCollege1234567890abcd"}


def post(page_id, n, text, day="2026-10-20"):
    return {"id": f"{page_id}_{n}", "message": text, "created_time": f"{day}T09:00:00+0000",
            "permalink_url": f"https://www.facebook.com/synthetic/posts/{n}"}


class Graph:
    """Simulated /{page-id}/posts: page id -> list of posts, or an error code."""

    def __init__(self, posts=None, errors=None):
        self.posts, self.errors, self.calls = posts or {}, errors or {}, []

    def __call__(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        page_id = url.rstrip("/").split("/")[-2]
        r = Mock(status_code=200)
        if page_id in self.errors:
            r.status_code = 400
            r.json = Mock(return_value={"error": {"code": self.errors[page_id],
                                                  "message": f"synthetic failure, token {params['access_token']}"}})
        else:
            r.json = Mock(return_value={"data": self.posts.get(page_id, [])})
        return r


def config(tmp_path, entries=None):
    entries = entries or [
        {"key": "main", "name": "Synthetic Main Page", "page_id": "11111111111", "tier": "official",
         "token_env": "META_TOKEN_MAIN"},
        {"key": "college", "name": "Synthetic College Page", "page_id": "22222222222", "tier": "recognized",
         "token_env": "META_TOKEN_COLLEGE"}]
    path = tmp_path / "pages.json"
    path.write_text(json.dumps({"pages": entries}), encoding="utf-8")
    return str(path)


GRAPH = {"11111111111": [post("11111111111", 1, "Enrollment for the second semester opens on November 3."),
                         post("11111111111", 2, "The library is closed on November 1 for inventory.")],
         "22222222222": [post("22222222222", 7, "College of Nursing orientation is on November 5 at the gym.")]}


def reservation(tmp_path, day="2026-12-01"):
    path = tmp_path / "reservation.json"
    if not path.exists():
        pg.set_cutoff(str(path), day)
    return str(path)


def run(tmp_path, graph, env=TOKENS, cutoff="2026-12-01", **kw):
    archive = tmp_path / "archive"
    with patch("src.ingestion.adapters.requests.get", graph):
        record = pg.refresh_pages(pg.load_pages(config(tmp_path)), str(archive), environ=env,
                                  reservation=reservation(tmp_path, cutoff), **kw)
    return record, archive


def read(path):
    return path.read_bytes() if path.exists() else None


def test_two_pages_update_into_their_own_files_with_their_tier_and_note(tmp_path):
    record, archive = run(tmp_path, Graph(GRAPH))
    assert [(p["key"], p["status"], p["added"]) for p in record["pages"]] == [("main", "ok", 2), ("college", "ok", 1)]
    rows = [json.loads(l) for l in (archive / "main.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["credibility_tier"] for r in rows} == {"official"}
    assert all("Synthetic Main Page (Facebook page 'main'" in r["acquisition_note"] for r in rows)
    college = json.loads((archive / "college.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert college["credibility_tier"] == "recognized"


def test_one_page_failing_does_not_stop_the_other_and_leaves_its_archive_unchanged(tmp_path):
    _, archive = run(tmp_path, Graph(GRAPH))
    before = read(archive / "main.jsonl")
    grown = {**GRAPH, "22222222222": GRAPH["22222222222"] + [post("22222222222", 8, "Exam permits are released on November 9.")]}
    record, _ = run(tmp_path, Graph(grown, errors={"11111111111": 190}))
    main, college = record["pages"]
    assert main["status"] == "failed" and main["archive_unchanged"] and "expired or invalid" in main["error"]
    assert read(archive / "main.jsonl") == before
    assert college["status"] == "ok" and college["added"] == 1


@pytest.mark.parametrize("code", [4, 17, 32, 613])
def test_a_rate_limit_is_recorded_and_nothing_is_written(tmp_path, code):
    record, archive = run(tmp_path, Graph(GRAPH, errors={"11111111111": code, "22222222222": code}))
    assert {p["status"] for p in record["pages"]} == {"rate_limited"}
    assert all("retry later" in p["error"] for p in record["pages"])
    assert not (archive / "main.jsonl").exists() and not (archive / "college.jsonl").exists()


def test_a_page_whose_token_is_not_set_is_skipped(tmp_path):
    graph = Graph(GRAPH)
    record, archive = run(tmp_path, graph, env={"META_TOKEN_MAIN": TOKENS["META_TOKEN_MAIN"]})
    assert record["pages"][1] == {"key": "college", "name": "Synthetic College Page", "page_id": "22222222222",
                                  "tier": "recognized", "status": "skipped", "reason": "META_TOKEN_COLLEGE is not set"}
    assert all("22222222222" not in url for url, _ in graph.calls)


def test_rerunning_unchanged_pages_adds_nothing_and_keeps_the_files_byte_identical(tmp_path):
    _, archive = run(tmp_path, Graph(GRAPH))
    before = {n: read(archive / n) for n in ("main.jsonl", "college.jsonl")}
    record, _ = run(tmp_path, Graph(GRAPH))
    assert [p["added"] for p in record["pages"]] == [0, 0]
    assert {n: read(archive / n) for n in before} == before


def test_the_vanishing_page_guard_holds_per_page(tmp_path):
    many = {"11111111111": [post("11111111111", i, f"Notice {i}: room {i} schedule for November {i}.") for i in range(1, 7)],
            "22222222222": GRAPH["22222222222"]}
    _, archive = run(tmp_path, Graph(many))
    before = read(archive / "main.jsonl")
    record, _ = run(tmp_path, Graph({**many, "11111111111": many["11111111111"][:1]}))
    main = record["pages"][0]
    assert main["retired"] == 0 and "did not come back" in main["retire_refused"]
    assert read(archive / "main.jsonl") == before


def test_a_post_shared_on_two_pages_is_reported_not_removed(tmp_path):
    shared = "Classes are suspended on November 4 because of the typhoon."
    graph = Graph({"11111111111": [post("11111111111", 1, shared)], "22222222222": [post("22222222222", 9, shared)]})
    record, archive = run(tmp_path, graph)
    assert [p["added"] for p in record["pages"]] == [1, 1]                 # both archives keep their copy
    pairs = {(d["page"], d["other_page"]) for d in record["cross_page_duplicates"]}
    assert pairs == {("main", "college"), ("college", "main")}


def test_no_token_appears_in_the_record_the_run_file_or_the_output(tmp_path, monkeypatch):
    record, archive = run(tmp_path, Graph(GRAPH, errors={"11111111111": 190, "22222222222": 4}))
    path = pg.write_run_record(str(archive), record)
    text = open(path, encoding="utf-8").read() + json.dumps(record)
    assert all(t not in text for t in TOKENS.values())
    cfg = config(tmp_path)
    env = {**os.environ, **TOKENS, "PYTHONPATH": ROOT}
    env.pop("PYTHONIOENCODING", None)
    script = os.path.join(ROOT, "scripts", "ingest_pages.py")
    # The command itself, with the network patched out through a sitecustomize-free wrapper.
    wrapper = tmp_path / "run.py"
    wrapper.write_text(
        "import sys, runpy, json\n"
        "from unittest.mock import patch, Mock\n"
        "def fail(url, params=None, timeout=None):\n"
        "    r = Mock(status_code=400); r.json = Mock(return_value={'error': {'code': 190, 'message': 'bad ' + params['access_token']}}); return r\n"
        f"sys.argv = ['ingest_pages.py', '--config', {cfg!r}, '--archive', {str(tmp_path / 'cli_archive')!r}, "
        f"'--reservation', {reservation(tmp_path)!r}]\n"
        "with patch('src.ingestion.adapters.requests.get', fail):\n"
        f"    runpy.run_path({script!r}, run_name='__main__')\n", encoding="utf-8")
    out = subprocess.run([sys.executable, str(wrapper)], capture_output=True, text=True, env=env, cwd=ROOT)
    assert out.returncode == 1 and "failed" in out.stdout
    assert all(t not in out.stdout + out.stderr for t in TOKENS.values())
    runs = os.listdir(tmp_path / "cli_archive" / "runs")
    assert len(runs) == 1 and all(t not in open(tmp_path / "cli_archive" / "runs" / runs[0], encoding="utf-8").read()
                                  for t in TOKENS.values())


def test_a_dry_run_writes_no_posts(tmp_path):
    record, archive = run(tmp_path, Graph(GRAPH), dry_run=True)
    assert [p["added"] for p in record["pages"]] == [2, 1] and record["dry_run"]
    assert not (archive / "main.jsonl").exists()
    assert record["cross_page_duplicates"] == "not checked in a dry run"


@pytest.mark.parametrize("entries, message", [
    ([{"key": "main", "name": "A", "page_id": "11111111111", "tier": "official", "token_env": "META_TOKEN_A"},
      {"key": "main", "name": "B", "page_id": "22222222222", "tier": "official", "token_env": "META_TOKEN_B"}], "used twice"),
    ([{"key": "main", "name": "A", "page_id": "11111111111", "tier": "trusted", "token_env": "META_TOKEN_A"}], "tier"),
    ([{"key": "main", "name": "A", "page_id": "11111111111", "tier": "official", "token_env": "EAAabcdefghijklmnopqrstuvwxyz123"}],
     "looks like an access token"),
    ([{"key": "main", "name": "A", "page_id": "11111111111", "tier": "official", "token_env": "META_TOKEN_A",
       "token": "x"}], "unknown field"),
    ([{"key": "main", "name": "A", "page_id": "dyci", "tier": "official", "token_env": "META_TOKEN_A"}], "numeric ID"),
    ([{"key": "Main Page", "name": "A", "page_id": "11111111111", "tier": "official", "token_env": "META_TOKEN_A"}], "key"),
])
def test_the_config_is_checked(tmp_path, entries, message):
    with pytest.raises(pg.PagesConfigError, match=message):
        pg.load_pages(config(tmp_path, entries))


def test_an_unknown_page_key_is_refused(tmp_path):
    with pytest.raises(pg.PagesConfigError, match="unknown page key"):
        pg.refresh_pages(pg.load_pages(config(tmp_path)), str(tmp_path / "a"), only=["nope"], environ=TOKENS,
                         reservation=reservation(tmp_path))


# ---- final-evaluation reservation (D5 = (b)) ----------------------------------------------------------------------------
def test_nothing_is_collected_while_no_cutoff_is_recorded(tmp_path):
    graph = Graph(GRAPH)
    with patch("src.ingestion.adapters.requests.get", graph), \
            pytest.raises(pg.PagesConfigError, match="no final-evaluation cutoff"):
        pg.refresh_pages(pg.load_pages(config(tmp_path)), str(tmp_path / "a"), environ=TOKENS,
                         reservation=str(tmp_path / "missing.json"))
    assert graph.calls == []                                           # not even fetched


def test_posts_from_the_cutoff_on_are_never_written_and_only_counted(tmp_path):
    main = [post("11111111111", 1, "Enrollment for the second semester opens on November 3.", day="2026-10-20"),
            {**post("11111111111", 2, "RESERVED: midterm schedule announced."), "created_time": "2026-10-31T16:00:00+0000"},
            {**post("11111111111", 3, "Kept: last post before the cutoff."), "created_time": "2026-10-31T15:59:59+0000"},
            {"id": "11111111111_4", "message": "RESERVED: no date on this one.", "permalink_url": "https://x/4"}]
    record, archive = run(tmp_path, Graph({"11111111111": main, "22222222222": []}), cutoff="2026-11-01")
    p = record["pages"][0]
    assert p["added"] == 2 and p["reserved_for_final_evaluation"] == 2
    text = (archive / "main.jsonl").read_text(encoding="utf-8")
    assert "RESERVED" not in text and "Kept: last post" in text
    dumped = json.dumps(record)
    assert "RESERVED" not in dumped and "11111111111_2" not in dumped and "11111111111_4" not in dumped
    assert record["reservation_cutoff"] == "2026-11-01T00:00:00+08:00"


def test_reserved_posts_never_count_as_deleted_ones(tmp_path):
    early = [post("11111111111", i, f"Notice {i}: room {i} schedule for October {i}.", day=f"2026-10-0{i}")
             for i in range(1, 6)]
    run(tmp_path, Graph({"11111111111": early, "22222222222": []}), cutoff="2026-11-01")
    late = [{**post("11111111111", 9, "RESERVED new post."), "created_time": "2026-11-02T01:00:00+0000"}]
    record, _ = run(tmp_path, Graph({"11111111111": late + early, "22222222222": []}), cutoff="2026-11-01")
    p = record["pages"][0]
    assert p["retired"] == 0 and p["added"] == 0 and p["reserved_for_final_evaluation"] == 1


def test_the_cutoff_is_recorded_once_and_cannot_be_changed(tmp_path):
    path = str(tmp_path / "reservation.json")
    assert pg.set_cutoff(path, "2026-11-03").isoformat() == "2026-11-03T00:00:00+08:00"
    with pytest.raises(pg.PagesConfigError, match="cannot be changed"):
        pg.set_cutoff(path, "2027-01-01")
    assert pg.load_cutoff(path).isoformat() == "2026-11-03T00:00:00+08:00"


def test_the_command_records_the_cutoff_once(tmp_path):
    script, path = os.path.join(ROOT, "scripts", "ingest_pages.py"), str(tmp_path / "r.json")
    first = subprocess.run([sys.executable, script, "--reservation", path, "--reserve-from", "2026-11-03"],
                           capture_output=True, text=True, cwd=ROOT)
    again = subprocess.run([sys.executable, script, "--reservation", path, "--reserve-from", "2026-12-01"],
                           capture_output=True, text=True, cwd=ROOT)
    assert first.returncode == 0 and "2026-11-03T00:00:00+08:00" in first.stdout
    assert again.returncode == 2 and "cannot be changed" in again.stderr
    assert json.load(open(path, encoding="utf-8"))["cutoff_date"] == "2026-11-03"


def test_the_example_config_is_valid_and_includes_no_page():
    pages = pg.load_pages(os.path.join(ROOT, "data", "pages.example.json"))
    assert pages and not any(p.include for p in pages)


def test_an_undated_post_anywhere_in_the_list_is_reserved_without_shifting_the_others(tmp_path):
    main = [{"id": "11111111111_4", "message": "RESERVED: no date, listed first.", "permalink_url": "https://x/4"},
            post("11111111111", 1, "Kept: enrollment opens on November 3.", day="2026-10-20"),
            {**post("11111111111", 2, "RESERVED: after the cutoff."), "created_time": "2026-11-02T01:00:00+0000"},
            post("11111111111", 3, "Kept: library closed on October 30.", day="2026-10-21")]
    record, archive = run(tmp_path, Graph({"11111111111": main, "22222222222": []}), cutoff="2026-11-01")
    text = (archive / "main.jsonl").read_text(encoding="utf-8")
    assert record["pages"][0]["added"] == 2 and record["pages"][0]["reserved_for_final_evaluation"] == 2
    assert text.count("Kept:") == 2 and "RESERVED" not in text
