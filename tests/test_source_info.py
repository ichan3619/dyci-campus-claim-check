"""
Source dates and original links in responses (src/api/source_info.py). Display metadata only: retrieval and verdicts
never read it. Fast offline configuration (tests/conftest.py); temporary corpus files.
"""
import os
import time

from fastapi.testclient import TestClient

from src.api.source_info import build_source_info
from src.ingestion.corpus_storage import append_to_corpus
from src.ingestion.evidence import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(ROOT, "data", "dev", "sample_claims.csv")


def _post(text, ident, date, origin_url="", origin_date=""):
    return normalize(text=text, source_type="facebook", source_url=f"https://www.facebook.com/test/posts/{ident}",
                     published_date=date, fetched_at="2026-10-08T00:00:00+00:00", credibility_tier="unverified",
                     source_identifier=ident, origin_url=origin_url, origin_date=origin_date)


def test_development_notices_carry_their_date_and_are_marked_fictional():
    info = build_source_info("dev", ROOT, DEV)
    assert info["evt_001"].published_date == "2026-01-10" and info["evt_001"].synthetic is True


def test_ingested_posts_carry_the_post_date_and_the_original_announcement(tmp_path):
    corpus = tmp_path / "fb.jsonl"
    append_to_corpus(_post("The Cashier's Office is closed on October 20.", "123_1", "2026-10-07T08:00:00+0000",
                           "https://www.example.edu/notices/cashier", "2026-10-06"), str(corpus))
    info = build_source_info(f"dev;{corpus}", ROOT, DEV)
    doc = info["ingested:123_1"]
    assert (doc.published_date, doc.original_url, doc.original_date, doc.synthetic) == \
           ("2026-10-07T08:00:00+0000", "https://www.example.edu/notices/cashier", "2026-10-06", False)
    assert "evt_001" in info                                         # both sources indexed


def test_responses_include_the_metadata(api_settings):
    api_main = api_settings(corpus="dev")
    with TestClient(api_main.app) as c:
        body = c.post("/verify/text", json={"claim": "Classes are suspended tomorrow due to a typhoon."}).json()
    used = body["used_evidence"]
    assert used["document_id"] == "evt_001" and used["published_date"] == "2026-01-10" and used["synthetic"] is True
    assert all("published_date" in d for d in body["retrieved_evidence"])


def test_a_post_added_while_the_api_runs_gets_its_date_too(api_settings, tmp_path):
    corpus = tmp_path / "fb.jsonl"
    append_to_corpus(_post("Library hours change to 7 AM to 9 PM starting November 3.", "123_1", "2026-10-07T08:00:00+0000"), str(corpus))
    api_main = api_settings(corpus=f"dev;{corpus}", reload_seconds=0)
    with TestClient(api_main.app) as c:
        c.post("/verify/text", json={"claim": "library hours change november 3"})
        time.sleep(0.02)
        append_to_corpus(_post("The Registrar moves to Room 104 on October 27.", "123_2", "2026-10-08T09:00:00+0000",
                               "https://www.example.edu/notices/registrar"), str(corpus))
        os.utime(corpus, ns=(os.stat(corpus).st_atime_ns, os.stat(corpus).st_mtime_ns + 10_000_000))
        body = c.post("/verify/text", json={"claim": "the registrar moves to room 104 on october 27"}).json()
    by_id = {d["document_id"]: d for d in body["retrieved_evidence"]}
    assert by_id["ingested:123_2"]["published_date"] == "2026-10-08T09:00:00+0000"
    assert by_id["ingested:123_2"]["original_url"] == "https://www.example.edu/notices/registrar"


def test_frontend_shows_dates_links_and_the_fictional_label():
    app = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
    for text in ("function fmtDate", '"Posted"', '"Original announcement"', "Fictional example",
                 "not a real campus notice"):
        assert text in app, text


def test_every_document_says_how_it_was_acquired(tmp_path):
    from src.ingestion.corpus_storage import append_to_corpus as add
    corpus = tmp_path / "fb.jsonl"
    add(_post("Library hours change on November 3.", "123_1", "2026-10-07T08:00:00+0000"), str(corpus))
    manual = tmp_path / "manual.jsonl"
    add(normalize(text="Enrollment opens on August 10.", source_type="manual_entry",
                  source_url="https://www.facebook.com/dyci/posts/1", published_date="2026-08-01",
                  fetched_at="2026-10-08T00:00:00+00:00", credibility_tier="unverified", source_identifier="N99"), str(manual))
    info = build_source_info(f"dev;{corpus};{manual}", ROOT, DEV, source_map_path=str(tmp_path / "absent.csv"))
    assert info["ingested:123_1"].provenance == "test_page_mirror"
    assert info["ingested:N99"].provenance == "manual_entry"
    assert info["evt_001"].provenance == "fictional"


def test_the_source_map_fills_a_missing_original_without_overriding_a_recorded_one(tmp_path):
    corpus = tmp_path / "fb.jsonl"
    append_to_corpus(_post("Library hours change on November 3.", "123_1", "2026-10-07T08:00:00+0000"), str(corpus))
    append_to_corpus(_post("The Registrar moves to Room 104.", "123_2", "2026-10-08T09:00:00+0000",
                           "https://www.example.edu/recorded", "2026-10-01"), str(corpus))
    source_map = tmp_path / "map.csv"
    source_map.write_text("mock_post,original_url,original_date\n"
                          "123_1,https://www.example.edu/library,2026-10-05\n"
                          "123_2,https://www.example.edu/other,2026-09-01\n", encoding="utf-8")
    info = build_source_info(str(corpus), ROOT, DEV, source_map_path=str(source_map))
    assert (info["ingested:123_1"].original_url, info["ingested:123_1"].original_date) == \
           ("https://www.example.edu/library", "2026-10-05")
    assert (info["ingested:123_2"].original_url, info["ingested:123_2"].original_date) == \
           ("https://www.example.edu/recorded", "2026-10-01")          # the record's own fields win


def test_responses_carry_the_provenance(api_settings, tmp_path):
    corpus = tmp_path / "fb.jsonl"
    append_to_corpus(_post("Library hours change to 7 AM to 9 PM starting November 3.", "123_1", "2026-10-07T08:00:00+0000"),
                     str(corpus))
    api_main = api_settings(corpus=str(corpus))
    with TestClient(api_main.app) as c:
        body = c.post("/verify/text", json={"claim": "Library hours change to 7 AM to 9 PM starting November 3."}).json()
    assert body["used_evidence"]["provenance"] == "test_page_mirror"
