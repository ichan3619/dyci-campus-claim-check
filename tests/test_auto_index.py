"""Rule I: DYCI's own pictures as the reference (results/round2/photo_auto_index_plan.md, Amendment 1)."""
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from src.image import auto_index as ai
from src.image.levels import assess

ORIGINAL = ("Dr. Yanga's Colleges, Inc. ANNOUNCEMENT 9 JULY 2030 ORIENTATION FOR NEW COLLEGE STUDENTS JULY 10, 2030 | "
            "8:00 AM ALL FRESHMEN ARE REQUIRED TO ATTEND AT THE DYCI GYMNASIUM dyci.edu.ph")


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(ai.ENV, raising=False)
    assert ai.enabled() is False


def test_page_label():
    assert ai.page_label("https://www.facebook.com/YangaCollege/posts/1") == "DYCI"
    assert ai.page_label("https://www.facebook.com/770077/posts/1") == "a Facebook page linked to DYCI"


@pytest.mark.parametrize("upload,kind,dyci,picture", [
    (ORIGINAL.replace("JULY 10, 2030", "JULY 12, 2030"), "date", ["July 10"], ["July 12"]),
    (ORIGINAL.replace("8:00 AM", "3:00 PM"), "time", ["8:00 AM"], ["3:00 PM"]),
    (ORIGINAL.replace("COLLEGE STUDENTS", "SENIOR HIGH SCHOOL STUDENTS"), "level", ["college"], ["senior high school"]),
])
def test_swaps_are_flagged(upload, kind, dyci, picture):
    e = ai.edited(ORIGINAL, upload)
    assert (e["kind"], e["dyci_values"], e["picture_values"]) == (kind, dyci, picture)


@pytest.mark.parametrize("upload", [
    ORIGINAL,                                                     # unchanged
    ORIGINAL.replace("8:00 AM", "8:0O AM"),                       # time misread: only one side has it
    ORIGINAL.replace("JULY 10, 2030 | ", ""),                     # event date missed by the reader
    ORIGINAL + " JULY 11, 2030",                                  # an extra value on one side only
    "something else entirely",
])
def test_no_swap_no_flag(upload):
    assert ai.edited(ORIGINAL, upload) is None


def _png(colour=(250, 250, 250)):
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), colour).save(buf, format="PNG")
    return buf.getvalue()


def test_collector_stores_once_and_load(tmp_path):
    reads = []

    class Engine:
        def read(self, image):
            reads.append(1)
            return SimpleNamespace(text=ORIGINAL)
    c = ai.PictureCollector(str(tmp_path / "live.jsonl.photos"), downloader=lambda u: (_png(), "image/png"),
                            engine=lambda: Engine())
    post = {"id": "77_P01", "message": "Please be guided.", "created_time": "2030-07-09T01:00:00+0000",
            "permalink_url": "https://www.facebook.com/77/posts/P01", "full_picture": "https://x.fbcdn.net/p.png"}
    c(post)
    c(post)
    c({"id": "77_P02", "created_time": "2030-07-10T01:00:00+0000"})               # no picture: nothing
    again = ai.PictureCollector(str(tmp_path / "live.jsonl.photos"), downloader=lambda u: pytest.fail("cached"),
                                engine=lambda: Engine())
    again(post)
    assert len(reads) == 1 and c.saved == ["77_P01"]
    record = json.load(open(tmp_path / "live.jsonl.photos" / "index.json", encoding="utf-8"))
    assert record["photos"][0]["post_date"] == "2030-07-09" and record["photos"][0]["reading"] == ORIGINAL
    entries, pages, readings = ai.load(str(tmp_path / "live.jsonl.photos"))
    assert [e.post_id for e in entries] == ["77_P01"] and readings["auto/77_P01.jpg"] == ORIGINAL
    assert pages["auto/77_P01.jpg"] == "a Facebook page linked to DYCI"
    assert ai.load(str(tmp_path / "missing")) == ([], {}, {})


def test_collector_failure_stores_nothing(tmp_path):
    def boom(url):
        raise ValueError("picture download failed (HTTP 403)")
    c = ai.PictureCollector(str(tmp_path / "f.photos"), downloader=boom, engine=lambda: None)
    c({"id": "77_P03", "full_picture": "https://x.fbcdn.net/p.png"})
    assert c.saved == [] and "403" in c.problems[0]["reason"]
    assert ai.load(str(tmp_path / "f.photos")) == ([], {}, {})


def test_level_1_for_an_edited_copy():
    edited = {"kind": "date", "summary": "This picture matches DYCI's picture of 9 July 2030, but it reads July 12 where "
                                         "DYCI's picture reads July 10. It may have been edited."}
    response = {"check_basis": "sentences", "sentence_summary": {"kind": "match", "matches": 1, "checked": 1},
                "sentence_checks": [], "image_signals": {"photo_match": {"post_id": "77_P01", "post_date": "2030-07-09",
                                                                         "page": "DYCI"}, "edited_copy": edited}}
    card = assess(response, None)
    assert card["level"] == 1 and card["reasons"][0] == edited["summary"]
    response["image_signals"].pop("edited_copy")
    assert assess(response, None)["level"] != 1


def test_fetch_requests_pictures_only_when_on(tmp_path, monkeypatch):
    from src.ingestion import facebook_page as fp
    from src.ingestion.adapters import MetaGraphAPIAdapter
    requested = []

    def get_json(self, url, params):
        requested.append(params["fields"])
        return {"data": [], "paging": {}}
    monkeypatch.setattr(MetaGraphAPIAdapter, "_get_json", get_json)
    corpus = tmp_path / "live.jsonl"
    corpus.write_text("", encoding="utf-8")
    monkeypatch.delenv(ai.ENV, raising=False)
    fp.refresh_from_mock_page(str(corpus), "9", "T")
    monkeypatch.setenv(ai.ENV, "1")
    fp.refresh_from_mock_page(str(corpus), "9", "T")
    assert "full_picture" not in requested[0] and "full_picture" in requested[1]
    fp._FETCH_LOG.clear()
