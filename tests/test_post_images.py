"""Fix A: read the picture in each fetched post (results/round2/post_images_plan.md)."""
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from src.ingestion import facebook_page as fp
from src.ingestion import post_images as pi
from src.ingestion.adapters import MetaGraphAPIAdapter


def png():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


class FakeEngine:
    def __init__(self, texts):
        self.texts, self.calls = texts, 0

    def read(self, image):
        self.calls += 1
        return SimpleNamespace(text=self.texts.pop(0) if self.texts else "")


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(pi.ENV, raising=False)
    assert pi.enabled() is False


@pytest.mark.parametrize("url,ok", [
    ("https://scontent.fmnl4-1.fna.fbcdn.net/v/t39/x.jpg?oh=1", True), ("https://www.facebook.com/photo.jpg", True),
    ("http://scontent.fbcdn.net/x.jpg", False), ("https://evil.com/fbcdn.net/x.jpg", False),
    ("https://fbcdn.net.evil.com/x.jpg", False), ("https://127.0.0.1/x.jpg", False), ("", False),
])
def test_allowed_url(url, ok):
    assert pi.allowed_url(url) is ok


def test_compose():
    assert pi.compose("Please be guided.", "NO CLASSES  JULY 10") == ("Please be guided.\n\nText in the post's picture: "
                                                                      "NO CLASSES  JULY 10")
    assert pi.compose("", "NO CLASSES") == "Text in the post's picture: NO CLASSES"
    assert pi.compose(None, "1 2 ~") is None                   # fewer than 3 letters
    assert pi.compose("Caption", "~~") == "Caption"
    assert pi.compose(None, None) is None


def test_reader_reads_once_and_caches(tmp_path):
    engine = FakeEngine(["NO CLASSES ON JULY 10, 2030"])
    reader = pi.PostImageReader(str(tmp_path / "c.json"), downloader=lambda u: (png(), "image/png"),
                                engine=lambda: engine)
    post = {"id": "1_2", "message": "Please be guided.", "full_picture": "https://x.fbcdn.net/a.png"}
    first = reader(post)
    again = pi.PostImageReader(str(tmp_path / "c.json"), downloader=lambda u: pytest.fail("no second download"),
                               engine=lambda: engine)(post)
    assert first == again == "Please be guided.\n\nText in the post's picture: NO CLASSES ON JULY 10, 2030"
    assert engine.calls == 1 and reader.read_now == ["1_2"]


def test_reader_failure_leaves_the_caption(tmp_path):
    def boom(url):
        raise ValueError("picture download failed (HTTP 403)")
    reader = pi.PostImageReader(str(tmp_path / "c.json"), downloader=boom, engine=lambda: FakeEngine([]))
    assert reader({"id": "1_3", "message": "Caption.", "full_picture": "https://x.fbcdn.net/a.png"}) == "Caption."
    assert reader({"id": "1_4", "full_picture": "https://x.fbcdn.net/a.png"}) is None
    assert [p["post_id"] for p in reader.problems] == ["1_3", "1_4"] and "403" in reader.problems[0]["reason"]


def _payload(posts):
    return {"data": posts, "paging": {}}


def test_fetch_and_sync_are_stable(tmp_path, monkeypatch):
    """Picture-only and thin-caption posts become evidence; a second fetch sees no edits and retires nothing."""
    posts = [
        {"id": "9_1", "created_time": "2030-07-09T01:00:00+0000", "permalink_url": "https://facebook.com/9/posts/1",
         "full_picture": "https://x.fbcdn.net/1.png"},
        {"id": "9_2", "message": "Please be guided. #DYCI", "created_time": "2030-07-10T01:00:00+0000",
         "permalink_url": "https://facebook.com/9/posts/2", "full_picture": "https://x.fbcdn.net/2.png"},
        {"id": "9_3", "created_time": "2030-07-11T01:00:00+0000", "permalink_url": "https://facebook.com/9/posts/3",
         "full_picture": "https://x.fbcdn.net/3.png"},
        {"id": "9_4", "message": "Classes resume on July 14, 2030.", "created_time": "2030-07-12T01:00:00+0000",
         "permalink_url": "https://facebook.com/9/posts/4"},
    ]
    readings = {"https://x.fbcdn.net/1.png": "ANNOUNCEMENT NO CLASSES IN ALL LEVELS JULY 10, 2030",
                "https://x.fbcdn.net/2.png": "LIBRARY CLOSED JULY 11, 2030 FOR INVENTORY",
                "https://x.fbcdn.net/3.png": "~ ."}
    calls = []

    class Engine:
        def read(self, image):
            return SimpleNamespace(text=readings[calls[-1]])

    def downloader(url):
        calls.append(url)
        return png(), "image/png"
    requested = []

    def get_json(self, url, params):
        requested.append(params)
        return _payload(json.loads(json.dumps(posts)))
    monkeypatch.setattr(MetaGraphAPIAdapter, "_get_json", get_json)
    monkeypatch.setenv(pi.ENV, "1")
    monkeypatch.setattr(pi, "ENGINE_PROVIDER", lambda: Engine())
    monkeypatch.setattr(pi, "download", downloader)
    monkeypatch.setattr(pi.PostImageReader.__init__, "__defaults__", (downloader, None))
    corpus = tmp_path / "live.jsonl"
    corpus.write_text("", encoding="utf-8")
    fp._FETCH_LOG.clear()

    first, _ = fp.refresh_from_mock_page(str(corpus), "9", "TOKEN")
    assert "full_picture" in requested[0]["fields"]
    texts = {json.loads(l)["source_identifier"]: json.loads(l)["text"] for l in corpus.read_text("utf-8").splitlines() if l}
    assert texts["9_1"] == "Text in the post's picture: ANNOUNCEMENT NO CLASSES IN ALL LEVELS JULY 10, 2030"
    assert texts["9_2"].endswith("LIBRARY CLOSED JULY 11, 2030 FOR INVENTORY")
    assert "9_3" not in texts and texts["9_4"] == "Classes resume on July 14, 2030."
    assert first.added == 3 and first.unusable_skipped == 1

    second, _ = fp.refresh_from_mock_page(str(corpus), "9", "TOKEN")
    assert (second.added, second.edited, second.retired) == (0, 0, 0)
    assert len(calls) == 3                              # read once each, including 9_3 (read, no text): cached
    log = fp.fetch_log()
    assert log[1]["pictures_read"] == 3 and log[0]["pictures_read"] == 0
    fp._FETCH_LOG.clear()


def test_off_requests_no_picture_and_keeps_old_behaviour(tmp_path, monkeypatch):
    requested = []

    def get_json(self, url, params):
        requested.append(params)
        return _payload([{"id": "9_1", "created_time": "2030-07-09T01:00:00+0000", "full_picture": "https://x.fbcdn.net/1.png"}])
    monkeypatch.setattr(MetaGraphAPIAdapter, "_get_json", get_json)
    monkeypatch.delenv(pi.ENV, raising=False)
    corpus = tmp_path / "live.jsonl"
    corpus.write_text("", encoding="utf-8")
    result, _ = fp.refresh_from_mock_page(str(corpus), "9", "TOKEN")
    assert "full_picture" not in requested[0]["fields"] and result.added == 0 and result.unusable_skipped == 1
    fp._FETCH_LOG.clear()
