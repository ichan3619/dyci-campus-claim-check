"""
The photo check inside POST /verify/image (Phase 2A, owner decision 2026-10-24), with a synthetic photo index:
A4 (verdicts and statements identical with the photo check on and off) and A5 (wording; nothing written to disk).
"""
import builtins
import io
import json
import os
import random
import re
import tempfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFilter

from src.image.ocr import OCRResult

MEASURED = {"found": 44, "queries": 58, "false_matches": 0, "negatives": 28, "formal_decision": "FAIL"}


def _photo(seed, size=(900, 600)):
    rnd = random.Random(seed)
    img = Image.new("RGB", size)
    d = ImageDraw.Draw(img)
    for y in range(size[1]):
        d.line([(0, y), (size[0], y)], fill=(rnd.randint(0, 60) + y % 40, 80, 120))
    for _ in range(120):
        x, y, w, h = rnd.randint(0, size[0]), rnd.randint(0, size[1]), rnd.randint(10, 120), rnd.randint(10, 120)
        color = tuple(rnd.randint(0, 255) for _ in range(3))
        (d.rectangle if rnd.random() < 0.5 else d.ellipse)([x, y, x + w, y + h], fill=color, outline=(0, 0, 0))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def _png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class _FakeOCR:
    def read(self, image):
        return OCRResult(text="Classes are suspended tomorrow due to a typhoon.", confidence=0.9, lines=1)


@pytest.fixture()
def photo_api(api_settings, monkeypatch, tmp_path):
    os.makedirs(tmp_path / "photos")
    _photo(1).save(tmp_path / "photos" / "P1.jpg", quality=95)
    _photo(2).save(tmp_path / "photos" / "P2.jpg", quality=95)
    (tmp_path / "index.json").write_text(json.dumps({
        "max_hamming": 0, "min_inliers": 121, "measured": MEASURED,
        "photos": [{"file": "photos/P1.jpg", "post_id": "P1", "post_url": "https://www.facebook.com/YangaCollege/posts/1",
                    "post_date": "2026-07-13"},
                   {"file": "photos/P2.jpg", "post_id": "P2", "post_url": "https://www.facebook.com/YangaCollege/posts/2",
                    "post_date": "2026-09-26"}]}), encoding="utf-8")
    api_main = api_settings(image_check=True)
    api_main.state["ocr"] = _FakeOCR()
    api_main.state.pop("photo_index_path", None)
    monkeypatch.setenv("CAMPUS_VERIFY_PHOTO_INDEX", str(tmp_path))
    return api_main, monkeypatch, str(tmp_path)


def _post(c, img):
    return c.post("/verify/image", content=_png(img), headers={"Content-Type": "image/png"}).json()


def test_a_dyci_photo_is_found_with_its_post_date_and_the_measured_error(photo_api):
    api_main, _, _ = photo_api
    with TestClient(api_main.app) as c:
        body = _post(c, Image.open(io.BytesIO(_png(_photo(1)))).convert("RGB"))
        assert c.get("/health").json()["image_signals"] is True
    s = body["image_signals"]
    assert s["photo_match"]["post_id"] == "P1" and s["photo_match"]["post_url"].endswith("/posts/1")
    assert s["summary"] == "This photo also appears in DYCI's post of 13 July 2026."
    assert "found 44 of 58 reused DYCI photos (76%)" in s["note"] and "wrongly matched 0 of 28" in s["note"]
    assert "photo_match" in body["timings_ms"]


def test_an_unknown_photo_is_reported_as_not_found(photo_api):
    api_main, _, _ = photo_api
    with TestClient(api_main.app) as c:
        s = _post(c, _photo(99))["image_signals"]
    assert s["photo_match"] is None and "not found among the DYCI photos" in s["summary"]
    assert "No match does not mean the photo is new" in s["note"]


def test_a4_verdicts_and_statements_are_identical_with_the_photo_check_on_and_off(photo_api):
    api_main, monkeypatch, folder = photo_api
    with TestClient(api_main.app) as c:
        on = _post(c, _photo(1))
        monkeypatch.setenv("CAMPUS_VERIFY_PHOTO_INDEX", "none")
        off = _post(c, _photo(1))
        assert c.get("/health").json()["image_signals"] is False
    assert on["image_signals"]["photo_match"] is not None and off["image_signals"] is None
    for body in (on, off):
        body.pop("image_signals"), body.pop("timings_ms"), body.pop("assessment")   # the summary card may cite the photo
    assert on == off


def test_a5_wording_never_claims_fake_real_authentic_or_proof(photo_api):
    api_main, _, _ = photo_api
    with TestClient(api_main.app) as c:
        texts = [_post(c, img)["image_signals"] for img in (_photo(1), _photo(99))]
    for s in texts:
        words = (s["summary"] + " " + s["note"]).lower()
        assert not re.search(r"\b(fake|real|authentic|proof|proves?)\b", words), words


def test_a5_nothing_is_written_to_disk_with_the_photo_check_on(photo_api):
    api_main, monkeypatch, _ = photo_api
    writes, real_open = [], builtins.open

    def spy_open(file, mode="r", *a, **k):
        if any(ch in mode for ch in "wax+"):
            writes.append(str(file))
        return real_open(file, mode, *a, **k)

    tmp_before = set(os.listdir(tempfile.gettempdir()))
    with TestClient(api_main.app) as c:
        _post(c, _photo(2))                                          # loads the index
        monkeypatch.setattr(builtins, "open", spy_open)
        body = _post(c, _photo(2))
        monkeypatch.setattr(builtins, "open", real_open)
    assert body["image_signals"]["photo_match"]["post_id"] == "P2"
    assert writes == [] and set(os.listdir(tempfile.gettempdir())) - tmp_before == set()


def test_the_default_index_folder_loads_and_holds_the_frozen_settings():
    from src.image.photo_match import load_index
    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "photo_index")
    index = load_index(folder)
    import json
    photos = json.load(open(os.path.join(folder, "index.json"), encoding="utf-8"))["photos"]
    built = [p for p in photos if "added" not in p]            # later additions: scripts/round2/photo_index_add.py
    assert (index.max_hamming, index.min_inliers) == (0, 121) and len(built) == 23
    assert len(index.entries) == len(photos)
    assert index.measured == MEASURED


def test_a_broken_index_switches_the_photo_check_off_without_breaking_the_check(api_settings, monkeypatch, tmp_path):
    (tmp_path / "index.json").write_text("{not json", encoding="utf-8")
    api_main = api_settings(image_check=True)
    api_main.state["ocr"] = _FakeOCR()
    monkeypatch.setenv("CAMPUS_VERIFY_PHOTO_INDEX", str(tmp_path))
    with TestClient(api_main.app) as c:
        body = _post(c, _photo(1))
        assert c.get("/health").json()["image_signals"] is False
    assert body["image_signals"] is None and body["text_check"] is not None


def test_every_image_check_carries_a_level_card(photo_api):
    api_main, _, _ = photo_api
    with TestClient(api_main.app) as c:
        card = _post(c, _photo(1))["assessment"]
    assert card["level"] in (1, 2, 3, 4, 5) and card["name"] and "not a probability" in card["explain"]
    assert card["photo_note"].startswith("The photo check compares")


def test_a_photo_from_a_student_page_is_not_called_dycis_post(photo_api):
    api_main, _, folder = photo_api
    record = json.load(open(os.path.join(folder, "index.json"), encoding="utf-8"))
    record["photos"][0]["page"] = "Vox Nostra, a DYCI student publication"
    json.dump(record, open(os.path.join(folder, "index.json"), "w", encoding="utf-8"))
    with TestClient(api_main.app) as c:
        s = _post(c, _photo(1))["image_signals"]
    assert s["summary"] == "This photo also appears in a post by Vox Nostra, a DYCI student publication, on 13 July 2026."
    assert s["photo_match"]["page"].startswith("Vox Nostra")


def test_the_one_post_cross_check_runs_only_when_the_post_text_is_known(photo_api):
    api_main, _, folder = photo_api
    with TestClient(api_main.app) as c:
        without = _post(c, _photo(1))["image_signals"]
    assert "photo_post_check" not in without                    # no caption stored for P1 yet
    record = json.load(open(os.path.join(folder, "index.json"), encoding="utf-8"))
    record["photos"][0]["text"] = "Classes are suspended tomorrow due to a typhoon. Please stay safe."
    json.dump(record, open(os.path.join(folder, "index.json"), "w", encoding="utf-8"))
    api_main.state.pop("photo_index_path", None)
    api_main.state["ocr"] = _FakeOCR()                          # the first client's shutdown cleared the app state
    with TestClient(api_main.app) as c:
        body = _post(c, _photo(1))
    x = body["image_signals"]["photo_post_check"]
    assert x["post_id"] == "P1" and x["checked"] >= 1 and len(x["statements"]) == x["checked"]
    assert "photo_post_check" in body["timings_ms"]
