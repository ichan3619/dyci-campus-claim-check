"""
Image Check option B reader (src/image/florence_reader.py). Synthetic regions and a stub reader only: no model is
downloaded or run here (the real model is measured in scripts/round2/image_b_feasibility.py, development images).
"""
import pytest
from fastapi.testclient import TestClient

from src.image import florence_reader as fr
from src.image import ui_filter as uf
from src.image.ocr import OCRResult
from tests.test_image_check import _on, _post_image


def quad(x, y, w=100, h=10):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def test_regions_are_put_in_reading_order_and_special_tokens_dropped():
    boxes = fr.boxes_from_regions([quad(0, 40), quad(150, 0), quad(0, 0), quad(0, 80)],
                                  ["second line</s>", "top right", "<s>top left", ""])
    assert [b.text for b in boxes] == ["top left", "top right", "second line"]
    assert uf.joined(boxes) == "top left top right second line"


def test_the_interface_rules_apply_to_florence_lines_but_no_confidence_rule():
    reader = fr.FlorenceReader()
    boxes = fr.boxes_from_regions([quad(0, 0), quad(0, 20), quad(0, 40)],
                                  ["September 26 at 11:03 PM", "Classes are suspended on September 26.", "Like Comment Share"])
    reader.read_boxes = lambda image: boxes
    raw, filtered = reader.read_filtered(object())
    assert raw.text == "September 26 at 11:03 PM Classes are suspended on September 26. Like Comment Share"
    assert raw.confidence is None and raw.lines == 3
    assert filtered.text == "Classes are suspended on September 26."
    assert [r["reason"] for r in filtered.removed] == ["post header (page name and time)", "interface line"]


def test_the_setting_defaults_to_easyocr_and_rejects_unknown_readers(monkeypatch):
    monkeypatch.delenv(fr.ENV, raising=False)
    assert fr.reader_name() == "easyocr"
    monkeypatch.setenv(fr.ENV, "Florence2")
    assert fr.reader_name() == "florence2"
    monkeypatch.setenv(fr.ENV, "tesseract")
    with pytest.raises(ValueError):
        fr.reader_name()


def test_the_model_is_pinned_to_an_exact_revision():
    assert fr.REPO == "florence-community/Florence-2-base" and len(fr.REVISION) == 40


def test_the_route_uses_florence_only_when_selected(api_settings, monkeypatch):
    class Stub:
        calls = 0

        def read_filtered(self, image):
            Stub.calls += 1
            boxes = fr.boxes_from_regions([quad(0, 0), quad(0, 20)],
                                          ["2h", "Classes are suspended tomorrow due to a typhoon."])
            return OCRResult(text=uf.joined(boxes), confidence=None, lines=2), uf.filter_boxes(boxes, None)
    api_main = _on(api_settings)
    monkeypatch.setenv(fr.ENV, "florence2")
    with TestClient(api_main.app) as c:
        api_main.state["florence"] = Stub()
        body = _post_image(c).json()
        typed = c.post("/verify/text", json={"claim": "Classes are suspended tomorrow due to a typhoon."}).json()
    assert Stub.calls == 1 and api_main.state.get("ocr") is None          # EasyOCR never created
    assert body["ocr"]["checked_text"] == "Classes are suspended tomorrow due to a typhoon."
    assert body["ocr"]["normalization"][0] == "interface filter (interface line): 2h"
    assert body["text_check"] == typed


def test_with_the_default_reader_florence_is_never_loaded(api_settings, monkeypatch):
    monkeypatch.delenv(fr.ENV, raising=False)
    api_main = _on(api_settings)
    with TestClient(api_main.app) as c:
        _post_image(c)
        assert api_main.state.get("florence") is None


@pytest.mark.parametrize("header", ["S School News 2.0 October 4 at 2:19PM", "Follow August 20 - August 20, 2016",
                                    "Dr. Yanga's Colleges, Inc. 2h", "Dr. Yanga's Colleges, Inc. Yesterday at 5:40 PM"])
def test_the_post_header_line_is_removed(header):
    boxes = fr.boxes_from_regions([quad(0, 0), quad(0, 20)],
                                  [header, "DYCI ADVISORY 10 September 2026 To all DYCIans, in-person classes are suspended today."])
    kept, removed = fr.remove_header(boxes)
    assert uf.joined(kept).startswith("DYCI ADVISORY") and removed[0]["reason"].startswith("post header")


@pytest.mark.parametrize("line", ["DYCI ADVISORY", "Announcement", "13 July 2026", "To all DYCIans,",
                                  "Followers of the club meet today", "Room 2h3 schedule"])
def test_announcement_lines_at_the_top_are_kept(line):
    kept, removed = fr.remove_header(fr.boxes_from_regions([quad(0, 0)], [line]))
    assert removed == [] and uf.joined(kept) == line


def test_the_header_rule_stops_at_the_first_long_line():
    long = "The program will start on October 4 at 2:00 PM at the Elida Convention Center for all students."
    boxes = fr.boxes_from_regions([quad(0, 0), quad(0, 20), quad(0, 40)], ["DYCI ADVISORY", long, "Follow us for updates"])
    kept, removed = fr.remove_header(boxes)
    assert removed == [] and uf.joined(kept) == f"DYCI ADVISORY {long} Follow us for updates"


@pytest.mark.parametrize("line", ["Classes are suspended on September 26.", "Enrollment until August 30, 2026",
                                  "Midterms on October 1"])
def test_a_short_dated_announcement_line_at_the_top_is_not_a_header(line):
    kept, removed = fr.remove_header(fr.boxes_from_regions([quad(0, 0)], [line]))
    assert removed == [] and uf.joined(kept) == line
