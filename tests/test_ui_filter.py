"""
Image Check v2 option A: the Facebook interface filter (src/image/ui_filter.py). Synthetic boxes and synthetic images
only; no evaluation image is read here.
"""
import pytest
from fastapi.testclient import TestClient

from src.image import ui_filter as uf
from src.image.ocr import OCRResult
from tests.test_image_check import _FakeOCR, _on, _post_image


def box(text, top, conf=0.9, left=0.0):
    return uf.Box(text, conf, float(top), float(top + 10), left, left + 10 * len(text))


@pytest.mark.parametrize("line", ["Jul", "September 26 at 11:03 PM", "July 13, 2025 at 4:12 PM", "Yesterday at 5:40 PM",
                                  "2h", "See more", "Like Comment Share", "Like · Comment · Share", "34 comments",
                                  "1.2K", "Write a comment...", "Follow", "Most relevant", "View more comments"])
def test_facebook_interface_lines_are_recognised(line):
    assert uf.UI_LINE.fullmatch(uf._canon(line))


@pytest.mark.parametrize("line", ["13 July 2026", "July 13, 2026", "Announcement", "Classes are suspended tomorrow.",
                                  "Like last year, the fair is open.", "Share your ideas at the forum on July 20.",
                                  "DYCI ADVISORY 20 August 2026"])
def test_announcement_lines_are_not_mistaken_for_interface(line):
    assert not uf.UI_LINE.fullmatch(uf._canon(line))


def test_with_no_rule_matching_the_text_is_exactly_the_joined_reading():
    boxes = [box("Classes", 0), box("are", 0, left=80), box("suspended.", 0, left=120), box("Enrollment is open.", 30)]
    result = uf.filter_boxes(boxes, uf.MIN_LINE_CONFIDENCE)
    assert result.text == uf.joined(boxes) == "Classes are suspended. Enrollment is open." and result.removed == []


def test_interface_lines_and_text_after_see_less_are_removed_and_reported():
    boxes = [box("Dr. Yanga's Colleges, Inc.", 0), box("September 26 at 11:03 PM", 15),
             box("Classes are suspended on September 26.", 30),
             box("Thank you.", 45), box("See less", 45, left=200), box("SCHEDULE OF ACTIVITIES", 45, left=300),
             box("Like", 60), box("Comment", 60, left=60), box("Share", 60, left=140)]
    result = uf.filter_boxes(boxes, uf.MIN_LINE_CONFIDENCE)
    assert result.text == "Dr. Yanga's Colleges, Inc. Classes are suspended on September 26. Thank you."
    assert [r["reason"] for r in result.removed] == ["interface line", "after See more / See less", "interface line"]
    assert result.removed[1]["text"] == "See less SCHEDULE OF ACTIVITIES"


def test_low_confidence_lines_are_removed_only_with_a_threshold():
    boxes = [box("Classes resume on Monday.", 0, conf=0.8), box("Yango collo90 9", 20, conf=0.05)]
    assert uf.filter_boxes(boxes, None).text == "Classes resume on Monday. Yango collo90 9"
    result = uf.filter_boxes(boxes, 0.20)
    assert result.text == "Classes resume on Monday." and result.removed[0]["reason"] == "low reading confidence"


def test_line_confidence_is_weighted_by_length():
    line = [box("Enrollment", 0, conf=0.9), box("x", 0, conf=0.0, left=120)]
    assert uf.line_confidence(line) > 0.8


def test_boxes_on_the_same_height_form_one_line():
    lines = uf.group_lines([box("A", 0), box("B", 2, left=30), box("C", 20)])
    assert [[b.text for b in l] for l in lines] == [["A", "B"], ["C"]]


def test_the_setting_is_off_by_default(monkeypatch):
    monkeypatch.delenv(uf.ENV, raising=False)
    assert uf.enabled() is False
    monkeypatch.setenv(uf.ENV, "1")
    assert uf.enabled() is True


# ---- the image route ------------------------------------------------------------------------------------------------
def test_with_the_filter_off_the_route_is_unchanged(api_settings, monkeypatch):
    monkeypatch.delenv(uf.ENV, raising=False)
    api_main = _on(api_settings)
    with TestClient(api_main.app) as c:
        api_main.state["ocr"] = _FakeOCR("September 26 at 11:03 PM Classes are suspended tomorrow due to a typhoon.")
        body = _post_image(c).json()
    assert body["ocr"]["checked_text"].startswith("September 26 at 11:03 PM")
    assert not any(n.startswith("interface filter") for n in body["ocr"]["normalization"])


def test_with_the_filter_on_removed_lines_are_reported_and_only_the_rest_is_checked(api_settings, monkeypatch):
    monkeypatch.setenv(uf.ENV, "1")
    boxes = [box("September 26 at 11:03 PM", 0), box("Classes are suspended tomorrow due to a typhoon.", 20)]

    def fake_read_filtered(engine, image):
        return (OCRResult(text=uf.joined(boxes), confidence=0.9, lines=2), uf.filter_boxes(boxes, uf.MIN_LINE_CONFIDENCE))
    monkeypatch.setattr(uf, "read_filtered", fake_read_filtered)
    api_main = _on(api_settings)
    with TestClient(api_main.app) as c:
        body = _post_image(c).json()
        typed = c.post("/verify/text", json={"claim": "Classes are suspended tomorrow due to a typhoon."}).json()
    assert body["ocr"]["text"] == "September 26 at 11:03 PM Classes are suspended tomorrow due to a typhoon."  # full reading
    assert body["ocr"]["checked_text"] == "Classes are suspended tomorrow due to a typhoon."
    assert body["ocr"]["normalization"][0] == "interface filter (interface line): September 26 at 11:03 PM"
    assert body["text_check"] == typed
