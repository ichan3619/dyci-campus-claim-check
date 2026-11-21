"""
Image Check source rule (src/image/source_rule.py; results/round2/image_source_rule_plan.md, Amendment 1). Synthetic
header lines and a stub reader only.
"""
import re

import pytest
from fastapi.testclient import TestClient

from src.image import florence_reader as fr
from src.image import source_rule as sr
from src.image import ui_filter as uf
from src.image.ocr import OCRResult
from tests.test_image_check import _on, _post_image
from tests.test_florence_reader import quad


@pytest.mark.parametrize("header, page", [
    ("Bulacan State University September 10 at 4:00 PM", "Bulacan State University"),
    ("BulSU Official · Follow", "BulSU Official"),
    ("Dr. Yanga's Colleges, Inc. and 3 others · Follow", "Dr. Yanga's Colleges, Inc"),
    ("St. Paul University Surigao Yesterday at 5:40 PM", "St. Paul University Surigao"),
    ("Meycauayan College 3h", "Meycauayan College"),
])
def test_the_page_name_is_read_from_the_header_line(header, page):
    assert sr.page_name(header) == page


@pytest.mark.parametrize("header", ["Bulacan State University September 10 at 4:00 PM", "BulSU Official · Follow",
                                    "St. Paul University Surigao Yesterday at 5:40 PM", "Meycauayan College 3h"])
def test_a_header_naming_another_institution_is_another_page(header):
    result = sr.judge([header])
    assert result["status"] == "other_page" and result["institutions"]
    assert "not one of DYCI's pages" in result["note"] and result["page_name"] in result["note"]


@pytest.mark.parametrize("header", ["Dr. Yanga's Colleges, Inc. 2h", "DYCL for Sustainability - Follow",
                                    "S School News 2.0 October 4 at 2:19PM", "DYCI Senior High School · Follow",
                                    "Elida Campus · Follow", "PAGASA-DOST · Follow", "3rd Colleges, Inc. and 3 others . Follow"])
def test_dyci_pages_test_pages_and_unknown_pages_are_checked(header):
    assert sr.judge([header])["status"] == "header_read"


def test_no_header_is_checked_with_a_note():
    result = sr.judge([])
    assert result["status"] == "no_header" and result["note"] == sr.NO_HEADER_NOTE


def test_the_setting_is_off_by_default(monkeypatch):
    monkeypatch.delenv(sr.ENV, raising=False)
    assert sr.enabled() is False


# ---- the image route ------------------------------------------------------------------------------------------------
def _stub(header):
    class Stub:
        def read_filtered(self, image):
            boxes = fr.boxes_from_regions([quad(0, 0), quad(0, 20)],
                                          [header, "Classes are suspended tomorrow due to a typhoon."])
            kept, removed = fr.remove_header(boxes)
            filtered = uf.filter_boxes(kept, None)
            filtered.removed[:0] = removed
            return OCRResult(text=uf.joined(boxes), confidence=None, lines=2), filtered
    return Stub()


def _image(api_settings, monkeypatch, header, rule="1"):
    monkeypatch.setenv(fr.ENV, "florence2")
    monkeypatch.setenv(sr.ENV, rule)
    api_main = _on(api_settings)
    with TestClient(api_main.app) as c:
        api_main.state["florence"] = _stub(header)
        return _post_image(c).json()


def test_another_institutions_screenshot_gets_no_verdict_and_the_closest_dyci_posts(api_settings, monkeypatch):
    body = _image(api_settings, monkeypatch, "Bulacan State University September 10 at 4:00 PM")
    assert body["status"] == "other_page" and body["text_check"] is None and body["sentence_checks"] == []
    assert body["check_basis"] == "none" and "Bulacan State University" in body["note"]
    assert body["source_check"]["status"] == "other_page" and body["source_check"]["closest"]
    assert all({"document_id", "text"} <= set(d) for d in body["source_check"]["closest"])


def test_a_dyci_screenshot_is_checked_as_before(api_settings, monkeypatch):
    body = _image(api_settings, monkeypatch, "Dr. Yanga's Colleges, Inc. 2h")
    assert body["status"] == "checked" and body["source_check"]["status"] == "header_read"


def test_with_the_rule_off_nothing_changes(api_settings, monkeypatch):
    body = _image(api_settings, monkeypatch, "Bulacan State University September 10 at 4:00 PM", rule="0")
    assert body["status"] == "checked" and body["source_check"] is None


def test_the_page_shows_the_note_and_closest_posts_and_never_sets_a_verdict():
    import os
    app = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "js", "app.js"),
               encoding="utf-8").read()
    block = app[app.index("// Source rule:"):app.index("var found = data.image_signals")]
    assert 'src.status === "other_page"' in block and "Closest DYCI announcements" in block and "src.note" in block
    assert not re.search(r"verdict\s*=(?!=)", block)
