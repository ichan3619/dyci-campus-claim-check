"""
Tests for src/ingestion/html_extraction.py (Milestone 8.2).

All tests here use LOCAL FIXTURE HTML strings/files -- zero network
access, zero mocking required, since extract_content() is a pure
function of an HTML string.
"""

import os

from src.ingestion.html_extraction import extract_content, ExtractedContent

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def read_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def test_extract_content_returns_extracted_content():
    html = read_fixture("sample_page.html")
    result = extract_content(html)
    assert isinstance(result, ExtractedContent)


def test_title_extraction():
    html = read_fixture("sample_page.html")
    result = extract_content(html)
    assert "Library Extended Hours" in result.title


def test_main_text_extraction():
    """
    Note: extract_content() deliberately does NOT collapse internal
    whitespace/newlines itself -- that's src.data.preprocessing's
    normalize_whitespace(), already applied one layer downstream inside
    src.ingestion.evidence.normalize(), reused rather than duplicated
    here. So this test normalizes whitespace on the raw extracted text
    before asserting, to check CONTENT presence, not exact formatting
    (which is intentionally NOT this function's responsibility -- see
    the full pipeline test in tests/test_website_adapter.py, where text
    HAS already passed through normalize() and reads cleanly).
    """
    from src.data.preprocessing import normalize_whitespace

    html = read_fixture("sample_page.html")
    result = extract_content(html)
    cleaned = normalize_whitespace(result.body_text)

    assert "24/7 access during the final examination period" in cleaned
    assert "valid university ID" in cleaned


def test_script_content_excluded():
    html = read_fixture("sample_page.html")
    result = extract_content(html)
    assert "console.log" in html  # confirm the fixture actually has it
    assert "console.log" not in result.body_text


def test_style_content_excluded():
    html = read_fixture("sample_page.html")
    result = extract_content(html)
    assert "font-family" not in result.body_text


def test_navigation_excluded():
    html = read_fixture("sample_page.html")
    result = extract_content(html)
    assert "Contact" not in result.body_text  # nav link text, not article content


def test_footer_excluded():
    html = read_fixture("sample_page.html")
    result = extract_content(html)
    assert "All rights reserved" not in result.body_text


def test_published_date_extraction_from_meta_tag():
    html = read_fixture("sample_page.html")
    result = extract_content(html)
    assert result.published_date == "2026-05-01T09:00:00+00:00"


def test_published_date_missing_returns_empty_string():
    html = "<html><head><title>No date here</title></head><body><p>Some text.</p></body></html>"
    result = extract_content(html)
    assert result.published_date == ""


def test_malformed_html_does_not_crash():
    """BeautifulSoup should tolerate unclosed tags gracefully -- this is
    exactly why it was chosen over a strict parser."""
    html = read_fixture("malformed_page.html")
    result = extract_content(html)
    assert "Classes are suspended tomorrow" in result.body_text


def test_page_with_no_useful_content_returns_empty_body():
    html = read_fixture("empty_content_page.html")
    result = extract_content(html)
    assert result.body_text == ""
    assert result.title == ""


def test_missing_title_tag_returns_empty_string():
    html = "<html><body><p>Just some content, no title tag at all.</p></body></html>"
    result = extract_content(html)
    assert result.title == ""
