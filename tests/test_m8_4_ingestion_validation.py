"""
Tests for Milestone 8.4 -- ingestion validation against real, permitted
public campus/university content.

IMPORTANT PROVENANCE: tests/fixtures/dyci_announcements_page.html's
announcement TEXT is real, verified public content from DYCI's official
announcements page (https://dyci.edu.ph/announcements) -- see that
fixture file's own header comment for the full provenance note and why
the surrounding HTML structure is reconstructed rather than a byte-exact
copy of the live page. These are DETERMINISTIC, AUTOMATED tests -- they
use this local fixture and mocked HTTP (same pattern as
tests/test_website_adapter.py), never a live network call, so they never
depend on dyci.edu.ph's actual availability and will not make the
193+-test suite flaky.

A SEPARATE, manual, explicitly-live validation path exists in
scripts/demo_ingestion.py --url <real URL>, deliberately NOT part of
this automated suite -- see docs/methodology.md's Milestone 8.4 section
for why direct live fetching could not be exercised end-to-end from this
project's own sandboxed development environment, and what running that
manual command on a machine with normal internet access would add.
"""

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch, Mock

import pytest

from src.ingestion.html_extraction import extract_content
from src.ingestion.evidence import normalize
from src.ingestion.adapters import WebsiteSourceAdapter, WebsiteFetchError
from src.ingestion.corpus_storage import append_to_corpus, load_corpus, load_corpus_for_retrieval
from src.retrieval.retriever import Retriever

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
DYCI_FIXTURE = os.path.join(FIXTURES_DIR, "dyci_announcements_page.html")


def read_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def mock_response(status_code=200, text=""):
    response = Mock()
    response.status_code = status_code
    response.text = text
    response.headers = {}
    response.iter_content = Mock(return_value=[text.encode("utf-8")])   # adapters read the body in chunks
    return response


# --------------------------------------------------------------------
# Extraction quality on real-content fixture
# --------------------------------------------------------------------

def test_dyci_fixture_extracts_real_announcement_text():
    """The three REAL, verified DYCI announcements must all survive
    extraction -- this is the actual content usefulness check, not just
    a 'did it return something non-empty' check."""
    html = read_fixture("dyci_announcements_page.html")
    result = extract_content(html)

    assert "Removal of Incomplete (INC) grades" in result.body_text
    assert "College Enrollment is ongoing" in result.body_text
    assert "Enrollment is Open" in result.body_text


def test_dyci_fixture_title_extraction():
    html = read_fixture("dyci_announcements_page.html")
    result = extract_content(html)
    assert "DYCI Official Website" in result.title


def test_dyci_fixture_published_date_extraction():
    html = read_fixture("dyci_announcements_page.html")
    result = extract_content(html)
    assert result.published_date == "2026-06-04T00:00:00+00:00"


def test_dyci_fixture_script_and_style_excluded():
    html = read_fixture("dyci_announcements_page.html")
    result = extract_content(html)
    assert "console.log" not in result.body_text
    assert "font-family" not in result.body_text


def test_dyci_fixture_nav_and_footer_excluded():
    html = read_fixture("dyci_announcements_page.html")
    result = extract_content(html)
    assert "Admissions" not in result.body_text  # nav link text
    assert "All rights reserved" not in result.body_text  # footer text


def test_dyci_fixture_cookie_banner_KNOWN_LIMITATION_leaks_through():
    """
    DOCUMENTED FINDING (Milestone 8.4), not a bug being silently fixed:
    the cookie-consent banner is a <div>, not one of the four tags
    (script/style/nav/header/footer) extract_content() strips, so its
    text currently leaks into extracted evidence. This test PINS that
    real, observed behavior so it's visible and tracked, rather than
    being an assumption anyone has to rediscover later. See
    docs/methodology.md's Milestone 8.4 section for the full finding and
    why it was not silently patched in this milestone.
    """
    html = read_fixture("dyci_announcements_page.html")
    result = extract_content(html)
    assert "cookies to enhance your browsing experience" in result.body_text


# --------------------------------------------------------------------
# Full chain: real-content fixture -> adapter (mocked HTTP) -> normalize
# -> corpus storage -> reload -> retrieval, with realistic claims
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_end_to_end_dyci_content_ingestion_and_retrieval(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("dyci_announcements_page.html"))
    adapter = WebsiteSourceAdapter(
        url="https://dyci.edu.ph/announcements",
        source_type="official_site",
        credibility_tier="official",
    )
    evidence = adapter.fetch()

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        append_to_corpus(evidence, path)
        documents = load_corpus_for_retrieval(path)
        retriever = Retriever(documents)

        # Realistic claims derived from the real announcement content.
        on_topic = retriever.retrieve("Is enrollment currently open at DYCI?", top_k=1)
        off_topic = retriever.retrieve("Is there a robotics competition happening?", top_k=1)

        assert len(on_topic) == 1
        assert on_topic[0].document_id == evidence.document_id
        assert on_topic[0].score > 0.0

        # A claim about something never mentioned in this evidence should
        # score near zero -- retrieval RELEVANCE discrimination, not a
        # classifier-accuracy claim (that distinction matters -- see
        # module docstring).
        assert len(off_topic) == 1  # still returned (only doc in corpus)
        assert off_topic[0].score == 0.0
    finally:
        os.remove(path)


def test_metadata_survives_full_round_trip():
    """Required metadata (source_url, source_type, credibility_tier,
    published_date) must survive extract -> normalize -> save -> reload
    with zero loss -- checked field by field, not just 'it loaded'."""
    html = read_fixture("dyci_announcements_page.html")
    extracted = extract_content(html)
    combined_text = f"{extracted.title}\n\n{extracted.body_text}"

    evidence = normalize(
        text=combined_text,
        source_type="official_site",
        source_url="https://dyci.edu.ph/announcements",
        published_date=extracted.published_date,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        credibility_tier="official",
    )

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        append_to_corpus(evidence, path)
        reloaded = load_corpus(path)

        assert len(reloaded) == 1
        assert reloaded[0].source_url == "https://dyci.edu.ph/announcements"
        assert reloaded[0].source_type == "official_site"
        assert reloaded[0].credibility_tier == "official"
        assert reloaded[0].published_date == "2026-06-04T00:00:00+00:00"
        assert "Removal of Incomplete" in reloaded[0].text
    finally:
        os.remove(path)


# --------------------------------------------------------------------
# Failure handling must not corrupt the corpus
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_fetch_failure_never_reaches_corpus_storage(mock_get):
    """A failed fetch must raise before append_to_corpus is ever called
    -- confirmed here by checking the corpus file is never even
    created, not just that an exception was raised somewhere."""
    mock_get.return_value = mock_response(403, "")
    adapter = WebsiteSourceAdapter(url="https://dyci.edu.ph/blocked-page")

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(path)  # start with the file genuinely not existing

    try:
        with pytest.raises(WebsiteFetchError):
            evidence = adapter.fetch()
            append_to_corpus(evidence, path)  # must never be reached

        assert not os.path.exists(path)  # corpus was never touched
    finally:
        if os.path.exists(path):
            os.remove(path)


@patch("src.ingestion.adapters.requests.get")
def test_existing_corpus_untouched_by_a_later_failed_fetch(mock_get):
    """A failed fetch for a SECOND source must not corrupt or alter a
    corpus that already has valid entries from earlier successful
    ingestion -- a stronger check than just 'no file was created'."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        # First, a real successful ingestion.
        mock_get.return_value = mock_response(200, read_fixture("dyci_announcements_page.html"))
        good_adapter = WebsiteSourceAdapter(url="https://dyci.edu.ph/announcements")
        good_evidence = good_adapter.fetch()
        append_to_corpus(good_evidence, path)

        before = load_corpus(path)
        assert len(before) == 1

        # Then, a failed fetch for a different URL.
        mock_get.return_value = mock_response(500, "")
        bad_adapter = WebsiteSourceAdapter(url="https://dyci.edu.ph/other-page")
        with pytest.raises(WebsiteFetchError):
            bad_adapter.fetch()

        # The existing corpus must be completely unaffected.
        after = load_corpus(path)
        assert after == before
    finally:
        os.remove(path)


# --------------------------------------------------------------------
# Duplicate ingestion behavior with real content (M8.3 policy, applied
# to a real-content source specifically)
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_reingesting_same_dyci_url_is_skipped_as_duplicate(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("dyci_announcements_page.html"))
    adapter = WebsiteSourceAdapter(url="https://dyci.edu.ph/announcements")

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        first = append_to_corpus(adapter.fetch(), path)
        second = append_to_corpus(adapter.fetch(), path)  # re-ingest same URL

        assert first is True
        assert second is False  # correctly skipped, per M8.3's conservative policy

        documents = load_corpus(path)
        assert len(documents) == 1  # no duplicate accumulated
    finally:
        os.remove(path)
