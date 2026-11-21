"""
Tests for WebsiteSourceAdapter (Milestone 8.2).

Per the Milestone 8.2 spec, these tests must NOT depend on a live
external website. The ONLY thing mocked is requests.get itself -- the
actual network boundary -- using controlled local HTML fixtures as the
mocked response body. Everything downstream of that (extraction,
normalization, document ID derivation, corpus storage, retrieval) is
REAL code, not mocked, matching this project's established pattern from
earlier milestones (e.g. Milestone 6's real classifier + fake-network
NLI tests) of mocking only the actual external boundary.
"""

import io
import os
import tempfile
from unittest.mock import patch, Mock

import pytest
import requests

from src.ingestion.adapters import WebsiteSourceAdapter, WebsiteFetchError
from src.ingestion.evidence import NormalizedEvidence
from src.ingestion.corpus_storage import save_corpus, load_corpus, load_corpus_for_retrieval

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


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


def real_response_with_undeclared_charset(status_code, utf8_bytes, content_type="text/html"):
    """
    Constructs an ACTUAL requests.Response (not a bare Mock) with real
    UTF-8 bytes and a Content-Type header, with .encoding set EXACTLY
    the way requests' real Session.send()/adapter machinery sets it for
    a genuine network response (via get_encoding_from_headers) -- NOT
    left unset. This distinction matters: a manually-constructed
    Response() that never has .encoding explicitly set will silently
    fall back to apparent_encoding regardless of whether the adapter's
    own fix code runs, which would make a test built on it pass
    regardless of whether the real bug is present or fixed -- exactly
    the mistake made and caught during this investigation before this
    helper was corrected.

    This is DELIBERATELY different from mock_response() above:
    mock_response() sets .text directly on a Mock, bypassing requests'
    encoding logic entirely -- it could never have caught the
    Milestone 8.4 encoding bug either, for the same underlying reason.
    """
    response = requests.Response()
    response.status_code = status_code
    response.headers = requests.structures.CaseInsensitiveDict({"Content-Type": content_type})
    # The adapter streams the body (stream=True) to cap its size, so the bytes arrive through .raw exactly as from a
    # real connection, not pre-attached as ._content.
    response.raw = io.BytesIO(utf8_bytes)
    # This is the critical line: set .encoding exactly as requests'
    # real internals do for a genuine response, BEFORE any adapter code
    # runs -- reproducing the true starting state a real WebsiteSourceAdapter
    # would receive from requests.get().
    response.encoding = requests.utils.get_encoding_from_headers(response.headers)
    return response


# --------------------------------------------------------------------
# 1-6: successful fetch, title/text extraction, URL/date preservation,
# non-content exclusion (via the real adapter, not just extract_content directly)
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_successful_fetch_produces_normalized_evidence(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("sample_page.html"))
    adapter = WebsiteSourceAdapter(url="https://university.example.edu/notice/204")

    result = adapter.fetch()

    assert isinstance(result, NormalizedEvidence)


@patch("src.ingestion.adapters.requests.get")
def test_title_is_folded_into_text(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("sample_page.html"))
    adapter = WebsiteSourceAdapter(url="https://university.example.edu/notice/204")

    result = adapter.fetch()

    assert "Library Extended Hours" in result.text


@patch("src.ingestion.adapters.requests.get")
def test_main_content_is_extracted(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("sample_page.html"))
    adapter = WebsiteSourceAdapter(url="https://university.example.edu/notice/204")

    result = adapter.fetch()

    assert "24/7 access during the final examination period" in result.text


@patch("src.ingestion.adapters.requests.get")
def test_source_url_is_preserved(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("sample_page.html"))
    adapter = WebsiteSourceAdapter(url="https://university.example.edu/notice/204")

    result = adapter.fetch()

    assert result.source_url == "https://university.example.edu/notice/204"


@patch("src.ingestion.adapters.requests.get")
def test_published_date_is_extracted_when_present(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("sample_page.html"))
    adapter = WebsiteSourceAdapter(url="https://university.example.edu/notice/204")

    result = adapter.fetch()

    assert result.published_date == "2026-05-01T09:00:00+00:00"


@patch("src.ingestion.adapters.requests.get")
def test_script_and_nav_content_excluded_via_adapter(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("sample_page.html"))
    adapter = WebsiteSourceAdapter(url="https://university.example.edu/notice/204")

    result = adapter.fetch()

    assert "console.log" not in result.text
    assert "All rights reserved" not in result.text


# --------------------------------------------------------------------
# 7: malformed/minimal HTML handling
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_malformed_html_is_still_handled(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("malformed_page.html"))
    adapter = WebsiteSourceAdapter(url="https://university.example.edu/notice/1")

    result = adapter.fetch()

    assert "Classes are suspended tomorrow" in result.text


# --------------------------------------------------------------------
# 8: HTTP failure handling -- must be LOUD, never silently empty
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_connection_error_raises_website_fetch_error(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError("simulated connection failure")
    adapter = WebsiteSourceAdapter(url="https://unreachable.example.edu/notice/1")

    with pytest.raises(WebsiteFetchError, match="connection error"):
        adapter.fetch()


@patch("src.ingestion.adapters.requests.get")
def test_timeout_raises_website_fetch_error(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout("simulated timeout")
    adapter = WebsiteSourceAdapter(url="https://slow.example.edu/notice/1")

    with pytest.raises(WebsiteFetchError, match="timed out"):
        adapter.fetch()


@patch("src.ingestion.adapters.requests.get")
def test_404_status_raises_website_fetch_error(mock_get):
    mock_get.return_value = mock_response(404, "")
    adapter = WebsiteSourceAdapter(url="https://university.example.edu/notice/does-not-exist")

    with pytest.raises(WebsiteFetchError, match="404"):
        adapter.fetch()


@patch("src.ingestion.adapters.requests.get")
def test_500_status_raises_website_fetch_error(mock_get):
    mock_get.return_value = mock_response(500, "")
    adapter = WebsiteSourceAdapter(url="https://university.example.edu/notice/1")

    with pytest.raises(WebsiteFetchError, match="500"):
        adapter.fetch()


# --------------------------------------------------------------------
# 9: empty/no-useful-content handling
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_page_with_no_useful_content_raises_value_error(mock_get):
    """Reuses normalize()'s existing empty-content check -- not a
    separately reimplemented rule."""
    mock_get.return_value = mock_response(200, read_fixture("empty_content_page.html"))
    adapter = WebsiteSourceAdapter(url="https://university.example.edu/blank")

    with pytest.raises(ValueError, match="empty"):
        adapter.fetch()


# --------------------------------------------------------------------
# 10: deterministic document identity
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_same_url_produces_same_document_id_across_fetches(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("sample_page.html"))
    adapter = WebsiteSourceAdapter(url="https://university.example.edu/notice/204")

    result_1 = adapter.fetch()
    result_2 = adapter.fetch()  # a second, independent fetch of the "same" URL

    assert result_1.document_id == result_2.document_id


@patch("src.ingestion.adapters.requests.get")
def test_different_urls_produce_different_document_ids(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("sample_page.html"))

    result_a = WebsiteSourceAdapter(url="https://university.example.edu/notice/1").fetch()
    result_b = WebsiteSourceAdapter(url="https://university.example.edu/notice/2").fetch()

    assert result_a.document_id != result_b.document_id


# --------------------------------------------------------------------
# 11: compatibility with existing corpus storage
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_fetched_evidence_can_be_saved_and_reloaded(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("sample_page.html"))
    adapter = WebsiteSourceAdapter(url="https://university.example.edu/notice/204")
    evidence = adapter.fetch()

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        save_corpus([evidence], path)
        loaded = load_corpus(path)
        assert len(loaded) == 1
        assert loaded[0] == evidence
    finally:
        os.remove(path)


# --------------------------------------------------------------------
# 12: full end-to-end -- controlled HTML -> adapter -> normalized
# document -> JSONL corpus -> reload -> REAL Retriever
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_end_to_end_website_ingestion_to_retrieval(mock_get):
    from src.retrieval.retriever import Retriever

    mock_get.return_value = mock_response(200, read_fixture("sample_page.html"))
    adapter = WebsiteSourceAdapter(
        url="https://university.example.edu/announcements/204",
        source_type="official_site",
        credibility_tier="official",
    )
    evidence = adapter.fetch()

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        save_corpus([evidence], path)
        documents = load_corpus_for_retrieval(path)

        retriever = Retriever(documents)
        results = retriever.retrieve("Is the library open late during finals?", top_k=1)

        assert len(results) == 1
        assert results[0].document_id == evidence.document_id
    finally:
        os.remove(path)


# --------------------------------------------------------------------
# Fetch mechanics: timeout and User-Agent are actually configured
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_fetch_passes_timeout_and_user_agent(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("sample_page.html"))
    adapter = WebsiteSourceAdapter(url="https://university.example.edu/notice/1", timeout_seconds=15)

    adapter.fetch()

    _, kwargs = mock_get.call_args
    assert kwargs["timeout"] == 15
    assert "User-Agent" in kwargs["headers"]
    assert kwargs["headers"]["User-Agent"] != ""


# --------------------------------------------------------------------
# Encoding regression test (Milestone 8.4 finding)
#
# ROOT CAUSE: requests' get_encoding_from_headers() defaults ANY
# "text/*" Content-Type lacking an explicit charset parameter to
# ISO-8859-1 (a longstanding requests behavior). Most real servers --
# confirmed against the real DYCI site during Milestone 8.4 -- send
# `Content-Type: text/html` with NO charset parameter, relying on the
# page's own <meta charset> tag instead, which requests never inspects.
# Real UTF-8 bytes (e.g. a curly apostrophe, U+2019) were therefore
# silently mis-decoded as Latin-1, producing mojibake like "Yangaâ€™s"
# instead of "Yanga's".
#
# FIX: WebsiteSourceAdapter._fetch_html() now sets
# response.encoding = response.apparent_encoding (which sniffs the
# ACTUAL byte content) before reading response.text, instead of trusting
# requests' unreliable header-based default. This is a real decoding fix
# at the correct layer -- NOT a downstream string-replacement patch on
# already-corrupted text.
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_utf8_content_without_declared_charset_decodes_correctly(mock_get):
    """
    The exact real-world condition that caused the Milestone 8.4 DYCI
    mojibake: a server sends real UTF-8 bytes containing a non-ASCII
    character, with a Content-Type header that does NOT declare a
    charset. Before the fix, this decoded as Latin-1 and corrupted the
    apostrophe. After the fix, it must decode correctly.

    Uses real_response_with_undeclared_charset() (a REAL
    requests.Response, not a bare Mock) specifically because a Mock
    with .text pre-set would bypass the exact code path this bug lived
    in -- see that helper's docstring.
    """
    html_with_curly_apostrophe = (
        "<html><head><title>Dr. Yanga\u2019s Colleges, Inc.</title></head>"
        "<body><p>Dr. Yanga\u2019s Colleges, Inc. announcement text.</p></body></html>"
    )
    mock_get.return_value = real_response_with_undeclared_charset(
        200, html_with_curly_apostrophe.encode("utf-8"), content_type="text/html"
    )

    adapter = WebsiteSourceAdapter(url="https://dyci.edu.ph/announcements")
    evidence = adapter.fetch()

    assert "Yanga\u2019s" in evidence.text  # correctly decoded: "Yanga's" (curly apostrophe)
    assert "Yang" + "\u00e2" not in evidence.text  # the mojibake artifact ("â") must NOT appear
    assert "\u00e2\u0080\u0099" not in evidence.text  # the raw mis-decoded byte sequence must NOT appear


@patch("src.ingestion.adapters.requests.get")
def test_utf8_content_with_correctly_declared_charset_still_works(mock_get):
    """The fix must not break the (less common) case where a server
    DOES correctly declare charset=utf-8 -- apparent_encoding should
    still sniff UTF-8 correctly in that case too."""
    html = "<html><head><title>Dr. Yanga\u2019s Colleges</title></head><body>Text.</body></html>"
    mock_get.return_value = real_response_with_undeclared_charset(
        200, html.encode("utf-8"), content_type="text/html; charset=utf-8"
    )

    adapter = WebsiteSourceAdapter(url="https://dyci.edu.ph/announcements")
    evidence = adapter.fetch()

    assert "Yanga\u2019s" in evidence.text


# --------------------------------------------------------------------
# follow_redirects=False (user-submitted URLs, pre-deployment review H1)
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_redirects_are_followed_by_default(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("sample_page.html"))
    WebsiteSourceAdapter(url="https://dyci.edu.ph/announcements").fetch()
    assert mock_get.call_args.kwargs["allow_redirects"] is True


@patch("src.ingestion.adapters.requests.get")
def test_with_follow_redirects_off_a_redirect_is_refused_not_followed(mock_get):
    redirect = mock_response(302, "")
    redirect.headers = {"Location": "http://169.254.169.254/"}
    mock_get.return_value = redirect
    with pytest.raises(WebsiteFetchError, match="not followed"):
        WebsiteSourceAdapter(url="https://example.com/a", follow_redirects=False).fetch()
    assert mock_get.call_count == 1 and mock_get.call_args.kwargs["allow_redirects"] is False
    redirect.close.assert_called_once()


@patch("src.ingestion.adapters.requests.get")
def test_with_follow_redirects_off_a_normal_page_still_works(mock_get):
    mock_get.return_value = mock_response(200, read_fixture("sample_page.html"))
    evidence = WebsiteSourceAdapter(url="https://example.com/a", follow_redirects=False).fetch()
    assert evidence.text
