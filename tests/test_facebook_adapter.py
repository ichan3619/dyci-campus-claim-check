"""
Tests for Milestone 8.5 -- MetaGraphAPIAdapter (src/ingestion/adapters.py).

IMPORTANT PROVENANCE: every fixture used here
(tests/fixtures/facebook_graph_api_posts_response*.json) is SYNTHETIC,
shaped after Meta's documented Graph API response format. None of it is
real DYCI Facebook content, and none of these tests make a live network
call -- `requests.get` is mocked in every test, same pattern as
tests/test_website_adapter.py and tests/test_m8_4_ingestion_validation.py.
This project has no Meta Graph API credentials and no authorized access
to the real DYCI Facebook Page; real-Page validation is Milestone
8.5-B / 8.6, not this file's job.
"""

import json
import os

import pytest
import requests
from unittest.mock import patch, Mock

from src.ingestion.evidence import NormalizedEvidence
from src.ingestion.adapters import MetaGraphAPIAdapter, GraphAPIFetchError

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
POSTS_FIXTURE = os.path.join(FIXTURES_DIR, "facebook_graph_api_posts_response.json")
POSTS_FIXTURE_PAGE2 = os.path.join(FIXTURES_DIR, "facebook_graph_api_posts_response_page2.json")


def read_fixture_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mock_response(status_code=200, payload=None):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    return response


# --------------------------------------------------------------------
# Happy path: multiple posts, one call
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_fetch_posts_normalizes_valid_posts(mock_get):
    mock_get.return_value = mock_response(200, read_fixture_json(POSTS_FIXTURE))
    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token="test-token")

    evidence_list = adapter.fetch_posts()

    # Fixture has 4 posts: 2 valid messages, 1 empty message, 1 missing
    # message entirely -- only the 2 valid ones should come back.
    assert len(evidence_list) == 2
    assert all(isinstance(e, NormalizedEvidence) for e in evidence_list)
    assert all(e.source_type == "facebook" for e in evidence_list)
    texts = [e.text for e in evidence_list]
    assert any("College Enrollment is ongoing" in t for t in texts)
    assert any("Enrollment is Open" in t for t in texts)


@patch("src.ingestion.adapters.requests.get")
def test_fetch_posts_preserves_post_metadata(mock_get):
    mock_get.return_value = mock_response(200, read_fixture_json(POSTS_FIXTURE))
    adapter = MetaGraphAPIAdapter(
        page_id="1000000000000", access_token="test-token", credibility_tier="official",
    )

    evidence_list = adapter.fetch_posts()
    first = next(e for e in evidence_list if "College Enrollment" in e.text)

    assert first.source_url == "https://www.facebook.com/example.page/posts/2000000000001"
    assert first.published_date == "2026-06-01T09:00:00+0000"
    assert first.source_identifier == "1000000000000_2000000000001"
    assert first.credibility_tier == "official"
    assert first.document_id == "ingested:1000000000000_2000000000001"


# --------------------------------------------------------------------
# Malformed-post handling: skip, don't crash, don't silently ignore
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_malformed_post_is_skipped_not_crashing_the_batch(mock_get):
    """A post with an empty or missing 'message' must not raise, and
    must not stop valid posts in the SAME response from being
    normalized."""
    mock_get.return_value = mock_response(200, read_fixture_json(POSTS_FIXTURE))
    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token="test-token")

    evidence_list = adapter.fetch_posts()  # must not raise

    assert len(evidence_list) == 2  # the 2 valid posts still came through


@patch("src.ingestion.adapters.requests.get")
def test_malformed_posts_are_recorded_in_skipped_posts(mock_get):
    """The skip must be observable/deterministic, not silent -- checked
    here field by field, not just 'something non-empty exists'."""
    mock_get.return_value = mock_response(200, read_fixture_json(POSTS_FIXTURE))
    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token="test-token")

    adapter.fetch_posts()

    assert len(adapter.skipped_posts) == 2
    skipped_ids = {s["post_id"] for s in adapter.skipped_posts}
    assert skipped_ids == {"1000000000000_2000000000002", "1000000000000_2000000000003"}
    assert all("message" in s["reason"] for s in adapter.skipped_posts)


@patch("src.ingestion.adapters.requests.get")
def test_skipped_posts_resets_on_each_call(mock_get):
    """skipped_posts must reflect only the MOST RECENT fetch_posts()
    call, not accumulate silently across repeated calls."""
    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token="test-token")

    mock_get.return_value = mock_response(200, read_fixture_json(POSTS_FIXTURE))
    adapter.fetch_posts()
    assert len(adapter.skipped_posts) == 2

    all_valid_payload = {"data": [
        {"id": "x_1", "message": "Fine.", "created_time": "2026-06-05T00:00:00+0000",
         "permalink_url": "https://www.facebook.com/example.page/posts/x_1"},
    ]}
    mock_get.return_value = mock_response(200, all_valid_payload)
    adapter.fetch_posts()
    assert adapter.skipped_posts == []  # reset, not still showing the earlier run's skips


@patch("src.ingestion.adapters.requests.get")
def test_whitespace_only_message_is_skipped(mock_get):
    payload = {"data": [
        {"id": "x_2", "message": "   \n\t  ", "created_time": "2026-06-06T00:00:00+0000"},
    ]}
    mock_get.return_value = mock_response(200, payload)
    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token="test-token")

    evidence_list = adapter.fetch_posts()

    assert evidence_list == []
    assert len(adapter.skipped_posts) == 1
    assert adapter.skipped_posts[0]["post_id"] == "x_2"


# --------------------------------------------------------------------
# Pagination: explicit, deterministic, single-page-only for M8.5
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_fetch_posts_does_not_follow_pagination(mock_get):
    """M8.5 fetches exactly one page. The second page's post must NOT
    appear, and requests.get must be called exactly once even though
    the mocked response contains a paging.next URL."""
    mock_get.return_value = mock_response(200, read_fixture_json(POSTS_FIXTURE))
    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token="test-token")

    evidence_list = adapter.fetch_posts()

    assert mock_get.call_count == 1
    assert not any("second page" in e.text for e in evidence_list)


@patch("src.ingestion.adapters.requests.get")
def test_second_page_content_is_reachable_only_via_a_second_explicit_call(mock_get):
    """Demonstrates the actual (documented) scope: getting page 2's
    content requires the caller to explicitly call fetch_posts() again
    against that page's data -- this adapter does not do it implicitly."""
    mock_get.return_value = mock_response(200, read_fixture_json(POSTS_FIXTURE_PAGE2))
    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token="test-token")

    evidence_list = adapter.fetch_posts()

    assert len(evidence_list) == 1
    assert "second page" in evidence_list[0].text
    assert mock_get.call_count == 1  # still exactly one call for this one explicit invocation


# --------------------------------------------------------------------
# Failure handling: loud, never silently empty, never touching the corpus
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_non_200_status_raises_graph_api_fetch_error(mock_get):
    mock_get.return_value = mock_response(500, {"data": []})
    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token="test-token")

    with pytest.raises(GraphAPIFetchError):
        adapter.fetch_posts()


@patch("src.ingestion.adapters.requests.get")
def test_api_error_object_raises_graph_api_fetch_error(mock_get):
    """A 200-status response that nonetheless contains an {"error": ...}
    body (a real, documented Graph API failure mode) must be treated as
    a failure, not parsed as if it contained posts."""
    error_payload = {"error": {"message": "Invalid OAuth access token.", "code": 190}}
    mock_get.return_value = mock_response(200, error_payload)
    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token="bad-token")

    with pytest.raises(GraphAPIFetchError, match="190"):
        adapter.fetch_posts()


@patch("src.ingestion.adapters.requests.get")
def test_timeout_raises_graph_api_fetch_error(mock_get):
    import requests as requests_module
    mock_get.side_effect = requests_module.exceptions.Timeout("timed out")
    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token="test-token")

    with pytest.raises(GraphAPIFetchError):
        adapter.fetch_posts()


@patch("src.ingestion.adapters.requests.get")
def test_empty_data_list_returns_empty_list_not_an_error(mock_get):
    mock_get.return_value = mock_response(200, {"data": []})
    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token="test-token")

    evidence_list = adapter.fetch_posts()

    assert evidence_list == []
    assert adapter.skipped_posts == []


# --------------------------------------------------------------------
# Token-leak regression tests (M8.5-A fix)
#
# IMPORTANT: a DUMMY token only -- never a real one. These simulate the
# real-world leak shape directly: requests/urllib3 exception messages
# frequently embed the full failed request URL, including its query
# string (and therefore access_token=...), in their own str(). Before
# the fix, GraphAPIFetchError's messages interpolated that raw exception
# text (`{e}`) verbatim, which would have carried the token straight
# into any log, console, or caller's exception handler.
# --------------------------------------------------------------------

DUMMY_TOKEN = "DUMMY_TEST_TOKEN_do_not_use_ABC123xyz456"


@patch("src.ingestion.adapters.requests.get")
def test_connection_error_does_not_leak_token(mock_get):
    leaky_urllib3_style_message = (
        "HTTPSConnectionPool(host='graph.facebook.com', port=443): "
        "Max retries exceeded with url: /v21.0/123456/posts?fields=id,message,created_time"
        f"&limit=25&access_token={DUMMY_TOKEN} (Caused by NewConnectionError(...))"
    )
    mock_get.side_effect = requests.exceptions.ConnectionError(leaky_urllib3_style_message)
    adapter = MetaGraphAPIAdapter(page_id="123456", access_token=DUMMY_TOKEN)

    with pytest.raises(GraphAPIFetchError) as excinfo:
        adapter.fetch_posts()

    message = str(excinfo.value)
    assert DUMMY_TOKEN not in message
    assert "[REDACTED]" in message
    assert "123456" in message  # page_id still present -- still diagnostic


@patch("src.ingestion.adapters.requests.get")
def test_timeout_does_not_leak_token(mock_get):
    leaky_message = f"Read timed out. (read timeout=10) url=.../posts?access_token={DUMMY_TOKEN}"
    mock_get.side_effect = requests.exceptions.Timeout(leaky_message)
    adapter = MetaGraphAPIAdapter(page_id="123456", access_token=DUMMY_TOKEN)

    with pytest.raises(GraphAPIFetchError) as excinfo:
        adapter.fetch_posts()

    assert DUMMY_TOKEN not in str(excinfo.value)
    assert "[REDACTED]" in str(excinfo.value)


@patch("src.ingestion.adapters.requests.get")
def test_generic_request_exception_does_not_leak_token(mock_get):
    leaky_message = f"some other requests-level failure referencing access_token={DUMMY_TOKEN} in its text"
    mock_get.side_effect = requests.exceptions.RequestException(leaky_message)
    adapter = MetaGraphAPIAdapter(page_id="123456", access_token=DUMMY_TOKEN)

    with pytest.raises(GraphAPIFetchError) as excinfo:
        adapter.fetch_posts()

    assert DUMMY_TOKEN not in str(excinfo.value)
    assert "[REDACTED]" in str(excinfo.value)


@patch("src.ingestion.adapters.requests.get")
def test_scrubbed_error_messages_remain_diagnostic(mock_get):
    """
    The fix must not turn error messages into useless noise: the page_id
    and the nature of the failure (e.g. "connection error") must remain
    readable after scrubbing -- only the token itself is removed.
    """
    mock_get.side_effect = requests.exceptions.ConnectionError(
        f"connection refused while fetching ...access_token={DUMMY_TOKEN}"
    )
    adapter = MetaGraphAPIAdapter(page_id="999888777", access_token=DUMMY_TOKEN)

    with pytest.raises(GraphAPIFetchError) as excinfo:
        adapter.fetch_posts()

    message = str(excinfo.value)
    assert "999888777" in message
    assert "connection error" in message.lower()
    assert DUMMY_TOKEN not in message


@patch("src.ingestion.adapters.requests.get")
def test_token_not_leaked_when_no_token_configured(mock_get):
    """Edge case: an adapter with an empty access_token must not crash
    the scrub helper (empty string is falsy, so the exact-match pass is
    skipped; the regex pass alone still runs safely)."""
    mock_get.side_effect = requests.exceptions.ConnectionError("connection refused, no query string here")
    adapter = MetaGraphAPIAdapter(page_id="123456", access_token="")

    with pytest.raises(GraphAPIFetchError) as excinfo:
        adapter.fetch_posts()

    assert "123456" in str(excinfo.value)


@patch("src.ingestion.adapters.requests.get")
def test_successful_fetch_is_unaffected_by_the_scrubbing_fix(mock_get):
    """
    Regression check: the success path (no exception at all) must be
    completely unchanged by this fix -- the scrub helper is only ever
    invoked inside the three exception-handling branches.
    """
    mock_get.return_value = mock_response(200, read_fixture_json(POSTS_FIXTURE))
    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token=DUMMY_TOKEN)

    evidence_list = adapter.fetch_posts()

    assert len(evidence_list) == 2
    assert all(isinstance(e, NormalizedEvidence) for e in evidence_list)


# --------------------------------------------------------------------
# Integration with existing (unmodified) corpus storage + retrieval
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_fetched_posts_integrate_with_existing_corpus_storage_and_retrieval(mock_get, tmp_path):
    """Same integration shape M8.4 proved for the website adapter: this
    adapter's output needs zero special-case handling anywhere
    downstream -- it goes straight into the existing, unmodified
    corpus_storage/Retriever pipeline."""
    from src.ingestion.corpus_storage import append_to_corpus, load_corpus_for_retrieval
    from src.retrieval.retriever import Retriever

    mock_get.return_value = mock_response(200, read_fixture_json(POSTS_FIXTURE))
    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token="test-token")
    evidence_list = adapter.fetch_posts()

    corpus_path = str(tmp_path / "facebook_test_corpus.jsonl")
    for evidence in evidence_list:
        append_to_corpus(evidence, corpus_path)

    documents = load_corpus_for_retrieval(corpus_path)
    assert len(documents) == 2

    retriever = Retriever(documents)
    results = retriever.retrieve("Is enrollment open?", top_k=1)
    assert len(results) == 1
    assert results[0].score > 0.0
