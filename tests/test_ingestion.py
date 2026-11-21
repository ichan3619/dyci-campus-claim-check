"""
Tests for src/ingestion/ (Milestone 8).

Covers: successful normalization, required/optional metadata handling,
invalid/empty content rejection, adapter behavior (the one working
adapter, and confirming the two stubs correctly refuse to run rather
than silently doing nothing), and the raw-vs-normalized separation.
"""

import os

import pytest

from src.ingestion.evidence import NormalizedEvidence, normalize
from src.ingestion.adapters import (
    LocalFileSourceAdapter, WebsiteSourceAdapter, MetaGraphAPIAdapter,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_FILE = os.path.join(FIXTURES_DIR, "sample_announcement.txt")
EMPTY_FILE = os.path.join(FIXTURES_DIR, "empty_content.txt")


# --------------------------------------------------------------------
# normalize() -- successful normalization
# --------------------------------------------------------------------

def test_normalize_produces_normalized_evidence():
    result = normalize(text="Classes are suspended tomorrow.", source_type="official_site")
    assert isinstance(result, NormalizedEvidence)


def test_normalize_collapses_whitespace_but_preserves_content():
    """Same conservative normalization stance as src/data/preprocessing.py
    -- collapse whitespace, but don't touch dates/numbers/names."""
    messy = "  Classes  are\tsuspended   on August 10.  "
    result = normalize(text=messy, source_type="official_site")
    assert result.text == "Classes are suspended on August 10."


def test_normalize_derives_document_id_from_source_url_when_no_identifier_given():
    result = normalize(text="Some evidence.", source_type="official_site",
                        source_url="https://example.edu/notice/1")
    assert result.document_id.startswith("ingested:")
    assert len(result.document_id) > len("ingested:")


def test_normalize_document_id_is_deterministic():
    """Re-normalizing the SAME source_url twice must produce the SAME
    document_id -- required so re-ingesting a page doesn't create
    duplicate corpus entries."""
    r1 = normalize(text="Text A", source_type="official_site", source_url="https://x.edu/1")
    r2 = normalize(text="Text B (different!)", source_type="official_site", source_url="https://x.edu/1")
    assert r1.document_id == r2.document_id  # same URL -> same ID, regardless of text


def test_normalize_prefers_explicit_source_identifier_over_url():
    result = normalize(text="Some evidence.", source_type="official_site",
                        source_url="https://example.edu/notice/1",
                        source_identifier="notice-204")
    assert "notice-204" in result.document_id


# --------------------------------------------------------------------
# Required and optional metadata
# --------------------------------------------------------------------

def test_normalize_preserves_all_provided_metadata():
    result = normalize(
        text="Classes are suspended tomorrow.",
        source_type="official_site",
        source_url="https://example.edu/notice/1",
        published_date="2026-01-10",
        fetched_at="2026-01-15T10:00:00+00:00",
        credibility_tier="official",
        source_identifier="notice-1",
    )
    assert result.source_url == "https://example.edu/notice/1"
    assert result.published_date == "2026-01-10"
    assert result.fetched_at == "2026-01-15T10:00:00+00:00"
    assert result.credibility_tier == "official"
    assert result.source_identifier == "notice-1"


def test_normalize_handles_missing_optional_metadata():
    """Only text and source_type are required -- everything else should
    default cleanly to empty string, not crash or require None-handling
    downstream."""
    result = normalize(text="Some evidence.", source_type="manual_entry")
    assert result.source_url == ""
    assert result.published_date == ""
    assert result.fetched_at == ""
    assert result.credibility_tier == ""
    assert result.source_identifier == ""
    # document_id must still be derived even with nothing else provided
    assert result.document_id.startswith("ingested:")


# --------------------------------------------------------------------
# Invalid / empty content
# --------------------------------------------------------------------

def test_normalize_rejects_empty_text():
    with pytest.raises(ValueError, match="empty"):
        normalize(text="", source_type="official_site")


def test_normalize_rejects_whitespace_only_text():
    with pytest.raises(ValueError, match="empty"):
        normalize(text="   \n\t  ", source_type="official_site")


def test_normalize_rejects_invalid_source_type():
    """Reuses src.data.schema.VALID_SOURCE_TYPES -- an ingested document
    claiming to be from an unrecognized source type must be rejected,
    same rule already enforced for the classifier training CSV."""
    with pytest.raises(ValueError, match="not one of"):
        normalize(text="Some evidence.", source_type="university_website")  # not a valid value


def test_normalize_rejects_invalid_credibility_tier():
    with pytest.raises(ValueError, match="not one of"):
        normalize(text="Some evidence.", source_type="official_site",
                  credibility_tier="very_trustworthy")  # not a recognized tier


def test_normalize_accepts_empty_credibility_tier():
    """Unlike source_type, credibility_tier is genuinely optional --
    empty must NOT raise."""
    result = normalize(text="Some evidence.", source_type="official_site", credibility_tier="")
    assert result.credibility_tier == ""


# --------------------------------------------------------------------
# to_retriever_document() -- the integration point with retrieval
# --------------------------------------------------------------------

def test_to_retriever_document_matches_retriever_expected_shape():
    """Confirms the exact dict shape src.retrieval.retriever.Retriever's
    constructor expects -- this is what proves Retriever needed ZERO
    changes to accept ingested evidence."""
    result = normalize(text="Some evidence.", source_type="official_site",
                        source_url="https://example.edu/1")
    doc = result.to_retriever_document()

    assert set(doc.keys()) == {"document_id", "text", "source_type", "source_url"}
    assert doc["text"] == "Some evidence."
    assert doc["source_type"] == "official_site"


def test_ingested_evidence_actually_works_with_real_retriever():
    """End-to-end proof: normalize a document, feed it to a REAL
    Retriever (not a mock), and confirm it's actually searchable."""
    from src.retrieval.retriever import Retriever

    evidence = normalize(
        text="Official notice: the library will be open 24/7 during finals week.",
        source_type="official_site",
        source_url="https://example.edu/library-hours",
    )
    retriever = Retriever([evidence.to_retriever_document()])
    results = retriever.retrieve("Is the library open late during finals?", top_k=1)

    assert len(results) == 1
    assert results[0].document_id == evidence.document_id


# --------------------------------------------------------------------
# Source adapters
# --------------------------------------------------------------------

def test_local_file_adapter_fetches_and_normalizes():
    adapter = LocalFileSourceAdapter(
        file_path=SAMPLE_FILE,
        source_type="official_site",
        source_url="https://example.edu/announcements/204",
    )
    result = adapter.fetch()

    assert isinstance(result, NormalizedEvidence)
    assert "24/7" in result.text
    assert result.source_type == "official_site"
    assert result.source_url == "https://example.edu/announcements/204"


def test_local_file_adapter_sets_fetched_at_automatically():
    adapter = LocalFileSourceAdapter(file_path=SAMPLE_FILE, source_type="official_site")
    result = adapter.fetch()
    assert result.fetched_at != ""  # a real timestamp was recorded, not left blank


def test_local_file_adapter_uses_file_path_as_identifier_when_none_given():
    adapter = LocalFileSourceAdapter(file_path=SAMPLE_FILE, source_type="official_site")
    result = adapter.fetch()
    assert SAMPLE_FILE in result.source_identifier


def test_local_file_adapter_raises_on_missing_file():
    adapter = LocalFileSourceAdapter(file_path="/nonexistent/path.txt", source_type="official_site")
    with pytest.raises(FileNotFoundError):
        adapter.fetch()


def test_local_file_adapter_raises_on_empty_file_content():
    """The file exists, but has no real content -- normalize() should
    still reject it, proving validation runs even for a successfully
    'fetched' (but empty) source."""
    adapter = LocalFileSourceAdapter(file_path=EMPTY_FILE, source_type="official_site")
    with pytest.raises(ValueError, match="empty"):
        adapter.fetch()


def test_local_file_adapter_raises_on_invalid_source_type():
    adapter = LocalFileSourceAdapter(file_path=SAMPLE_FILE, source_type="not_a_real_type")
    with pytest.raises(ValueError, match="not one of"):
        adapter.fetch()


def test_website_adapter_is_implemented_as_of_milestone_8_2():
    """
    Superseded assertion, updated: WebsiteSourceAdapter was a
    NotImplementedError placeholder in Milestone 8.1; implementing it
    was the entire point of Milestone 8.2, so asserting it still raises
    NotImplementedError would now be asserting something false.

    Full behavioral coverage (successful fetch, HTTP failure handling,
    extraction, document identity, corpus storage integration, and the
    end-to-end retrieval test) lives in tests/test_website_adapter.py --
    this test only confirms the class no longer raises
    NotImplementedError, without duplicating that other file's mocking
    setup here.
    """
    from unittest.mock import patch, Mock

    adapter = WebsiteSourceAdapter(url="https://university.example.edu/notice/1")

    with patch("src.ingestion.adapters.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><head><title>T</title></head><body><p>Some real content here.</p></body></html>"
        mock_response.headers = {}
        mock_response.iter_content = Mock(return_value=[mock_response.text.encode("utf-8")])   # body is read in chunks
        mock_get.return_value = mock_response

        result = adapter.fetch()  # must NOT raise NotImplementedError

    assert result.text  # produced real (if minimal) evidence, not a stub failure


def test_facebook_adapter_is_implemented_as_of_milestone_8_5():
    """
    Superseded assertion, updated: FacebookGraphAPIAdapter was a
    NotImplementedError placeholder in Milestone 8.1 (per project policy,
    not to be built until real Graph API credentials/access existed).
    Milestone 8.5 replaced that placeholder with MetaGraphAPIAdapter,
    developed and tested entirely against controlled/mock Graph API
    responses (this project still has no real Meta credentials or DYCI
    Facebook access) -- so asserting NotImplementedError here would now
    be asserting something false.

    Note the interface itself changed, not just the implementation
    status: MetaGraphAPIAdapter does NOT implement SourceAdapter's
    single-document fetch() -- a Facebook Page's /posts endpoint
    naturally returns many posts per call, not one (see
    src.ingestion.adapters.MetaGraphAPIAdapter's own docstring for the
    full reasoning). It exposes fetch_posts() -> List[NormalizedEvidence]
    instead.

    Full behavioral coverage (successful multi-post fetch, malformed-post
    skip handling, pagination scope, HTTP/API-error handling, and corpus
    storage/retrieval integration) lives in tests/test_facebook_adapter.py
    -- this test only confirms the class is real and callable, without
    duplicating that other file's mocking setup here.
    """
    from unittest.mock import patch, Mock

    adapter = MetaGraphAPIAdapter(page_id="1000000000000", access_token="test-token")

    with patch("src.ingestion.adapters.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [
            {"id": "x_1", "message": "Some real post content.",
             "created_time": "2026-06-01T00:00:00+0000"},
        ]}
        mock_get.return_value = mock_response

        result = adapter.fetch_posts()  # must NOT raise NotImplementedError

    assert len(result) == 1
    assert result[0].text  # produced real (if minimal) evidence, not a stub failure


# --------------------------------------------------------------------
# Separation between raw source content and normalized evidence
# --------------------------------------------------------------------

def test_adapter_output_is_always_the_normalized_type_not_raw_text():
    """No adapter should ever hand back a bare string -- every adapter's
    fetch() must go through normalize() and return a NormalizedEvidence,
    keeping 'raw fetched content' and 'validated evidence' structurally
    distinct rather than conflated."""
    adapter = LocalFileSourceAdapter(file_path=SAMPLE_FILE, source_type="official_site")
    result = adapter.fetch()
    assert not isinstance(result, str)
    assert isinstance(result, NormalizedEvidence)


def test_ingestion_module_does_not_import_retrieval_internals():
    """Structural check, same pattern as Milestones 4/6/7: ingestion
    must stay independent of retrieval's implementation (TF-IDF,
    cosine similarity) -- it only produces documents in the shape
    Retriever expects, never reaching into how Retriever works."""
    import src.ingestion.evidence as evidence_module
    import src.ingestion.adapters as adapters_module
    import inspect

    for module in (evidence_module, adapters_module):
        import_lines = [
            line for line in inspect.getsource(module).splitlines()
            if line.strip().startswith("import ") or line.strip().startswith("from ")
        ]
        source_of_imports = "\n".join(import_lines)
        assert "TfidfVectorizer" not in source_of_imports
        assert "cosine_similarity" not in source_of_imports


def test_ingestion_module_does_not_import_classification_or_verification():
    import src.ingestion.evidence as evidence_module
    import src.ingestion.adapters as adapters_module
    import inspect

    for module in (evidence_module, adapters_module):
        import_lines = [
            line for line in inspect.getsource(module).splitlines()
            if line.strip().startswith("import ") or line.strip().startswith("from ")
        ]
        source_of_imports = "\n".join(import_lines)
        assert "src.classification" not in source_of_imports
        assert "src.verification" not in source_of_imports
