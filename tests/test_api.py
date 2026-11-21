"""
Tests for Milestone 10.1 -- the FastAPI backend (src/api/main.py).

IMPORTANT: these tests force CAMPUS_VERIFY_CLASSIFIER=tfidf_lr (see
src/api/main.py's _build_classifier()) BEFORE importing src.api.main, so
the whole suite runs against ClaimEvidenceClassifier -- trained fresh
from the local dev CSV, no network access and no model download required
-- rather than the real runtime default, NLIClassifier, which needs to
download cross-encoder/nli-deberta-v3-small from huggingface.co. This
mirrors the exact reasoning already established in
tests/test_nli_classifier.py (skip/avoid real model download in
automated tests) and in the M8.5 Facebook adapter tests (mock network
access rather than depend on it). A small, separate, explicitly-skippable
test at the bottom of this file exercises the REAL default (NLI) end to
end, skipped when the model can't be loaded -- so the actual production
default is still covered when network access is available.
"""

import os

os.environ["CAMPUS_VERIFY_CLASSIFIER"] = "tfidf_lr"

import requests as requests_module
import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from src.api.main import app


def mock_html_response(status_code=200, text="<html><body><p>Some fetched content.</p></body></html>"):
    response = Mock()
    response.status_code = status_code
    response.text = text
    response.headers = {}
    response.iter_content = Mock(return_value=[text.encode("utf-8")])   # adapters read the body in chunks
    response.apparent_encoding = "utf-8"
    response.encoding = None
    # NOTE: src.ingestion.adapters and src.api.ssrf_guard both `import
    # requests`, so patching "src.ingestion.adapters.requests.get" patches
    # the SAME underlying attribute ssrf_guard's own `requests.get` call
    # sees too. Every mocked response therefore needs to look like a
    # sensible answer to BOTH call sites: is_redirect/close (read by the
    # SSRF pre-flight probe) and status_code/text/... (read by
    # WebsiteSourceAdapter's real fetch) -- not just whichever one a
    # given test happens to be thinking about.
    response.is_redirect = False
    response.close = Mock()
    return response


@pytest.fixture()
def client():
    # TestClient used as a context manager triggers the app's lifespan
    # (startup: build Retriever + classifier + ClaimVerifier once; shutdown:
    # state.clear()) exactly the way a real deployed process would.
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------
# /health
# --------------------------------------------------------------------

def test_health_reports_ok_and_data_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["data_status"] == "development"
    assert body["classifier"] == "ClaimEvidenceClassifier"


# --------------------------------------------------------------------
# /verify/text
# --------------------------------------------------------------------

def test_verify_text_happy_path_returns_full_verification_result(client):
    response = client.post("/verify/text", json={"claim": "Are classes suspended tomorrow?"})
    assert response.status_code == 200
    body = response.json()

    # Field-for-field presence check against VerificationResult's shape --
    # this is a thin wrapper, so nothing should be missing or renamed.
    for field in ("claim", "verdict", "evidence_status", "retrieved_evidence",
                  "used_evidence", "classification_result", "explanation", "data_status"):
        assert field in body

    assert body["claim"] == "Are classes suspended tomorrow?"
    assert body["verdict"] in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")
    assert body["data_status"] == "development"
    assert isinstance(body["retrieved_evidence"], list)


def test_verify_text_rejects_empty_claim(client):
    response = client.post("/verify/text", json={"claim": ""})
    assert response.status_code == 422  # caught by Pydantic's field_validator


def test_verify_text_rejects_whitespace_only_claim(client):
    response = client.post("/verify/text", json={"claim": "   \n\t  "})
    assert response.status_code == 422


def test_verify_text_rejects_missing_claim_field(client):
    response = client.post("/verify/text", json={})
    assert response.status_code == 422


# --------------------------------------------------------------------
# /verify/url -- success (mocked fetch, no live network call)
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_verify_url_success_uses_fetched_content_as_evidence(mock_get, client):
    mock_get.return_value = mock_html_response(
        text="<html><body><p>Official Notice: Classes are suspended tomorrow due to a typhoon.</p></body></html>"
    )

    response = client.post(
        "/verify/url",
        json={"claim": "Are classes suspended tomorrow?", "url": "https://example.edu/announcement"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data_status"] == "development"
    # The temporary, single-document corpus means the ONLY possible
    # retrieved/used evidence is the one just fetched -- confirms the
    # ephemeral Retriever is genuinely built from the fetched page, not
    # accidentally reusing the permanent startup corpus.
    assert body["used_evidence"] is not None
    assert body["used_evidence"]["source_url"] == "https://example.edu/announcement"
    assert len(body["retrieved_evidence"]) == 1


@patch("src.ingestion.adapters.requests.get")
def test_verify_url_never_persists_to_the_permanent_corpus(mock_get, client, tmp_path):
    """
    Directly verifies the EPHEMERAL requirement: no corpus_storage file is
    written anywhere as a side effect of /verify/url, and the permanent,
    startup-built Retriever's document count is unaffected afterward.
    """
    from src.api import main as api_main

    permanent_retriever = api_main.state["manager"].current().retriever
    document_count_before = len(permanent_retriever._documents)

    mock_get.return_value = mock_html_response(
        text="<html><body><p>A brand new announcement never seen before.</p></body></html>"
    )
    response = client.post(
        "/verify/url",
        json={"claim": "Is this announced?", "url": "https://example.edu/new-announcement"},
    )
    assert response.status_code == 200

    # The SAME permanent Retriever object built at startup must be
    # untouched -- /verify/url must never rebuild or mutate it.
    assert api_main.state["manager"].current().retriever is permanent_retriever
    assert len(permanent_retriever._documents) == document_count_before


# --------------------------------------------------------------------
# /verify/url -- failure modes: never silently become INSUFFICIENT_EVIDENCE
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
def test_verify_url_connection_error_returns_502_not_a_verdict(mock_get, client):
    mock_get.side_effect = requests_module.exceptions.ConnectionError("boom")

    response = client.post(
        "/verify/url",
        json={"claim": "Are classes suspended tomorrow?", "url": "https://example.edu/unreachable"},
    )

    assert response.status_code == 502
    assert "Failed to fetch" in response.json()["detail"]


@patch("src.ingestion.adapters.requests.get")
def test_verify_url_timeout_returns_502(mock_get, client):
    mock_get.side_effect = requests_module.exceptions.Timeout("timed out")

    response = client.post(
        "/verify/url",
        json={"claim": "Are classes suspended tomorrow?", "url": "https://example.edu/slow"},
    )

    assert response.status_code == 502


@patch("src.ingestion.adapters.requests.get")
def test_verify_url_non_200_status_returns_502(mock_get, client):
    mock_get.return_value = mock_html_response(status_code=404, text="Not Found")

    response = client.post(
        "/verify/url",
        json={"claim": "Are classes suspended tomorrow?", "url": "https://example.edu/missing-page"},
    )

    assert response.status_code == 502


@patch("src.ingestion.adapters.requests.get")
def test_verify_url_empty_page_content_returns_502_not_insufficient_evidence(mock_get, client):
    """
    A page that fetches successfully but has no extractable text is an
    UPSTREAM CONTENT problem, not a client request-format problem, and
    must NOT be silently reinterpreted as "evidence was found and it's
    insufficient" (that would misrepresent a fetch/content failure as an
    evidence finding).
    """
    mock_get.return_value = mock_html_response(text="<html><body>   </body></html>")

    response = client.post(
        "/verify/url",
        json={"claim": "Are classes suspended tomorrow?", "url": "https://example.edu/blank-page"},
    )

    assert response.status_code == 502
    assert response.status_code != 200


def test_verify_url_rejects_empty_claim(client):
    response = client.post("/verify/url", json={"claim": "", "url": "https://example.edu/x"})
    assert response.status_code == 422


def test_verify_url_rejects_empty_url(client):
    response = client.post("/verify/url", json={"claim": "Some claim.", "url": ""})
    assert response.status_code == 422


# --------------------------------------------------------------------
# /verify/url -- SSRF protection (Milestone 10.2)
#
# These test the FULL endpoint wiring (request -> SSRF guard -> 400,
# never reaching WebsiteSourceAdapter at all). Unit-level coverage of the
# guard's own logic (every unsafe-IP category, redirect-chain walking,
# scheme rejection) lives in tests/test_ssrf_guard.py -- these confirm
# main.py actually calls it before fetching, not a second copy of that
# logic.
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_verify_url_allows_public_destination(mock_getaddrinfo, mock_get, client):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 0))]
    # One shared mock serves BOTH the SSRF pre-flight probe (is_redirect/
    # close) and the real WebsiteSourceAdapter fetch (status_code/text/...)
    # -- see mock_html_response()'s note on why this is one object, not two.
    mock_get.return_value = mock_html_response(
        text="<html><body><p>A real public announcement.</p></body></html>"
    )

    response = client.post(
        "/verify/url",
        json={"claim": "Is this announced?", "url": "https://example.com/announcement"},
    )

    assert response.status_code == 200


def test_verify_url_rejects_localhost(client):
    response = client.post(
        "/verify/url",
        json={"claim": "Are classes suspended tomorrow?", "url": "http://localhost:8000/health"},
    )
    assert response.status_code == 400
    assert "Refusing to fetch" in response.json()["detail"]


@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_verify_url_rejects_127_0_0_1(mock_getaddrinfo, client):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 0))]
    response = client.post(
        "/verify/url",
        json={"claim": "Are classes suspended tomorrow?", "url": "http://127.0.0.1:9999/internal"},
    )
    assert response.status_code == 400


@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_verify_url_rejects_private_address(mock_getaddrinfo, client):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("10.0.0.5", 0))]
    response = client.post(
        "/verify/url",
        json={"claim": "Are classes suspended tomorrow?", "url": "http://internal.example/secret"},
    )
    assert response.status_code == 400


def test_verify_url_rejects_non_http_https_scheme(client):
    response = client.post(
        "/verify/url",
        json={"claim": "Are classes suspended tomorrow?", "url": "file:///etc/passwd"},
    )
    assert response.status_code == 400


@patch("src.ingestion.adapters.requests.get")
@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_verify_url_rejects_redirect_to_private_destination(mock_getaddrinfo, mock_get, client):
    """
    The endpoint-level version of the redirect-chain attack: a URL that
    LOOKS public at the top level 302-redirects to a private address.
    Must be rejected (400), and the real fetch (which shares this SAME
    mocked requests.get -- see mock_html_response()'s note) must be
    called exactly once: the SSRF guard's pre-flight probe. If
    WebsiteSourceAdapter's own real fetch had also run, call_count would
    be 2 or more.
    """
    def getaddrinfo_side_effect(hostname, *args, **kwargs):
        if hostname == "public.example.com":
            return [(2, 1, 6, "", ("93.184.216.34", 0))]
        if hostname == "internal.example.com":
            return [(2, 1, 6, "", ("10.0.0.5", 0))]
        raise AssertionError(f"unexpected hostname: {hostname}")

    mock_getaddrinfo.side_effect = getaddrinfo_side_effect

    redirect_response = Mock()
    redirect_response.is_redirect = True
    redirect_response.headers = {"Location": "http://internal.example.com/secret"}
    redirect_response.close = Mock()
    mock_get.return_value = redirect_response

    response = client.post(
        "/verify/url",
        json={"claim": "Are classes suspended tomorrow?", "url": "https://public.example.com/redirector"},
    )

    assert response.status_code == 400
    assert mock_get.call_count == 1  # only the SSRF guard's own probe ran; the real fetch never happened


# --------------------------------------------------------------------
# CORS (Milestone 10.2)
# --------------------------------------------------------------------

def test_cors_allows_configured_local_dev_origin(client):
    response = client.options(
        "/verify/text",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_unconfigured_origin(client):
    response = client.options(
        "/verify/text",
        headers={
            "Origin": "https://not-an-allowed-origin.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    # Starlette's CORS middleware simply omits the allow-origin header
    # for a disallowed origin, rather than returning an error status --
    # a browser enforces the actual block client-side based on this
    # header being absent.
    assert response.headers.get("access-control-allow-origin") is None


# --------------------------------------------------------------------
# Real default (NLIClassifier) -- skipped if the model can't be loaded
# (no network / not cached), same skip discipline as
# tests/test_nli_classifier.py. Not part of the main suite's fast path.
# --------------------------------------------------------------------

def _nli_available():
    try:
        from src.classification.nli_classifier import NLIClassifier
        NLIClassifier()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _nli_available(), reason="NLI model not available locally (no network / not cached)")
def test_health_reports_nli_when_it_is_the_configured_default():
    # Deliberately does NOT use the module-level `client` fixture (which
    # forces tfidf_lr) -- this test re-imports with the real default to
    # confirm the actual production configuration also works end to end.
    import importlib
    os.environ["CAMPUS_VERIFY_CLASSIFIER"] = "nli"
    from src.api import main as api_main
    importlib.reload(api_main)

    with TestClient(api_main.app) as client:
        response = client.get("/health")
        assert response.json()["classifier"] == "NLIClassifier"

    # restore for any tests that run after this one in the same process
    os.environ["CAMPUS_VERIFY_CLASSIFIER"] = "tfidf_lr"
    importlib.reload(api_main)


# --------------------------------------------------------------------
# /verify/url -- the real fetch never follows a redirect the guard did not check (pre-deployment review H1)
# --------------------------------------------------------------------

@patch("src.ingestion.adapters.requests.get")
@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_verify_url_refuses_a_redirect_that_appears_only_after_the_safety_check(mock_getaddrinfo, mock_get, client):
    """A server answers the guard's walk with 200 and the real fetch with a redirect to the cloud metadata address.
    The redirect must be refused (502), never followed, and nothing but the checked URL may be requested."""
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 0))]
    checked = mock_html_response(text="<html><body><p>Looks harmless.</p></body></html>")
    switched = mock_html_response(status_code=302, text="")
    switched.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}
    mock_get.side_effect = [checked, switched]

    response = client.post(
        "/verify/url",
        json={"claim": "Are classes suspended tomorrow?", "url": "https://example.com/announcement"},
    )

    assert response.status_code == 502
    assert "redirect" in response.json()["detail"] and "not followed" in response.json()["detail"]
    assert [c.args[0] for c in mock_get.call_args_list] == ["https://example.com/announcement"] * 2
    assert all(c.kwargs.get("allow_redirects") is False for c in mock_get.call_args_list)
