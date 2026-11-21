"""
API protection (src/api/protection.py, wired in src/api/main.py) and the fetched-page size cap (src/ingestion/adapters.py).
No network: /verify/url refusals come from the SSRF guard, and page fetches are mocked.
"""
import threading
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.protection import Busy, ConcurrencyGate, RateLimiter, api_key_ok, client_key
from src.ingestion import adapters
from src.ingestion.adapters import WebsiteFetchError, WebsiteSourceAdapter

ORIGIN = "http://127.0.0.1:5173"


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# ---- units -------------------------------------------------------------------------------------------------------------
def test_rate_limiter_allows_n_per_window_per_client_and_slides():
    clock = _Clock()
    rl = RateLimiter(2, window_seconds=60, clock=clock)
    assert rl.check("a") == (True, 0) and rl.check("a") == (True, 0)
    allowed, retry = rl.check("a")
    assert not allowed and 1 <= retry <= 61
    assert rl.check("b")[0]                                   # another client is not affected
    clock.t = 61
    assert rl.check("a")[0]                                   # the window slid


def test_rate_limiter_memory_is_bounded_and_zero_disables():
    rl = RateLimiter(1, max_clients=3)
    for i in range(10):
        rl.check(f"client-{i}")
    assert len(rl._hits) == 3
    off = RateLimiter(0)
    assert all(off.check("x")[0] for _ in range(100))


def test_concurrency_gate_refuses_when_full_and_frees_the_slot():
    gate = ConcurrencyGate(1, wait_seconds=0.05)
    held, release = threading.Event(), threading.Event()

    def hold():
        with gate.slot():
            held.set(); release.wait(2)

    t = threading.Thread(target=hold); t.start(); held.wait(2)
    with pytest.raises(Busy):
        with gate.slot():
            pass
    release.set(); t.join()
    with gate.slot():                                          # free again
        pass
    with ConcurrencyGate(0, 0).slot():                         # 0 = no cap
        pass


def test_client_identity_ignores_forwarded_headers_unless_a_proxy_is_trusted():
    headers = {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}
    assert client_key(headers, "127.0.0.1", trusted_proxies=0) == "127.0.0.1"
    assert client_key({}, None, trusted_proxies=0) == "unknown"


def test_with_trusted_proxies_the_client_is_the_entry_the_outermost_proxy_added():
    # One proxy (a host's front door): it appends the address it saw, so the client is the LAST entry.
    assert client_key({"x-forwarded-for": "198.51.100.7"}, "10.0.0.2", trusted_proxies=1) == "198.51.100.7"
    # A client that writes its own header cannot pick its identity: only the proxy's own entry counts.
    assert client_key({"x-forwarded-for": "1.2.3.4, 198.51.100.7"}, "10.0.0.2", trusted_proxies=1) == "198.51.100.7"
    # Two proxies: the client is the second entry from the right.
    assert client_key({"x-forwarded-for": "6.6.6.6, 198.51.100.7, 10.0.0.9"}, "10.0.0.2", trusted_proxies=2) == "198.51.100.7"
    # Fewer entries than proxies (the request did not come through them): the connecting address.
    assert client_key({"x-forwarded-for": "198.51.100.7"}, "10.0.0.2", trusted_proxies=2) == "10.0.0.2"
    assert client_key({}, "10.0.0.2", trusted_proxies=1) == "10.0.0.2"


def test_the_setting_is_a_proxy_count_and_the_old_value_1_still_means_one_proxy(monkeypatch):
    from src.api.protection import ProtectionSettings
    for value, expected in [("", 0), ("0", 0), ("1", 1), ("2", 2), ("-3", 0)]:
        monkeypatch.setenv("CAMPUS_VERIFY_TRUST_PROXY", value)
        assert ProtectionSettings.from_env().trusted_proxies == expected


def test_api_key_check():
    assert api_key_ok("", None) and api_key_ok("secret", "secret")
    assert not api_key_ok("secret", None) and not api_key_ok("secret", "wrong")


# ---- through the API ---------------------------------------------------------------------------------------------------
def test_rate_limit_returns_429_with_retry_after_and_cors(api_protection):
    api_main = api_protection(rate_per_minute=2)
    with TestClient(api_main.app) as c:
        for _ in range(2):
            assert c.post("/verify/text", json={"claim": "kailan ang enrollment?"}).status_code == 200
        r = c.post("/verify/text", json={"claim": "kailan ang enrollment?"}, headers={"Origin": ORIGIN})
        assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1 and "Too many requests" in r.json()["detail"]
        assert r.headers.get("access-control-allow-origin") == ORIGIN     # the page can read the refusal
        assert all(c.get("/health").status_code == 200 for _ in range(5))   # /health is never limited


def test_behind_a_proxy_faked_forwarded_addresses_do_not_escape_the_limit(api_protection):
    api_main = api_protection(rate_per_minute=2, trusted_proxies=1)
    with TestClient(api_main.app) as c:
        def post(fake):   # the client writes a new fake address each time; the proxy appends the real one
            return c.post("/verify/text", json={"claim": "kailan ang enrollment?"},
                          headers={"X-Forwarded-For": f"{fake}, 198.51.100.7"})
        assert [post(f"1.1.1.{i}").status_code for i in range(3)] == [200, 200, 429]
        other = c.post("/verify/text", json={"claim": "kailan ang enrollment?"}, headers={"X-Forwarded-For": "198.51.100.8"})
        assert other.status_code == 200                       # a different real visitor keeps their own limit


def test_url_checks_have_their_own_stricter_bucket(api_protection):
    api_main = api_protection(rate_per_minute=5, url_rate_per_minute=1)
    with TestClient(api_main.app) as c:
        body = {"claim": "Are classes suspended tomorrow?", "url": "http://localhost:8000/health"}
        assert c.post("/verify/url", json=body).status_code == 400          # refused by the SSRF guard, still counted
        assert c.post("/verify/url", json=body).status_code == 429
        assert c.post("/verify/text", json={"claim": "may pasok?"}).status_code == 200


def test_body_size_limits(api_protection):
    api_main = api_protection(max_body_bytes=60)
    with TestClient(api_main.app) as c:
        assert c.post("/verify/text", json={"claim": "x" * 100}).status_code == 413
        chunked = c.post("/verify/text", content=iter([b'{"claim": "may pasok?"}']), headers={"content-type": "application/json"})
        assert chunked.status_code == 411                                    # no declared size: not read at all


def test_overlong_claims_and_urls_are_rejected_before_any_model_runs(api_protection):
    api_main = api_protection()
    with TestClient(api_main.app) as c:
        assert c.post("/verify/text", json={"claim": "a" * 1001}).status_code == 422
        assert c.post("/verify/url", json={"claim": "a", "url": "https://x.edu/" + "p" * 2100}).status_code == 422
        assert c.post("/verify/text", json={"claim": "a" * 1000}).status_code == 200


def test_optional_api_key(api_protection):
    api_main = api_protection(api_key="demo-key")
    with TestClient(api_main.app) as c:
        assert c.post("/verify/text", json={"claim": "may pasok?"}).status_code == 401
        assert c.post("/verify/text", json={"claim": "may pasok?"}, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/verify/text", json={"claim": "may pasok?"}, headers={"X-API-Key": "demo-key"}).status_code == 200
        assert c.get("/health").status_code == 200


def test_busy_checker_returns_503_never_a_verdict(api_protection):
    api_main = api_protection(max_concurrent=1, queue_seconds=0.05)
    with TestClient(api_main.app) as c:
        sem = api_main.state["gate"]._sem
        sem.acquire()                                                        # another check is running
        try:
            r = c.post("/verify/text", json={"claim": "Classes are suspended tomorrow."})
        finally:
            sem.release()
        assert r.status_code == 503 and r.headers["Retry-After"] == "5" and "busy" in r.json()["detail"]
        assert c.post("/verify/text", json={"claim": "Classes are suspended tomorrow."}).status_code == 200


def test_security_headers(api_protection):
    api_main = api_protection()
    with TestClient(api_main.app) as c:
        h, v = c.get("/health").headers, c.post("/verify/text", json={"claim": "may pasok?"}).headers
    for headers in (h, v):
        assert headers["X-Content-Type-Options"] == "nosniff" and headers["Referrer-Policy"] == "no-referrer"
    assert v["Cache-Control"] == "no-store"


# ---- fetched-page size cap ---------------------------------------------------------------------------------------------
def _page(chunks, declared=None):
    response = Mock(status_code=200, text="<html><body><p>x</p></body></html>", apparent_encoding="utf-8")
    response.headers = {"Content-Length": str(declared)} if declared is not None else {}
    response.iter_content = Mock(return_value=chunks)
    return response


@patch("src.ingestion.adapters.requests.get")
def test_oversized_pages_are_refused(mock_get, monkeypatch):
    monkeypatch.setattr(adapters, "MAX_PAGE_BYTES", 10)
    mock_get.return_value = _page([b"12345", b"67890", b"X"])
    with pytest.raises(WebsiteFetchError, match="limit"):
        WebsiteSourceAdapter(url="https://example.edu/a").fetch()
    declared = _page([b"x"], declared=999)
    mock_get.return_value = declared
    with pytest.raises(WebsiteFetchError, match="too large"):
        WebsiteSourceAdapter(url="https://example.edu/a").fetch()
    declared.iter_content.assert_not_called()                                # refused before reading anything
    assert mock_get.call_args.kwargs["stream"] is True


def test_public_mode_hides_the_api_documentation_and_trims_health(api_protection, monkeypatch):
    """Pre-deployment review L3 (owner 2026-10-25): with CAMPUS_VERIFY_PUBLIC=1 the documentation pages are gone and
    /health shows only what the page reads; checking still works. Without it, everything is shown as before."""
    api_main = api_protection()
    with TestClient(api_main.app) as c:
        full = c.get("/health").json()
        assert {"classifier", "corpus_source", "corpus_sources"} <= set(full)
        assert all(c.get(p).status_code == 200 for p in ("/docs", "/redoc", "/openapi.json"))
        monkeypatch.setenv("CAMPUS_VERIFY_PUBLIC", "1")
        assert all(c.get(p).status_code == 404 for p in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"))
        h = c.get("/health")
        assert h.status_code == 200 and h.json() == {"status": "ok", "image_check": full["image_check"],
                                                     "image_signals": full["image_signals"]}
        assert h.headers.get("x-content-type-options") == "nosniff"
        assert c.post("/verify/text", json={"claim": "kailan ang enrollment?"}).status_code == 200
