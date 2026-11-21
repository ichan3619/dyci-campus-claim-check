"""Quick wins from docs/evaluation/system_design_review.md: CAMPUS_VERIFY_HARDENING (Content-Security-Policy, HSTS,
refusing cross-site state changes; default off) and per-account rate limits for signed-in DYCI accounts."""
from fastapi.testclient import TestClient

from test_auth import _client, signin, token          # noqa: F401  (the signed-in app fixture)

CLAIM = {"claim": "The midterm exams are on September 28 to 30."}


def test_off_by_default_nothing_changes(api_settings, monkeypatch):
    monkeypatch.delenv("CAMPUS_VERIFY_HARDENING", raising=False)
    app = api_settings()
    with TestClient(app.app, base_url="https://dycicheck.example.ts.net") as c:
        r = c.post("/verify/text", json=CLAIM, headers={"Origin": "https://evil.example"})
        assert r.status_code == 200
        assert "content-security-policy" not in r.headers and "strict-transport-security" not in r.headers


def test_headers_when_on(api_settings, monkeypatch):
    monkeypatch.setenv("CAMPUS_VERIFY_HARDENING", "1")
    app = api_settings()
    with TestClient(app.app, base_url="https://dycicheck.example.ts.net") as c:
        h = c.get("/health").headers
    csp = h["content-security-policy"]
    for part in ("default-src 'self'", "script-src 'self' https://accounts.google.com/gsi/client",
                 "frame-src https://accounts.google.com/gsi/", "object-src 'none'", "frame-ancestors 'none'"):
        assert part in csp, part
    assert h["strict-transport-security"].startswith("max-age=")


def test_no_hsts_on_the_pc_itself(api_settings, monkeypatch):
    monkeypatch.setenv("CAMPUS_VERIFY_HARDENING", "1")
    app = api_settings()
    with TestClient(app.app, base_url="http://localhost:8000") as c:
        assert "strict-transport-security" not in c.get("/health").headers


def test_cross_site_posts_are_refused_and_own_or_public_origin_pass(api_settings, monkeypatch):
    monkeypatch.setenv("CAMPUS_VERIFY_HARDENING", "1")
    monkeypatch.setenv("CAMPUS_VERIFY_PUBLIC_ORIGIN", "https://dycicheck.example.ts.net")
    app = api_settings()
    with TestClient(app.app, base_url="http://127.0.0.1:8000") as c:          # the tunnel's view: host is local
        assert c.post("/verify/text", json=CLAIM, headers={"Origin": "https://evil.example"}).status_code == 403
        assert c.post("/verify/text", json=CLAIM, headers={"Origin": "https://dycicheck.example.ts.net"}).status_code == 200
        assert c.post("/verify/text", json=CLAIM, headers={"Origin": "http://127.0.0.1:8000"}).status_code == 200
        assert c.post("/verify/text", json=CLAIM).status_code == 200                       # no Origin: not a browser form
        assert c.get("/health", headers={"Origin": "https://evil.example"}).status_code == 200   # reading is not refused


def test_each_signed_in_account_has_its_own_budget(signin, api_protection):
    app = api_protection(rate_per_minute=2)                # sign-in settings from the signin fixture stay in place
    with _client(app) as a, _client(app) as b:
        a.post("/auth/google", json={"credential": token(email="a.student@dyci.edu.ph")})
        b.post("/auth/google", json={"credential": token(email="b.student@dyci.edu.ph")})
        assert [a.post("/verify/text", json=CLAIM).status_code for _ in range(3)] == [200, 200, 429]
        assert b.post("/verify/text", json=CLAIM).status_code == 200                # a's use does not block b
