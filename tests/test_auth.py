"""Sign in with Google, DYCI accounts only (src/api/auth.py; CAMPUS_VERIFY_AUTH=google, default off). Tokens are signed
here with a test RSA key that stands in for Google's published keys, so nothing goes online."""
import base64
import json
import time

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi.testclient import TestClient

from src.api import auth, db

CLIENT = "test-client.apps.googleusercontent.com"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _jwk(key, kid="k1"):
    n = key.public_key().public_numbers()
    return {"kty": "RSA", "kid": kid, "alg": "RS256", "use": "sig", "e": _b64(n.e.to_bytes(3, "big")),
            "n": _b64(n.n.to_bytes((n.n.bit_length() + 7) // 8, "big"))}


def token(email="juan.delacruz@dyci.edu.ph", hd="dyci.edu.ph", aud=CLIENT, exp=None, verified=True, key=KEY, kid="k1",
          name="Juan Dela Cruz"):
    header = {"alg": "RS256", "kid": kid, "typ": "JWT"}
    claims = {"iss": "https://accounts.google.com", "aud": aud, "sub": "1234", "email": email, "email_verified": verified,
              "name": name, "iat": int(time.time()), "exp": int(time.time()) + 3600 if exp is None else exp}
    if hd is not None:
        claims["hd"] = hd
    signing = _b64(json.dumps(header).encode()) + "." + _b64(json.dumps(claims).encode())
    sig = key.sign(signing.encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    return signing + "." + _b64(sig)


@pytest.fixture
def signin(api_settings, monkeypatch, tmp_path):
    """The app with sign-in on, a fresh database and the test key as Google's key."""
    monkeypatch.setenv(auth.ENV, "google")
    monkeypatch.setenv(auth.CLIENT_ID_ENV, CLIENT)
    monkeypatch.setenv(auth.ADMINS_ENV, "admin.user@dyci.edu.ph")
    monkeypatch.setenv(db.ENV, str(tmp_path / "app.db"))
    monkeypatch.setattr(auth, "_keys", auth.GoogleKeys(fetch=lambda: ({"keys": [_jwk(KEY)]}, 3600)))
    app = api_settings()
    app.state.pop("db", None)
    app.state.pop("stats", None)
    return app


def _client(app):
    return TestClient(app.app, base_url="https://testserver")


def test_a_dyci_account_signs_in_and_stays_signed_in(signin):
    with _client(signin) as c:
        r = c.post("/auth/google", json={"credential": token()})
        assert r.status_code == 200 and r.json()["user"]["email"] == "juan.delacruz@dyci.edu.ph"
        cookie = r.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=lax" in cookie and "secure" in cookie
        assert c.get("/auth/me").json()["user"]["name"] == "Juan Dela Cruz"


@pytest.mark.parametrize("bad, status", [
    (dict(email="someone@gmail.com", hd=None), 403),              # a personal Gmail account
    (dict(email="x@otherschool.edu.ph", hd="otherschool.edu.ph"), 403),
    (dict(hd="dyci.edu.ph", email="trick@gmail.com"), 403),       # hd claim alone is not enough
    (dict(verified=False), 403),
    (dict(aud="someone-else.apps.googleusercontent.com"), 401),
    (dict(exp=int(time.time()) - 3600), 401),
    (dict(key=OTHER_KEY), 401),                                    # not signed by Google's key
    (dict(kid="unknown"), 401),
])
def test_anything_but_a_valid_dyci_token_is_refused(signin, bad, status):
    with _client(signin) as c:
        r = c.post("/auth/google", json={"credential": token(**bad)})
        assert r.status_code == status and "set-cookie" not in r.headers
        assert c.get("/auth/me").status_code == 401


def test_any_username_before_the_domain_is_accepted(signin):
    with _client(signin) as c:
        for email in ("2023-00123@dyci.edu.ph", "maria.santos.bsit@dyci.edu.ph"):
            assert c.post("/auth/google", json={"credential": token(email=email)}).status_code == 200


def test_only_a_hash_of_the_session_token_is_stored(signin, tmp_path):
    with _client(signin) as c:
        c.post("/auth/google", json={"credential": token()})
        raw = c.cookies.get(auth.COOKIE)
    stored = db.Database(str(tmp_path / "app.db")).conn.execute("SELECT token_hash FROM sessions").fetchall()
    assert raw and stored and all(raw not in row[0] for row in stored) and len(stored[0][0]) == 64


def test_sign_out_ends_the_session(signin):
    with _client(signin) as c:
        c.post("/auth/google", json={"credential": token()})
        raw = c.cookies.get(auth.COOKIE)
        c.post("/auth/logout")
        c.cookies.set(auth.COOKIE, raw)                            # an old copy of the cookie no longer works
        assert c.get("/auth/me").status_code == 401


def test_the_history_preference_is_the_students_own_choice(signin):
    with _client(signin) as c:
        c.post("/auth/google", json={"credential": token()})
        assert c.get("/auth/me").json()["user"]["history_enabled"] is False          # off until they choose
        assert c.post("/account/history", json={"history_enabled": True}).json()["user"]["history_enabled"] is True


def test_admins_manage_accounts_and_students_cannot(signin):
    with _client(signin) as student, _client(signin) as admin:
        student.post("/auth/google", json={"credential": token()})
        assert student.get("/admin/users").status_code == 403
        admin.post("/auth/google", json={"credential": token(email="admin.user@dyci.edu.ph", name="Admin")})
        users = admin.get("/admin/users").json()["users"]
        juan = next(u for u in users if u["email"] == "juan.delacruz@dyci.edu.ph")
        me = next(u for u in users if u["is_admin"])
        assert admin.post(f"/admin/users/{me['id']}/active", json={"active": False}).status_code == 400   # not yourself
        assert admin.post(f"/admin/users/{juan['id']}/active", json={"active": False}).json()["user"]["active"] is False
        assert student.get("/auth/me").status_code == 401                            # disabled: signed out at once
        assert student.post("/auth/google", json={"credential": token()}).status_code == 403
        admin.post(f"/admin/users/{juan['id']}/active", json={"active": True})
        assert student.post("/auth/google", json={"credential": token()}).status_code == 200


def test_required_mode_keeps_checks_for_signed_in_students(signin, monkeypatch):
    monkeypatch.setenv(auth.REQUIRED_ENV, "1")
    with _client(signin) as c:
        assert c.post("/verify/text", json={"claim": "The midterms are on September 28 to 30."}).status_code == 401
        c.post("/auth/google", json={"credential": token()})
        assert c.post("/verify/text", json={"claim": "The midterms are on September 28 to 30."}).status_code == 200
        assert c.get("/health").json()["auth"] == {"provider": "google", "domain": "dyci.edu.ph", "required": True}


def test_with_sign_in_off_nothing_changes(api_settings, monkeypatch):
    monkeypatch.delenv(auth.ENV, raising=False)
    app = api_settings()
    with TestClient(app.app) as c:
        assert "auth" not in c.get("/health").json()
        for path in ("/auth/config", "/auth/me", "/admin/users"):
            assert c.get(path).status_code == 404
        assert c.post("/verify/text", json={"claim": "The midterms are on September 28 to 30."}).status_code == 200


def test_the_client_id_is_the_only_setting_the_page_sees(signin):
    with _client(signin) as c:
        assert c.get("/auth/config").json() == {"client_id": CLIENT, "domain": "dyci.edu.ph", "required": False}


def test_with_sign_in_on_the_dashboard_is_for_administrators(signin, monkeypatch):
    monkeypatch.setenv("CAMPUS_VERIFY_STATS", "1")
    with _client(signin) as student, _client(signin) as admin:
        assert student.get("/stats").status_code == 401
        student.post("/auth/google", json={"credential": token()})
        assert student.get("/stats").status_code == 403
        admin.post("/auth/google", json={"credential": token(email="admin.user@dyci.edu.ph")})
        assert admin.get("/stats").status_code == 200


def test_the_page_never_decides_who_may_sign_in():
    import os
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = open(os.path.join(root, "frontend", "js", "app.js"), encoding="utf-8").read()
    block = app[app.index("// ---- Sign in with Google"):app.index("// ---- startup: GET /health")]
    assert "dyci.edu.ph" not in block                       # the domain comes from the server
    assert "innerHTML" not in block and not re.search(r"verdict\s*=(?!=)", block)
    assert "https://accounts.google.com/gsi/client" in block


@pytest.mark.parametrize("base, secure", [("http://localhost:8000", False), ("http://dycicheck.example.ts.net", True),
                                          ("https://dycicheck.example.ts.net", True)])
def test_the_cookie_is_secure_everywhere_but_on_the_pc_itself(signin, base, secure):
    with TestClient(signin.app, base_url=base) as c:
        cookie = c.post("/auth/google", json={"credential": token()}).headers["set-cookie"].lower()
    assert ("; secure" in cookie) is secure


def test_the_public_health_tells_the_page_that_sign_in_is_on(signin, monkeypatch):
    monkeypatch.setenv("CAMPUS_VERIFY_PUBLIC", "1")                    # what start_public.ps1 uses
    with _client(signin) as c:
        assert c.get("/health").json()["auth"]["provider"] == "google"


def test_students_can_delete_their_own_account(signin, tmp_path):
    with _client(signin) as c:
        c.post("/auth/google", json={"credential": token()})
        assert c.post("/account/delete", json={"confirm": "delete", "credential": token()}).json() == {"deleted": True}
        assert c.get("/auth/me").status_code == 401
    database = db.Database(str(tmp_path / "app.db"))
    assert database.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    assert database.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_the_privacy_notice_is_served_with_the_page_and_names_the_rights():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page = " ".join(open(os.path.join(root, "frontend", "privacy.html"), encoding="utf-8").read().split())
    for words in ("Republic Act No. 10173", "Delete my account", "National Privacy Commission", "400 days", "30 days"):
        assert words in page
    assert "<script" not in page



def test_deleting_needs_the_typed_word_and_a_fresh_google_confirmation_by_the_same_account(signin):
    old = int(time.time()) - 3600
    with _client(signin) as c:
        c.post("/auth/google", json={"credential": token()})
        assert c.post("/account/delete", json={}).status_code == 422                          # nothing sent
        assert c.post("/account/delete", json={"confirm": "yes", "credential": token()}).status_code == 400
        other = token(email="someone.else@dyci.edu.ph")
        assert c.post("/account/delete", json={"confirm": "DELETE", "credential": other}).status_code == 401
        stale = _stale_token(old)
        assert c.post("/account/delete", json={"confirm": "DELETE", "credential": stale}).status_code == 401
        assert c.post("/account/delete", json={"confirm": "DELETE", "credential": token(key=OTHER_KEY)}).status_code == 401
        assert c.get("/auth/me").status_code == 200                                         # still there


def _stale_token(iat):
    header = {"alg": "RS256", "kid": "k1", "typ": "JWT"}
    claims = {"iss": "https://accounts.google.com", "aud": CLIENT, "email": "juan.delacruz@dyci.edu.ph",
              "email_verified": True, "hd": "dyci.edu.ph", "iat": iat, "exp": int(time.time()) + 3600}
    signing = _b64(json.dumps(header).encode()) + "." + _b64(json.dumps(claims).encode())
    return signing + "." + _b64(KEY.sign(signing.encode("ascii"), padding.PKCS1v15(), hashes.SHA256()))


def test_fake_key_ids_cannot_make_the_server_fetch_google_keys_over_and_over():
    calls = []
    keys = auth.GoogleKeys(fetch=lambda: calls.append(1) or ({"keys": [_jwk(KEY)]}, 3600))
    assert keys.get("k1") is not None
    for i in range(50):
        assert keys.get(f"made-up-{i}") is None
    assert len(calls) == 1                         # one real fetch; the 50 unknown ids did not trigger more


# ---- Inactivity time-out (CAMPUS_VERIFY_SESSION_IDLE_MINUTES, default off) ------------------------------------------
class _Clock:
    def __init__(self):
        import datetime as dt
        self.t = dt.datetime.now(dt.timezone.utc)

    def __call__(self):
        return self.t

    def wait(self, minutes):
        import datetime as dt
        self.t += dt.timedelta(minutes=minutes)


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(auth, "_now", c)
    return c


def test_idle_timeout_is_off_by_default(signin, clock, monkeypatch):
    monkeypatch.delenv(auth.IDLE_ENV, raising=False)
    assert auth.idle_minutes() == 0
    with _client(signin) as c:
        c.post("/auth/google", json={"credential": token()})
        clock.wait(60 * 24 * 29)                                    # 29 days unused: still signed in
        assert c.get("/auth/me").json()["user"]["email"] == "juan.delacruz@dyci.edu.ph"


@pytest.mark.parametrize("value, minutes", [("60", 60), ("0", 0), ("-5", 0), ("soon", 0), ("", 0)])
def test_idle_setting(monkeypatch, value, minutes):
    monkeypatch.setenv(auth.IDLE_ENV, value)
    assert auth.idle_minutes() == minutes


def test_an_unused_session_ends_and_is_deleted(signin, clock, monkeypatch, tmp_path):
    monkeypatch.setenv(auth.IDLE_ENV, "60")
    with _client(signin) as c:
        c.post("/auth/google", json={"credential": token()})
        clock.wait(59)
        assert c.get("/auth/me").status_code == 200                 # used within the hour
        clock.wait(61)                                              # then unused for 61 minutes
        assert c.get("/auth/me").status_code == 401
        clock.wait(-61)                                             # even with the clock back, the session is gone
        assert c.get("/auth/me").status_code == 401
    left = db.Database(str(tmp_path / "app.db")).conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert left == 0


def test_each_use_resets_the_idle_timer(signin, clock, monkeypatch):
    monkeypatch.setenv(auth.IDLE_ENV, "60")
    with _client(signin) as c:
        c.post("/auth/google", json={"credential": token()})
        for _ in range(5):                                          # used every 50 minutes for over 4 hours
            clock.wait(50)
            assert c.get("/auth/me").status_code == 200


def test_the_overall_limit_still_applies(signin, clock, monkeypatch):
    monkeypatch.setenv(auth.IDLE_ENV, "60")
    monkeypatch.setenv(auth.SESSION_DAYS_ENV, "1")
    with _client(signin) as c:
        c.post("/auth/google", json={"credential": token()})
        for _ in range(29):                                         # in use every 50 minutes, past 24 hours
            clock.wait(50)
        assert c.get("/auth/me").status_code == 401


def test_idle_checks_need_sign_in_again_when_required(signin, clock, monkeypatch):
    monkeypatch.setenv(auth.IDLE_ENV, "30")
    monkeypatch.setenv(auth.REQUIRED_ENV, "1")
    with _client(signin) as c:
        c.post("/auth/google", json={"credential": token()})
        clock.wait(31)
        r = c.post("/verify/text", json={"claim": "no classes on november 12"})
        assert r.status_code == 401 and "sign in" in r.json()["detail"].lower()
        c.post("/auth/google", json={"credential": token()})        # signing in again works at once
        assert c.get("/auth/me").status_code == 200


def test_a_session_from_before_the_upgrade_counts_from_its_sign_in(signin, clock, monkeypatch, tmp_path):
    monkeypatch.setenv(auth.IDLE_ENV, "60")
    with _client(signin) as c:
        c.post("/auth/google", json={"credential": token()})
        database = db.Database(str(tmp_path / "app.db"))
        database.conn.execute("UPDATE sessions SET last_seen_at = NULL")   # as a v7 row would be
        database.close()
        clock.wait(30)
        assert c.get("/auth/me").status_code == 200
        clock.wait(61)
        assert c.get("/auth/me").status_code == 401
