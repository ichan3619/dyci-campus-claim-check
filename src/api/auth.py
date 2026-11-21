"""
Sign in with Google, DYCI accounts only (CAMPUS_VERIFY_AUTH=google, default off; owner and panel request 2026-10-28).

The page shows Google's own "Sign in with Google" button (Google Identity Services). Google returns a signed ID token
to the page, the page posts it to POST /auth/google, and this module checks it before anyone is signed in:
    - the RS256 signature, against Google's published keys (fetched and cached; `cryptography`, already pinned)
    - iss is Google, aud is our client ID (CAMPUS_VERIFY_GOOGLE_CLIENT_ID), the token has not expired
    - email_verified is true, the hosted domain (hd) is CAMPUS_VERIFY_AUTH_DOMAIN (default dyci.edu.ph) and the email
      ends with @<that domain>: any username before the @ is accepted
Then a random session token is set as an HttpOnly cookie (SameSite=Lax; Secure on https) and only its SHA-256 is
stored. Passwords never reach this app. A disabled account cannot sign in and loses its sessions.

Settings (all read per request):
    CAMPUS_VERIFY_AUTH              google | unset (off: every /auth, /account, /admin route answers 404)
    CAMPUS_VERIFY_GOOGLE_CLIENT_ID  the OAuth client ID from Google Cloud (public; no client secret is needed)
    CAMPUS_VERIFY_AUTH_DOMAIN       dyci.edu.ph (default)
    CAMPUS_VERIFY_AUTH_REQUIRED     1: /verify/* need a signed-in DYCI account; default 0 (sign-in optional)
    CAMPUS_VERIFY_ADMIN_EMAILS      space- or comma-separated emails that get the admin flag at sign-in
    CAMPUS_VERIFY_SESSION_DAYS      30 (default): the overall limit, counted from sign-in
    CAMPUS_VERIFY_SESSION_IDLE_MINUTES  0 (default, off): a session unused for this many minutes ends (it is deleted on
                                    the server) and the student signs in again; each use resets the timer, the overall
                                    limit stays (NIST SP 800-63B-4 session timeouts; OWASP idle timeout)
Needs CAMPUS_VERIFY_DB_FILE (src/api/db.py, schema v2).
"""
import base64
import datetime as _dt
import hashlib
import json
import os
import re
import secrets
import threading
import time
from typing import Optional

ENV = "CAMPUS_VERIFY_AUTH"
CLIENT_ID_ENV = "CAMPUS_VERIFY_GOOGLE_CLIENT_ID"
DOMAIN_ENV = "CAMPUS_VERIFY_AUTH_DOMAIN"
REQUIRED_ENV = "CAMPUS_VERIFY_AUTH_REQUIRED"
ADMINS_ENV = "CAMPUS_VERIFY_ADMIN_EMAILS"
SESSION_DAYS_ENV = "CAMPUS_VERIFY_SESSION_DAYS"
IDLE_ENV = "CAMPUS_VERIFY_SESSION_IDLE_MINUTES"
SEEN_EVERY_SECONDS = 60                  # last_seen_at is written at most once a minute per session
COOKIE = "campus_session"
GOOGLE_CERTS = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("accounts.google.com", "https://accounts.google.com")
CLOCK_SKEW_SECONDS = 60
REAUTH_SECONDS = 300                     # deleting an account needs a Google confirmation made in the last 5 minutes


class AuthError(Exception):
    """A refused sign-in. `reason` is a short code; the message is safe to show."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def enabled() -> bool:
    return os.environ.get(ENV, "").strip().lower() == "google"


def required() -> bool:
    return enabled() and os.environ.get(REQUIRED_ENV, "0").strip() == "1"


def client_id() -> str:
    return os.environ.get(CLIENT_ID_ENV, "").strip()


def domain() -> str:
    return os.environ.get(DOMAIN_ENV, "dyci.edu.ph").strip().lower() or "dyci.edu.ph"


def admin_emails() -> set:
    return {e.lower() for e in re.split(r"[\s,]+", os.environ.get(ADMINS_ENV, "")) if e}


def session_days() -> int:
    try:
        return max(1, int(os.environ.get(SESSION_DAYS_ENV, "30")))
    except ValueError:
        return 30


def idle_minutes() -> int:
    """CAMPUS_VERIFY_SESSION_IDLE_MINUTES: 0 (default) is off; anything not a positive whole number is off too."""
    try:
        return max(0, int(os.environ.get(IDLE_ENV, "0")))
    except ValueError:
        return 0


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso(t: _dt.datetime) -> str:
    return t.replace(microsecond=0).isoformat()


def _b64(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


# ---- Google ID token ---------------------------------------------------------------------------------------------
class GoogleKeys:
    """Google's public signing keys, cached for the time Google's Cache-Control allows (at most a day)."""

    MIN_REFETCH_SECONDS = 60             # an unknown key id refetches at most this often (no fetch storm from fake tokens)

    def __init__(self, fetch=None):
        self._fetch = fetch or self._download
        self._keys, self._until, self._fetched_at = {}, 0.0, 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _download():
        import requests
        r = requests.get(GOOGLE_CERTS, timeout=10)
        r.raise_for_status()
        m = re.search(r"max-age=(\d+)", r.headers.get("Cache-Control", ""))
        return r.json(), min(int(m.group(1)), 86400) if m else 3600

    def get(self, kid: str):
        with self._lock:
            now = time.time()
            expired = now >= self._until
            unknown = kid not in self._keys and now - self._fetched_at >= self.MIN_REFETCH_SECONDS
            if expired or unknown:
                data, ttl = self._fetch()
                self._keys = {k["kid"]: k for k in data.get("keys", []) if k.get("kty") == "RSA"}
                self._until, self._fetched_at = now + ttl, now
            return self._keys.get(kid)


_keys = GoogleKeys()


def verify_google_token(token: str, keys: Optional[GoogleKeys] = None, now: Optional[float] = None) -> dict:
    """The token's claims if it is a valid Google ID token for our client and a verified DYCI email; AuthError if not."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    if not client_id():
        raise AuthError("not_configured", "Sign-in is not set up on this server.")
    try:
        head_b64, body_b64, sig_b64 = (token or "").split(".")
        header, claims = json.loads(_b64(head_b64)), json.loads(_b64(body_b64))
        signature = _b64(sig_b64)
    except (ValueError, TypeError):
        raise AuthError("malformed", "The sign-in could not be read. Please try again.")
    if header.get("alg") != "RS256" or not header.get("kid"):
        raise AuthError("malformed", "The sign-in could not be read. Please try again.")
    try:
        jwk = (keys or _keys).get(header["kid"])
    except Exception:
        raise AuthError("keys_unavailable", "Google could not be reached to confirm the sign-in. Please try again.")
    if jwk is None:
        raise AuthError("unknown_key", "The sign-in could not be confirmed. Please try again.")
    public = rsa.RSAPublicNumbers(int.from_bytes(_b64(jwk["e"]), "big"), int.from_bytes(_b64(jwk["n"]), "big")).public_key()
    try:
        public.verify(signature, f"{head_b64}.{body_b64}".encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature:
        raise AuthError("bad_signature", "The sign-in could not be confirmed. Please try again.")
    t = time.time() if now is None else now
    if claims.get("iss") not in GOOGLE_ISSUERS or claims.get("aud") != client_id():
        raise AuthError("wrong_audience", "This sign-in was not made for this site.")
    if not isinstance(claims.get("exp"), (int, float)) or claims["exp"] + CLOCK_SKEW_SECONDS < t:
        raise AuthError("expired", "The sign-in expired. Please try again.")
    email = str(claims.get("email", "")).lower()
    if claims.get("email_verified") is not True or claims.get("hd", "").lower() != domain() \
            or not email.endswith("@" + domain()):
        raise AuthError("wrong_domain", f"Please sign in with your DYCI account (@{domain()}).")
    return claims


# ---- accounts and sessions ---------------------------------------------------------------------------------------
def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _user_row(row) -> Optional[dict]:
    if row is None:
        return None
    keys = ("id", "email", "name", "created_at", "last_login_at", "is_admin", "active", "history_enabled", "tour_seen")
    u = dict(zip(keys, row))
    for k in ("is_admin", "active", "history_enabled", "tour_seen"):
        u[k] = bool(u[k])
    return u


_USER_COLUMNS = "id, email, name, created_at, last_login_at, is_admin, active, history_enabled, tour_seen"


def sign_in(database, claims: dict) -> tuple:
    """(user, raw session token) for verified claims. AuthError when the account is disabled."""
    email, name = claims["email"].lower(), str(claims.get("name") or "")[:120]
    now = _iso(_now())
    admin = 1 if email in admin_emails() else 0
    with database.lock:
        c = database.conn
        c.execute("BEGIN")
        try:
            row = c.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE email = ?", (email,)).fetchone()
            if row is None:
                c.execute("INSERT INTO users (email, name, created_at, last_login_at, is_admin) VALUES (?, ?, ?, ?, ?)",
                          (email, name, now, now, admin))
            elif not row[6]:
                c.execute("ROLLBACK")
                raise AuthError("disabled", "This account has been disabled. Please contact the system administrator.")
            else:
                c.execute("UPDATE users SET name = ?, last_login_at = ?, is_admin = ? WHERE email = ?",
                          (name or row[2], now, admin, email))
            user = _user_row(c.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE email = ?", (email,)).fetchone())
            token = secrets.token_urlsafe(32)
            c.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            c.execute("INSERT INTO sessions (token_hash, user_id, created_at, expires_at, last_seen_at) "
                      "VALUES (?, ?, ?, ?, ?)",
                      (_hash(token), user["id"], now, _iso(_now() + _dt.timedelta(days=session_days())), now))
            c.execute("COMMIT")
        except AuthError:
            raise
        except Exception:
            c.execute("ROLLBACK")
            raise
    return user, token


def user_for_token(database, token: Optional[str]) -> Optional[dict]:
    if not token:
        return None
    idle = idle_minutes()
    now = _now()
    with database.lock:
        row = database.conn.execute(
            f"SELECT {', '.join('u.' + c for c in _USER_COLUMNS.split(', '))}, "
            "COALESCE(s.last_seen_at, s.created_at) FROM sessions s JOIN users u "
            "ON u.id = s.user_id WHERE s.token_hash = ? AND s.expires_at >= ? AND u.active = 1",
            (_hash(token), _iso(now))).fetchone()
        if row is None or not idle:
            return _user_row(row[:-1] if row else None)
        seen = _dt.datetime.fromisoformat(row[-1])
        if now - seen > _dt.timedelta(minutes=idle):        # unused too long: the session ends on the server
            database.conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash(token),))
            return None
        if now - seen >= _dt.timedelta(seconds=SEEN_EVERY_SECONDS):
            database.conn.execute("UPDATE sessions SET last_seen_at = ? WHERE token_hash = ?",
                                  (_iso(now), _hash(token)))
    return _user_row(row[:-1])


def sign_out(database, token: Optional[str]) -> None:
    if token:
        with database.lock:
            database.conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash(token),))


def set_history(database, user_id: int, on: bool) -> dict:
    with database.lock:
        database.conn.execute("UPDATE users SET history_enabled = ? WHERE id = ?", (1 if on else 0, user_id))
        return _user_row(database.conn.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone())


def set_tour_seen(database, user_id: int) -> dict:
    """The student finished or skipped the guided tour: it no longer starts by itself for this account."""
    with database.lock:
        database.conn.execute("UPDATE users SET tour_seen = 1 WHERE id = ?", (user_id,))
        return _user_row(database.conn.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone())


def check_reauth(user: dict, claims: dict, now: Optional[float] = None) -> None:
    """A fresh Google confirmation by the same account (deleting needs it). AuthError if not."""
    if str(claims.get("email", "")).lower() != user["email"]:
        raise AuthError("reauth_other_account", "Please confirm with the same DYCI account you are signed in with.")
    t = time.time() if now is None else now
    if not isinstance(claims.get("iat"), (int, float)) or claims["iat"] < t - REAUTH_SECONDS:
        raise AuthError("reauth_stale", "That confirmation is too old. Please confirm with Google again.")


def delete_user(database, user_id: int) -> None:
    """The student's own request to erase their account: the user row, its sessions and its saved checks go.
    The admin log keeps only the numeric id of an account an admin once enabled or disabled."""
    with database.lock:
        database.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))  # sessions, checks follow (ON DELETE CASCADE)


def list_users(database) -> list:
    with database.lock:
        rows = database.conn.execute(f"SELECT {_USER_COLUMNS} FROM users ORDER BY last_login_at DESC").fetchall()
    return [_user_row(r) for r in rows]


def set_active(database, admin: dict, user_id: int, active: bool) -> dict:
    """Enable or disable an account (admins only; an admin cannot disable themself). Disabling ends its sessions."""
    if user_id == admin["id"] and not active:
        raise AuthError("self", "You cannot disable your own account.")
    with database.lock:
        c = database.conn
        c.execute("BEGIN")
        try:
            row = c.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                c.execute("ROLLBACK")
                raise AuthError("not_found", "No such account.")
            c.execute("UPDATE users SET active = ? WHERE id = ?", (1 if active else 0, user_id))
            if not active:
                c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            c.execute("INSERT INTO admin_actions (at, admin_user_id, action, target_user_id) VALUES (?, ?, ?, ?)",
                      (_iso(_now()), admin["id"], "enable" if active else "disable", user_id))
            user = _user_row(c.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone())
            c.execute("COMMIT")
        except AuthError:
            raise
        except Exception:
            c.execute("ROLLBACK")
            raise
    return user


def public_user(user: dict) -> dict:
    """What the page may see about the signed-in account."""
    return {k: user[k] for k in ("email", "name", "is_admin", "history_enabled", "tour_seen", "created_at")}
