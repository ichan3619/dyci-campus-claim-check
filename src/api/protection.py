"""
API protection for the prototype: per-client rate limits, a cap on concurrent model work, a request-size limit and an
optional API key. Stdlib only; no FastAPI imports, so it is testable on its own (src/api/main.py wires it in).

SETTINGS (environment variables; 0 disables a limit)
    CAMPUS_VERIFY_RATE_LIMIT_PER_MINUTE      30   requests per client per minute to /verify/text
    CAMPUS_VERIFY_URL_RATE_LIMIT_PER_MINUTE  10   to /verify/url (each request also fetches an external page)
    CAMPUS_VERIFY_MAX_CONCURRENT             2    verifications running at once (each holds the models' working memory)
    CAMPUS_VERIFY_QUEUE_SECONDS              15   how long a request waits for a free slot before HTTP 503
    CAMPUS_VERIFY_MAX_BODY_BYTES             16384  larger request bodies get HTTP 413
    CAMPUS_VERIFY_MAX_IMAGE_BYTES            5242880  body limit for POST /verify/image, ONLY while CAMPUS_VERIFY_IMAGE
                                                      is on (otherwise that path keeps the 16384-byte limit, as in v1.3)
    CAMPUS_VERIFY_IMAGE_RATE_LIMIT_PER_MINUTE 6   per client to /verify/image while CAMPUS_VERIFY_IMAGE is on
    CAMPUS_VERIFY_API_KEY                    (unset)  when set, /verify/* require header X-API-Key with this value
    CAMPUS_VERIFY_TRUST_PROXY                0    number of proxies in front of the app that append to X-Forwarded-For
                                                   (a host's front door is usually 1). The client is the entry the
                                                   outermost of them added: the N-th from the RIGHT. Entries to its left
                                                   were written by the client and are ignored (they can be faked). 0 =
                                                   the connecting address; never set it without such a proxy, since
                                                   any client could then pick its own identity.

WHAT THIS IS NOT
    Not authentication of students, and not a defence against a distributed attack: limits are per process and in
    memory. It keeps one client, a script, or a burst of requests from exhausting the machine (the server process used
    about 1.25 GB with the three models loaded and 1.45 GB after a few checks, measured 2026-10-25; Image Check adds
    OCR) or abusing /verify/url as a page fetcher. An API key set here is visible to anyone who can read a page that
    sends it, so it is for non-browser clients or a gated demo, not a secret in the frontend.
"""
import hmac
import os
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple


def _int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


@dataclass(frozen=True)
class ProtectionSettings:
    rate_per_minute: int = 30
    url_rate_per_minute: int = 10
    max_concurrent: int = 2
    queue_seconds: float = 15.0
    max_body_bytes: int = 16384
    max_image_bytes: int = 5 * 1024 * 1024
    image_rate_per_minute: int = 6
    api_key: str = ""
    trusted_proxies: int = 0

    @classmethod
    def from_env(cls) -> "ProtectionSettings":
        return cls(
            rate_per_minute=_int("CAMPUS_VERIFY_RATE_LIMIT_PER_MINUTE", 30),
            url_rate_per_minute=_int("CAMPUS_VERIFY_URL_RATE_LIMIT_PER_MINUTE", 10),
            max_concurrent=_int("CAMPUS_VERIFY_MAX_CONCURRENT", 2),
            queue_seconds=float(os.environ.get("CAMPUS_VERIFY_QUEUE_SECONDS", "15") or 15),
            max_body_bytes=_int("CAMPUS_VERIFY_MAX_BODY_BYTES", 16384),
            max_image_bytes=_int("CAMPUS_VERIFY_MAX_IMAGE_BYTES", 5 * 1024 * 1024),
            image_rate_per_minute=_int("CAMPUS_VERIFY_IMAGE_RATE_LIMIT_PER_MINUTE", 6),
            api_key=os.environ.get("CAMPUS_VERIFY_API_KEY", "").strip(),
            trusted_proxies=max(0, _int("CAMPUS_VERIFY_TRUST_PROXY", 0)),
        )


class RateLimiter:
    """Sliding one-minute window per (client, bucket). Memory is bounded: at most `max_clients` keys are tracked."""

    def __init__(self, per_minute: int, window_seconds: float = 60.0, max_clients: int = 10000, clock=time.monotonic):
        self.per_minute, self.window, self.max_clients, self._clock = per_minute, window_seconds, max_clients, clock
        self._hits: "OrderedDict[str, Deque[float]]" = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, int]:
        """(allowed, seconds until the next request would be allowed). Counts the request when allowed."""
        if self.per_minute <= 0:
            return True, 0
        now = self._clock()
        with self._lock:
            hits = self._hits.pop(key, None) or deque()
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            allowed = len(hits) < self.per_minute
            if allowed:
                hits.append(now)
            self._hits[key] = hits                       # most recently seen at the end
            while len(self._hits) > self.max_clients:
                self._hits.popitem(last=False)
            retry = 0 if allowed else max(1, int(self.window - (now - hits[0])) + 1)
            return allowed, retry


class Busy(Exception):
    """No verification slot became free within the queue time."""


class ConcurrencyGate:
    """At most `limit` verifications run at once; others wait up to `wait_seconds`, then Busy."""

    def __init__(self, limit: int, wait_seconds: float):
        self.limit, self.wait = limit, wait_seconds
        self._sem = threading.BoundedSemaphore(limit) if limit > 0 else None

    @contextmanager
    def slot(self):
        if self._sem is None:
            yield
            return
        if not self._sem.acquire(timeout=self.wait):
            raise Busy()
        try:
            yield
        finally:
            self._sem.release()


def client_key(headers: Dict[str, str], peer: Optional[str], trusted_proxies: int) -> str:
    """Who is asking. With N trusted proxies, each appends the address it received the request from to X-Forwarded-For,
    so the N-th entry from the right is the client as the outermost proxy saw it; anything further left came from the
    client and is ignored (pre-deployment review M3: the first entry used before could be faked to dodge the limits).
    Fewer entries than proxies means the request did not come through them: the connecting address is used."""
    if trusted_proxies > 0:
        entries = [e.strip() for e in headers.get("x-forwarded-for", "").split(",") if e.strip()]
        if len(entries) >= trusted_proxies:
            return entries[-trusted_proxies]
    return peer or "unknown"


def api_key_ok(expected: str, given: Optional[str]) -> bool:
    """True when no key is configured, or the given key matches (constant-time comparison)."""
    return not expected or hmac.compare_digest(expected.encode(), (given or "").encode())
