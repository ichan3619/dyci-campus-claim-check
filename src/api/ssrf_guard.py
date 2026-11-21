"""
Milestone 10.2 -- SSRF protection for POST /verify/url.

Responsibility of this module, and ONLY this module: decide whether a
user-submitted URL (and everywhere its redirect chain leads) is safe to
fetch. This is deliberately NOT a DYCI-domain allowlist -- /verify/url is
intentionally a user-submitted, ephemeral evidence endpoint (see the
M10.1 methodology section), so this module blocks destination
*categories* (loopback, private, link-local, etc.), not specific domains.

This module does not fetch content, extract HTML, or normalize evidence
-- that remains entirely WebsiteSourceAdapter's job (src/ingestion/
adapters.py), untouched by this milestone. This module only decides
GO/NO-GO before src/api/main.py ever calls that adapter.
"""

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests

ALLOWED_SCHEMES = {"http", "https"}
MAX_REDIRECT_HOPS = 5
PREFLIGHT_TIMEOUT_SECONDS = 10


class UnsafeUrlError(Exception):
    """
    Raised when a submitted URL, or a redirect it leads to, targets a
    disallowed scheme or destination. Caught in src/api/main.py and
    mapped to HTTP 400 -- a policy refusal, not a fetch failure (that
    remains WebsiteFetchError -> 502, unchanged from M10.1).
    """
    pass


def _is_unsafe_ip(ip_str: str) -> bool:
    """
    True if `ip_str` falls into any category that should never be
    reachable from a user-submitted evidence URL: loopback (127.0.0.0/8,
    ::1), private ranges (10/8, 172.16/12, 192.168/16, and their IPv6
    equivalents), link-local (169.254.0.0/16, fe80::/10), unspecified
    (0.0.0.0, ::), other IANA-reserved ranges, or multicast. Python's
    ipaddress module already encodes the exact CIDR boundaries for all
    of these -- reimplementing them by hand here would just be a second
    place for that arithmetic to be subtly wrong.
    """
    ip = ipaddress.ip_address(ip_str)
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_reserved
        or ip.is_multicast
    )


def _resolve_hostname_ips(hostname: str) -> list:
    """
    Resolves `hostname` to every IP address it currently maps to (a
    hostname can legitimately resolve to more than one; ALL of them are
    checked, not just the first, since an attacker-controlled DNS record
    could otherwise return a safe IP first and an unsafe one second).
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise UnsafeUrlError(f"could not resolve host {hostname!r}: {e}")
    return sorted({info[4][0] for info in infos})


def check_url_is_safe(url: str) -> None:
    """
    Validates a SINGLE URL (no redirect-following). Raises UnsafeUrlError
    if:
      - the scheme is anything other than http/https (blocks file://,
        ftp://, gopher://, data:, etc.), or
      - the hostname is literally "localhost", or
      - the hostname resolves to a loopback/private/link-local/
        unspecified/reserved/multicast address.

    Does NOT catch a redirect to an unsafe destination on its own --
    see check_redirect_chain_is_safe() for that.
    """
    parsed = urlparse(url)

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"scheme {parsed.scheme!r} is not allowed (only http/https)")

    if not parsed.hostname:
        raise UnsafeUrlError("URL has no hostname")

    hostname = parsed.hostname.lower()
    if hostname == "localhost":
        raise UnsafeUrlError("'localhost' is not an allowed destination")

    for ip_str in _resolve_hostname_ips(hostname):
        if _is_unsafe_ip(ip_str):
            raise UnsafeUrlError(f"{hostname!r} resolves to disallowed address {ip_str}")


def check_redirect_chain_is_safe(url: str, *, max_hops: int = MAX_REDIRECT_HOPS) -> str:
    """
    Validates `url` AND every hop of its redirect chain, one at a time,
    without ever automatically following a redirect blind. This is the
    specific gap check_url_is_safe() alone leaves open: a URL that is
    perfectly safe at the top level could 301/302/303/307/308-redirect
    to an internal address, and a naive check of only the user-typed URL
    would never see that.

    Uses requests.get(..., allow_redirects=False, stream=True) rather
    than HEAD, since some real servers don't support HEAD the same way
    they support GET -- stream=True means only headers/status are read
    here, not the response body, so this stays lightweight. If a
    non-safety error occurs mid-chain (timeout, connection error, a
    method/response quirk unrelated to destination safety), this stops
    validating further hops and returns the last-validated URL rather
    than raising -- the REAL fetch that follows (WebsiteSourceAdapter's
    own GET, unchanged) will surface that failure properly as a 502, not
    have it misreported as an SSRF block.

    Returns the final destination URL once no further redirect is found.

    The returned URL is fetched by WebsiteSourceAdapter with
    follow_redirects=False (src/api/main.py), so a server that answers this
    walk with 200 and the real fetch with a redirect (e.g. to an internal
    address) is refused, not followed (pre-deployment review H1, 2026-10-25).

    KNOWN LIMITATION (documented, not silently accepted): there is a
    small time-of-check-to-time-of-use gap between this pre-flight walk
    and WebsiteSourceAdapter's own later, independent fetch of the same
    URL, which re-resolves DNS. A precisely-timed DNS-rebinding attack
    could in principle change a hostname's resolution between these two
    steps. Fully closing that
    gap would require validating the IP at the moment of each live
    socket connection (e.g. a custom requests transport), which is a
    larger change than this milestone's explicit "harden the API layer,
    do not redesign the backend" scope calls for.
    """
    current_url = url

    for _ in range(max_hops + 1):
        check_url_is_safe(current_url)

        try:
            response = requests.get(
                current_url,
                allow_redirects=False,
                stream=True,
                timeout=PREFLIGHT_TIMEOUT_SECONDS,
            )
            response.close()
        except requests.exceptions.RequestException:
            return current_url

        if not response.is_redirect:
            return current_url

        location = response.headers.get("Location")
        if not location:
            return current_url

        current_url = urljoin(current_url, location)

    raise UnsafeUrlError(f"too many redirects (> {max_hops}) while validating {url!r}")
