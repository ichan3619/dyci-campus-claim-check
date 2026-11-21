"""
Tests for Milestone 10.2's SSRF guard (src/api/ssrf_guard.py).

IMPORTANT: DNS resolution and any live-request checks in this file mock
`socket.getaddrinfo` and `requests.get` directly -- no real network call
is ever made here, same discipline as every other network-adjacent test
in this project (test_website_adapter.py, test_facebook_adapter.py).
"""

import socket

import pytest
from unittest.mock import patch, Mock

from src.api.ssrf_guard import (
    check_url_is_safe,
    check_redirect_chain_is_safe,
    UnsafeUrlError,
)


def _addrinfo_for(*ips):
    """Build a fake socket.getaddrinfo() return value for the given IP(s)."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]


# --------------------------------------------------------------------
# check_url_is_safe -- scheme and single-URL destination checks
# --------------------------------------------------------------------

@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_allowed_public_https_url_passes(mock_getaddrinfo):
    mock_getaddrinfo.return_value = _addrinfo_for("93.184.216.34")  # a real public-style address
    check_url_is_safe("https://example.com/announcement")  # must not raise


@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_allowed_public_http_url_passes(mock_getaddrinfo):
    mock_getaddrinfo.return_value = _addrinfo_for("93.184.216.35")
    check_url_is_safe("http://example.com/announcement")  # must not raise


def test_localhost_hostname_is_rejected():
    with pytest.raises(UnsafeUrlError, match="localhost"):
        check_url_is_safe("http://localhost:8000/health")


@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_loopback_ip_is_rejected(mock_getaddrinfo):
    mock_getaddrinfo.return_value = _addrinfo_for("127.0.0.1")
    with pytest.raises(UnsafeUrlError, match="127.0.0.1"):
        check_url_is_safe("http://127.0.0.1:8000/health")


@pytest.mark.parametrize("private_ip", ["10.0.0.5", "172.16.0.5", "192.168.1.10"])
@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_private_ipv4_ranges_are_rejected(mock_getaddrinfo, private_ip):
    mock_getaddrinfo.return_value = _addrinfo_for(private_ip)
    with pytest.raises(UnsafeUrlError, match=private_ip):
        check_url_is_safe(f"http://internal.example/{private_ip}")


@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_link_local_ip_is_rejected(mock_getaddrinfo):
    mock_getaddrinfo.return_value = _addrinfo_for("169.254.169.254")  # cloud metadata endpoint address
    with pytest.raises(UnsafeUrlError):
        check_url_is_safe("http://metadata.example/")


@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_unspecified_ip_is_rejected(mock_getaddrinfo):
    mock_getaddrinfo.return_value = _addrinfo_for("0.0.0.0")
    with pytest.raises(UnsafeUrlError):
        check_url_is_safe("http://zero.example/")


@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_ipv6_loopback_is_rejected(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0))]
    with pytest.raises(UnsafeUrlError):
        check_url_is_safe("http://v6loop.example/")


@pytest.mark.parametrize("scheme_url", ["ftp://example.com/file", "file:///etc/passwd", "gopher://example.com/"])
def test_non_http_https_scheme_is_rejected(scheme_url):
    with pytest.raises(UnsafeUrlError, match="not allowed"):
        check_url_is_safe(scheme_url)


def test_unresolvable_hostname_is_rejected():
    with patch("src.api.ssrf_guard.socket.getaddrinfo", side_effect=socket.gaierror("Name or service not known")):
        with pytest.raises(UnsafeUrlError, match="could not resolve"):
            check_url_is_safe("http://this-does-not-exist.invalid/")


def test_url_with_no_hostname_is_rejected():
    with pytest.raises(UnsafeUrlError, match="no hostname"):
        check_url_is_safe("http:///just-a-path")


# --------------------------------------------------------------------
# check_redirect_chain_is_safe -- the redirect-to-internal case
# --------------------------------------------------------------------

def _mock_direct_response():
    """A non-redirect response (is_redirect False), for the terminal hop."""
    response = Mock()
    response.is_redirect = False
    response.close = Mock()
    response.headers = {}
    response.iter_content = Mock(return_value=[b""])
    return response


def _mock_redirect_response(location):
    response = Mock()
    response.is_redirect = True
    response.headers = {"Location": location}
    response.close = Mock()
    return response


@patch("src.api.ssrf_guard.requests.get")
@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_public_url_with_no_redirect_returns_itself(mock_getaddrinfo, mock_get):
    mock_getaddrinfo.return_value = _addrinfo_for("93.184.216.34")
    mock_get.return_value = _mock_direct_response()

    result = check_redirect_chain_is_safe("https://example.com/announcement")

    assert result == "https://example.com/announcement"


@patch("src.api.ssrf_guard.requests.get")
@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_redirect_to_private_destination_is_rejected(mock_getaddrinfo, mock_get):
    """
    The exact attack this milestone targets: a URL that is safe at the
    top level (public IP) 302-redirects to an internal address. The
    chain walk must catch this on the SECOND hop, not just validate the
    first URL and stop.
    """
    def getaddrinfo_side_effect(hostname, *args, **kwargs):
        if hostname == "public.example.com":
            return _addrinfo_for("93.184.216.34")
        if hostname == "internal.example.com":
            return _addrinfo_for("10.0.0.5")
        raise AssertionError(f"unexpected hostname resolved: {hostname}")

    mock_getaddrinfo.side_effect = getaddrinfo_side_effect
    mock_get.return_value = _mock_redirect_response("http://internal.example.com/secret")

    with pytest.raises(UnsafeUrlError, match="10.0.0.5"):
        check_redirect_chain_is_safe("https://public.example.com/redirector")


@patch("src.api.ssrf_guard.requests.get")
@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_redirect_to_localhost_is_rejected(mock_getaddrinfo, mock_get):
    mock_getaddrinfo.return_value = _addrinfo_for("93.184.216.34")
    mock_get.return_value = _mock_redirect_response("http://localhost/admin")

    with pytest.raises(UnsafeUrlError, match="localhost"):
        check_redirect_chain_is_safe("https://public.example.com/redirector")


@patch("src.api.ssrf_guard.requests.get")
@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_multi_hop_public_redirect_chain_resolves_to_final_url(mock_getaddrinfo, mock_get):
    mock_getaddrinfo.return_value = _addrinfo_for("93.184.216.34")
    mock_get.side_effect = [
        _mock_redirect_response("https://hop2.example.com/"),
        _mock_redirect_response("https://hop3.example.com/final"),
        _mock_direct_response(),
    ]

    result = check_redirect_chain_is_safe("https://hop1.example.com/")

    assert result == "https://hop3.example.com/final"


@patch("src.api.ssrf_guard.requests.get")
@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_too_many_redirects_raises(mock_getaddrinfo, mock_get):
    mock_getaddrinfo.return_value = _addrinfo_for("93.184.216.34")
    # Always redirect to a new public-looking hop -- never terminates.
    mock_get.side_effect = lambda *a, **k: _mock_redirect_response("https://example.com/next")

    with pytest.raises(UnsafeUrlError, match="too many redirects"):
        check_redirect_chain_is_safe("https://example.com/start", max_hops=3)


@patch("src.api.ssrf_guard.requests.get")
@patch("src.api.ssrf_guard.socket.getaddrinfo")
def test_preflight_network_error_does_not_raise_unsafe(mock_getaddrinfo, mock_get):
    """
    A non-safety network error during the pre-flight walk (e.g. the
    server doesn't like a GET-with-stream probe) must NOT be reported as
    an SSRF rejection -- it should fall through and let the real fetch
    (WebsiteSourceAdapter) report the actual failure as a 502 later.
    """
    import requests as requests_module
    mock_getaddrinfo.return_value = _addrinfo_for("93.184.216.34")
    mock_get.side_effect = requests_module.exceptions.ConnectionError("boom")

    result = check_redirect_chain_is_safe("https://example.com/flaky")

    assert result == "https://example.com/flaky"
