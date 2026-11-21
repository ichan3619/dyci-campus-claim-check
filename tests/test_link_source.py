"""Rule U, links that are not DYCI's (results/round2/link_source_plan.md)."""
from types import SimpleNamespace

import pytest

from src.api import link_source as ls


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(ls.ENV, raising=False)
    assert ls.enabled() is False
    monkeypatch.setenv(ls.ENV, "1")
    assert ls.enabled() is True


@pytest.mark.parametrize("url", [
    "https://dyci.edu.ph/news/advisory", "https://www.dyci.edu.ph/", "http://college.dyci.edu.ph/a?b=c",
    "https://DYCI.EDU.PH./x",
])
def test_official_domain(url):
    assert ls.is_official(url)


@pytest.mark.parametrize("url", [
    "https://dyci-edu-ph.com/news", "https://dyci.edu.ph.update-news.com/x", "https://dycii.edu.ph/",
    "https://dyci-official.blogspot.com/2030/07/x.html", "https://notdyci.edu.ph/", "https://bulacan-daily.com/dyci",
    "https://evil.com/?u=https://dyci.edu.ph/", "not a url",
])
def test_not_official(url):
    assert not ls.is_official(url)


def test_facebook_pages(monkeypatch):
    monkeypatch.setenv(ls.PAGES_ENV, "DYCIOfficial, 1245921188615218")
    assert ls.is_official("https://www.facebook.com/DYCIOfficial/posts/123")
    assert ls.is_official("https://m.facebook.com/dycIofficial")
    assert ls.is_official("https://www.facebook.com/profile.php?id=1245921188615218")
    assert ls.is_official("https://www.facebook.com/1245921188615218/posts/1")
    assert not ls.is_official("https://www.facebook.com/DYCIUpdates/posts/123")
    assert not ls.is_official("https://www.facebook.com/share/p/abc123/")
    assert not ls.is_official("https://www.facebook.com/groups/DYCIOfficial/")
    assert not ls.is_official("https://facebook.com.evil.com/DYCIOfficial")


def test_facebook_pages_empty_by_default(monkeypatch):
    monkeypatch.delenv(ls.PAGES_ENV, raising=False)
    assert not ls.is_official("https://www.facebook.com/DYCIOfficial/posts/123")


def test_info_read_and_unread():
    page = SimpleNamespace(verdict="SUPPORTED", selected_passage=SimpleNamespace(text="Classes are suspended."),
                           used_evidence=None)
    i = ls.info("https://bulacan-daily.com/x", page)
    assert i == {"version": ls.VERSION, "host": "bulacan-daily.com", "official": False, "read": True,
                 "page_verdict": "SUPPORTED", "page_says": "supports", "page_sentence": "Classes are suspended."}
    j = ls.info("https://bulacan-daily.com/x", None, "Failed to fetch")
    assert j["read"] is False and j["reason"] == "Failed to fetch"
    assert ls.page_says(None) == "does not settle" and ls.page_says("CONTRADICTED") == "contradicts"


def test_route_uses_dyci_posts_for_other_links(monkeypatch):
    import src.api.main as api
    from src.api.schemas import VerifyUrlRequest
    monkeypatch.setenv(ls.ENV, "1")
    monkeypatch.setattr(api, "check_redirect_chain_is_safe", lambda u: u)
    typed = SimpleNamespace(model_copy=lambda update: {"from": "dyci", **update})
    monkeypatch.setattr(api, "verify_text", lambda req: typed)
    page = SimpleNamespace(verdict="SUPPORTED", selected_passage=None, used_evidence=SimpleNamespace(text="Fake news."))
    monkeypatch.setattr(api, "_check_fetched_page", lambda url, claim: page)
    out = api.verify_url(VerifyUrlRequest(url="https://campusbuzz.ph/dyci", claim="no classes on july 10"))
    assert out["from"] == "dyci" and out["link_source"]["page_says"] == "supports"

    def unreadable(url, claim):
        raise api.HTTPException(status_code=502, detail="Failed to fetch the submitted URL: boom")
    monkeypatch.setattr(api, "_check_fetched_page", unreadable)
    out = api.verify_url(VerifyUrlRequest(url="https://campusbuzz.ph/dyci", claim="no classes on july 10"))
    assert out["link_source"]["read"] is False and "boom" in out["link_source"]["reason"]


def test_route_unchanged_for_official_links_and_when_off(monkeypatch):
    import src.api.main as api
    from src.api.schemas import VerifyUrlRequest
    monkeypatch.setattr(api, "check_redirect_chain_is_safe", lambda u: u)
    monkeypatch.setattr(api, "verify_text", lambda req: pytest.fail("typed check must not run"))
    monkeypatch.setattr(api, "_check_fetched_page", lambda url, claim: "page-only")
    monkeypatch.setenv(ls.ENV, "1")
    assert api.verify_url(VerifyUrlRequest(url="https://dyci.edu.ph/a", claim="no classes on july 10")) == "page-only"
    monkeypatch.setenv(ls.ENV, "0")
    assert api.verify_url(VerifyUrlRequest(url="https://campusbuzz.ph/a", claim="no classes on july 10")) == "page-only"
