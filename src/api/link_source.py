"""
Rule U: links that are not DYCI's (results/round2/link_source_plan.md). CAMPUS_VERIFY_LINK_SOURCE=1 (default off).

A link is DYCI's when its host is dyci.edu.ph or a subdomain of it, or when it is a Facebook link whose page name or
page ID is listed in CAMPUS_VERIFY_OFFICIAL_PAGES (comma-separated). For any other link the verdict comes from DYCI's
own posts (exactly the /verify/text answer for the same message); what the linked page says is reported beside it.
"""
from __future__ import annotations

import os
from typing import Optional
from urllib.parse import parse_qs, urlsplit

ENV = "CAMPUS_VERIFY_LINK_SOURCE"
PAGES_ENV = "CAMPUS_VERIFY_OFFICIAL_PAGES"
VERSION = "link-source-v1"
OFFICIAL_DOMAIN = "dyci.edu.ph"
FACEBOOK_HOSTS = frozenset({"facebook.com", "www.facebook.com", "m.facebook.com", "web.facebook.com",
                            "mbasic.facebook.com", "fb.com", "www.fb.com"})


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def official_pages() -> frozenset:
    raw = os.environ.get(PAGES_ENV, "")
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


def host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def facebook_page(url: str) -> Optional[str]:
    """The page name or ID a Facebook link names, or None (a share link, a group, or not Facebook)."""
    parts = urlsplit(url)
    if (parts.hostname or "").lower().rstrip(".") not in FACEBOOK_HOSTS:
        return None
    segments = [s for s in parts.path.split("/") if s]
    if not segments:
        return None
    first = segments[0].lower()
    if first in ("profile.php", "permalink.php", "story.php"):
        query = parse_qs(parts.query)
        ids = query.get("id") or []
        return ids[0].lower() if ids else None
    if first in ("share", "groups", "events", "watch", "photo", "photo.php", "reel", "login", "sharer"):
        return None
    return first


def is_official(url: str) -> bool:
    host = host_of(url)
    if host == OFFICIAL_DOMAIN or host.endswith("." + OFFICIAL_DOMAIN):
        return True
    page = facebook_page(url)
    return page is not None and page in official_pages()


def page_says(verdict: Optional[str]) -> str:
    return {"SUPPORTED": "supports", "CONTRADICTED": "contradicts"}.get(verdict or "", "does not settle")


def info(url: str, page_result=None, error: Optional[str] = None) -> dict:
    """The response's link_source field for a link that is not DYCI's."""
    out = {"version": VERSION, "host": host_of(url), "official": False, "read": page_result is not None}
    if page_result is not None:
        passage, used = getattr(page_result, "selected_passage", None), getattr(page_result, "used_evidence", None)
        sentence = getattr(passage, "text", None) or (passage.get("text") if isinstance(passage, dict) else None)
        if not sentence and used is not None:
            sentence = (getattr(used, "text", "") or "")[:300]
        out.update({"page_verdict": page_result.verdict, "page_says": page_says(page_result.verdict),
                    "page_sentence": sentence or None})
    else:
        out["reason"] = error
    return out
