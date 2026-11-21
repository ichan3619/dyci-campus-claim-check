"""
HTML content extraction (Milestone 8.2).

Responsibility of this module, and ONLY this module: given raw HTML
text (already fetched by someone else), extract a title, main
human-readable body text, and a best-effort publication date.

This module performs NO network access at all -- it is a pure function
of an HTML string, deliberately kept separate from
src/ingestion/adapters.py's WebsiteSourceAdapter (which handles the
actual HTTP fetch) so extraction logic can be tested directly against
fixture HTML strings, with no request mocking required at all.

============================================================================
SCOPE: DELIBERATELY SIMPLE, NOT A UNIVERSAL CONTENT EXTRACTOR
============================================================================
This is a heuristic, not a solved problem. It strips the obvious
non-content elements (script, style, nav, header, footer) and returns
whatever text remains -- it does NOT attempt article-body detection,
readability scoring, or boilerplate removal beyond those specific tags.
Real-world pages vary enormously; a page that puts its actual content
inside a <nav> or wraps everything in unusual markup will not extract
cleanly. This limitation is intentional and documented (see
docs/methodology.md's Milestone 8.2 section) rather than papered over
with a heavier, more "magic" extraction library.

Publication date extraction is similarly best-effort: it checks a
couple of common, well-established patterns
(<meta property="article:published_time">, <meta name="date">, and
<time datetime="...">) and returns an empty string if none match --
NOT an error, since most pages won't have any of these.
"""

from dataclasses import dataclass

from bs4 import BeautifulSoup

# Tags that are never part of human-readable page content, regardless of
# site. Removed entirely before extracting text -- not just ignored,
# since leaving them in place could leak script/CSS source text into the
# "body" if get_text() were called on the untouched tree.
NON_CONTENT_TAGS = ["script", "style", "nav", "header", "footer"]


@dataclass
class ExtractedContent:
    """
    The result of parsing one HTML document.

    title: the <title> tag's text, or "" if absent.
    body_text: whitespace-joined visible text remaining after stripping
        NON_CONTENT_TAGS -- may be empty for a page with no real content
        (e.g. only navigation/scripts), which the caller (adapters.py)
        treats as a real failure, not silently accepted.
    published_date: a best-effort ISO-ish date string if a recognized
        pattern was found, else "".
    """
    title: str
    body_text: str
    published_date: str


def extract_content(html: str) -> ExtractedContent:
    """
    Parse `html` and extract title, body text, and (if findable) a
    publication date.

    Does not raise on malformed/minimal HTML -- BeautifulSoup is
    deliberately tolerant of broken markup (this is one of the specific
    reasons it was chosen over hand-rolling a strict parser -- see
    docs/methodology.md's Milestone 8.2 dependency justification). A
    truly empty or content-free document simply produces an empty
    body_text; it is the CALLER's responsibility (WebsiteSourceAdapter,
    via the existing normalize() function's empty-text check) to decide
    that's a failure -- this function does not duplicate that check.
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    published_date = _extract_published_date(soup)

    for tag_name in NON_CONTENT_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    body_text = soup.get_text(separator=" ", strip=True)

    return ExtractedContent(title=title, body_text=body_text, published_date=published_date)


def _extract_published_date(soup: BeautifulSoup) -> str:
    """
    Best-effort publication date lookup across a few common, widely-used
    patterns. Returns "" if none match -- this is expected and normal
    for most pages, not a warning-worthy condition.
    """
    meta_published = soup.find("meta", attrs={"property": "article:published_time"})
    if meta_published and meta_published.get("content"):
        return meta_published["content"].strip()

    meta_date = soup.find("meta", attrs={"name": "date"})
    if meta_date and meta_date.get("content"):
        return meta_date["content"].strip()

    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        return time_tag["datetime"].strip()

    return ""
