"""
Image Check source rule (results/round2/image_source_rule_plan.md, approved 2026-10-26, Amendment 1): a screenshot
whose Facebook header names ANOTHER institution is not checked against DYCI's announcements.
Setting CAMPUS_VERIFY_IMAGE_SOURCE_RULE, default off; it applies with the Florence-2 reader, whose header rule
(src/image/florence_reader.py remove_header) finds the header line.

Amendment 1 (owner, 2026-10-26): DYCI has many pages and no complete list is known, so the rule does NOT use a list of
DYCI pages. It uses the existing, tested institution detector (src/api/school_scope.py detect(), detector v2, which
never reports DYCI's names, campuses or units) on the page name in the header:
    another institution named in a header line  -> "other_page": no verdict; the closest DYCI announcements are shown
    a header line, no other institution named   -> "header_read": checked as now (DYCI pages under any name, test pages,
                                                   unknown pages)
    no header line                               -> "no_header": checked as now, with the note that the source page
                                                   could not be read (Q1)
It can only withhold a verdict, never create one. Errors lean towards checking: a wrong withhold would hide a correct
answer about a genuine DYCI screenshot; a missed other page behaves as before.
"""
import os
import re
from typing import List, Optional

from src.api.school_scope import detect

ENV = "CAMPUS_VERIFY_IMAGE_SOURCE_RULE"
OTHER_PAGE_NOTE = ("This screenshot is from “{page}”, not one of DYCI's pages, so it was not checked against DYCI's "
                   "announcements. The closest DYCI announcements are shown so you can compare them yourself.")
NO_HEADER_NOTE = "The source page could not be read from this screenshot."

_TAIL = re.compile(r"(?i)\s*(?:[·•|\-]\s*)*(?:follow(?:ing)?|(?:and\s+\d+\s+others?))\b.*$")
_TIMESTAMP = re.compile(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s\d{1,2}\b.*$|"
                        r"\b(?:yesterday|today)\s+at\b.*$|\bjust\s+now\b.*$|\b\d{1,2}\s?(?:m|h|d|w|mins?|hrs?)\b.*$")


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def page_name(header_line: str) -> str:
    """The page name part of a header line: what comes before "Follow", "and 3 others" and the timestamp."""
    name = _TAIL.sub("", header_line or "")
    name = _TIMESTAMP.sub("", name)
    return name.strip(" ·•|-,.").strip()


_KEYWORDS = frozenset("high school academy college colleges university institute of the de del ng and".split())


def _distinctive(name: str) -> bool:
    """A detected institution counts only with a distinctive word of 3+ letters: a page header's "S School News 2.0"
    (the profile-picture letter read as a word) is not an institution called "S School"."""
    return any(len(w) >= 3 and w.lower() not in _KEYWORDS for w in re.findall(r"[A-Za-z]+", name))


def judge(header_lines: List[str]) -> dict:
    """The source decision for the header lines the header rule removed (in reading order)."""
    if not header_lines:
        return {"status": "no_header", "page_name": None, "institutions": [], "note": NO_HEADER_NOTE}
    names = [page_name(line) for line in header_lines]
    for name in names:
        found = [i for i in detect(name) if _distinctive(i)] if name else []
        if found:
            return {"status": "other_page", "page_name": name, "institutions": found,
                    "note": OTHER_PAGE_NOTE.format(page=name)}
    first = next((n for n in names if n), None)
    return {"status": "header_read", "page_name": first, "institutions": [], "note": None}
