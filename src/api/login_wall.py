"""
Login-wall check for /verify/url (defence scenario 10; results/round2/scenario_probes_report.md, probe 10a). Setting
CAMPUS_VERIFY_LOGIN_WALL, on by default since v1.9 ("0" switches it off).

A Facebook link fetched without a login often returns Facebook's login page instead of the post. Read as evidence, it
gives "not confirmed, no relevant evidence", which looks like an answer about the claim. With the setting on, such a page
is reported like any other page that could not be read (HTTP 502), so the student knows to copy the post's text instead.

    login wall   at least 2 different login phrases AND fewer than RESIDUAL_WORDS words left once they are removed
    unavailable  Facebook's "This content isn't available" page, with fewer than RESIDUAL_WORDS other words

A real post that Facebook wraps in a "Log in / Create new account" banner keeps its own text, so it is read as before.
"""
import os
import re

ENV = "CAMPUS_VERIFY_LOGIN_WALL"
RESIDUAL_WORDS = 60
LOGIN_PHRASES = [re.compile(p, re.I) for p in (
    r"\blog\s?in(?:to)?\s+(?:to\s+)?facebook\b", r"\byou\s+must\s+log\s?in\b", r"\blog\s?in\s+or\s+sign\s+up\b",
    r"\bcreate\s+(?:a\s+)?new\s+account\b", r"\bforgot\s+(?:your\s+)?password\b",
    r"\bemail\s+(?:address\s+)?or\s+(?:phone|mobile)(?:\s+number)?\b", r"\bsee\s+more\s+(?:of\s+.{0,60}?\s+)?on\s+facebook\b",
    r"\blog\s?in\s+to\s+continue\b", r"\bpassword\b")]
UNAVAILABLE = re.compile(r"\bthis\s+content\s+isn['’]?t\s+available\b|\bthis\s+page\s+isn['’]?t\s+available\b", re.I)
MESSAGE = ("Could not read this page: Facebook showed a login page (or said the post is not available) instead of the post. "
           "Copy the post's text and check it as a message instead.")


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"         # on by default since v1.9 (owner, 2026-10-27)


def _residual_words(text: str, patterns) -> int:
    for p in patterns:
        text = p.sub(" ", text)
    return len(re.findall(r"[^\W\d_]{2,}", text))


def detect(text: str) -> bool:
    """True when the fetched text is a login wall or an unavailable-content page rather than a post."""
    text = text or ""
    found = [p for p in LOGIN_PHRASES if p.search(text)]
    if UNAVAILABLE.search(text):
        return _residual_words(text, found + [UNAVAILABLE]) < RESIDUAL_WORDS
    return len(found) >= 2 and _residual_words(text, found) < RESIDUAL_WORDS
