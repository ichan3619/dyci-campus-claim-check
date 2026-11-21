"""
Rule O: a student organisation's own posts can decide about its own matters (results/round2/org_pages_plan.md).
CAMPUS_VERIFY_ORG_PAGES=1 (default off); CAMPUS_VERIFY_ORG_SOURCES="key=Display name;key2=Name 2" names the archive
pages (data/corpus/archive/<key>.jsonl, or CAMPUS_VERIFY_ORG_DIR) that count.

DYCI first: only when DYCI's posts give INSUFFICIENT_EVIDENCE (not withheld for language) is the message checked again
against the organisation's posts, through the same _answer with every adopted rule. A school-wide message (the
owner's list: classes, suspensions, online or face-to-face classes, dismissal, office or campus closures, exams,
enrollment, tuition and fees) never gets the organisation's verdict; its closest post may be shown as related.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

ENV = "CAMPUS_VERIFY_ORG_PAGES"
SOURCES_ENV = "CAMPUS_VERIFY_ORG_SOURCES"
DIR_ENV = "CAMPUS_VERIFY_ORG_DIR"
VERSION = "org-pages-v1"
DEFAULT_DIR = os.path.join("data", "corpus", "archive")

# The owner's list (2026-11-18), as written in the plan before the set was seen.
SCHOOL_WIDE = re.compile(
    r"\bclass(?:es)?\b|\bsuspen(?:d|ded|ds|sion|sions)\b|\bwalang\s+pasok\b|\bonline\s+(?:class\w*|learning)\b|"
    r"\bface[\s-]to[\s-]face\b|\bf2f\b|\bdismiss\w*\b|"
    r"\b(?:office|offices|campus|school)\b[^.?!]{0,40}\b(?:closed|close|closes|closure)\b|"
    r"\b(?:closed|close|closure)\b[^.?!]{0,40}\b(?:office|offices|campus|school)\b|"
    r"\bexam\w*\b|\benrol(?:l)?(?:ment|ments|ing|ed)?\b|\btuition\b|\bfees?\b",
    re.IGNORECASE)


def mode() -> str:
    """"off" (default), "verdict" (CAMPUS_VERIFY_ORG_PAGES=1: rule O v1, FAIL, not adopted), "related"
    (CAMPUS_VERIFY_ORG_PAGES=related: owner 2026-11-18; the page's closest post is shown, it never decides) or "verdict2"
    (CAMPUS_VERIFY_ORG_PAGES=2: rule O v2, results/round2/org_pages_v2_plan.md)."""
    value = os.environ.get(ENV, "0").strip().lower()
    return {"1": "verdict", "2": "verdict2", "3": "verdict3", "31": "verdict31", "related": "related"}.get(value, "off")


# Rule O v3.1 fallback (owner 2026-11-18): with ORG_PAGES at 3 or 31, when the rule-based match shows no page post (the
# message names no date, or no recent post names it), the page's closest post by meaning is shown as related, as in
# "related" mode. Display only: it never changes a verdict.
RELATED_FALLBACK_ENV = "CAMPUS_VERIFY_ORG_RELATED_FALLBACK"


def related_fallback() -> bool:
    return os.environ.get(RELATED_FALLBACK_ENV, "0").strip() == "1"


# Rule O v2 (owner 2026-11-18): the school-wide list gains break, holiday, vacation, finals, semester; recent posts only;
# emoji and hashtags removed before comparison; the "department announcement only" label.
RECENT_DAYS = 90
LABEL = "department announcement only"
SCHOOL_WIDE_V2_EXTRA = re.compile(r"\bbreaks?\b|\bholidays?\b|\bvacations?\b|\bfinals\b|\bsemest(?:er|ers|ral)\b",
                                  re.IGNORECASE)
_EMOJI = re.compile("[\U0001F000-\U0001FAFF⌀-⏿☀-➿⬀-⯿︎️‍⃣]")
_HASHTAG = re.compile(r"#\w+")


def clean(text: str) -> str:
    """V2-b: emoji, pictographs and hashtags removed (styled letters are already folded by load)."""
    return re.sub(r"\s+", " ", _HASHTAG.sub(" ", _EMOJI.sub(" ", text or ""))).strip()


def school_wide_v2(message: str) -> bool:
    return school_wide(message) or bool(SCHOOL_WIDE_V2_EXTRA.search(message or ""))


def recent(document: dict, today) -> bool:
    """V2-a: posted within RECENT_DAYS before `today` (a date), and not after it."""
    import datetime as dt
    try:
        posted = dt.date.fromisoformat((document.get("published_date") or "")[:10])
    except ValueError:
        return False
    return 0 <= (today - posted).days <= RECENT_DAYS


def enabled() -> bool:
    return mode() != "off"


def sources() -> Dict[str, str]:
    """{archive key: display name} from CAMPUS_VERIFY_ORG_SOURCES."""
    out = {}
    for part in os.environ.get(SOURCES_ENV, "").split(";"):
        if "=" in part:
            key, name = part.split("=", 1)
            if key.strip() and name.strip():
                out[key.strip()] = name.strip()
    return out


def school_wide(message: str) -> bool:
    return bool(SCHOOL_WIDE.search(message or ""))


def folder(project_root: str) -> str:
    value = os.environ.get(DIR_ENV, "").strip() or DEFAULT_DIR
    return value if os.path.isabs(value) else os.path.join(project_root, value)


def load(project_root: str):
    """(documents with styled letters folded, {document_id: (key, name)}, stamp) for the named organisation pages."""
    from src.normalization.fold import fold
    docs, owner, stamp = [], {}, []
    for key, name in sorted(sources().items()):
        path = os.path.join(folder(project_root), f"{key}.jsonl")
        if not os.path.isfile(path):
            continue
        stamp.append((key, os.path.getmtime(path)))
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                doc = dict(r, text=fold(r.get("text") or ""))
                docs.append(doc)
                owner[doc["document_id"]] = (key, name)
    return docs, owner, tuple(stamp)


def related(result, owner: dict) -> Optional[dict]:
    """The organisation's closest post that passed the relevance gate, for display, or None."""
    used = getattr(result, "used_evidence", None)
    if used is None or getattr(result, "evidence_status", "") != "RELEVANT_EVIDENCE_FOUND":
        return None
    key, name = owner.get(used.document_id, ("", ""))
    return {"key": key, "name": name, "document_id": used.document_id, "text": (used.text or "")[:300],
            "post_url": getattr(used, "source_url", "") or "", "post_date": (getattr(used, "published_date", "") or "")[:10]}


# Rule O-SW (results/round2/org_school_wide_plan.md; owner 2026-11-19): with rule O v3/v3.1 on, a department page may
# also confirm a school-wide message, labelled as the department's word. DYCI first and the strict match are unchanged.
SCHOOL_WIDE_ENV = "CAMPUS_VERIFY_ORG_SCHOOL_WIDE"


def school_wide_enabled() -> bool:
    return os.environ.get(SCHOOL_WIDE_ENV, "0").strip() == "1"


def school_wide_note(name: str) -> str:
    return f"This answer comes from a post by {name}, a DYCI department page, not from DYCI's official page."


def note(name: str, verdict: str, v2: bool = False) -> str:
    word = {"SUPPORTED": "Supported", "CONTRADICTED": "Contradicted"}.get(verdict, verdict)
    if v2:
        return f" {word}: {LABEL}. From {name}, not an official DYCI announcement."
    return f" {word} according to {name}, not an official DYCI announcement."

# ---- Rule O v3 (results/round2/org_pages_v3_plan.md): a rule-based match instead of the classifier -------------------
COVERAGE = 0.85
STOPWORDS = frozenset("""
a an the and or but of in on at to for from by with as is are was were be been being will would shall can could should
may might must has have had do does did it its this that these those there their they we our you your ur u i my me he
she his her them us so if then than also just only very about into up over after before next last dyci dycian dycians
sa ng ang na po pa daw raw ba got get gets going gonna which who whom what when where how all any each every some
s via per
""".split())
MONTH_WORDS = frozenset("""january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec""".split())
_TIME_WORD = re.compile(r"\b\d{1,4}(?::\d{2})?\s*(?:a\.?\s?m\.?|p\.?\s?m\.?)(?![a-z])", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z]+|\d[\d,]*")


def _tokens(text: str) -> list:
    return [t.replace(",", "") if t[0].isdigit() else t for t in _TOKEN.findall((text or "").lower())]


def content_words(message: str) -> list:
    """The message's content words (plan step 4): common words, month names, years, day numbers and times removed."""
    out = []
    for t in _tokens(_TIME_WORD.sub(" ", message or "")):
        if t in STOPWORDS or t in MONTH_WORDS or t in ("am", "pm", "nn", "noon"):
            continue
        if t.isdigit() and (int(t) <= 31 or 1900 <= int(t) <= 2100):
            continue
        out.append(t)
    return out


def _appears(word: str, post_words: set) -> bool:
    if word in post_words:
        return True
    if not word.isdigit():
        if word.endswith("s") and word[:-1] in post_words or word + "s" in post_words or word + "es" in post_words:
            return True
        if len(word) >= 5 and any(p.startswith(word[:5]) for p in post_words):
            return True
    return False


def match(message: str, post_text: str, strict_numbers: bool = False) -> Optional[dict]:
    """None, or {"coverage", "words", "missing"} when the (folded, cleaned) post confirms the message (plan step 4).
    strict_numbers (rule O v3.1, results/round2/org_pages_v31_plan.md): any number among the content words that is not
    in the post fails the match, whatever the coverage."""
    from src.verification.date_check import extract_dates
    from src.verification.unit_check import conflict as unit_conflict
    from src.verification.value_check import find_conflicts
    wanted = set(extract_dates(message or ""))
    if not wanted or not wanted & set(extract_dates(post_text or "")):
        return None
    words = content_words(message)
    if not words:
        return None
    post_words = set(_tokens(post_text))
    missing = [w for w in words if not _appears(w, post_words)]
    if strict_numbers and any(w.isdigit() for w in missing):
        return None
    coverage = 1 - len(missing) / len(words)
    if coverage < COVERAGE:
        return None
    if find_conflicts(message, post_text, post_text) or unit_conflict(message, post_text, post_text) is not None:
        return None
    return {"coverage": round(coverage, 3), "words": len(words), "missing": missing}


def recent_clean_posts(project_root: str, today):
    """(posts, owner) for v3: the named pages' posts from the last RECENT_DAYS days, text folded and cleaned, each with
    its original text kept as "shown_text"; newest first."""
    docs, owner, _ = load(project_root)
    out = [dict(d, shown_text=d["text"], text=clean(d["text"])) for d in docs if recent(d, today)]
    out.sort(key=lambda d: d.get("published_date") or "", reverse=True)
    return out, owner


def dates_in(text: str) -> set:
    from src.verification.date_check import extract_dates
    return set(extract_dates(text or ""))
