"""
Rule G: picture text, the school's own name and one stray misread word do not withhold the verdict
(results/round2/poster_language_plan.md). CAMPUS_VERIFY_POSTER_LANGUAGE=1 (default off). Picture check only; typed
messages and links are unchanged.

A text reader turns DYCI's seal "Dr. Yanga's Colleges" into "Dr. Ya nga' s Colleges" and "NO" into "MO"; both spell
words on the verdict gate's extra Tagalog list (src/api/verdict_policy.py), so an English poster's verdict is withheld.

- G-a school name: for the language decision only, a run of 1 to 4 consecutive tokens whose letters (lowercase, letters
  only, joined) are exactly one of SCHOOL_NAMES is left out. The text that is checked is not changed.
- G-b one stray word: after G-a, the verdict is not withheld for language when the router's own Tagalog check does not
  fire, at most one token matches the extra list (or a hyphenated Tagalog prefix), and at least MIN_TOKENS remain.
"""
from __future__ import annotations

import contextvars
import os
import re
from typing import List, Optional

ENV = "CAMPUS_VERIFY_POSTER_LANGUAGE"
VERSION = "poster-language-v1"
SCHOOL_NAMES = frozenset({"yanga", "yangas", "dryanga", "dryangas"})
MAX_NAME_TOKENS = 4
MIN_TOKENS = 8

# Set by the picture check while it runs its text checks with the setting on; RELEASED collects what G released.
ACTIVE = contextvars.ContextVar("poster_language_active", default=False)
RELEASED = contextvars.ContextVar("poster_language_released", default=None)


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def _letters(token: str) -> str:
    return re.sub(r"[^a-z]", "", token.lower())


def school_name_positions(toks: List[str]) -> set:
    """Indexes of tokens that spell the school's own name (G-a), longest run first."""
    out, i = set(), 0
    while i < len(toks):
        for n in range(min(MAX_NAME_TOKENS, len(toks) - i), 0, -1):
            if _letters("".join(toks[i:i + n])) in SCHOOL_NAMES:
                out.update(range(i, i + n))
                i += n
                break
        else:
            i += 1
    return out


def decide(text: str) -> Optional[dict]:
    """None when G does not release the verdict; otherwise {"ignored": [...], "stray": [...]}."""
    from src.api.verdict_policy import EXTRA_TAGALOG_WORDS, TAGALOG_PREFIXES
    from src.routing.router import is_tagalog_or_taglish, tokens
    text = text if isinstance(text, str) else ""
    if is_tagalog_or_taglish(text):
        return None
    toks = tokens(text)
    name = school_name_positions(toks)
    rest = [t for i, t in enumerate(toks) if i not in name]
    stray = [t for t in rest if t in EXTRA_TAGALOG_WORDS or ("-" in t and t.split("-")[0] in TAGALOG_PREFIXES)]
    if len(stray) > 1 or len(rest) < MIN_TOKENS:
        return None
    return {"ignored": [toks[i] for i in sorted(name)], "stray": stray}


def released_by_rule(text: str) -> bool:
    """Called by the verdict gate's wrapper only when the gate would withhold: True when G releases, and noted."""
    if not ACTIVE.get():
        return False
    found = decide(text)
    if found is None:
        return False
    notes = RELEASED.get()
    if notes is not None:
        notes.append({"version": VERSION, "released": True, **found})
    return True
