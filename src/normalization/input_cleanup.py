"""
Input hardening (results/round2/input_hardening_plan.md): two closed rewrites of the processing copy of a message, run
right after the Unicode fold and before the language gate. The response's `claim` stays as typed.

H-a CAMPUS_VERIFY_ACADEMIC_YEAR (default off): a capital "AY" directly followed by a year range ("AY 2026-2027") becomes
    "A.Y.", so the frozen router's Tagalog word list does not read it as Tagalog "ay". Lowercase "ay" is never touched.
H-b CAMPUS_VERIFY_LOOKALIKE (default off): Cyrillic and Greek letters that look like Latin letters (LOOKALIKE below) become
    those Latin letters (1) inside every word that also has a Latin letter, and (2) in the whole message when it has
    Latin letters and every Cyrillic/Greek letter in it is in the table. Real Russian or Greek text is left alone.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Optional, Tuple

VERSION = "input-cleanup-v1"
ENV_YEAR = "CAMPUS_VERIFY_ACADEMIC_YEAR"
ENV_LOOKALIKE = "CAMPUS_VERIFY_LOOKALIKE"

_AY = re.compile(r"(?<![\w.])AY(?=\s*\d{4}\s*[-–—]\s*\d{2,4}\b)")

LOOKALIKE = {
    # Cyrillic
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "У": "Y", "І": "I", "Ј": "J",
    "Ѕ": "S",
    # Greek
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "ο": "o",
}
_TABLE = str.maketrans(LOOKALIKE)
_WORD = re.compile(r"[^\W\d_]+")


def year_enabled() -> bool:
    return os.environ.get(ENV_YEAR, "0").strip() == "1"


def lookalike_enabled() -> bool:
    return os.environ.get(ENV_LOOKALIKE, "0").strip() == "1"


def _script(ch: str) -> str:
    name = unicodedata.name(ch, "")
    return "latin" if name.startswith("LATIN") else "cyrillic_greek" if name.startswith(("CYRILLIC", "GREEK")) else ""


def academic_year(text: str) -> Tuple[str, int]:
    """H-a: ('AY 2026-2027' -> 'A.Y. 2026-2027', number of changes)."""
    return _AY.subn("A.Y.", text or "")


def lookalike(text: str) -> Tuple[str, int]:
    """H-b: (text with lookalike letters mapped, number of letters changed)."""
    text = text or ""
    letters = [c for c in text if c.isalpha()]
    has_latin = any(_script(c) == "latin" for c in letters)
    foreign = [c for c in letters if _script(c) == "cyrillic_greek"]
    if not has_latin or not foreign:
        return text, 0
    if all(c in LOOKALIKE for c in foreign):                                   # (2) the whole message
        out = text.translate(_TABLE)
    else:                                                                       # (1) mixed words only
        out = _WORD.sub(lambda m: m.group(0).translate(_TABLE)
                        if any(_script(c) == "latin" for c in m.group(0)) else m.group(0), text)
    return out, sum(a != b for a, b in zip(text, out))


def clean(message: str) -> Tuple[str, Optional[dict]]:
    """Both rules as enabled: (processing copy, input_cleanup info or None when nothing changed)."""
    looks = years = 0
    if lookalike_enabled():
        message, looks = lookalike(message)
    if year_enabled():
        message, years = academic_year(message)
    if not (looks or years):
        return message, None
    return message, {"version": VERSION, "academic_year": years, "lookalike": looks}
