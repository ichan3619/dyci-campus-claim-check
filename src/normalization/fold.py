"""
Styled-text folding (CAMPUS_VERIFY_UNICODE_FOLD; results/round2/unicode_fold_plan_draft.md revision 3, §2).

A character is folded if and only if (1) its code point is in U+1D400-U+1D7FF (Mathematical Alphanumeric Symbols),
U+FF10-U+FF19, U+FF21-U+FF3A or U+FF41-U+FF5A (full-width digits and Latin letters), (2) it is assigned, and (3) its
NFKC form is exactly one character in [0-9A-Za-z]. It is replaced by that character; every other character is left
exactly as it is. No whole-string NFKC is applied: full-width punctuation, ligatures, circled numbers, superscripts,
currency signs, accented letters, combining marks, emoji and styled Greek stay unchanged. Folding never changes the
length of a text, so positions in the folded text are positions in the original.
"""
import hashlib
import string
import unicodedata
from typing import Dict

RANGES = ((0x1D400, 0x1D7FF), (0xFF10, 0xFF19), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A))
_TARGET = frozenset(string.ascii_letters + string.digits)


def _build() -> Dict[int, str]:
    table = {}
    for lo, hi in RANGES:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            if unicodedata.category(ch) == "Cn":
                continue
            n = unicodedata.normalize("NFKC", ch)
            if len(n) == 1 and n in _TARGET:
                table[cp] = n
    return table


ALLOWLIST: Dict[int, str] = _build()
ALLOWLIST_SHA256 = hashlib.sha256(
    "\n".join(f"{cp:06X}={ALLOWLIST[cp]}" for cp in sorted(ALLOWLIST)).encode("ascii")).hexdigest()
VERSION = "fold-v1"


def fold(text: str) -> str:
    """The text with every allowlisted styled letter/digit replaced by its plain form (same length)."""
    return (text or "").translate(ALLOWLIST) if text else (text or "")


def has_foldable(text: str) -> bool:
    return any(ord(ch) in ALLOWLIST for ch in text or "")
