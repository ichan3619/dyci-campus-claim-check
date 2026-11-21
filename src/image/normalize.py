"""
Conservative clean-up of OCR text before it is checked (Image Check Phase 1, image path only).

Only layout noise that OCR adds and a person would not type is changed; every change is reported, and the raw OCR text is
kept. Nothing else is touched: no letters, digits, words, case or punctuation marks are added, removed or reinterpreted,
apart from the one time-separator rule below.

Rules (in order):
- whitespace: runs of spaces, tabs and line breaks become one space; leading/trailing whitespace is removed
- space_before_punctuation: a space before . , ; : ! ? or ) is removed when the mark ends a word ("Center ." -> "Center.")
- time_separator: a clock time read with a dot before AM/PM becomes a colon ("1.00 PM" -> "1:00 PM"); only hours 1-12 and
  minutes 00-59 followed by AM/PM, so amounts ("PHP 1.00") and decimals are never changed

The /verify/text route and `_answer()` are not affected: the text check runs unchanged on the cleaned text.
"""
import re
from dataclasses import dataclass, field
from typing import List

_WHITESPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r" ([.,;:!?)])(?=\s|$)")
_TIME_DOT = re.compile(r"\b(1[0-2]|0?[1-9])\.([0-5]\d)(\s?)([AaPp]\.?[Mm]\b\.?)")


@dataclass
class NormalizedText:
    text: str
    changes: List[str] = field(default_factory=list)   # rule names that changed something, in order


def normalize_ocr_text(raw: str) -> NormalizedText:
    text, changes = raw or "", []
    collapsed = _WHITESPACE.sub(" ", text).strip()
    if collapsed != text:
        changes.append("whitespace")
    text = collapsed
    for name, pattern, repl in (("space_before_punctuation", _SPACE_BEFORE_PUNCT, r"\1"),
                                ("time_separator", _TIME_DOT, r"\1:\2\3\4")):
        new = pattern.sub(repl, text)
        if new != text:
            changes.append(name)
        text = new
    return NormalizedText(text=text, changes=changes)


def agreement_variant(text: str):
    """The same text with its final full stop toggled (removed if present, added after a final letter or digit), or None
    when the text ends in other punctuation. OCR often drops or adds a final dot, so a verdict that changes with it is
    not a reliable reading of the image. Used by the image route's agreement check; no threshold is involved."""
    text = text.rstrip()
    if text.endswith("."):
        return text[:-1].rstrip() or None
    if text and text[-1].isalnum():
        return text + "."
    return None
