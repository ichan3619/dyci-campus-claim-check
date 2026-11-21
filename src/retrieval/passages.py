"""
Sentence-level passage splitting (shared by the E5 retriever and the passage selector).
Kept in its own module so retrieval and selection can both use it without importing each other.
"""
import re
from typing import List

# Tokens whose trailing period is NOT a sentence end ("Circular Memorandum No. 136").
_ABBREVIATIONS = {
    "no", "nos", "mr", "mrs", "ms", "dr", "prof", "engr", "atty", "sr", "jr", "st", "vs", "etc",
    "inc", "dept", "fr", "hon", "gen", "col", "rm", "bldg", "tel", "ext", "approx",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _ends_with_abbreviation(piece: str) -> bool:
    m = re.search(r"([A-Za-z]+)\.$", piece.rstrip())
    if not m:
        return False
    word = m.group(1)
    return word.lower() in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())   # "J." initials


def split_into_passages(text: str, min_words: int = 3) -> List[str]:
    """
    Split text into sentence-sized passages. Whitespace (including newlines) is collapsed first,
    so a headline and its first sentence stay together -- in M12.1 the first sentence WITH its
    banner ("DYCI Advisory 27 September 2026 ... Please be informed that ...") gave the correct
    NLI label. Fragments shorter than `min_words` are dropped; if nothing survives, the whole
    text is returned as one passage so a document is never silently lost.
    """
    flat = re.sub(r"\s+", " ", text or "").strip()
    if not flat:
        return []
    pieces: List[str] = []
    for raw in _BOUNDARY.split(flat):
        if pieces and _ends_with_abbreviation(pieces[-1]):
            pieces[-1] = pieces[-1] + " " + raw          # false boundary after "No." etc.
        else:
            pieces.append(raw)
    kept = [p.strip() for p in pieces if len(p.split()) >= min_words]
    return kept or [flat]
