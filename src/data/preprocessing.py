"""
Conservative text normalization.

We deliberately do very little here. Aggressive cleaning (lowercasing,
stripping punctuation/numbers/URLs, removing proper nouns) would destroy
exactly the details that distinguish a true campus claim from a false one
-- e.g. "August 10" vs "August 15", or "Dr. Santos" vs no name at all.

The only thing we normalize at this stage is whitespace, because
inconsistent whitespace (double spaces, tabs, trailing spaces from manual
CSV entry) is a data-entry artifact, not meaningful content.

Further normalization (e.g. lowercasing for TF-IDF) is left to whichever
downstream component actually needs it, so it can be applied reversibly
and only where required -- not baked irreversibly into stored data now.
"""

import re


def normalize_whitespace(text: str) -> str:
    """
    Collapse repeated whitespace (spaces, tabs, newlines) into a single
    space, and strip leading/trailing whitespace.

    Example:
        "Classes  are\\tsuspended   tomorrow.  " -> "Classes are suspended tomorrow."
    """
    if text is None:
        return ""
    # \s+ matches any run of whitespace characters; collapse to one space.
    collapsed = re.sub(r"\s+", " ", text)
    return collapsed.strip()
