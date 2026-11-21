"""
Option B: a picture whose whole text conflicts with a DYCI post it closely matches is put at Level 1
(results/round2/text_conflict_level_plan.md). CAMPUS_VERIFY_TEXT_CONFLICT_LEVEL=1 (default off). Picture check only,
after the level and rule F, before option A's line; it changes the picture's level and reasons, never a verdict.

Fires when:
  B-a the whole-text check (text_check) is CONTRADICTED and not withheld;
  B-b the level is 2, 3 or 4;
  B-c at least half of the picture's content words appear in the post the check used (rule F's overlap, as it is:
      2 x found >= words, words >= 1).
Then: Level 1 ("Conflicts with DYCI") with the reason first; why_not_sure, closest and check cleared, as rule F does.
"""
from __future__ import annotations

import os
from typing import Optional

ENV = "CAMPUS_VERIFY_TEXT_CONFLICT_LEVEL"
VERSION = "text-conflict-level-v1"
FROM_LEVELS = (2, 3, 4)
REASON = "Checked as one piece of text, the words read from this picture conflict with {post}."


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def check(picture_text: str, text_check, level: Optional[int]) -> Optional[dict]:
    """The option B finding, or None."""
    from src.image.changed_date import overlap
    from src.image.text_line import post_name
    if text_check is None or getattr(text_check, "verdict_withheld", None):
        return None
    if getattr(text_check, "verdict", None) != "CONTRADICTED":                                        # B-a
        return None
    if level not in FROM_LEVELS:                                                                     # B-b
        return None
    ev = getattr(text_check, "used_evidence", None)
    post_text = getattr(ev, "text", None) if ev is not None else None
    if not post_text:
        return None
    found, words = overlap(picture_text, post_text)
    if words == 0 or 2 * found < words:                                                             # B-c
        return None
    name, post_dated, _ = post_name(ev)
    return {"version": VERSION, "from_level": level, "post": ev.document_id, "post_dated": post_dated,
            "post_name": name, "overlap": f"{found} of {words}"}


def apply(card: dict, info: dict) -> dict:
    """The level card at Level 1 with option B's reason first; nothing else in the response changes."""
    from src.image.levels import CHECK, NAMES as LEVEL_NAMES
    reason = REASON.format(post=info["post_name"])
    return {**card, "level": 1, "name": LEVEL_NAMES[1], "reasons": [reason] + list(card.get("reasons") or []),
            "why_not_sure": None, "closest": [], "check": CHECK}
