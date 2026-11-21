"""
Date notice (display only, added 2026-10-12): when the user's message names an explicit date and a post shown does not
contain that date anywhere in its text, the response says so ("This post does not mention September 27.").

    message dates  src.verification.date_check.extract_dates on the RAW message, unchanged (no new formats: "9/27",
                   "bukas", weekdays and other relative dates give no dates, so no notice)
    post dates     extract_dates on the post's full text, header date included: the notice is about textual absence
                   only, so a date that appears anywhere in the post counts as mentioned. No metadata (e.g. a post's
                   publication date field) is read, and no event date is inferred.
    ranges         each date is checked on its own; only the dates the post does not mention are listed

It never changes a verdict, the ranking, retrieval, routing or the language policy: it is computed from the response's
posts after every decision has been made.
"""
from typing import Iterable, List, Optional

from src.verification.date_check import describe, extract_dates


def _join(names: List[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


def date_notice(message: str, posts: Iterable) -> Optional[dict]:
    """{"message_dates": [...], "posts": {document_id: {"missing": [...], "text": "This post does not mention ..."}}}
    for the shown posts (objects with .document_id and .text) that miss at least one message date; None otherwise."""
    claimed = extract_dates(message or "")
    if not claimed:
        return None
    out = {}
    for post in posts:
        if post is None or post.document_id in out:
            continue
        missing = sorted(claimed - extract_dates(post.text or ""))
        if missing:
            names = [describe([d]) for d in missing]
            out[post.document_id] = {"missing": names, "text": f"This post does not mention {_join(names)}."}
    if not out:
        return None
    return {"message_dates": [describe([d]) for d in sorted(claimed)], "posts": out}
