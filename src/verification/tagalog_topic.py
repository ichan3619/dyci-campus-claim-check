"""
Topic guard for Tagalog/Taglish verdicts (CAMPUS_VERIFY_TAGALOG_TOPIC_GUARD, default off; route A,
results/round2/tagalog_verdicts_plan.md). Only reached with CAMPUS_VERIFY_VERDICT_LANGUAGES=all.

A SUPPORTED or CONTRADICTED verdict on a message the language policy flags as Tagalog/Taglish becomes
INSUFFICIENT_EVIDENCE unless the message shares at least one topic word with the post used. Topic words are the English
wrong-post rule's (wrong_post.topic_words); the Tagalog word lists are removed from the message's side. It never
touches INSUFFICIENT_EVIDENCE, never creates a verdict and never reverses one.
"""
import os
from dataclasses import replace

from src.api.verdict_policy import EXTRA_TAGALOG_WORDS
from src.routing.router import TAGALOG_MARKERS
from src.verification.wrong_post import topic_words

ENV = "CAMPUS_VERIFY_TAGALOG_TOPIC_GUARD"
VERSION = "tagalog-topic-v1"
REASON = (" Topic check: the closest post does not mention what this message is about, so it cannot settle this.")


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def message_topic_words(message: str) -> set:
    return topic_words(message) - TAGALOG_MARKERS - EXTRA_TAGALOG_WORDS


def check(message: str, result):
    """(result, info); info is None when the verdict is not SUPPORTED/CONTRADICTED or no post was used."""
    used = result.used_evidence
    if result.verdict not in ("SUPPORTED", "CONTRADICTED") or used is None:
        return result, None
    shared = sorted(message_topic_words(message) & topic_words(used.text))
    info = {"version": VERSION, "post_used": used.document_id, "shared_topic_words": shared,
            "withheld": not shared, "original_verdict": result.verdict}
    if shared:
        return result, info
    return replace(result, verdict="INSUFFICIENT_EVIDENCE", explanation=result.explanation + REASON), info
