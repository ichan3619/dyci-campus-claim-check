"""
Wrong-post rule (1c; results/round2/wrong_post_plan.md, committed 47b59a6; PASS with Amendment 1,
results/round2/wrong_post_report.md). Setting CAMPUS_VERIFY_WRONG_POST, on by default since v1.7 (owner, 2026-10-26);
"0" switches it off. Wired after the other-school check in src/api/main.py; English messages only.

A CONTRADICTED verdict becomes INSUFFICIENT_EVIDENCE when the post it rests on is probably about something else:
    (a) the message names an explicit date and the post used mentions none of the message's dates, and
    (b) either (A) another shown post mentions every message date (a better-dated post exists), or
               (B) the post used shares no topic word with the message (TOPIC_STOP lists the words that do not count).
It never touches SUPPORTED or INSUFFICIENT_EVIDENCE, never creates a verdict, and never reverses one. A post that
mentions a nearby date of the same event (a fake with one detail changed) keeps its CONTRADICTED through (b).
"""
import os
import re
from dataclasses import replace
from typing import Iterable, Optional

from src.verification.date_check import describe, extract_dates

ENV = "CAMPUS_VERIFY_WRONG_POST"
VERSION = "wrong-post-v1"
TOPIC_STOP = frozenset("""the a an and or but of on in at to for from by with is are was were be been will would shall should
can could may might must have has had do does did not no yes this that these those there their they them it its as into
about over under after before during than then so such also only all any each every some more most other very just own
same dyci yanga yangas colleges college inc school students student 2026 2025 day days date dates held hold heard
january february march april may june july august september october november december""".split())
REASON = (" Wrong-post check: the closest post is about something else (it does not mention {dates}), so this could "
          "not be checked against DYCI's announcements.")


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"         # on by default since v1.7 (owner, 2026-10-26)


def topic_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in TOPIC_STOP}


def check(message: str, result, shown: Iterable = ()):
    """(result, info). `shown` are the posts shown with the answer (objects with .document_id and .text)."""
    used = result.used_evidence
    if result.verdict != "CONTRADICTED" or used is None:
        return result, None
    dates = extract_dates(message or "")
    if not dates or dates & extract_dates(used.text or ""):
        return result, None
    better: Optional[str] = next((d.document_id for d in shown if d is not None and d.document_id != used.document_id
                                  and dates <= extract_dates(d.text or "")), None)
    shared = sorted(topic_words(message) & topic_words(used.text))
    reason = "better_dated_post" if better else ("no_shared_topic_word" if not shared else None)
    info = {"version": VERSION, "message_dates": describe(dates), "post_used": used.document_id,
            "better_dated_post": better, "shared_topic_words": shared, "withheld": reason is not None,
            "reason": reason, "original_verdict": result.verdict}
    if reason is None:
        return result, info
    return replace(result, verdict="INSUFFICIENT_EVIDENCE",
                   explanation=result.explanation + REASON.format(dates=describe(dates))), info
