"""
English "Has/Have <subject> been ...?" -> statement (R1; results/round2/english_fixes_plan.md, committed 1015def).
Setting CAMPUS_VERIFY_EN_BEEN_RULE, on by default since v1.8 (PASS with Amendment 1,
results/round2/english_fixes_report.md); "0" switches it off.

en-question-v1 (english_question.py, unchanged) allows subjects of at most 5 words and refuses longer ones with
"no_subject_boundary" ("Has the Memorandum of Agreement with SPUS been signed?"). In a has/have question, "been" marks the
end of the subject unambiguously, so this rule handles exactly those refusals: the subject is every word between the
auxiliary and "been" (up to MAX_SUBJECT). Words are only reordered: nothing is added, removed or replaced.
"""
import hashlib
import json
import os
import re
from typing import Optional

from src.normalization.english_question import DETERMINERS, NEGATIONS, WH_WORDS, _bare

ENV = "CAMPUS_VERIFY_EN_BEEN_RULE"
VERSION = "en-been-v1"
MAX_SUBJECT = 12
RULES_SHA256 = hashlib.sha256(json.dumps({"version": VERSION, "max_subject": MAX_SUBJECT, "aux": ["has", "have"],
                                          "marker": "been", "wh": sorted(WH_WORDS), "neg": sorted(NEGATIONS),
                                          "det": sorted(DETERMINERS)}, sort_keys=True).encode("utf-8")).hexdigest()


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"         # on by default since v1.8 (owner, 2026-10-27)


def rewrite(text: str) -> Optional[str]:
    """The statement, or None when the rule does not apply."""
    s = (text or "").strip()
    if not s.endswith("?") or re.search(r"[.?!]\s", s[:-1]):
        return None
    tokens = s[:-1].rstrip("?").split()
    bares = [_bare(t) for t in tokens]
    if len(tokens) < 4 or bares[0] not in ("has", "have"):
        return None
    if any(b in WH_WORDS or b in NEGATIONS or b.endswith("n't") for b in bares):
        return None
    try:
        i = bares.index("been", 2, MAX_SUBJECT + 2)
    except ValueError:
        return None
    subject = tokens[1:i]
    if _bare(subject[-1]) in DETERMINERS:
        return None
    first = subject[0][:1].upper() + subject[0][1:]
    return " ".join([first] + subject[1:] + [tokens[0].lower()] + tokens[i:]) + "."
