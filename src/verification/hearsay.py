"""
Hearsay and prediction notice (defence scenario 7; results/round2/scenario_probes_report.md). Setting
CAMPUS_VERIFY_HEARSAY_NOTE, on by default since v1.9 ("0" switches it off). Warn-only: it never changes a verdict,
retrieval, routing or the language policy.

When the message is phrased as a prediction or as something heard ("probably", "I heard", "baka", "daw" ...), the answer
carries a note that only an official announcement settles it. The words are matched as whole words, English and Tagalog.
"""
import os
import re
from typing import Optional

ENV = "CAMPUS_VERIFY_HEARSAY_NOTE"
VERSION = "hearsay-v1"
PHRASES = [
    "probably", "maybe", "perhaps", "possibly", "might", "likely", "i heard", "heard that", "rumor", "rumors", "rumour",
    "rumours", "allegedly", "supposedly", "reportedly", "they say", "people say", "someone said",
    "daw", "baka", "siguro", "yata", "sabi nila", "sabi ng", "balita ko", "narinig ko", "chika",
]   # "raw" (the form of "daw" after a vowel) is left out: it is also an English word ("raw materials")
_RX = re.compile(r"(?<![\w'])(" + "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in
                                            sorted(PHRASES, key=len, reverse=True)) + r")(?![\w'])", re.I)
TEXT = ("Your message sounds like a prediction or something heard (“{word}”). Only an official DYCI announcement "
        "settles it: check that the post shown says this was decided.")


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"         # on by default since v1.9 (owner, 2026-10-27)


def notice(message: str) -> Optional[dict]:
    words = []
    for m in _RX.finditer(message or ""):
        w = re.sub(r"\s+", " ", m.group(1).lower())
        if w not in words:
            words.append(w)
    if not words:
        return None
    return {"version": VERSION, "words": words, "text": TEXT.format(word=words[0])}
