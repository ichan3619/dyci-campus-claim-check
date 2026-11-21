"""
Local translation of Tagalog/Taglish messages into English (CAMPUS_VERIFY_TAGALOG_TRANSLATE, on by default since v2.3, "0"
switches it off; route B, results/round2/tagalog_translate_plan.md; v2 PASS in tagalog_translate_v2_report.md).

translate(message) asks the local llama.cpp server (CAMPUS_VERIFY_LLM_URL, 127.0.0.1 only) running the pinned
SEA-LION model for one English sentence, then applies the plan's checks C1-C3. It returns an info dict whose
`accepted` says whether the English text may replace the message; anything else (server down, timeout, a failed
check) keeps today's evidence-only response. Nothing leaves the machine and nothing here changes a verdict.
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Optional

ENV = "CAMPUS_VERIFY_TAGALOG_TRANSLATE"
URL_ENV = "CAMPUS_VERIFY_LLM_URL"
VERSION = "tagalog-translate-v1.1"
# v1.1 (2026-11-04, results/round2/v2_3_1_decision.md): llama.cpp's prompt cache is off for every request. With it on,
# a request that reuses part of an earlier prompt is computed in a different order, so the same message could come back
# as different English even at temperature 0 (7 of 66 Tagalog development messages after days of uptime).
CACHE_PROMPT = False
MODEL = "aisingapore/Llama-SEA-LION-v3-8B-IT-GGUF@f7042ba3 Llama-SEA-LION-v3-8B-IT-Q4_K_M.gguf"
MODEL_SHA256 = "01c39efada0c86698ef7496c711ea74fd7c8750e9f550ce85405c87fd28e26a6"
INSTRUCTION = ("Translate the student's message from Tagalog/Taglish into one plain English sentence. Keep every "
               "number, date, name and English word exactly. Keep \"today\", \"tomorrow\" and \"yesterday\" as words; "
               "do not turn them into dates. Keep a question as a question. Do not add, explain or answer anything. "
               "If the message cannot be translated with confidence, reply only UNCLEAR. Campus words: \"DYCI\" and "
               "\"Yanga\" are the school's name; \"f2f\" or \"ftf\" means face-to-face classes; \"ga\" means general "
               "assembly; \"pasok\" and \"klase\" mean classes.")
INSTRUCTION_SHA256 = hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest()
TIMEOUT_SECONDS = 30
MAX_WORDS = 40
NEG_MESSAGE = frozenset("wala walang hindi di 'di ayaw no not cancelled suspended".split())
NEG_ENGLISH = frozenset("no not none never without cancelled canceled suspended".split())
_WORDS = re.compile(r"[a-z']+")


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"         # on by default since v2.3 (owner, 2026-10-29)


_CALENDAR = {
    **{m: m for m in "january february march april june july august september october november december".split()},
    **{f[:3]: f for f in "january february march april june july august september october november december".split()},
    "sept": "september", "mayo": "may", "enero": "january", "pebrero": "february", "marso": "march", "abril": "april",
    "hunyo": "june", "hulyo": "july", "agosto": "august", "setyembre": "september", "oktubre": "october",
    "nobyembre": "november", "disyembre": "december",
    **{d: d for d in "monday tuesday wednesday thursday friday saturday sunday".split()},
    **{d[:3]: d for d in "monday tuesday wednesday thursday friday saturday sunday".split()},
    "lunes": "monday", "martes": "tuesday", "miyerkules": "wednesday", "huwebes": "thursday", "biyernes": "friday",
    "sabado": "saturday", "linggo": "sunday",
}


def _numbers(text: str) -> Counter:
    """Free-standing numbers only: the 2 in "f2f" is not a number (amendment B1)."""
    return Counter(re.findall(r"(?<![A-Za-z])\d+", text or ""))


def _calendar(text: str, english: bool) -> set:
    """Months and weekdays named. "May" in a translation is the month only when a day number follows it."""
    found = set()
    for m in re.finditer(r"[a-z]+", (text or "").lower()):
        w = m.group()
        if w == "may":
            if english and re.match(r"\s+\d", text[m.end():]):
                found.add("may")
            continue
        if w in _CALENDAR:
            found.add(_CALENDAR[w])
    return found


def _negated(text: str, words: frozenset) -> bool:
    toks = _WORDS.findall((text or "").lower().replace("’", "'"))
    return any(t in words or t.endswith("n't") for t in toks)


def checks(message: str, english: str, withheld) -> dict:
    """The plan's C1-C3 and amendment B1's C4 on a candidate translation; `withheld` is the language policy's test."""
    words = english.split()
    c1 = ("\n" not in english and 1 <= len(words) <= MAX_WORDS and english.strip().upper() != "UNCLEAR"
          and not withheld(english))
    c2 = _numbers(message) == _numbers(english)
    c3 = _negated(message, NEG_MESSAGE) == _negated(english, NEG_ENGLISH)
    c4 = _calendar(english, True) <= _calendar(message, False)          # amendment B1: no invented month or weekday
    return {"C1_form": c1, "C2_numbers": c2, "C3_negation": c3, "C4_calendar": c4}


def _ask(message: str) -> str:
    url = os.environ.get(URL_ENV, "http://127.0.0.1:8081").rstrip("/") + "/v1/chat/completions"
    body = json.dumps({"messages": [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": message}],
                       "temperature": 0, "seed": 42, "max_tokens": 96, "stream": False,
                       "cache_prompt": CACHE_PROMPT}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as r:
        data = json.loads(r.read().decode("utf-8"))
    return (data["choices"][0]["message"]["content"] or "").strip().strip('"“”').strip()


def translate(message: str, withheld) -> Optional[dict]:
    """Info dict (accepted True/False), or None when the server could not be reached or answered badly."""
    started = time.monotonic()
    try:
        english = _ask(message)
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, ValueError):
        return None
    result = checks(message, english, withheld)
    return {"version": VERSION, "model": MODEL, "model_sha256": MODEL_SHA256, "instruction_sha256": INSTRUCTION_SHA256,
            "translation": english, "checks": result, "accepted": all(result.values()),
            "seconds": round(time.monotonic() - started, 2)}
