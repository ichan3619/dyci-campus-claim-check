"""
Question focus (CAMPUS_VERIFY_QUESTION_FOCUS, on by default since v2.3, "0" switches it off; v1
results/round2/question_focus_plan.md FAIL, v2 results/round2/question_focus_v2_plan.md PASS in question_focus_v2_report.md).

"Will there be classes tomorrow? It's Friday." was judged as a whole, and the side remark "It's Friday" matched the
post's "(Friday)", so a message about classes on a no-class day was SUPPORTED. When a message has two or more sentences
and exactly one of them is a question, that question is what gets rewritten and judged; everything else (routing, the
date and post rules, the notes, the claim shown) still reads the whole message.

v2: an abbreviation ("St. Paul", "Dr. Yanga") never ends a sentence, and a tag question ("No classes tomorrow. Is that
true?") carries no claim of its own, so the statement around it is judged instead.
"""
import os
import re
from typing import List, Optional

ENV = "CAMPUS_VERIFY_QUESTION_FOCUS"
VERSION = "question-focus-v2"
ABBREVIATIONS = frozenset("""st sta sto dr mr mrs ms sr jr prof engr atty rev fr mt inc co corp ltd dept univ vs no nos
jan feb mar apr jun jul aug sep sept oct nov dec""".split())
TAG_WORDS = frozenset("""is isn't isnt that this it true real legit right really correct confirmed sure fake or not no yes
ba po totoo daw""".split())
_END = re.compile(r"[?!.]+(?=\s+|$)")


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"         # on by default since v2.3 (owner, 2026-10-29)


def sentences(text: str) -> List[str]:
    """Sentences, cut after ? ! or . followed by a space, but never after an abbreviation or a single initial."""
    text = (text or "").strip()
    out, start = [], 0
    for m in _END.finditer(text):
        if m.group().endswith(".") and "?" not in m.group() and "!" not in m.group():
            word = re.search(r"([A-Za-z]+)$", text[start:m.start()])
            if word and (word.group(1).lower() in ABBREVIATIONS or len(word.group(1)) == 1):
                continue
        piece = text[start:m.end()].strip()
        if piece:
            out.append(piece)
        start = m.end()
    rest = text[start:].strip()
    if rest:
        out.append(rest)
    return out


def is_tag_question(sentence: str) -> bool:
    words = re.findall(r"[a-z']+", sentence.lower())
    return bool(words) and all(w in TAG_WORDS for w in words)


def pick(text: str) -> Optional[dict]:
    """{'version', 'sentences', 'question', 'used', 'checked'} when the message has 2+ sentences and exactly one question;
    'checked' is that question, or (for a tag question) the other sentences. None otherwise."""
    parts = sentences(text)
    questions = [s for s in parts if s.endswith("?")]
    if len(parts) < 2 or len(questions) != 1:
        return None
    question = questions[0]
    if is_tag_question(question):
        statement = " ".join(s for s in parts if s is not question)
        return {"version": VERSION, "sentences": len(parts), "question": question, "used": "statement",
                "checked": statement}
    return {"version": VERSION, "sentences": len(parts), "question": question, "used": "question", "checked": question}
