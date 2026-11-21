"""
Tagalog/Taglish fixed-phrase reader (CAMPUS_VERIFY_TAGALOG_PHRASES, default off; results/round2/tagalog_phrases_plan.md).

read(message, vocabulary) returns an English sentence for a withheld Tagalog/Taglish message that fits one of the plan's
closed frames T1-T8 with every word understood, or None. It never guesses: an unknown word, a reaction word, a second
clause or a question word means None, and the message keeps today's evidence-only response. Relative day words become
the English ones ("bukas" -> "tomorrow") so relative_resolve turns them into Philippine-time dates and its nearby-date
guard applies. Nothing here changes a verdict.
"""
import hashlib
import os
import re
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from src.api.verdict_policy import EXTRA_TAGALOG_WORDS
from src.routing.router import TAGALOG_MARKERS

ENV = "CAMPUS_VERIFY_TAGALOG_PHRASES"
VERSION = "tagalog-phrases-v1"

FILLER: FrozenSet[str] = frozenset("ba po na pa ulit nga daw raw".split())
REFUSE: FrozenSet[str] = frozenset("""
akala diba eh talaga sabi yata kaya
hindi di 'di
at pero kasi tapos and or
ano sino saan san kailan magkano ilan bakit paano anong what when where who how
""".split())
TRAILING_TAGS = ("no", "diba")
HAVE = frozenset("may meron mayroon".split())
NONE_ = frozenset("wala walang".split())
CLASS = frozenset("pasok klase class classes ftf f2f face-to-face".split())
FACE_TO_FACE = frozenset("ftf f2f face-to-face".split())
RELATIVE = {"bukas": "tomorrow", "ngayon": "today", "kahapon": "yesterday", "mamaya": "tonight"}
WEEKDAYS = {"monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday", "thursday": "Thursday",
            "friday": "Friday", "saturday": "Saturday", "sunday": "Sunday", "lunes": "Monday", "martes": "Tuesday",
            "miyerkules": "Wednesday", "huwebes": "Thursday", "biyernes": "Friday", "sabado": "Saturday",
            "linggo": "Sunday"}
_MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December")
_TAGALOG_MONTHS = ("enero", "pebrero", "marso", "abril", "mayo", "hunyo", "hulyo", "agosto", "setyembre", "oktubre",
                   "nobyembre", "disyembre")
MONTHS: Dict[str, str] = {}
for _i, _name in enumerate(_MONTH_NAMES):
    MONTHS[_name.lower()] = MONTHS[_name.lower()[:3]] = MONTHS[_TAGALOG_MONTHS[_i]] = _name
MONTHS["sept"] = "September"
del MONTHS["may"]                                   # "may" is the frame word "there is"; the month is read only as "mayo"
GLOSSARY: Dict[str, str] = {
    "misa": "mass", "simbahan": "church", "estudyante": "students", "istudyante": "students", "guro": "teachers",
    "magulang": "parents", "lahat": "all", "paaralan": "school", "eskwelahan": "school", "pagsusulit": "exam",
    "bakasyon": "break", "seremonya": "ceremony", "pista": "fiesta", "parada": "parade", "paligsahan": "contest",
    "pagpupulong": "meeting", "pagdiriwang": "celebration", "bisita": "visit", "bagyo": "typhoon", "baha": "flood",
}
_TOKEN = re.compile(r"[\w'’-]+|,")
_WORD = re.compile(r"[a-z][a-z'’-]+")
_NOT_ENGLISH = (TAGALOG_MARKERS | EXTRA_TAGALOG_WORDS | FILLER | REFUSE | HAVE | NONE_ | frozenset(RELATIVE)
                | frozenset(_TAGALOG_MONTHS) | frozenset(k for k, v in WEEKDAYS.items() if k != v.lower())
                | frozenset("nagkaroon ng ang si ngayong".split()))


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


def build_vocabulary(texts: Iterable[str]) -> Tuple[FrozenSet[str], str]:
    """(words of the collection's posts, lower-cased, >= 3 letters, minus every Tagalog-listed word; its SHA-256)."""
    words = set()
    for t in texts:
        words.update(w.replace("’", "'") for w in _WORD.findall((t or "").lower()) if len(w) >= 3)
    vocab = frozenset(words - _NOT_ENGLISH)
    return vocab, hashlib.sha256("\n".join(sorted(vocab)).encode("utf-8")).hexdigest()


_CACHE: Dict[Tuple[int, int], Tuple[FrozenSet[str], str]] = {}


def vocabulary_for(verifier) -> Tuple[FrozenSet[str], str]:
    """The vocabulary of the verifier's own collection, built once per collection object."""
    docs = getattr(getattr(verifier, "_retriever", None), "_documents", None) or []
    key = (id(docs), len(docs))
    if key not in _CACHE:
        _CACHE[key] = build_vocabulary(d.get("text", "") if isinstance(d, dict) else getattr(d, "text", "") for d in docs)
    return _CACHE[key]


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _english(words: List[str], vocab: FrozenSet[str]) -> Optional[str]:
    """1-4 understood words -> English, or None."""
    if not 1 <= len(words) <= 4:
        return None
    out = []
    for w in words:
        if w in GLOSSARY:
            out.append(GLOSSARY[w])
        elif w in vocab:
            out.append(w)
        else:
            return None
    return " ".join(out)


def _modifiers(words: List[str], original: Dict[str, str], vocab: FrozenSet[str]) -> Optional[Tuple[str, str]]:
    """(when, where) from the frame's tail, each '' when absent; None if any word is left over or a slot repeats."""
    when = where = ""
    i = 0
    while i < len(words):
        w = words[i]
        nxt = words[i + 1] if i + 1 < len(words) else None
        if w in RELATIVE and not when:
            when, i = RELATIVE[w], i + 1
        elif w == "ngayong" and nxt == "araw" and not when:
            when, i = "today", i + 2
        elif w == "ngayong" and nxt in MONTHS and not when:
            when, i = f"in {MONTHS[nxt]}", i + 2
        elif w == "sa" and nxt is not None:
            rest = words[i + 1:]
            got = _date_after_sa(rest)
            if got and not when:
                when, used = got
                i += 1 + used
                continue
            place = _place(rest, original, vocab)
            if place and not where:
                where, used = place
                i += 1 + used
                continue
            return None
        else:
            return None
    return when, where


def _date_after_sa(rest: List[str]) -> Optional[Tuple[str, int]]:
    a = rest[0]
    b = rest[1] if len(rest) > 1 else None
    if a in WEEKDAYS:
        return f"on {WEEKDAYS[a]}", 1
    if a in MONTHS and b is not None and b.isdigit() and 1 <= int(b) <= 31:
        return f"on {MONTHS[a]} {int(b)}", 2
    if a.isdigit() and 1 <= int(a) <= 31 and b in MONTHS:
        return f"on {MONTHS[b]} {int(a)}", 2
    if a in MONTHS:
        return f"in {MONTHS[a]}", 1
    if a.isdigit() and 1 <= int(a) <= 31:
        return f"on the {_ordinal(int(a))}", 1
    return None


def _place(rest: List[str], original: Dict[str, str], vocab: FrozenSet[str]) -> Optional[Tuple[str, int]]:
    """One or two words: understood, or capitalised as typed (a name)."""
    for n in (2, 1):
        part = rest[:n]
        if len(part) < n:
            continue
        shown = []
        for w in part:
            typed = original.get(w, w)
            if typed[:1].isupper() and w not in _NOT_ENGLISH:
                shown.append(typed)
            elif w in GLOSSARY or w in vocab:
                shown.append(GLOSSARY.get(w, w))
            else:
                shown = None
                break
        if shown:
            return "at " + " ".join(shown), n
    return None


def _split_thing(words: List[str]) -> Tuple[List[str], List[str]]:
    """THING runs up to the first WHEN/WHERE marker."""
    for i, w in enumerate(words):
        if w == "sa" or w == "ngayong" or w in RELATIVE:
            return words[:i], words[i:]
    return words, []


def _sentence(*parts: str) -> str:
    s = " ".join(p for p in parts if p).strip()
    return s[0].upper() + s[1:] + "."


def read(message: str, vocabulary: Tuple[FrozenSet[str], str]) -> Optional[dict]:
    """{'version', 'frame', 'reading', 'vocabulary_sha256', 'vocabulary_size'} or None (not understood)."""
    vocab, sha = vocabulary
    raw = _TOKEN.findall(message or "")
    if not raw or "," in raw:
        return None
    original = {t.lower().replace("’", "'"): t for t in raw}
    words = [t.lower().replace("’", "'") for t in raw]
    while words and words[-1] in TRAILING_TAGS:
        words.pop()
    if any(w in REFUSE for w in words):
        return None
    words = [w for w in words if w not in FILLER]
    found = _frame(words, original, vocab)
    if found is None:
        return None
    frame, reading = found
    return {"version": VERSION, "frame": frame, "reading": reading, "vocabulary_sha256": sha,
            "vocabulary_size": len(vocab)}


def _frame(words: List[str], original: Dict[str, str], vocab: FrozenSet[str]) -> Optional[Tuple[str, str]]:
    if not words:
        return None
    head, rest = words[0], words[1:]
    # T3: suspended [ang] CLASS [WHEN] / CLASS suspended [WHEN]
    if head == "suspended" or (head in CLASS and rest[:1] == ["suspended"]):
        tail = rest[1:] if head in CLASS else rest
        if head == "suspended":
            if tail[:1] == ["ang"]:
                tail = tail[1:]
            if not tail or tail[0] not in CLASS:
                return None
            tail = tail[1:]
        mods = _modifiers(tail, original, vocab)
        if mods is None or mods[1]:
            return None
        return "T3", _sentence("classes are suspended", mods[0])
    # T4: (online|ftf|f2f|face-to-face) [class|classes] [ang] [lahat] [WHEN]
    if head in ("online",) or head in FACE_TO_FACE:
        tail = rest
        if tail[:1] and tail[0] in ("class", "classes"):
            tail = tail[1:]
        if tail[:1] == ["ang"]:
            tail = tail[1:]
        everyone = tail[:1] == ["lahat"]
        if everyone:
            tail = tail[1:]
        mods = _modifiers(tail, original, vocab)
        if mods is None or mods[1]:
            return None
        mode = "online" if head == "online" else "face-to-face"
        return "T4", _sentence("classes are", mode, "for all levels" if everyone else "", mods[0])
    if head in HAVE or head in NONE_:
        negative = head in NONE_
        # T1/T2: CLASS [WHEN]
        if rest[:1] and rest[0] in CLASS:
            mods = _modifiers(rest[1:], original, vocab)
            if mods is None or mods[1]:
                return None
            kind = "face-to-face classes" if rest[0] in FACE_TO_FACE else "classes"
            return ("T2" if negative else "T1"), _sentence("there are", "no" if negative else "", kind, mods[0])
        thing, tail = _split_thing(rest)
        # T8: (may|meron) THING (ang|si) NAME
        if not negative:
            for j, w in enumerate(thing):
                if w in ("ang", "si"):
                    have, name = _english(thing[:j], vocab), _english(thing[j + 1:], vocab)
                    if have is None or name is None or tail:
                        return None
                    return "T8", _sentence(name, "has", have)
        thing_en = _english(thing, vocab)
        mods = _modifiers(tail, original, vocab)
        if thing_en is None or mods is None:
            return None
        return ("T6" if negative else "T5"), _sentence("there is", "no" if negative else "", thing_en, *mods)
    # T7: nagkaroon [ng] THING [WHEN] [WHERE]
    if head == "nagkaroon":
        tail = rest[1:] if rest[:1] == ["ng"] else rest
        thing, tail = _split_thing(tail)
        thing_en = _english(thing, vocab)
        mods = _modifiers(tail, original, vocab)
        if thing_en is None or mods is None:
            return None
        return "T7", _sentence("there was", thing_en, *mods)
    return None
