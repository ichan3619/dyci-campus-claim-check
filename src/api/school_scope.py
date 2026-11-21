"""
School / institution detection shared by C1 (scope notice, display only) and C2 (other-school check, off by default).
Detector v2 (results/round2/other_school_v2_plan_draft.md, revision 2), 2026-10-12.

    detect(text)     non-DYCI institutions named in a text, in order, from
                     (a) the fixed list SCHOOLS (any-case aliases; capitals-only aliases for abbreviations that are also
                         ordinary words, e.g. "UP" not "up");
                     (b) NAME + KEYWORD, KEYWORD in High School / School / Academy / College(s) / University / Institute,
                         Title Case or ALL CAPS ("Philippine Science High School", "HOGWARTS ACADEMY");
                     (c) "University of <Name>" / "Institute of <Name>". "College of ..." and "School of ..." are DYCI-style
                         department forms and are never counted.
                     Guards (all only reduce detection): DYCI names (including its campuses Elida and Marciano) and
                     units are never reported; a name whose only distinctive word is DYCI's town ("Bocaue High School")
                     is uncertain and not reported; a name made only of
                     generic words is dropped (GENERIC_WORDS: "The University", "Junior High School"), except the fixed
                     "National University"; in an all-caps message capitals-only abbreviations are not matched; a name
                     directly followed by a place word ("Ateneo Avenue", "UP Town Center") is dropped.
    mentions(post, name)   the post names that institution: same detected name, a fixed short form (SHORT_FORMS), or the
                     name's distinctive (non-generic) words; a distinctive part of one word of 3 letters or fewer is
                     uncertain and counts as mentioned.
    notice(message, posts) C1: one sentence for the named institutions that no shown post mentions; None otherwise.
                     Added only to /verify/text responses. Never changes a decision.
"""
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

# (display name, case-insensitive aliases, capitals-only aliases)
SCHOOLS: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    ("Bulacan State University", ("bulacan state university", "bulsu"), ()),
    ("BSU", (), ("BSU",)),                                   # ambiguous (several "State Universities"): shown as typed
    ("University of the Philippines", ("university of the philippines",), ("UP", "U.P.")),
    ("Polytechnic University of the Philippines", ("polytechnic university of the philippines",), ("PUP",)),
    ("University of Santo Tomas", ("university of santo tomas",), ("UST",)),
    ("De La Salle", ("de la salle", "la salle"), ("DLSU",)),
    ("Ateneo", ("ateneo",), ("ADMU",)),
    ("STI", ("sti college",), ("STI",)),
    ("AMA", ("ama computer college", "ama university"), ("AMA",)),
    ("Far Eastern University", ("far eastern university",), ("FEU",)),
    ("Centro Escolar University", ("centro escolar university",), ("CEU",)),
    ("Mapúa", ("mapua", "mapúa"), ()),
    ("St. Paul University", ("st. paul university", "st paul university", "saint paul university"), ("SPUS",)),
    ("La Consolacion University", ("la consolacion university",), ("LCUP",)),
    ("Our Lady of Fatima University", ("our lady of fatima university",), ("OLFU",)),
    ("Baliuag University", ("baliuag university", "baliwag university"), ()),
)
AMBIGUOUS = frozenset({"BSU"})
SHORT_FORMS: Dict[str, Tuple[str, ...]] = {
    "St. Paul University": ("st. paul", "st paul", "saint paul"),
    "Bulacan State University": ("bulacan state",),
    "De La Salle": ("salle",),
    "University of Santo Tomas": ("santo tomas",),
    "Our Lady of Fatima University": ("fatima university",),
    "La Consolacion University": ("la consolacion",),
    "Centro Escolar University": ("centro escolar",),
    "Far Eastern University": ("far eastern",),
}
DYCI_NAMES = ("dyci", "yanga", "francisco balagtas academy", "elida", "marciano")    # incl. DYCI's campuses (owner, 2026-10-12)
DYCI_TOWN_WORDS = frozenset({"bocaue"})     # DYCI's town: "Bocaue High School" may be DYCI's, so it is uncertain
DYCI_UNITS = ("technology and engineering college", "junior high school", "senior high school", "grade school",
              "graduate school", "college community")
GENERIC_WORDS = frozenset("""
the a an this that these those our your my his her its their every each any some other another same
state national private public city municipal provincial local community catholic christian partner host home main new old
senior junior grade elementary primary secondary graduate basic integrated night summer online open
law medical medicine nursing engineering technology business arts science sciences education accountancy hospitality tourism
psychology computer information
and & of de del ng
is are was were will would can could did does do has have had may should must kailan saan ano sino bakit paano totoo sabi
daw ba sa ang mga at
""".split())
PLACE_WORDS = frozenset("avenue ave street st road rd highway town center centre village subdivision plaza station mall bridge".split())
GENERIC_NAMED = ("National University",)       # the one real institution kept although its name is generic-only

_KEYWORD = r"(?:High\s+School|HIGH\s+SCHOOL|School|SCHOOL|Academy|ACADEMY|Colleges|COLLEGES|College|COLLEGE|University|UNIVERSITY|Institute|INSTITUTE)"
_KW_WORDS = r"(?:High\s+School|HIGH\s+SCHOOL|School|SCHOOL|Academy|ACADEMY|Colleges?|COLLEGES?|University|UNIVERSITY|Institute|INSTITUTE)\b"
_WORD = r"(?!" + _KW_WORDS + r")[A-Z][\w.'\u2019&-]*"        # a name word is never itself an institution keyword
_CONN = r"(?:de|del|ng|of|the|and|&|DE|DEL|NG|OF|THE|AND)"
_NAME_KEYWORD = re.compile(r"((?:" + _WORD + r"|" + _CONN + r")(?:\s+(?:" + _WORD + r"|" + _CONN + r")){0,7})\s+(" + _KEYWORD + r")\b")
_KEYWORD_OF = re.compile(r"\b(University|UNIVERSITY|Institute|INSTITUTE)\s+(?:of|OF)\s+(?:(?:the|THE)\s+)?(" + _WORD
                         + r"(?:\s+" + _WORD + r"){0,3})")
_TOKEN = re.compile(r"[^\s]+")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").replace("’", "'")


def _key(name: str) -> str:
    return " ".join(_norm(name).lower().split())


def _all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) >= 0.6


def _next_word(text: str, end: int) -> str:
    m = _TOKEN.match(text, end + (1 if end < len(text) and text[end].isspace() else 0))
    return re.sub(r"[^\w]", "", m.group(0)).lower() if m else ""


def _prev_word(text: str, start: int) -> str:
    words = text[:start].split()
    return words[-1] if words else ""


def _is_dyci(name: str) -> bool:
    k = _key(name)
    return any(d in k for d in DYCI_NAMES) or any(k == u or k.startswith(u + " ") for u in DYCI_UNITS)


STOP_WORDS = frozenset("at in on for to from with by about sa kay ni si para is are was were will did does do has have".split())


def _strip_leading(words: List[str]) -> List[str]:
    """Keep only the words after the last preposition/auxiliary ("CLASSES AT PHILIPPINE SCIENCE" -> "PHILIPPINE SCIENCE"),
    then drop leading generic words ("The", "Did")."""
    for i in range(len(words) - 1, -1, -1):
        if words[i].lower().strip(".,") in STOP_WORDS:
            words = words[i + 1:]
            break
    while words and words[0].lower().strip(".,") in GENERIC_WORDS:
        words = words[1:]
    return words


def _generic_only(words: List[str]) -> bool:
    return all(w.lower().strip(".,") in GENERIC_WORDS for w in words)


def detect(text: str) -> List[str]:
    """Display names of the non-DYCI institutions named in `text`, in order of first appearance, without repeats."""
    t = _norm(text)
    low = t.lower()
    caps_text = _all_caps(t)
    found: List[Tuple[int, str]] = []
    covered: List[Tuple[int, int]] = []

    def take(span, name):
        found.append((span[0], name))
        covered.append(span)

    for name, ci, caps in SCHOOLS:
        hits = [m.span() for a in ci for m in re.finditer(r"(?<![\w.])" + re.escape(a) + r"(?![\w])", low)]
        if not caps_text:
            hits += [m.span() for a in caps for m in re.finditer(r"(?<![\w.])" + re.escape(a) + r"(?![\w])", t)]
        hits = [h for h in hits if _next_word(t, h[1]) not in PLACE_WORDS]
        if hits:
            take(min(hits), name)
            covered.extend(hits)
    for name in GENERIC_NAMED:                                    # "National University", not "Seoul National University"
        for form in (name, name.upper()):
            for m in re.finditer(r"(?<![\w.])" + re.escape(form) + r"(?![\w])", t):
                prev = _prev_word(t, m.start())
                if not (prev[:1].isupper() and prev.lower().strip(".,") not in GENERIC_WORDS) \
                        and _next_word(t, m.end()) not in PLACE_WORDS:
                    take(m.span(), name)
    for m in _NAME_KEYWORD.finditer(t):
        s, e = m.span()
        words = _strip_leading(m.group(1).split())
        while words and words[-1].lower() in ("of", "the", "and", "&", "de", "del", "ng"):
            words = words[:-1]
        if not words or _generic_only(words) or all(w.lower().strip(".,") in GENERIC_WORDS | DYCI_TOWN_WORDS for w in words):
            continue
        name = " ".join(words + [" ".join(m.group(2).split())])
        start = e - len(name) if t[s:e].endswith(name) else s
        if any(start < ce and cs < e for cs, ce in covered) or _is_dyci(name) or _next_word(t, e) in PLACE_WORDS:
            continue
        take((start, e), name)
    for m in _KEYWORD_OF.finditer(t):
        s, e = m.span()
        if any(s < ce and cs < e for cs, ce in covered):
            continue
        name = " ".join(m.group(0).split())
        if _is_dyci(name) or _next_word(t, e) in PLACE_WORDS:
            continue
        take((s, e), name)
    out: List[str] = []
    for _, name in sorted(found):
        if _key(name) not in {_key(o) for o in out}:
            out.append(name)
    return out


def _distinctive(name: str) -> List[str]:
    """The name's distinctive part: from its first to its last non-generic word, keyword removed ("Mater Dei Academy" ->
    "Mater Dei", "Philippine Science High School" -> "Philippine"). Listed schools use their aliases/short forms instead."""
    if name in SHORT_FORMS or any(name == n for n, _, _ in SCHOOLS):
        return []
    words = re.sub(r"\s+(High\s+School|School|Academy|Colleges?|University|Institute)$", "", _norm(name), flags=re.I).split()
    idx = [i for i, w in enumerate(words) if w.lower().strip(".,") not in GENERIC_WORDS]
    return words[idx[0]: idx[-1] + 1] if idx else []


def _contains(low_text: str, phrase: str) -> bool:
    return re.search(r"(?<![\w.])" + re.escape(phrase.lower()) + r"(?!\w)", low_text) is not None


def mentions(post_text: str, name: str) -> bool:
    """The post names the institution (see module docstring). Uncertain partial matches count as mentioned."""
    if _key(name) in {_key(d) for d in detect(post_text)}:
        return True
    low = " ".join(_norm(post_text).lower().split())
    if any(_contains(low, s) for s in SHORT_FORMS.get(name, ())):
        return True
    core = _distinctive(name)
    if core:
        if len(core) == 1 and len(core[0]) <= 3:
            return True                                   # too short to tell: uncertain, so treated as mentioned
        return _contains(low, " ".join(core))
    return False


def _join(names: List[str]) -> str:
    owned = [n + "'s" for n in names]
    return owned[0] if len(owned) == 1 else ", ".join(owned[:-1]) + " or " + owned[-1]


def notice(message: str, posts: Iterable = ()) -> Optional[dict]:
    """{"schools": [...], "text": "..."} for the institutions the message names that no shown post mentions, else None."""
    named = detect(message)
    if not named:
        return None
    shown = [p for p in posts if p is not None]
    schools = [s for s in named if not any(mentions(p.text or "", s) for p in shown)]
    if not schools:
        return None
    return {"schools": schools, "text": "This checker only covers DYCI announcements. The posts shown are DYCI’s, "
                                        f"not {_join(schools)}."}
