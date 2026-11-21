"""
ClaimRouter -- decides, BEFORE verification, whether a user's raw input is a CHECKABLE_CLAIM or INFO_SEEKING.

SPECIFICATION (decisions D1-D3 in docs/12_3/m12_3_decisions.md)
    D1  The frozen category rule in scripts/m12_retrieval/m12_3_case_classification.json (`_readme`) is the spec:
          INFO_SEEKING     asks for an unknown value (when, what time, where, how much, who, what are the
                           dates/schedule/deadline) and states no proposition whose truth could be judged.
          CHECKABLE_CLAIM  states, or asks a yes/no question about, a specific proposition: statements, yes/no
                           questions (including Tagalog 'ba'), and fragments or keyword phrases that name an event
                           AND the attribute to confirm (a day, time, date, cause or value).
          A bare topic or schedule word without a stated value is INFO_SEEKING.
    D2  The router reads the user's RAW text. It does not import or depend on src/normalization.
    D3  Exactly two outputs. Input with no identifiable claim ('po', '???') is INFO_SEEKING.

RULES (first match wins; each is a closed word list or a fixed pattern, all listed below and hashed into
ROUTER_CONFIG_SHA256 so a frozen router can be identified exactly)
    1. no content word left after removing particles, function words,
       auxiliaries and negators ('po', '???', 'wala po', 'is it?')            -> INFO_SEEKING  no_identifiable_claim
    2. a wh-word, English or Tagalog (what, when, kailan, saan, magkano ...)  -> INFO_SEEKING  wh_question
    3. an information-request frame with no embedded yes/no proposition
       ('can you tell me', 'patanong', 'pakisabi' ...)                        -> INFO_SEEKING  info_request
       A check frame ('is it true that', 'totoo ba na', 'can you confirm if') or a request frame followed by
       if / whether / kung is stripped, and the REST of the input is routed by rules 1 and 4-6.
    4. an assertion marker: English auxiliary/copula/modal, Tagalog predicate word (ay, may, wala, hindi,
       pwede ...), or 'no'/'not' before a word ('no classes'). The particle 'ba' turns a statement into a
       yes/no question but is not itself a proposition: 'enrollment ba?' is still a bare topic.
                                                                             -> CHECKABLE_CLAIM  assertion_marker
    4b. (v1.1) an English main verb with a subject: a regular past tense ('-ed'), an irregular past tense or a
       present-tense announcement verb from closed lists, DIRECTLY after a content word (the subject) and with a
       content word somewhere after it ('the delegation visited SPUS', 'the personnel attended a seminar'). The
       subject requirement keeps participles inside noun phrases out: 'list of passed students' (after 'of'),
       'the updated schedule' (after 'the'), 'required documents' (nothing before) stay bare topics.
                                                                             -> CHECKABLE_CLAIM  main_verb
    5. a stated value: day word, month, a number (not an ordinal like '2nd'), clock time, cause word,
       or a state word (suspended, cancelled, open, sarado ...)               -> CHECKABLE_CLAIM  stated_value
    6. otherwise (a bare topic / schedule word)                               -> INFO_SEEKING  no_stated_value

    When rules 2-3 and an assertion both appear, INFO_SEEKING wins. That direction is deliberate: the primary
    acceptance criterion is zero gold-INFO_SEEKING inputs sent to verification.

WHAT THIS IS NOT
    A parser or a language identifier. Known gaps, stated so they can be measured rather than hidden:
      * v1.1 recognises an English main verb only from its '-ed' form or the closed lists below, and only with a
        subject word directly before it; other plain verbs ('the registrar issues IDs') are still routed
        INFO_SEEKING;
      * Tagalog verbs are not recognised by their affixes ('lilipat ang registrar'), for the same reason;
      * a leading filler 'ano' ('ano, may pasok ba?') is read as the wh-word 'ano'.
    Routing accuracy may only be reported from the frozen independent set (m12_3_router_eval.py --set independent).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

ROUTER_VERSION = "router-v1.1-rules"

CHECKABLE_CLAIM = "CHECKABLE_CLAIM"
INFO_SEEKING = "INFO_SEEKING"
ROUTE_STATUSES: Tuple[str, ...] = (CHECKABLE_CLAIM, INFO_SEEKING)

# Letters/digits, with internal apostrophes, slashes, colons, hyphens and dots kept ("what's", "24/7", "1:00",
# "mag-aral"). Lower-cased before matching.
TOKEN_RE = re.compile(r"[^\W_]+(?:['’/:.\-][^\W_]+)*", re.UNICODE)


def _words(s: str) -> FrozenSet[str]:
    return frozenset(s.split())


# ---- rule 1: words that carry no claim on their own ----------------------------------------------------------------
FILLER_WORDS = _words(
    "po opo ho oho ba bang lang lamang naman nga daw raw pala kaya yata muna din rin ha huh eh e uy oy hoy "
    "hello hi hey pls plz please thanks thank salamat ty sir maam ma'am guys pre bro sis hehe haha lol ok okay "
    "uhm um hmm ah oh yes yeah oo"
)
FUNCTION_WORDS = _words(
    # English
    "the a an of to in on at for by with from and or but this that these those it its i me my we our you your "
    "u he she they them their there here about as if so than then also just very"
    # Tagalog
    " ang ng sa mga si ni kay nina sina kina na nang at o pero kung kasi ko mo niya natin namin ninyo nila "
    "ako ikaw ka siya kami tayo kayo sila ito iyan iyon yan yon yun yung ung dito diyan doon dun"
)

# ---- rule 2: wh-words (unknown-value questions) -----------------------------------------------------------------------
WH_WORDS = _words(
    "what whats what's when whens when's where wheres where's who whos who's whom whose which why how hows how's "
    "ano anong anu anung kailan kelan kilan saan nasaan asan nasan sino sinong bakit bkt paano pano panu "
    "magkano ilan ilang gaano alin aling kanino"
)

# ---- rule 3: frames ---------------------------------------------------------------------------------------------------
_YOU = r"(?:you|u|po)"
_PLEASE = r"(?:(?:please|pls|plz|po)\s+)?"
# Asking to be TOLD something. Without if/whether/kung after it, no proposition is stated.
INFO_REQUEST_PATTERNS: Tuple[str, ...] = (
    rf"\b(?:can|could|would|will)\s+{_YOU}\s+{_PLEASE}(?:tell|send|give|share|post|show|explain|update|inform|help)\b",
    r"\btell\s+me\b", r"\blet\s+me\s+know\b", r"\b(?:do|does)\s+(?:you|anyone|anybody)\s+know\b",
    r"\banyone\s+know\b", r"\b(?:may|can)\s+i\s+ask\b", r"\bpa\s?-?ask\b", r"\bpatanong\b", r"\bpakisabi\b",
    r"\bpasabi\b", r"\bpakibigay\b", r"\bneed\s+(?:the\s+)?(?:info|information|details)\b",
    r"\b(?:info|information|details|updates?)\s+(?:on|about|sa|regarding)\b",
)
# Asking to CHECK something. The rest of the input is the proposition (or nothing, e.g. 'totoo ba?').
CHECK_FRAME_PATTERNS: Tuple[str, ...] = (
    rf"\b(?:can|could|would|will)\s+{_YOU}\s+{_PLEASE}(?:confirm|check|verify)\b",
    r"\bis\s+it\s+true\b", r"\b(?:totoo|true|legit|confirmed)\s+(?:ba|bang)\b", r"\bconfirm\s+ko\s+lang\b",
)
EMBEDDED_PROPOSITION_WORDS = _words("if whether kung that na")

# ---- rule 4: assertion markers ----------------------------------------------------------------------------------------
ASSERTION_MARKERS = _words(
    # English auxiliaries / copulas / modals and their negative contractions
    "is are was were am be been being will would shall should can could may might must do does did has have had "
    "isn't aren't wasn't weren't won't wont can't cant couldn't shouldn't doesn't don't didn't hasn't haven't "
    "isnt arent wasnt werent doesnt dont didnt hasnt havent not"
    # Tagalog predicate words. The yes/no particle 'ba' is NOT here: it only marks a question, so the proposition
    # must come from the rest of the input ('half day ba bukas?' -> 'bukas'), and 'enrollment ba?' stays a bare topic.
    " ay mayroon meron mayroong merong wala walang wla wlang hindi di hnd pwede puwede pede pwd bawal kailangan dapat"
)
NEGATOR_BEFORE_WORD = _words("no")          # 'no classes tomorrow': 'no' + a following word asserts absence

# ---- rule 4b (v1.1): English main verbs -------------------------------------------------------------------------------
PAST_ED_RE = re.compile(r"^[a-z]{3,}ed$")
NOT_PAST_ED = _words("hundred need indeed speed seed bed feed shed red bred sacred naked wicked kindred breed creed "
                     "steed weed exceed proceed succeed")
IRREGULAR_PAST = _words(
    "came went held gave took made met left began won got brought sent led paid spoke taught became chose saw said "
    "told ran wrote kept lost built bought caught fell grew knew sold spent stood understood wore flew drove ate "
    "drank sang read heard found felt thought"
)
PRESENT_VERBS = _words(
    "starts begins ends opens closes resumes releases requires includes offers holds runs continues hosts visits "
    "announces covers needs gives provides allows accepts"
)

# ---- rule 5: stated values --------------------------------------------------------------------------------------------
DAY_WORDS = _words(
    "today tonight tomorrow tom tmrw tmr yesterday weekend monday tuesday wednesday thursday friday saturday sunday "
    "mon tue tues wed thu thur thurs fri sat "
    "bukas ngayon mamaya kahapon kanina lunes martes miyerkules miyerkoles huwebes biyernes sabado linggo"
)
MONTH_WORDS = _words(
    # 'may' is left out on purpose: it is already an assertion marker (English modal, Tagalog existential).
    "january february march april june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec "
    "enero pebrero marso abril mayo hunyo hulyo agosto setyembre oktubre nobyembre disyembre"
)
TIME_WORDS = _words("am pm a.m p.m noon midnight tanghali madaling-araw")
CAUSE_WORDS = _words("due because dahil typhoon storm bagyo rain ulan flood flooding baha earthquake lindol")
STATE_WORDS = _words(
    "suspended suspension cancelled canceled cancellation postponed rescheduled resched moved closed open opened "
    "extended resumed suspendido kanselado sarado"
)
ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$")
DIGIT_RE = re.compile(r"\d")

GUIDANCE: Dict[str, str] = {
    "no_identifiable_claim": "There is no claim to check in this input. Type what you heard, for example "
                             "'walang pasok bukas' or 'Classes are suspended tomorrow.'",
    "wh_question": "This asks for information rather than stating a claim, so no truth verdict is given. "
                   "The most relevant campus notices are shown instead.",
    "info_request": "This asks for information rather than stating a claim, so no truth verdict is given. "
                    "The most relevant campus notices are shown instead.",
    "no_stated_value": "This names a topic but states nothing to check, so no truth verdict is given. The most "
                       "relevant campus notices are shown instead. To check something, add the detail you heard "
                       "(a day, date, time or what happened).",
}

TAGALOG_MARKERS = _words(
    "ang ng sa mga ba bang po na ay wala walang hindi pwede puwede dahil kasi naman lang daw raw yung bukas ngayon "
    "pasok klase kailan saan sino magkano ano anong paano bakit"
)


def _config_record() -> Dict[str, object]:
    return {
        "version": ROUTER_VERSION,
        "filler": sorted(FILLER_WORDS), "function": sorted(FUNCTION_WORDS), "wh": sorted(WH_WORDS),
        "info_request": list(INFO_REQUEST_PATTERNS), "check_frame": list(CHECK_FRAME_PATTERNS),
        "embedded": sorted(EMBEDDED_PROPOSITION_WORDS), "assertion": sorted(ASSERTION_MARKERS),
        "negator": sorted(NEGATOR_BEFORE_WORD), "not_past_ed": sorted(NOT_PAST_ED), "past_ed": PAST_ED_RE.pattern,
        "irregular_past": sorted(IRREGULAR_PAST), "present_verbs": sorted(PRESENT_VERBS),
        "day": sorted(DAY_WORDS), "month": sorted(MONTH_WORDS),
        "time": sorted(TIME_WORDS), "cause": sorted(CAUSE_WORDS), "state": sorted(STATE_WORDS),
        "ordinal": ORDINAL_RE.pattern, "token": TOKEN_RE.pattern,
    }


# Identifies this exact rule set. Record it when the router is frozen (validation_protocol.md).
ROUTER_CONFIG_SHA256 = hashlib.sha256(json.dumps(_config_record(), sort_keys=True).encode("utf-8")).hexdigest()

WORD_LIST_SIZES: Dict[str, int] = {k: len(v) for k, v in _config_record().items() if isinstance(v, list)}


@dataclass(frozen=True)
class RouteDecision:
    status: str
    reasons: Tuple[str, ...]
    caveats: Tuple[str, ...]
    guidance: str
    signals: Dict[str, object]
    router_version: str = ROUTER_VERSION

    @property
    def is_checkable(self) -> bool:
        return self.status == CHECKABLE_CLAIM

    def to_dict(self) -> Dict[str, object]:
        """JSON-safe. Keys must stay in sync with RouteInfoModel in src/api/schemas.py (tests enforce it)."""
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "caveats": list(self.caveats),
            "guidance": self.guidance,
            "signals": dict(self.signals),
            "router_version": self.router_version,
        }


ROUTE_DECISION_KEYS: Tuple[str, ...] = ("status", "reasons", "caveats", "guidance", "signals", "router_version")


def tokens(text: str) -> List[str]:
    return [m.group().lower().replace("’", "'").rstrip(".") for m in TOKEN_RE.finditer(text or "")]


def _first_match(patterns, text: str) -> Optional[re.Match]:
    hits = [m for m in (re.search(p, text) for p in patterns) if m]
    return min(hits, key=lambda m: m.start()) if hits else None


def is_tagalog_or_taglish(text: str) -> bool:
    """True if the input contains a Tagalog function word, particle or common word (closed list, not a language ID).
    Used for the 'tagalog_or_taglish' caveat and by LanguageRoutedClassifier to pick the NLI model."""
    return any(t in TAGALOG_MARKERS for t in tokens(text if isinstance(text, str) else ""))


class ClaimRouter:
    """Stateless; `route(raw_text)` never raises on a string and never mutates anything."""

    def route(self, text: str) -> RouteDecision:
        raw = text if isinstance(text, str) else ""
        lowered = " ".join(tokens(raw))
        toks = lowered.split()
        signals: Dict[str, object] = {"token_count": len(toks), "frame": None}
        caveats = ("tagalog_or_taglish",) if is_tagalog_or_taglish(raw) else ()

        if not self._content(toks):
            return self._decide(INFO_SEEKING, "no_identifiable_claim", caveats, signals)
        if any(t in WH_WORDS for t in toks):
            return self._decide(INFO_SEEKING, "wh_question", caveats, signals)

        check = _first_match(CHECK_FRAME_PATTERNS, lowered)
        request = _first_match(INFO_REQUEST_PATTERNS, lowered)
        if check:
            signals["frame"] = "check"
            toks = self._drop_conjunction(lowered[check.end():].split())
        elif request:
            rest = lowered[request.end():].split()
            if not rest or rest[0] not in EMBEDDED_PROPOSITION_WORDS:
                signals["frame"] = "info_request"
                return self._decide(INFO_SEEKING, "info_request", caveats, signals)
            signals["frame"] = "request_with_embedded_proposition"
            toks = self._drop_conjunction(rest)
        if signals["frame"] and not self._content(toks):
            return self._decide(INFO_SEEKING, "no_identifiable_claim", caveats, signals)

        marker = self._assertion(toks)
        value = self._value(toks)
        signals.update(assertion_marker=marker, stated_value=value)
        if marker:
            return self._decide(CHECKABLE_CLAIM, "assertion_marker", caveats, signals)
        verb = self._main_verb(toks)
        if verb:
            signals["main_verb"] = verb
            return self._decide(CHECKABLE_CLAIM, "main_verb", caveats, signals)
        if value:
            return self._decide(CHECKABLE_CLAIM, "stated_value", caveats + ("fragment",), signals)
        return self._decide(INFO_SEEKING, "no_stated_value", caveats, signals)

    # -- internals ---------------------------------------------------------------------------------------------
    @staticmethod
    def _decide(status: str, reason: str, caveats: Tuple[str, ...], signals: Dict[str, object]) -> RouteDecision:
        return RouteDecision(status=status, reasons=(reason,), caveats=tuple(caveats),
                             guidance=GUIDANCE.get(reason, ""), signals=dict(signals))

    @staticmethod
    def _content(toks: List[str]) -> List[str]:
        """Words that can carry a claim: not a particle, function word, auxiliary or negator on its own."""
        return [t for t in toks if t not in FILLER_WORDS and t not in FUNCTION_WORDS
                and t not in ASSERTION_MARKERS and t not in NEGATOR_BEFORE_WORD]

    @staticmethod
    def _drop_conjunction(rest: List[str]) -> List[str]:
        return rest[1:] if rest and rest[0] in EMBEDDED_PROPOSITION_WORDS else rest

    @staticmethod
    def _assertion(toks: List[str]) -> Optional[str]:
        for i, t in enumerate(toks):
            if t in ASSERTION_MARKERS:
                return t
            if t in NEGATOR_BEFORE_WORD and i + 1 < len(toks) and toks[i + 1] not in FILLER_WORDS:
                return t + " " + toks[i + 1]
        return None

    @classmethod
    def _main_verb(cls, toks: List[str]) -> Optional[str]:
        """Rule 4b: an English verb form directly after a subject word, with content after it."""
        is_content = lambda t: bool(cls._content([t]))
        for i in range(1, len(toks) - 1):
            t = toks[i]
            verb = t in IRREGULAR_PAST or t in PRESENT_VERBS or (PAST_ED_RE.match(t) and t not in NOT_PAST_ED)
            if verb and is_content(toks[i - 1]) and toks[i - 1] not in WH_WORDS and any(is_content(x) for x in toks[i + 1:]):
                return t
        return None

    @staticmethod
    def _value(toks: List[str]) -> Optional[str]:
        for i, t in enumerate(toks):
            if t in DAY_WORDS or t in MONTH_WORDS or t in TIME_WORDS or t in CAUSE_WORDS or t in STATE_WORDS:
                return t
            if DIGIT_RE.search(t) and not ORDINAL_RE.match(t):
                return t
            if t == "half" and i + 1 < len(toks) and toks[i + 1] == "day":
                return "half day"
        return None
