"""
English yes/no question -> statement (round 2; opt-in via CAMPUS_VERIFY_EN_QUESTION_REWRITE, default off).

WHY
    The NLI models judge a statement against a passage; on an English yes/no question they return neutral even when the
    judged passage contains the answer (gate trace: Y01-Y03). The frozen Option B rule (R4) inverts only one- or
    two-word subjects ('Are classes suspended...?', 'Is there...?') and refuses the rest. This step handles longer
    subjects, by reordering words only: nothing is added, removed or replaced.

WHEN IT RUNS (the API decides): only for a claim that the verdict policy treats as English, and only when Option B left
    the text unchanged. It never touches statements, Tagalog/Taglish messages or questions read as INFO_SEEKING.

THE RULE (fixed before evaluation; closed lists below)
    input   one sentence ending in '?', starting with is/are/was/were/has/have/will/can; no wh-word; no negation
    is/are/was/were/has/have: the subject is the 1-5 words before the first word that starts a predicate: a preposition,
            be/been/being/still/also/already/now/again, a state word, a participle (-ed, 5+ letters), or a number/date
    will/can: the subject is the 1-5 words before the first base verb from MODAL_VERBS
    output  subject + auxiliary + rest, '?' -> '.', first letter re-cased ('Will classes on September 28 end at 1 PM?'
            -> 'Classes on September 28 will end at 1 PM.')
    Anything else is refused (returned unchanged) with the reason, so a refusal is today's behaviour.
"""
import hashlib
import json
import re
from typing import List, Optional, Tuple

VERSION = "en-question-v1"
COPULA_AUX = frozenset("is are was were has have".split())
MODAL_AUX = frozenset("will can".split())
WH_WORDS = frozenset("what when where who whom whose which why how".split())
NEGATIONS = frozenset("not no never isn't aren't wasn't weren't hasn't haven't won't can't cannot".split())
DETERMINERS = frozenset("the a an this that these those its their our".split())
PREPOSITIONS = frozenset("in on at until from for by to within during before after near inside outside".split())
PREDICATE_WORDS = frozenset("be been being still also already now again".split())
STATE_WORDS = frozenset("""open closed suspended cancelled canceled postponed scheduled held free online offline required
allowed available mandatory invited signed announced extended moved rescheduled""".split())
NOT_PARTICIPLES = frozenset("hundred need indeed speed seed bed red sacred naked wicked kindred".split())
MONTHS = frozenset("""january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec""".split())
MODAL_VERBS = frozenset("""be end start begin resume open close continue hold shift have run take push return remain stay
happen move last finish""".split())
MAX_SUBJECT = 5
RULES_SHA256 = hashlib.sha256(json.dumps({k: sorted(v) for k, v in {
    "copula": COPULA_AUX, "modal": MODAL_AUX, "wh": WH_WORDS, "neg": NEGATIONS, "det": DETERMINERS, "prep": PREPOSITIONS,
    "pred": PREDICATE_WORDS, "state": STATE_WORDS, "not_part": NOT_PARTICIPLES, "months": MONTHS, "verbs": MODAL_VERBS}.items()},
    sort_keys=True).encode("utf-8")).hexdigest()


def _bare(token: str) -> str:
    return re.sub(r"^[^\w]+|[^\w%]+$", "", token.lower().replace("’", "'"))


def _starts_predicate(token: str) -> bool:
    b = _bare(token)
    return (b in PREPOSITIONS or b in PREDICATE_WORDS or b in STATE_WORDS or b in MONTHS
            or (len(b) > 4 and b.endswith("ed") and b not in NOT_PARTICIPLES) or bool(re.search(r"\d", b)))


def rewrite_english_question(text: str) -> Tuple[str, Optional[str]]:
    """(statement, None) when rewritten; (text unchanged, refusal reason) otherwise."""
    s = (text or "").strip()
    if not s.endswith("?") or re.search(r"[.?!]\s", s[:-1]):
        return text, "not_a_single_question"
    tokens: List[str] = s[:-1].rstrip("?").split()
    if len(tokens) < 3:
        return text, "too_short"
    bares = [_bare(t) for t in tokens]
    aux = bares[0]
    if aux not in COPULA_AUX and aux not in MODAL_AUX:
        return text, "auxiliary_not_handled"
    if any(b in WH_WORDS for b in bares) or any(b in NEGATIONS or b.endswith("n't") for b in bares):
        return text, "wh_or_negative_question"
    test = (lambda i: bares[i] in MODAL_VERBS) if aux in MODAL_AUX else (lambda i: _starts_predicate(tokens[i]))
    boundary = next((i for i in range(2, min(len(tokens), MAX_SUBJECT + 2)) if test(i)), None)
    if boundary is None:
        return text, "no_subject_boundary"
    subject, rest = tokens[1:boundary], tokens[boundary:]
    if not subject or _bare(subject[-1]) in DETERMINERS or _bare(subject[0]) in PREPOSITIONS or not rest:
        return text, "no_subject_boundary"
    first = subject[0]
    subject = [first[:1].upper() + first[1:]] + subject[1:]
    return " ".join(subject + [tokens[0].lower()] + rest) + ".", None
