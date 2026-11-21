"""
Which claims may receive a truth verdict (CAMPUS_VERIFY_VERDICT_LANGUAGES=english).

withhold_for_language(text) is True for a Tagalog/Taglish claim. It is the router's own check
(src/routing/router.is_tagalog_or_taglish, which also chooses the NLI model and is NOT changed here) OR a wider closed list
used ONLY by this verdict gate: Tagalog function words, pronouns and particles that do not occur in English claims, and
hyphenated Tagalog verb prefixes ('nagka-mass'). Words that are also common English words are left out on purpose
('may', 'don', 'at', 'no', 'man'), so a short Tagalog fragment built only from those ('may free food?') can still pass
as English; that limitation is recorded in docs/12_3/m12_3_decisions.md.
"""
from typing import FrozenSet

from src.routing.router import is_tagalog_or_taglish, tokens

EXTRA_TAGALOG_WORDS: FrozenSet[str] = frozenset("""
nung noong nang yun yon iyon iyan ito dito diyan doon dun dyan
ko mo niya natin namin ninyo nila kami tayo kayo sila siya silang sina nina kay
akala talaga sabi diba pala din rin nga lamang muna yata kaya
ilan ilang sinong saang alin kanino kamusta kumusta
meron mayroon mayroong merong
ngayong kahapon kanina mamaya
kasama lahat naging nagka nagkaroon
""".split())
TAGALOG_PREFIXES: FrozenSet[str] = frozenset("nag mag nagka magka pag ipag pinag maki naka nakaka".split())


def withhold_for_language(text: str) -> bool:
    if is_tagalog_or_taglish(text):
        return True
    for t in tokens(text if isinstance(text, str) else ""):
        if t in EXTRA_TAGALOG_WORDS:
            return True
        if "-" in t and t.split("-")[0] in TAGALOG_PREFIXES:
            return True
    return False
