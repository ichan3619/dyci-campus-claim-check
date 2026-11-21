"""
TaglishNormalizer -- lexicon-driven normalization of informal Taglish / Tagalog / English claims.

WHERE THIS SITS
    Sprint-1 module. It is a NEW module, deliberately separate from
    scripts/m12_retrieval/m12_3_normalizer.py (Option B). Option B is frozen: it is hashed into the M12.3
    result files and its design rule is "never add a content word". This module does substitute words
    (hnd -> hindi, wp -> walang pasok), so it is a different experimental condition ("Option D") and must be
    reported as one. It is stdlib-only and not wired into /verify/text.

DESIGN RULES (carried over from M12.3, adapted)
    1. The original input is never modified. normalize() returns BOTH strings. If no edit applies,
       normalized is the original string, byte for byte.
    2. Every edit is a lexicon lookup or an elongation collapse validated against a vocabulary. There is no
       free-form rewriting, no translation, no model.
    3. Ambiguous surfaces (tier 0 in the lexicon) are NEVER rewritten; they are flagged.
    4. Context-dependent entries (tier 2) fire only when their condition holds (an adjacent number, or a
       nearby campus-context word). Otherwise they are left alone and flagged 'unresolved_context:<surface>'.
    5. Two invariants are checked mechanically on every output; if either fails, the edits are discarded and
       the input is returned unchanged with a 'refused:<invariant>' flag:
         numbers_changed             the sequence of purely numeric tokens differs
         negation_count_changed      the number of negator tokens differs (an entry may declare
                                     inherent_negation when its surface already carries one, e.g. wp)
    6. Every edit records its rule, tier and the lexicon entry status, so results are auditable and the
       lexicon can be hashed and frozen (Lexicon.sha256) before an independent evaluation.

WHAT THIS IS NOT
    The seed lexicon (data/taglish_seed_lexicon.csv) is an UNREVIEWED starting point, not a validated
    resource. Do not report results from it as thesis findings until a Tagalog speaker has reviewed every row
    and the file's SHA-256 has been recorded. Language profiling here is a closed-class word-count heuristic,
    not a language identifier.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional

NORMALIZER_VERSION = "taglish-v0.2-seed"

TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)

DEFAULT_LEXICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "taglish_seed_lexicon.csv")

# Closed-class words only. Copied from the M12.3 gate lexicon so behaviour matches; consolidate into one
# shared module when the gate is promoted out of scripts/. Words that are also common English words
# (at, may, no, to, pa, man, para, o, so, si, di) are excluded on purpose.
TAGALOG_FUNCTION_WORDS: FrozenSet[str] = frozenset((
    "ang ng sa mga ni kay nina sina kina na ay nang ba bang po daw raw lang lamang din rin naman pala nga "
    "kasi kaya yata muna ako ikaw ka siya kami tayo kayo sila ko mo niya natin namin ninyo nila akin iyo amin "
    "atin kanila ito iyan iyon yan yon yun yung ung dito diyan doon dun dyan wala walang hindi huwag wag "
    "mayroon mayroong meron merong dahil kung pero kapag pag"
).split())

ENGLISH_FUNCTION_WORDS: FrozenSet[str] = frozenset((
    "the a an is are was were will be been of and or for in on by with from this that these those it not"
).split())

# Common Tagalog CONTENT words (days/time words, school words, question words, a few modals). SEED, UNREVIEWED.
# Used only as language evidence for the code_switched feature -- never to rewrite text. Words that are also English
# words are left out on purpose. Extend it with a real Tagalog wordlist when one is available.
TAGALOG_COMMON_WORDS: FrozenSet[str] = frozenset((
    "bukas ngayon kahapon pasok klase bagyo hanggang araw gabi umaga hapon tanghali linggo buwan taon paaralan guro "
    "estudyante aral sarado bayad pwede puwede sige salamat paano kailan saan bakit sino ano kailangan dapat madaling "
    "talaga lahat bawal nasaan opo oo"
).split())

# Negators in both surface and canonical form, so hnd -> hindi leaves the count unchanged.
NEGATORS: FrozenSet[str] = frozenset("hindi hnd nde wala wla walang wlang huwag wag not no never".split())

SHORT_CLAIM_MAX_CONTENT_TOKENS = 3


# --------------------------------------------------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LexiconEntry:
    surface: str
    canonical: str
    tier: int                         # 1 = unconditional variant, 2 = conditional (see context fields)
    context_any: FrozenSet[str]       # tier 2: fire only if one of these appears within the context window
    needs_adjacent_number: bool       # tier 2: fire only if the previous or next token is a number
    status: str                       # SEED_UNREVIEWED | REVIEWED
    note: str
    inherent_negation: int = 0        # negators already carried by the SURFACE (wp = walang pasok -> 1)


class Lexicon:
    def __init__(self, entries: Dict[str, LexiconEntry], ambiguous: Dict[str, str], sha256: str, source: str):
        self.entries = entries
        self.ambiguous = ambiguous    # surface -> reason; these are never rewritten
        self.sha256 = sha256
        self.source = source

    @classmethod
    def from_rows(cls, rows: Iterable[dict], sha256: str = "in-memory", source: str = "in-memory") -> "Lexicon":
        entries: Dict[str, LexiconEntry] = {}
        ambiguous: Dict[str, str] = {}
        for n, r in enumerate(rows, start=2):
            surface = r["surface"].strip().lower()
            tier = int(r["tier"])
            if not surface or tier not in (0, 1, 2):
                raise ValueError(f"lexicon row {n}: bad surface/tier {r!r}")
            if surface in entries or surface in ambiguous:
                raise ValueError(f"lexicon row {n}: duplicate surface {surface!r}")
            if tier == 0:
                ambiguous[surface] = r.get("note", "")
                continue
            canonical = r["canonical"].strip()
            if not canonical:
                raise ValueError(f"lexicon row {n}: empty canonical for {surface!r}")
            ctx = frozenset(w for w in (r.get("context_any") or "").lower().split("|") if w)
            adj = str(r.get("needs_adjacent_number", "0")).strip() in ("1", "true", "True")
            if tier == 2 and not ctx and not adj:
                raise ValueError(f"lexicon row {n}: tier-2 entry {surface!r} has no condition")
            if tier == 1 and (ctx or adj):
                raise ValueError(f"lexicon row {n}: tier-1 entry {surface!r} must be unconditional")
            entries[surface] = LexiconEntry(surface, canonical, tier, ctx, adj, r.get("status", "SEED_UNREVIEWED"),
                                            r.get("note", ""), int(r.get("inherent_negation") or 0))
        return cls(entries, ambiguous, sha256, source)

    @classmethod
    def from_csv(cls, path: str = DEFAULT_LEXICON_PATH) -> "Lexicon":
        with open(path, "rb") as fh:
            raw = fh.read()
        rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
        return cls.from_rows(rows, sha256=hashlib.sha256(raw).hexdigest(), source=path)

    def vocabulary(self) -> FrozenSet[str]:
        words = set(self.entries)
        for e in self.entries.values():
            words.update(e.canonical.lower().split())
        return frozenset(words)


# --------------------------------------------------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Edit:
    start: int                        # span in NormalizationResult.cleaned (NFKC + whitespace-collapsed original)
    end: int
    original: str
    replacement: str
    rule: str                         # 'lex:<surface>' | 'elongation' | 'elongation+lex:<surface>'
    tier: int                         # 0 for a pure elongation collapse
    entry_status: str


@dataclass
class NormalizationResult:
    original: str                     # exactly what the user typed, never modified
    normalized: str                   # == original when no edit applied
    cleaned: str                      # NFKC / zero-width / whitespace cleanup; spans in `edits` index into this
    edits: List[Edit] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    features: Dict[str, object] = field(default_factory=dict)
    version: str = NORMALIZER_VERSION
    lexicon_sha256: str = ""

    @property
    def changed(self) -> bool:
        return self.normalized != self.original

    def to_dict(self) -> dict:
        return {
            "original_user_input": self.original,
            "normalized_claim": self.normalized,
            "changed": self.changed,
            "edits": [e.__dict__ for e in self.edits],
            "flags": list(self.flags),
            "features": dict(self.features),
            "normalizer_version": self.version,
            "lexicon_sha256": self.lexicon_sha256,
        }


# --------------------------------------------------------------------------------------------------------------------
# Normalizer
# --------------------------------------------------------------------------------------------------------------------
class TaglishNormalizer:
    def __init__(self, lexicon: Optional[Lexicon] = None, extra_vocab: Iterable[str] = (),
                 allow_unreviewed: bool = True, context_window: int = 4, english_vocab: Iterable[str] = ()):
        """
        extra_vocab: words that validate an elongation collapse ('bukaaas' -> 'bukas'). Pass the token set of the
            EVIDENCE CORPUS here, not of any evaluation set, so the vocabulary is not tuned to test inputs.
        english_vocab: words that count as ENGLISH evidence for the code_switched feature, typically the token set of the
            English evidence corpus (alphabetic, >= 3 letters are kept). Without it only English function words count, and
            a Taglish claim whose English part is all content words ("suspended ba ang klase") is NOT detected as mixed.
            Tagalog evidence always wins: a word in the Tagalog sets is never English evidence, even if the corpus has it.
        allow_unreviewed: set False for any frozen / reported run; entries still marked SEED_UNREVIEWED are then
            skipped and flagged instead of applied.
        """
        self.lexicon = lexicon or Lexicon.from_csv()
        self.allow_unreviewed = allow_unreviewed
        self.context_window = context_window
        self._vocab = (self.lexicon.vocabulary() | TAGALOG_FUNCTION_WORDS
                       | frozenset(w.lower() for w in extra_vocab))
        self._tl_evidence = TAGALOG_FUNCTION_WORDS | TAGALOG_COMMON_WORDS
        eng = frozenset(w.lower() for w in english_vocab if w.isalpha() and len(w) >= 3)
        self._en_evidence = (ENGLISH_FUNCTION_WORDS | eng) - self._tl_evidence - frozenset(self.lexicon.ambiguous)

    # -- public API --------------------------------------------------------------------------------------------
    def normalize(self, text: str) -> NormalizationResult:
        cleaned = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        matches = list(TOKEN_RE.finditer(cleaned))
        lowers = [m.group().lower() for m in matches]
        resolved = list(lowers)           # tokens after elongation resolution, before lexicon substitution
        edits: List[Edit] = []
        flags: List[str] = []
        inherent_neg = 0                  # negators hidden inside abbreviations that were expanded

        for i, m in enumerate(matches):
            tok, low = m.group(), lowers[i]
            if tok.isdigit():
                continue
            key, elong = low, False
            if low not in self.lexicon.entries and low not in self.lexicon.ambiguous and self._has_run(low):
                cand = self._resolve_elongation(low)
                if cand is not None:
                    key, elong = cand, True
                    resolved[i] = cand
            if key in self.lexicon.ambiguous:
                flags.append(f"ambiguous:{key}")
                resolved[i] = low
                continue
            entry = self.lexicon.entries.get(key)
            replacement: Optional[str] = None
            rule, tier, status = "", 0, ""
            if entry is not None:
                if entry.status != "REVIEWED" and not self.allow_unreviewed:
                    flags.append(f"unreviewed_skipped:{key}")
                elif not self._conditions_met(entry, i, lowers):
                    flags.append(f"unresolved_context:{key}")
                else:
                    replacement = entry.canonical
                    rule = ("elongation+lex:" if elong else "lex:") + key
                    tier, status = entry.tier, entry.status
                    inherent_neg += entry.inherent_negation
            if replacement is None and elong:
                replacement, rule, tier, status = key, "elongation", 0, "n/a"
            if replacement is None:
                continue
            edits.append(Edit(m.start(), m.end(), tok, _match_case(tok, replacement), rule, tier, status))

        features = self._features(lowers)
        if not edits:
            return self._result(text, text, cleaned, [], flags, features)

        normalized = _splice(cleaned, edits)
        problems = self._invariant_problems(lowers, resolved, normalized, inherent_neg)
        if problems:
            flags.extend(f"refused:{p}" for p in problems)
            return self._result(text, text, cleaned, [], flags, features)
        return self._result(text, normalized, cleaned, edits, flags, features)

    def variants(self, text: str) -> List[str]:
        """[original] or [original, normalized]. For variant fusion at retrieval time (query both, merge ranks)."""
        r = self.normalize(text)
        return [r.original, r.normalized] if r.changed else [r.original]

    # -- internals ---------------------------------------------------------------------------------------------
    def _result(self, original, normalized, cleaned, edits, flags, features) -> NormalizationResult:
        return NormalizationResult(original=original, normalized=normalized, cleaned=cleaned, edits=edits,
                                   flags=flags, features=features, lexicon_sha256=self.lexicon.sha256)

    @staticmethod
    def _has_run(low: str) -> bool:
        return re.search(r"([^\W\d_])\1{2,}", low) is not None

    def _resolve_elongation(self, low: str) -> Optional[str]:
        groups = [(ch, len(list(g))) for ch, g in itertools.groupby(low)]
        if sum(1 for ch, n in groups if n >= 3 and not ch.isdigit()) > 3:
            return None
        options = [[ch * n] if (n < 3 or ch.isdigit()) else [ch, ch * 2] for ch, n in groups]
        for combo in itertools.product(*options):
            cand = "".join(combo)
            if cand != low and cand in self._vocab:
                return cand
        return None

    def _conditions_met(self, e: LexiconEntry, i: int, lowers: List[str]) -> bool:
        n = len(lowers)
        if e.needs_adjacent_number:
            nxt = i + 1 < n and lowers[i + 1].isdigit()
            prv = i > 0 and lowers[i - 1].isdigit()
            if not (nxt or prv):
                return False
        if e.context_any:
            lo, hi = max(0, i - self.context_window), min(n, i + self.context_window + 1)
            if not any(lowers[j] in e.context_any for j in range(lo, hi) if j != i):
                return False
        return True

    @staticmethod
    def _invariant_problems(before: List[str], resolved: List[str], normalized: str, inherent_neg: int = 0) -> List[str]:
        after = [m.group().lower() for m in TOKEN_RE.finditer(normalized)]
        problems = []
        if [t for t in before if t.isdigit()] != [t for t in after if t.isdigit()]:
            problems.append("numbers_changed")
        if sum(t in NEGATORS for t in resolved) + inherent_neg != sum(t in NEGATORS for t in after):
            problems.append("negation_count_changed")
        return problems

    def _features(self, lowers: List[str]) -> Dict[str, object]:
        tl_f = sum(t in TAGALOG_FUNCTION_WORDS for t in lowers)
        en_f = sum(t in ENGLISH_FUNCTION_WORDS for t in lowers)
        tl = sum(t in self._tl_evidence for t in lowers)
        en = sum(t in self._en_evidence for t in lowers)
        content = [t for t in lowers if not t.isdigit() and t not in TAGALOG_FUNCTION_WORDS
                   and t not in ENGLISH_FUNCTION_WORDS]
        return {
            "token_count": len(lowers),
            "content_token_count": len(content),
            "has_number": any(t.isdigit() for t in lowers),
            "tagalog_function_hits": tl_f,
            "english_function_hits": en_f,
            "tagalog_evidence_hits": tl,
            "english_evidence_hits": en,
            # Heuristic, not a language identifier: >= 1 Tagalog-evidence token AND >= 1 English-evidence token.
            "code_switched": tl >= 1 and en >= 1,
            "short_claim": len(content) <= SHORT_CLAIM_MAX_CONTENT_TOKENS,
        }


def _match_case(original: str, replacement: str) -> str:
    if original[:1].isupper() and not original.isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _splice(text: str, edits: List[Edit]) -> str:
    out, pos = [], 0
    for e in sorted(edits, key=lambda e: e.start):
        out.append(text[pos:e.start])
        out.append(e.replacement)
        pos = e.end
    out.append(text[pos:])
    return "".join(out)
