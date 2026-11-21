"""
M12.3 Option B: deterministic, rule-based claim normalizer.  EVALUATION-SIDE ONLY.

This module is NOT imported by src/ and NOT wired into /verify/text. It exists so the M12.3 raw-vs-normalized experiment can be run
and audited. It has no dependencies beyond the standard library, no model, no translation, no lexicon of Tagalog content words.

Design rules (from the M12.3 brief):
  * original_user_input is never modified; normalize() returns BOTH strings. If no rule applies, normalized == original.
  * A rule may only DELETE question markers, re-attach a Tagalog linker, reorder an English auxiliary, or swap '?' for '.'.
    It never adds a content word. verify_no_content_added() checks this mechanically for every output.
  * When the structure does not give enough information the normalizer REFUSES (returns the input unchanged, with a reason).
  * Wh-questions (when/what/where/kailan/magkano ...) ask for an unknown value; they are never turned into claims.
  * No rule mentions a case id, a whole sentence, or a corpus document. Every rule is a structural pattern.

Rule registry: RULES (documentation + unit-test examples). Refusal guards: GUARDS.
"""
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

NORMALIZER_VERSION = "m12_3-optionB-v1"

# ---------------------------------------------------------------------------------------------------------------------
# Closed word classes. These are grammatical function words, not domain vocabulary.
# ---------------------------------------------------------------------------------------------------------------------
WH_WORDS = frozenset({
    # English
    "who", "whom", "whose", "what", "when", "where", "why", "how", "which",
    "what's", "who's", "when's", "where's", "how's", "whats", "whos",
    # Tagalog / Taglish interrogatives
    "kailan", "kelan", "saan", "sino", "ano", "anong", "bakit", "paano", "pano", "magkano", "ilan", "alin", "kanino", "gaano",
})
COPULA_AUX = frozenset({"is", "are", "was", "were"})
MODAL_AUX = frozenset({"will", "can", "has", "have"})
INVERTIBLE_AUX = COPULA_AUX | MODAL_AUX
# Aux-first questions that cannot be inverted deterministically (do-support needs verb inflection; other modals change certainty/modality).
NON_INVERTIBLE_AUX = frozenset({"do", "does", "did", "should", "would", "could", "may", "might", "must", "shall", "am"})
NEG_CONTRACTIONS = frozenset({"isn't", "aren't", "wasn't", "weren't", "won't", "can't", "hasn't", "haven't", "isnt", "arent", "wont", "cant"})
DETERMINERS = frozenset({"the", "a", "an", "this", "that", "these", "those", "our", "their", "my", "your", "his", "her", "its", "all"})
# Closed class that may follow a one-word subject and show that the predicate has started.
PREDICATE_STARTERS = frozenset({"be", "been", "being", "still", "also", "already", "now", "again", "in", "on", "at", "until", "from", "for", "by", "to", "within"})
# Irregular participles / the adjective 'open' (regular participles are recognised by the suffix -ed). 10 words; reported as a lexicon.
IRREGULAR_PREDICATE_WORDS = frozenset({"open", "held", "given", "taken", "made", "paid", "sent", "done", "kept", "lost"})
VOWELS = "aeiouAEIOU"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    input_pattern: str
    output_transformation: str
    language: str
    examples: Tuple[Tuple[str, str], ...]   # (input, expected normalized claim)
    semantic_risk: str
    observed_in_dev: str                    # honest statement of where in the development data the pattern occurs


RULES: Tuple[Rule, ...] = (
    Rule("R1_terminal_question_mark",
         "Replace a terminal '?' by '.' so that a question-shaped input is presented as a statement. Changes punctuation only.",
         "input ends with '?' and no guard refuses it",
         "final '?' -> '.'",
         "any",
         (("Suspension tomorrow?", "Suspension tomorrow."), ("library 24 hours?", "library 24 hours.")),
         "LOW. No word is touched. By the project's category rule, a yes/no question about a proposition is checked as that proposition, so "
         "the proposition asked is the proposition verified. The speaker's uncertainty is not represented in the claim, by design.",
         "English fragments ending in '?' (3 checkable cases in the 38-case set), the Tagalog fragment, and every case that R2/R3 also touch."),
    Rule("R2_tl_question_particle_ba",
         "Delete the Tagalog yes/no question particle 'ba' (standalone token). All other tokens stay in place and in their original language.",
         "token 'ba' (case-insensitive), not in a wh-question",
         "drop the token 'ba'",
         "Tagalog / Taglish",
         (("may exam ba sa sept 28?", "may exam sa sept 28."), ("pwede na ba kunin ID?", "pwede na kunin ID.")),
         "LOW. 'ba' only marks the sentence as a question; it carries no date, person, place, event, action or quantity.",
         "5 checkable Taglish/Tagalog cases in the 38-case set."),
    Rule("R3_tl_question_particle_bang",
         "Delete 'ba' from the contracted form 'bang' (ba + linker -ng) and re-attach the linker to the preceding word: after a vowel '-ng', after 'n' '-g'. "
         "Refuse if the preceding word ends in another letter (the linker would be 'na', which cannot be derived without a lexicon).",
         "standalone token 'bang' preceded by a word ending in a vowel or 'n'",
         "'<word> bang' -> '<word>ng' (vowel) or '<word>g' (n)",
         "Tagalog / Taglish",
         (("wala bang pasok bukas dahil sa bagyo?", "walang pasok bukas dahil sa bagyo."),),
         "LOW. The linker is grammatical; the negation 'wala' is preserved, so 'wala bang pasok' and 'walang pasok' differ only in being a question.",
         "1 checkable Tagalog case in the 38-case set."),
    Rule("R4_en_aux_inversion",
         "English yes/no question that starts with is/are/was/were/will/can/has/have: move the auxiliary behind the subject. The subject boundary is "
         "accepted only when it is 'there', or a determiner + one word, or one word, AND the next token shows the predicate has started "
         "(a word ending in -ed of 5+ letters, one of 10 irregular participles/'open', or one of a closed list: be/been/still/also/already/now/in/on/at/until/from/for/by/to/within). "
         "Otherwise refuse (guard G4).",
         "aux + subject + predicate, unambiguous boundary",
         "aux S rest -> S aux rest; first letter re-cased; '?' -> '.'",
         "English",
         (("Are classes suspended tomorrow?", "Classes are suspended tomorrow."),
          ("Is there no class tomorrow?", "There is no class tomorrow."),
          ("Will the library be open 24 hours?", "The library will be open 24 hours.")),
         "LOW to MEDIUM. Pure reordering, no word added or removed. The risk is a wrong subject boundary (e.g. a noun ending in -ed such as 'speed' read as a participle); "
         "G4 refuses whenever the next token does not look like the start of a predicate, but the heuristic is not a parser. "
         "'Is there no class tomorrow?' stays a negative existential: it is NOT rewritten to 'classes are suspended' (that would be a lexical equivalence).",
         "NONE: no English inverted question exists in the 38-case development set (its English questions are all fragments). Included because the brief lists the "
         "category; its effect on development data is therefore nil, and it is covered by unit tests only."),
)

GUARDS: Dict[str, str] = {
    "G1_wh_question": "Any wh/interrogative token (when, what, where, who, how many, kailan, kelan, saan, sino, ano, magkano ...): asks for an unknown value, states no proposition. Left unchanged.",
    "G2_multiple_sentences": "More than one sentence terminator inside the input: the proposition boundaries are unclear. Left unchanged.",
    "G3_negative_polar": "Negative contraction in a question (isn't, aren't, won't, can't ...): carries a presupposition whose negation scope cannot be settled deterministically. Left unchanged.",
    "G4_ambiguous_subject_boundary": "English auxiliary-first question whose subject/predicate boundary is not certain (do-support questions, compound subjects, bare adjectives after the subject). Left unchanged.",
    "G5_tl_linker_unresolved": "'bang' after a word ending in a consonant other than 'n' (the linker would be 'na'), or 'bang' without a preceding word. Left unchanged.",
    "G6_too_short": "Fewer than two tokens would remain. Left unchanged.",
    "G7_multiple_particles": "More than one 'ba'/'bang' token. Left unchanged.",
}

# What the rules cannot do, stated so the thesis can cite it (see report).
KNOWN_LIMITS = (
    "Fragment/keyword questions ('library 24 hours?', 'Early dismissal tomorrow?') get punctuation only: a verb would have to be invented.",
    "No Tagalog-to-English lexical mapping (pasok -> classes, madaling araw -> overnight): that is translation (Option C), not Option B.",
    "No resolution of 'you/your' in the premise or of person/reference mismatches (the known second-person premise case): that is premise-side and out of Option B.",
    "Wh-questions are not turned into claims.",
)


@dataclass(frozen=True)
class NormalizationResult:
    original_user_input: str
    normalized_claim: str
    rules_applied: Tuple[str, ...] = ()
    refused: Optional[str] = None          # guard id when a pattern was recognised but the structure was not safe to rewrite
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        return self.normalized_claim != self.original_user_input


def _split_terminal(s: str) -> Tuple[str, str]:
    m = re.match(r"^(.*?)([?.!]*)\s*$", s, flags=re.S)
    return m.group(1).rstrip(), m.group(2)


def _tokens(body: str) -> List[str]:
    return body.split()


def _bare(tok: str) -> str:
    return re.sub(r"^[\W_]+|[\W_]+$", "", tok).lower()


def _has_predicate_start(tok: str) -> bool:
    b = _bare(tok)
    return b in PREDICATE_STARTERS or b in IRREGULAR_PREDICATE_WORDS or (len(b) > 4 and b.endswith("ed"))


def _refuse(text: str, guard: str) -> NormalizationResult:
    return NormalizationResult(text, text, (), guard)


def normalize(text: str, enabled: Optional[frozenset] = None, keep_disabled_particles: bool = False) -> NormalizationResult:
    """Return original + normalized claim. `enabled` = optional set of rule ids (for ablation); default all rules on.

    keep_disabled_particles (EVALUATION ABLATION ONLY, default False = behaviour unchanged): when True and the rule that would delete a
    Tagalog particle ('ba' -> R2, 'bang' -> R3) is disabled, the token is KEPT and the remaining enabled rules (R1) still apply to the input.
    Default behaviour: such an input is returned unchanged, which left 'R1 alone on a ba input' untested (M12.3 handoff, ablation gap)."""
    on = (lambda r: True) if enabled is None else (lambda r: r in enabled)
    original = text
    s = text.strip()
    if not s:
        return NormalizationResult(original, original)
    body, term = _split_terminal(s)
    is_question = term.endswith("?")
    toks = _tokens(body)
    bares = [_bare(t) for t in toks]
    has_particle = any(b in ("ba", "bang") for b in bares)

    # Only question-shaped inputs are touched. A plain statement stays exactly as typed.
    if not (is_question or has_particle):
        return NormalizationResult(original, original)

    if re.search(r"[?.!]\s+\S", body) or ("?" in body) or ("!" in body):
        return _refuse(original, "G2_multiple_sentences")
    if any(b in WH_WORDS for b in bares):
        return _refuse(original, "G1_wh_question")
    if any(b in NEG_CONTRACTIONS for b in bares):
        return _refuse(original, "G3_negative_polar")

    applied: List[str] = []
    out = list(toks)

    # --- Tagalog / Taglish particle rules (R2, R3) ---
    if has_particle:
        if sum(b in ("ba", "bang") for b in bares) > 1:
            return _refuse(original, "G7_multiple_particles")
        i = next(k for k, b in enumerate(bares) if b in ("ba", "bang"))
        if bares[i] == "ba":
            if not on("R2_tl_question_particle_ba"):
                if not keep_disabled_particles:
                    return NormalizationResult(original, original)
                # ablation: keep 'ba' in place; fall through so R1 can still apply
            else:
                del out[i]
                applied.append("R2_tl_question_particle_ba")
        else:
            if not on("R3_tl_question_particle_bang"):
                if not keep_disabled_particles:
                    return NormalizationResult(original, original)
                # ablation: keep 'bang' in place; fall through so R1 can still apply
            else:
                if i == 0:
                    return _refuse(original, "G5_tl_linker_unresolved")
                prev = out[i - 1]
                core = re.sub(r"[\W_]+$", "", prev)
                if not core:
                    return _refuse(original, "G5_tl_linker_unresolved")
                last = core[-1]
                if last in VOWELS:
                    out[i - 1] = core + "ng"
                elif last in "nN":
                    out[i - 1] = core + "g"
                else:
                    return _refuse(original, "G5_tl_linker_unresolved")
                del out[i]
                applied.append("R3_tl_question_particle_bang")
    # --- English auxiliary inversion (R4): only for English, aux-first questions, no Tagalog particle present ---
    elif is_question and bares and bares[0] in NON_INVERTIBLE_AUX:
        return _refuse(original, "G4_ambiguous_subject_boundary")
    elif is_question and bares and bares[0] in INVERTIBLE_AUX and on("R4_en_aux_inversion"):
        aux = out[0]
        rest = out[1:]
        rb = bares[1:]
        if len(rest) < 2:
            return _refuse(original, "G4_ambiguous_subject_boundary")
        if rb[0] == "there":
            subj_n = 1   # existential: the boundary is certain, whatever follows is the predicate
        elif rb[0] in DETERMINERS and len(rest) >= 3:
            subj_n = 2
        elif rb[0] not in DETERMINERS and len(rest) >= 2:
            subj_n = 1
        else:
            return _refuse(original, "G4_ambiguous_subject_boundary")
        nxt = rest[subj_n] if len(rest) > subj_n else None
        if nxt is None or (rb[0] != "there" and not _has_predicate_start(nxt)):
            return _refuse(original, "G4_ambiguous_subject_boundary")
        if bares[0] in MODAL_AUX and bares[0] in ("will", "can") and rb[0] != "there" and _bare(nxt) not in ("be", "been", "being"):
            # modal + bare verb: 'Will classes resume Monday?' cannot be told from 'Will midterm exams resume ...' without a parser
            return _refuse(original, "G4_ambiguous_subject_boundary")
        subj, pred = rest[:subj_n], rest[subj_n:]
        sentence_initial_cap = aux[:1].isupper()
        new_aux = aux.lower() if sentence_initial_cap else aux
        subj = list(subj)
        if sentence_initial_cap and subj:
            subj[0] = subj[0][:1].upper() + subj[0][1:]
        out = subj + [new_aux] + pred
        applied.append("R4_en_aux_inversion")

    if len(out) < 2:
        return _refuse(original, "G6_too_short")

    new_body = " ".join(out)
    if is_question and on("R1_terminal_question_mark"):
        new_s = new_body + "."
        applied.append("R1_terminal_question_mark")
    else:
        new_s = new_body + term
    # keep original leading/trailing whitespace behaviour simple: normalized is the stripped form
    if not applied:
        return NormalizationResult(original, original)
    return NormalizationResult(original, new_s, tuple(applied))


# ---------------------------------------------------------------------------------------------------------------------
# Mechanical semantic-preservation check: no content word may be added.
# ---------------------------------------------------------------------------------------------------------------------
def _content_tokens(s: str) -> List[str]:
    return [t for t in (_bare(x) for x in s.split()) if t and t not in ("ba", "bang")]


def verify_no_content_added(original: str, normalized: str) -> Tuple[bool, str]:
    """True if every normalized token (ignoring case/punctuation, the question particle, and the re-attached linker) occurs in the original.
    Passing this does NOT prove meaning is preserved (reordering could); it proves that nothing was invented."""
    if original == normalized:
        return True, "identical"
    orig = _content_tokens(original)
    pool = list(orig)
    for t in _content_tokens(normalized):
        if t in pool:
            pool.remove(t)
            continue
        # linker re-attachment: 'wala' + 'bang' -> 'walang' / 'pasok' +n+g
        stem = [o for o in pool if t == o + "ng" or t == o + "g"]
        if stem:
            pool.remove(stem[0])
            continue
        return False, f"token '{t}' not in original"
    dropped = [t for t in pool]
    return True, ("dropped: " + ",".join(dropped)) if dropped else "tokens preserved"


def rules_markdown() -> str:
    lines = [f"# M12.3 Option B rule registry ({NORMALIZER_VERSION})", "",
             "Evaluation-side only. Generated from `m12_3_normalizer.py` (`python m12_3_normalizer.py --dump-rules`).", ""]
    for r in RULES:
        lines += [f"## {r.rule_id}", f"- **description:** {r.description}", f"- **input_pattern:** {r.input_pattern}",
                  f"- **output_transformation:** {r.output_transformation}", f"- **language:** {r.language}",
                  "- **examples:**"] + [f"  - `{a}` -> `{b}`" for a, b in r.examples] + [
                  f"- **semantic_risk:** {r.semantic_risk}", f"- **observed_in_dev:** {r.observed_in_dev}", ""]
    lines += ["## Refusal guards (input returned unchanged)", ""] + [f"- **{k}:** {v}" for k, v in GUARDS.items()]
    lines += ["", "## Known limits (not attempted in Option B)", ""] + [f"- {x}" for x in KNOWN_LIMITS]
    return "\n".join(lines) + "\n"


def config_record(path: Optional[str] = None) -> dict:
    """Provenance record for a run: version, rule ids, guard ids and the SHA-256 of this file."""
    p = path or __file__
    return {"normalizer_version": NORMALIZER_VERSION, "rules": [r.rule_id for r in RULES], "guards": list(GUARDS),
            "normalizer_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}


if __name__ == "__main__":
    import sys
    if "--dump-rules" in sys.argv:
        sys.stdout.write(rules_markdown())
    else:
        for line in sys.stdin:
            r = normalize(line.rstrip("\n"))
            print(f"{r.original_user_input!r} -> {r.normalized_claim!r}  rules={list(r.rules_applied)} refused={r.refused}")
