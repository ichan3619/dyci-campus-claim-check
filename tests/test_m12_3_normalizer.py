"""
Unit tests for the M12.3 Option B normalizer (evaluation-side module scripts/m12_retrieval/m12_3_normalizer.py).

Standard library only (unittest style, so it runs under both `python -m unittest` and `pytest`). No model, no network.
Tagalog/Taglish inputs are taken ONLY from the development material (the 38-case set). English inverted-question examples are
the two given in the M12.3 brief plus structurally similar ones. No independent-validation data is used. Placeholder words
('foo', 'bar') are used for structure-only property tests.
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts", "m12_retrieval")
sys.path.insert(0, SCRIPTS)
import m12_3_normalizer as N  # noqa: E402


def n(text):
    return N.normalize(text)


class PositiveTransformations(unittest.TestCase):
    def test_english_copula_inversion_from_brief(self):
        self.assertEqual(n("Are classes suspended tomorrow?").normalized_claim, "Classes are suspended tomorrow.")

    def test_existential_negative_stays_negative(self):
        # must NOT become "Classes are suspended tomorrow." (lexical equivalence is not Option B)
        r = n("Is there no class tomorrow?")
        self.assertEqual(r.normalized_claim, "There is no class tomorrow.")
        self.assertNotIn("suspended", r.normalized_claim.lower())

    def test_copula_with_open_and_participle_subjects(self):
        self.assertEqual(n("Is the library open 24 hours?").normalized_claim, "The library is open 24 hours.")
        self.assertEqual(n("Are students allowed in the gym?").normalized_claim, "Students are allowed in the gym.")

    def test_will_be_inversion(self):
        self.assertEqual(n("Will the library be open 24 hours?").normalized_claim, "The library will be open 24 hours.")

    def test_tagalog_ba_dev(self):
        self.assertEqual(n("may exam ba sa sept 28?").normalized_claim, "may exam sa sept 28.")
        self.assertEqual(n("half day ba bukas?").normalized_claim, "half day bukas.")
        self.assertEqual(n("hanggang 1pm lang ba ang klase bukas?").normalized_claim, "hanggang 1pm lang ang klase bukas.")

    def test_taglish_ba_dev(self):
        self.assertEqual(n("suspended ba ang klase bukas?").normalized_claim, "suspended ang klase bukas.")
        self.assertEqual(n("pwede ba mag-aral sa library hanggang madaling araw pag finals?").normalized_claim,
                         "pwede mag-aral sa library hanggang madaling araw pag finals.")

    def test_tagalog_bang_linker_dev(self):
        r = n("wala bang pasok bukas dahil sa bagyo?")
        self.assertEqual(r.normalized_claim, "walang pasok bukas dahil sa bagyo.")
        self.assertIn("R3_tl_question_particle_bang", r.rules_applied)

    def test_fragment_gets_punctuation_only(self):
        for q, e in [("Suspension tomorrow?", "Suspension tomorrow."), ("Early dismissal tomorrow?", "Early dismissal tomorrow."),
                     ("library 24 hours?", "library 24 hours."), ("walang pasok bukas?", "walang pasok bukas.")]:
            r = n(q)
            self.assertEqual(r.normalized_claim, e)
            self.assertEqual(r.rules_applied, ("R1_terminal_question_mark",))

    def test_na_ba_keeps_na(self):  # 'na' (already) is content; only 'ba' goes
        self.assertEqual(n("pwede na ba kunin ID?").normalized_claim, "pwede na kunin ID.")


class NoOpCases(unittest.TestCase):
    STATEMENTS = [
        "Classes are suspended tomorrow.", "There will be no classes tomorrow.", "no classes tomorrow", "class suspension typhoon",
        "Tomorrow is a half day; students will be dismissed at 1 PM.", "library 24/7 finals", "enrollment august 10",
        "Students should bring their Certificate of Registration when making payment.",
    ]

    def test_statements_unchanged_and_identical_object_text(self):
        for s in self.STATEMENTS:
            r = n(s)
            self.assertEqual(r.normalized_claim, s)
            self.assertEqual(r.rules_applied, ())
            self.assertFalse(r.changed)

    def test_all_26_development_claims_are_noops(self):
        base = os.path.join(SCRIPTS)
        claims = json.load(open(os.path.join(base, "m12_1_gold_claims.json"), encoding="utf-8"))["claims"]
        claims += json.load(open(os.path.join(base, "m12_2_deep_fact_claims.json"), encoding="utf-8"))["claims"]
        self.assertEqual(len(claims), 26)
        for c in claims:
            self.assertEqual(n(c["claim"]).normalized_claim, c["claim"], c["id"])

    def test_empty_and_whitespace(self):
        self.assertEqual(n("").normalized_claim, "")
        self.assertEqual(n("   ").normalized_claim, "   ")

    def test_disabling_rules_gives_noop(self):
        r = N.normalize("Suspension tomorrow?", enabled=frozenset())
        self.assertEqual(r.normalized_claim, "Suspension tomorrow?")


class AmbiguousCasesRefuse(unittest.TestCase):
    def test_wh_questions_are_not_turned_into_claims(self):
        for q in ["kailan ang enrollment sa 2nd sem?", "kailan ang midterms?", "kailan pwede kunin ang bagong ID?",
                  "What's the enrollment deadline?", "What is the tuition fee for next semester?", "magkano ang tuition fee?",
                  "Who won the basketball game last night?", "When is enrollment?"]:
            r = n(q)
            self.assertEqual(r.normalized_claim, q, q)
            self.assertEqual(r.refused, "G1_wh_question", q)

    def test_enrollment_date_is_never_invented(self):
        for q in ["kelan enrollment?", "kailan ang enrollment sa 2nd sem?"]:
            r = n(q)
            self.assertEqual(r.normalized_claim, q)
            self.assertFalse(any(ch.isdigit() for ch in r.normalized_claim) and not any(ch.isdigit() for ch in q))

    def test_negative_polar_refused(self):
        r = n("Aren't classes suspended tomorrow?")
        self.assertEqual(r.refused, "G3_negative_polar")
        self.assertFalse(r.changed)

    def test_do_support_and_compound_subject_refused(self):
        for q in ["Does the library open at night?", "Will the midterm exams be held online?", "Will classes resume Monday?",
                  "Is the school building closed?", "Is the dress code enforced?"]:
            r = n(q)
            self.assertEqual(r.refused, "G4_ambiguous_subject_boundary", q)
            self.assertEqual(r.normalized_claim, q)

    def test_multiple_sentences_refused(self):
        r = n("Classes are suspended. Is that true?")
        self.assertEqual(r.refused, "G2_multiple_sentences")

    def test_bang_after_consonant_refused(self):
        r = n("foo bar bang baz?")  # linker would be 'na'; cannot be derived without a lexicon
        self.assertEqual(r.refused, "G5_tl_linker_unresolved")
        self.assertFalse(r.changed)

    def test_two_particles_refused(self):
        self.assertEqual(n("foo ba bar ba baz?").refused, "G7_multiple_particles")

    def test_degenerate_input_refused(self):
        self.assertFalse(n("ba?").changed)


class SemanticPreservation(unittest.TestCase):
    """Person, date, place, event, action, scope, quantity, certainty-markers survive; nothing is added."""

    DEV_QUESTIONS = ["Suspension tomorrow?", "walang pasok bukas?", "suspended ba ang klase bukas?", "wala bang pasok bukas dahil sa bagyo?",
                     "may exam ba sa sept 28?", "Early dismissal tomorrow?", "half day ba bukas?", "hanggang 1pm lang ba ang klase bukas?",
                     "library 24 hours?", "pwede ba mag-aral sa library hanggang madaling araw pag finals?",
                     "Are classes suspended tomorrow?", "Is there no class tomorrow?", "Will the library be open 24 hours?"]

    def test_no_content_word_added(self):
        for q in self.DEV_QUESTIONS:
            r = n(q)
            ok, why = N.verify_no_content_added(q, r.normalized_claim)
            self.assertTrue(ok, f"{q!r} -> {r.normalized_claim!r}: {why}")

    def test_dates_numbers_and_scope_words_preserved(self):
        pairs = {"may exam ba sa sept 28?": ["sept", "28"], "hanggang 1pm lang ba ang klase bukas?": ["hanggang", "1pm", "lang", "bukas"],
                 "pwede ba mag-aral sa library hanggang madaling araw pag finals?": ["library", "hanggang", "madaling", "araw", "finals"],
                 "wala bang pasok bukas dahil sa bagyo?": ["bukas", "bagyo"], "Is there no class tomorrow?": ["no", "tomorrow"],
                 "library 24 hours?": ["24", "hours"]}
        for q, must in pairs.items():
            out = n(q).normalized_claim.lower()
            for w in must:
                self.assertIn(w, out, f"{q!r} lost {w!r}: {out!r}")

    def test_negation_and_polarity_not_flipped(self):
        self.assertIn("walang", n("wala bang pasok bukas?").normalized_claim)   # wala = negative existential kept
        self.assertIn(" no ", " " + n("Is there no class tomorrow?").normalized_claim.lower() + " ")

    def test_no_invented_all_everyone_institution_or_date(self):
        banned = ["all ", "everyone", "dyci", "campus", "university", "monday", "tuesday", "2026", "typhoon"]
        for q in self.DEV_QUESTIONS:
            out = n(q).normalized_claim.lower()
            for b in banned:
                if b in out:
                    self.assertIn(b, q.lower(), f"{q!r} acquired {b!r}")

    def test_original_is_never_modified_and_both_are_returned(self):
        for q in self.DEV_QUESTIONS + NoOpCases.STATEMENTS:
            r = n(q)
            self.assertEqual(r.original_user_input, q)
            self.assertIsInstance(r.normalized_claim, str)

    def test_inversion_keeps_person_and_subject(self):
        r = n("Are students allowed in the gym?")
        self.assertTrue(r.normalized_claim.startswith("Students"))
        self.assertTrue(N.verify_no_content_added("Are students allowed in the gym?", r.normalized_claim)[0])

    def test_idempotent(self):
        for q in self.DEV_QUESTIONS:
            once = n(q).normalized_claim
            self.assertEqual(n(once).normalized_claim, once, q)


class NotCaseSpecific(unittest.TestCase):
    """Rules act on structure: placeholder words behave exactly like real ones; the module contains no case ids or corpus text."""

    def test_structure_only_placeholders(self):
        self.assertEqual(n("foo ba bar baz?").normalized_claim, "foo bar baz.")
        self.assertEqual(n("qux bang quux?").refused, "G5_tl_linker_unresolved")   # consonant before 'bang'
        self.assertEqual(n("fooa bang bar?").normalized_claim, "fooang bar.")      # vowel -> -ng
        self.assertEqual(n("foon bang bar?").normalized_claim, "foong bar.")       # n -> -g
        self.assertEqual(n("Are foos barred today?").normalized_claim, "Foos are barred today.")

    def test_source_has_no_case_ids_or_corpus_specific_strings(self):
        src = open(os.path.join(SCRIPTS, "m12_3_normalizer.py"), encoding="utf-8").read().lower()
        for banned in ["a1-", "a4-short", "x02", "evt_001", "fb_midterm", "certificate of registration", "typhoon", "dyci"]:
            self.assertNotIn(banned, src)

    def test_registry_is_small_and_documented(self):
        self.assertLessEqual(len(N.RULES), 5)
        for r in N.RULES:
            for field in ("rule_id", "description", "input_pattern", "output_transformation", "language", "examples", "semantic_risk"):
                self.assertTrue(getattr(r, field), f"{r.rule_id}.{field}")
            for src, expected in r.examples:   # the registry examples are executable documentation
                self.assertEqual(N.normalize(src).normalized_claim, expected, r.rule_id)


if __name__ == "__main__":
    unittest.main()
