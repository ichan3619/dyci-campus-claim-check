from src.normalization import Lexicon, TaglishNormalizer


def make():
    return TaglishNormalizer()


def test_noop_returns_original_exactly():
    r = make().normalize("Classes are suspended tomorrow.")
    assert r.normalized == r.original and not r.edits and not r.changed


def test_vowel_dropped_variants():
    r = make().normalize("wla pasok bkas")
    assert r.normalized == "wala pasok bukas"
    assert [e.rule for e in r.edits] == ["lex:wla", "lex:bkas"]


def test_case_is_carried_over():
    assert make().normalize("Hnd po").normalized == "Hindi po"


def test_elongation_needs_vocabulary_support():
    n = make()
    assert n.normalize("pleaseee").normalized == "please"
    assert n.normalize("bukaaas").normalized == "bukas"
    assert n.normalize("soooo").normalized == "soooo"              # 'so' is not in vocab -> untouched


def test_elongation_then_lexicon():
    assert make().normalize("plssss").normalized == "please"


def test_numbers_are_never_collapsed():
    r = make().normalize("room 1000 bukaaas")
    assert "1000" in r.normalized


def test_tier2_month_needs_adjacent_number():
    n = make()
    assert n.normalize("exam sept 28").normalized == "exam September 28"
    assert n.normalize("28 sept exam").normalized == "28 September exam"
    r = n.normalize("i like sept")
    assert r.normalized == "i like sept" and "unresolved_context:sept" in r.flags


def test_tier2_wp_needs_campus_context():
    n = make()
    assert n.normalize("wp bukas?").normalized == "walang pasok bukas?"
    assert n.normalize("wp bukas?").flags == []
    r = n.normalize("wp gg")
    assert not r.changed and "unresolved_context:wp" in r.flags


def test_ambiguous_surfaces_are_flagged_not_rewritten():
    r = make().normalize("may exam ba sa may 5")
    assert not r.changed
    assert r.flags.count("ambiguous:may") == 2


def test_negation_and_numbers_preserved_on_normal_edits():
    r = make().normalize("hnd 2morrow sept 28")
    assert r.normalized == "hindi tomorrow September 28"
    assert not any(f.startswith("refused") for f in r.flags)


def test_invariant_blocks_negation_flip():
    lex = Lexicon.from_rows([{"surface": "ok", "canonical": "hindi", "tier": "1"}])
    r = TaglishNormalizer(lexicon=lex).normalize("ok pasok")
    assert not r.changed and "refused:negation_count_changed" in r.flags


def test_invariant_blocks_number_change():
    lex = Lexicon.from_rows([{"surface": "xx", "canonical": "28", "tier": "1"}])
    r = TaglishNormalizer(lexicon=lex).normalize("xx 5")
    assert not r.changed and "refused:numbers_changed" in r.flags


def test_unreviewed_entries_can_be_disabled_for_frozen_runs():
    r = TaglishNormalizer(allow_unreviewed=False).normalize("hnd po")
    assert not r.changed and "unreviewed_skipped:hnd" in r.flags


def test_variants_and_features():
    n = make()
    assert n.variants("hnd po") == ["hnd po", "hindi po"]
    assert n.variants("hindi po") == ["hindi po"]
    f = n.normalize("wp bukas").features
    assert f["short_claim"] is True and f["code_switched"] is False


def test_lexicon_hash_is_stable_and_recorded():
    a, b = make().normalize("hnd"), make().normalize("hnd")
    assert a.lexicon_sha256 == b.lexicon_sha256 and len(a.lexicon_sha256) == 64


def test_bad_lexicon_rows_are_rejected():
    for row in (
        {"surface": "wp", "canonical": "walang pasok", "tier": "2"},                      # tier 2 without a condition
        {"surface": "hnd", "canonical": "hindi", "tier": "1", "context_any": "bukas"},    # tier 1 with a condition
        {"surface": "x", "canonical": "", "tier": "1"},                                   # empty canonical
    ):
        try:
            Lexicon.from_rows([row])
        except ValueError:
            continue
        raise AssertionError(f"accepted bad row {row}")


# ---------------- code_switched (Step 1 fix) ----------------
EN = ["typhoon", "suspended", "canceled", "classes", "library", "exam"]       # stand-in for the English evidence-corpus vocabulary


def test_code_switched_needs_english_vocabulary_for_content_word_mixes():
    plain = TaglishNormalizer()
    assert plain.normalize("walang klase bukas, canceled daw dahil sa typhoon").features["code_switched"] is False   # documented limit
    with_vocab = TaglishNormalizer(english_vocab=EN)
    assert with_vocab.normalize("walang klase bukas, canceled daw dahil sa typhoon").features["code_switched"] is True


def test_code_switched_detects_english_content_words_with_tagalog_particles():
    n = TaglishNormalizer(english_vocab=EN)
    assert n.normalize("canceled ba ang exam bukas?").features["code_switched"] is True
    assert n.normalize("pwede ba sa library mamaya").features["code_switched"] is True


def test_code_switched_is_false_for_pure_english_and_pure_tagalog():
    n = TaglishNormalizer(english_vocab=EN + ["bagyo"])
    assert n.normalize("Classes are suspended tomorrow due to the typhoon").features["code_switched"] is False
    f = n.normalize("walang pasok bukas dahil sa bagyo").features       # 'bagyo' is in the English vocab but Tagalog evidence wins
    assert f["code_switched"] is False and f["english_evidence_hits"] == 0


def test_ambiguous_and_short_tokens_are_never_english_evidence():
    n = TaglishNormalizer(english_vocab=["may", "san", "pm"])
    assert n.normalize("klase may bukas").features["code_switched"] is False   # 'may' is ambiguous -> excluded
    assert n.normalize("klase 1 pm").features["code_switched"] is False        # 'pm' shorter than 3 letters


def test_backward_compatible_feature_keys_and_counts():
    f = TaglishNormalizer(english_vocab=EN).normalize("canceled ba ang exam bukas?").features
    for k in ("tagalog_function_hits", "english_function_hits", "short_claim", "has_number", "code_switched"):
        assert k in f
    assert f["tagalog_evidence_hits"] >= 3 and f["english_evidence_hits"] >= 2


if __name__ == "__main__":      # runs without pytest: python tests/test_taglish_normalizer.py
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as exc:        # noqa: BLE001
                failed += 1; print("FAIL", name, repr(exc))
    sys.exit(1 if failed else 0)
