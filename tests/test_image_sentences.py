"""Image Check Part B Stage 1: sentence logic unit tests (synthetic strings and stubbed checks only; no OCR, no models)."""
import pytest
from fastapi import HTTPException

from src.image.sentences import (ABBREVIATIONS, MAX_CHECKED, SOFT_CONFLICT, STRONG_CONFLICT, Counts, aggregate,
                                 anchored_posts, check_statements, count_outcomes, is_eligible, plan_sentences,
                                 select_basis, split_segments)


# ---- S1: splitting -----------------------------------------------------------------------------------------------------
def test_splits_at_full_stop_exclamation_question_and_ellipsis_before_a_capital_or_digit():
    text = "Classes are suspended today. Stay safe! Is the exam moved? Wait for news... 2026 classes resume Monday."
    assert split_segments(text) == ["Classes are suspended today.", "Stay safe!", "Is the exam moved?",
                                    "Wait for news...", "2026 classes resume Monday."]


def test_splits_before_an_opening_quote_or_bracket():
    assert split_segments('The office said so. "Classes resume Monday," it added.') == [
        "The office said so.", '"Classes resume Monday," it added.']
    assert split_segments("Read the memo. (Copies are at the gate.)") == ["Read the memo.", "(Copies are at the gate.)"]


def test_no_split_before_a_lowercase_word():
    assert split_segments("The fee is due. or else nothing happens") == ["The fee is due. or else nothing happens"]


@pytest.mark.parametrize("abbr", sorted(ABBREVIATIONS))
def test_never_splits_after_an_approved_abbreviation(abbr):
    text = f"The note mentions {abbr} Cruz as the contact person."
    assert split_segments(text) == [text]


@pytest.mark.parametrize("text", [
    "Dr. Yanga's Colleges, Inc. hosted the Teachers' Day celebration.",
    "The drive is at Brgy. Turo near the campus gate.",
    "The fiesta is in Sta. Maria this Sunday.",
    "Mass will be led by Fr. Cruz at the chapel.",
    "Talk by Dra. Santos in the gym today.",
    "Memorandum Circular No. 135 suspended classes today.",
    "DR. CRUZ and MS. REYES will attend.",                       # case-insensitive
])
def test_local_and_title_abbreviations_do_not_split(text):
    assert split_segments(text) == [text]


@pytest.mark.parametrize("text", [
    "Classes end at 1.00 PM because of the program.",
    "The caroling raised PHP 1,218.50 for the beneficiaries.",
    "The average grade was 3.5 this term.",
    "Classes end at 1:00 PM because of the program.",
])
def test_decimals_amounts_and_times_are_never_split(text):
    assert split_segments(text) == [text]


def test_a_sentence_final_abbreviation_stays_merged_documented_trade_off():
    assert split_segments("The seminar starts at 9 a.m. Students should bring their ID.") == [
        "The seminar starts at 9 a.m. Students should bring their ID."]


def test_text_without_a_split_point_is_one_segment_and_order_is_preserved():
    assert split_segments("NO CLASSES ON AUGUST 21 NINOY AQUINO DAY") == ["NO CLASSES ON AUGUST 21 NINOY AQUINO DAY"]
    text = "First statement here now. Second statement here now. Third statement here now."
    assert split_segments(text) == ["First statement here now.", "Second statement here now.", "Third statement here now."]


def test_empty_and_whitespace_text_give_no_segments():
    assert split_segments("") == [] and split_segments("   ") == [] and split_segments(None) == []


# ---- S2 and the cap of 12 ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("segment, eligible", [
    ("DYCI ADVISORY", False), ("1.", False), ("Classes are suspended.", False),        # < 4 words
    ("Classes are suspended today.", True), ("1 2 3 4", False),                         # < 3 letters
    ("No classes on Monday.", True),
])
def test_eligibility_needs_four_words_and_three_letters(segment, eligible):
    assert is_eligible(segment) is eligible


def test_plan_marks_short_segments_and_the_cap():
    sentences = [f"Statement number {i} is here." for i in range(14)]
    plan = plan_sentences("DYCI ADVISORY. " + " ".join(sentences))
    assert plan[0].status == "not_checked_short" and plan[0].text == "DYCI ADVISORY."
    assert [s.status for s in plan[1:]] == ["to_check"] * MAX_CHECKED + ["not_checked_limit"] * 2
    assert [s.index for s in plan] == list(range(15))


def test_fewer_than_twelve_eligible_are_all_checked():
    plan = plan_sentences("Classes are suspended today. The office is closed today. Hi.")
    assert [s.status for s in plan] == ["to_check", "to_check", "not_checked_short"]


# ---- checking each statement (stubbed check function) -------------------------------------------------------------------
class _R:
    def __init__(self, verdict):
        self.verdict = verdict


class _Stub:
    def __init__(self, verdicts):
        self.verdicts, self.calls = verdicts, []

    def __call__(self, text):
        self.calls.append(text)
        return _R(self.verdicts[text])


def _plan(*sentences):
    return plan_sentences(" ".join(sentences))


def test_outcomes_map_three_way_verdicts_and_the_agreement_check():
    a, b, c, d, e = ("Classes resume on Monday.", "The fee is PHP 250.", "The seminar is at noon.",
                     "Where is the venue today?", "The office is closed today.")
    stub = _Stub({a: "SUPPORTED", a[:-1]: "SUPPORTED",                    # agreement agrees -> match
                  b: "CONTRADICTED", b[:-1]: "CONTRADICTED",              # agreement agrees -> contradiction
                  c: "SUPPORTED", c[:-1]: "INSUFFICIENT_EVIDENCE",        # agreement disagrees -> uncertain
                  d: None,                                                # question / language-withheld -> inconclusive
                  e: "INSUFFICIENT_EVIDENCE"})
    results = check_statements(_plan(a, b, c, d, e), stub)
    assert [(r.outcome, r.status) for r in results] == [("match", "checked"), ("contradiction", "checked"),
                                                        ("inconclusive", "uncertain"), ("inconclusive", "checked"),
                                                        ("inconclusive", "checked")]
    assert [r.verdict for r in results] == ["SUPPORTED", "CONTRADICTED", "SUPPORTED", None, "INSUFFICIENT_EVIDENCE"]
    assert results[2].agreement == {"rule": "final_full_stop_toggled", "agreed": False}
    assert results[4].agreement is None                                   # no agreement check for a non-S/C verdict
    assert stub.calls == [a, a[:-1], b, b[:-1], c, c[:-1], d, e]           # sequential; extra call only for S/C


def test_statement_ending_in_other_punctuation_gets_no_agreement_check_as_on_the_whole_text():
    s = "Classes really resume on Monday!"
    stub = _Stub({s: "SUPPORTED"})
    (r,) = check_statements(_plan(s), stub)
    assert r.agreement is None and r.outcome == "match" and stub.calls == [s]


def test_skipped_segments_are_never_checked_or_counted():
    plan = plan_sentences("DYCI ADVISORY. Classes are suspended. Classes resume on Monday.")
    stub = _Stub({"Classes resume on Monday.": "INSUFFICIENT_EVIDENCE"})
    results = check_statements(plan, stub)
    counts = count_outcomes(plan, results)
    assert stub.calls == ["Classes resume on Monday."]
    assert (counts.n, counts.k, counts.c, counts.i, counts.not_checked_short) == (1, 0, 0, 1, 2)


def test_a_failing_check_propagates_and_gives_no_partial_results():
    a, b = "Classes resume on Monday.", "The office is closed today."

    def check(text):
        if text == b:
            raise HTTPException(status_code=503, detail="busy")
        return _R("INSUFFICIENT_EVIDENCE")
    with pytest.raises(HTTPException) as e:
        check_statements(_plan(a, b), check)
    assert e.value.status_code == 503


# ---- counts and labels (C1/C3) ---------------------------------------------------------------------------------------
def test_counts_add_up():
    a, b, c = "Classes resume on Monday.", "The fee is PHP 250.", "The office is closed today."
    stub = _Stub({a: "SUPPORTED", a[:-1]: "SUPPORTED", b: "INSUFFICIENT_EVIDENCE", c: None})
    plan = _plan(a, b, c)
    counts = count_outcomes(plan, check_statements(plan, stub))
    assert (counts.n, counts.k, counts.c, counts.i) == (3, 1, 0, 2) and counts.n == counts.k + counts.c + counts.i


@pytest.mark.parametrize("k, i, kind, label", [
    (0, 3, "not_confirmed", "Not confirmed"),                                            # all inconclusive
    (1, 2, "partly_confirmed", "Partly confirmed: 1 of 3 checked statements match"),    # SUPPORTED + INSUFFICIENT
    (3, 0, "all_match", "All 3 checked statements match an announcement"),               # all SUPPORTED
])
def test_labels_without_contradiction(k, i, kind, label):
    assert aggregate(Counts(n=k + i, k=k, c=0, i=i)) == {"kind": kind, "label": label}


# ---- C2 (option B, anchored contradiction; owner decision 2026-10-15) -------------------------------------------------
class _Doc:
    def __init__(self, document_id):
        self.document_id = document_id


class _RP(_R):
    """A stub response that also names the post used, like VerificationResultModel.used_evidence."""
    def __init__(self, verdict, post=None):
        super().__init__(verdict)
        self.used_evidence = _Doc(post) if post else None


def _results(*specs):
    """specs: (verdict, post) per statement; the agreement check agrees for every S/C statement."""
    texts = [f"Statement number {i} is here." for i in range(len(specs))]
    table = {}
    for t, (verdict, post) in zip(texts, specs):
        table[t] = _RP(verdict, post)
        table[t[:-1]] = _RP(verdict, post)
    plan = plan_sentences(" ".join(texts))
    return plan, check_statements(plan, lambda text: table[text])


def test_contradiction_anchored_by_a_match_on_the_same_post_gets_the_strong_headline():
    plan, results = _results(("SUPPORTED", "N20"), ("SUPPORTED", "N20"), ("CONTRADICTED", "N20"))
    out = aggregate(count_outcomes(plan, results), results)
    assert out == {"kind": "conflict", "label": STRONG_CONFLICT, "anchored_posts": ["N20"], "option_a_kind": "conflict"}


@pytest.mark.parametrize("specs", [
    [("CONTRADICTED", "N16")],                                                   # alone (short image)
    [("SUPPORTED", "N20"), ("CONTRADICTED", "N16")],                            # matches a different post (Q12-like)
    [("INSUFFICIENT_EVIDENCE", "N16"), ("CONTRADICTED", "N16")],                # same post, but not a match
    [("SUPPORTED", None), ("CONTRADICTED", None)],                              # no post recorded
])
def test_unanchored_contradiction_gets_the_soft_headline_and_is_not_hidden(specs):
    plan, results = _results(*specs)
    counts = count_outcomes(plan, results)
    out = aggregate(counts, results)
    assert counts.c == 1 and out["kind"] == "possible_conflict" and out["label"] == SOFT_CONFLICT
    assert out["anchored_posts"] == [] and out["option_a_kind"] == "conflict"   # option A would have shown the strong one


def test_c2_never_changes_a_statement_verdict_or_outcome():
    plan, results = _results(("SUPPORTED", "N20"), ("CONTRADICTED", "N16"), ("INSUFFICIENT_EVIDENCE", "N20"))
    before = [(r.verdict, r.outcome, r.status) for r in results]
    aggregate(count_outcomes(plan, results), results)
    assert [(r.verdict, r.outcome, r.status) for r in results] == before
    assert before == [("SUPPORTED", "match", "checked"), ("CONTRADICTED", "contradiction", "checked"),
                      ("INSUFFICIENT_EVIDENCE", "inconclusive", "checked")]


def test_uncertain_statements_do_not_anchor_or_count_as_contradictions():
    a, b = "Classes resume on Monday.", "The fee is PHP 250."
    table = {a: _RP("SUPPORTED", "N20"), a[:-1]: _RP("INSUFFICIENT_EVIDENCE", "N20"),   # match fails agreement -> uncertain
             b: _RP("CONTRADICTED", "N20"), b[:-1]: _RP("CONTRADICTED", "N20")}
    plan = _plan(a, b)
    results = check_statements(plan, lambda text: table[text])
    assert anchored_posts(results) == []
    assert aggregate(count_outcomes(plan, results), results)["kind"] == "possible_conflict"


def test_several_contradictions_one_anchored_gives_the_strong_headline():
    plan, results = _results(("SUPPORTED", "N05"), ("CONTRADICTED", "N05"), ("CONTRADICTED", "N16"))
    out = aggregate(count_outcomes(plan, results), results)
    assert out["kind"] == "conflict" and out["anchored_posts"] == ["N05"]


def test_aggregate_refuses_zero_checked_statements():
    with pytest.raises(ValueError):
        aggregate(Counts(n=0))


def test_labels_never_call_the_image_true_or_fake():
    labels = [aggregate(Counts(n=k + i, k=k, c=0, i=i))["label"] for k, i in [(0, 2), (1, 1), (2, 0)]]
    for label in labels + [STRONG_CONFLICT, SOFT_CONFLICT]:
        words = label.lower().replace(".", " ").split()
        assert not any(w in words for w in ("true", "real", "fake", "false", "supported"))


# ---- B1: basis selection ------------------------------------------------------------------------------------------------
def test_basis_selection():
    assert select_basis("", plan_sentences("")) == "none"
    assert select_basis("12 34", plan_sentences("12 34")) == "none"                        # < 3 letters
    short = "Walang pasok bukas"                                                           # text, but no eligible sentence
    assert select_basis(short, plan_sentences(short)) == "whole_text_fallback"
    long = "DYCI ADVISORY. Classes are suspended today."
    assert select_basis(long, plan_sentences(long)) == "sentences"
