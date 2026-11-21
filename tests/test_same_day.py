"""Same-day advisory rule Z (src/verification/same_day.py, CAMPUS_VERIFY_SAME_DAY_ADVISORY default off;
results/round2/cross_post_conflict_plan.md, Amendment 2). Synthetic posts and a fake verifier only; test set Z is used
once, by the plan's run."""
import datetime as dt
from types import SimpleNamespace as NS

import pytest

from src.verification import same_day as sd
from src.verification.verifier import VerificationResult

TODAY = dt.date(2026, 3, 12)
EARLY = NS(document_id="early", score=0.86, text="DYCI ADVISORY 10 March 2026 No classes tomorrow, March 11. Classes "
                                                 "resume on March 12, 2026.")
LATER = NS(document_id="later", score=0.84, text="DYCI ADVISORY March 12, 2026 Classes at all levels are suspended "
                                                 "today because of a power outage.")
FOLLOW = NS(document_id="follow", score=0.84, text="DYCI 2ND ADVISORY March 12, 2026 Power has been restored. Classes "
                                                   "resume tomorrow.")
OLD_YEAR = NS(document_id="old", score=0.85, text="DYCI Advisory 12 March 2025 Classes are suspended today.")
LOW = NS(document_id="low", score=0.80, text="DYCI ADVISORY March 12, 2026 Classes are suspended today.")
OTHER = NS(document_id="other", score=0.85, text="DYCI ADVISORY 20 August 2026 No classes on August 21.")


class FakeVerifier:
    """Labels by document id; records nothing else. Mirrors the attributes same_day.judge reads."""
    def __init__(self, ranked, labels):
        self._ranked, self._labels = ranked, labels
        self._relevance_threshold, self._passage_selector = 0.833, None
        self._premise_mode, self._date_check, self._value_check = "window", False, False
        by_text = {d.text: d.document_id for d in ranked}
        self._classifier = NS(classify=lambda claim, premise: NS(label=labels[by_text[premise]], confidence=0.9))
        self.retrieved_with = None

    @property
    def _retriever(self):
        outer = self

        def retrieve(text, top_k):
            outer.retrieved_with = (text, top_k)
            return outer._ranked[:top_k]
        return NS(retrieve=retrieve)


def _result(verdict, used, shown):
    return VerificationResult(claim="c", verdict=verdict, evidence_status="RELEVANT_EVIDENCE_FOUND",
                              retrieved_evidence=list(shown), used_evidence=used, classification_result=None,
                              explanation="E.")


def _run(message, verdict, used, shown, ranked, labels):
    v = FakeVerifier(ranked, labels)
    out, info = sd.check(message, message, _result(verdict, used, shown), v, TODAY)
    return out, info, v


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(sd.ENV, raising=False)
    assert not sd.enabled()
    monkeypatch.setenv(sd.ENV, "1")
    assert sd.enabled()


def test_header_date_must_be_the_message_day_and_year():
    assert sd.is_same_day_advisory(LATER.text, 3, 12, 2026)
    assert not sd.is_same_day_advisory(EARLY.text, 3, 12, 2026)        # mentions March 12, but dated March 10
    assert not sd.is_same_day_advisory(OLD_YEAR.text, 3, 12, 2026)     # same day, different year
    assert not sd.is_same_day_advisory("Classes are suspended on March 12, 2026.", 3, 12, 2026)   # no header


def test_a_stale_contradiction_becomes_supported_by_the_same_day_advisory():
    out, info, v = _run("no classes on march 12", "CONTRADICTED", EARLY, [EARLY, OTHER],
                        [EARLY, OTHER, LATER], {"later": "SUPPORTED"})
    assert out.verdict == "SUPPORTED" and out.used_evidence is LATER
    assert info["action"] == "supported" and info["original_verdict"] == "CONTRADICTED"
    assert [d.document_id for d in out.retrieved_evidence] == ["later", "early", "other"]
    assert v.retrieved_with == ("no classes on march 12", sd.TOP_N)


def test_a_cleared_follow_up_never_turns_it_into_contradicted():
    out, info, _ = _run("no classes on march 12 due to a power outage", "INSUFFICIENT_EVIDENCE", EARLY, [EARLY],
                        [EARLY, FOLLOW, LATER], {"follow": "CONTRADICTED", "later": "INSUFFICIENT_EVIDENCE"})
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["action"] == "shown_only"
    assert [d.document_id for d in out.retrieved_evidence][:2] == ["follow", "later"]


def test_same_day_advisories_that_disagree_withhold():
    out, info, _ = _run("no classes on march 12", "CONTRADICTED", EARLY, [EARLY],
                        [EARLY, LATER, FOLLOW], {"later": "SUPPORTED", "follow": "CONTRADICTED"})
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["action"] == "disagree_withheld"


def test_a_decisive_verdict_from_an_earlier_post_is_withheld_when_the_advisory_is_not_decisive():
    out, info, _ = _run("classes resume on march 12", "SUPPORTED", EARLY, [EARLY],
                        [EARLY, LATER], {"later": "INSUFFICIENT_EVIDENCE"})
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["action"] == "stale_withheld"


@pytest.mark.parametrize("verdict", ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"])
def test_never_creates_contradicted(verdict):
    out, _, _ = _run("no classes on march 12", verdict if verdict != "CONTRADICTED" else "INSUFFICIENT_EVIDENCE",
                     EARLY, [EARLY], [EARLY, LATER], {"later": "CONTRADICTED"})
    assert out.verdict != "CONTRADICTED"


@pytest.mark.parametrize("message, ranked", [
    ("no classes on march 12", [EARLY, OTHER]),                 # no same-day advisory
    ("no classes on march 12", [EARLY, LOW]),                   # below the relevance gate
    ("no classes on march 12", [EARLY, OLD_YEAR]),              # another year
    ("no classes next week", [EARLY, LATER]),                   # no explicit date
    ("no classes on march 12 and march 13", [EARLY, LATER]),    # two dates
])
def test_does_not_apply(message, ranked):
    r = _result("CONTRADICTED", EARLY, [EARLY])
    out, info = sd.check(message, message, r, FakeVerifier(ranked, {"later": "SUPPORTED"}), TODAY)
    assert info is None and out is r


def test_does_nothing_when_the_post_used_is_the_advisory():
    r = _result("SUPPORTED", LATER, [LATER])
    out, info = sd.check("no classes on march 12", "no classes on march 12", r,
                         FakeVerifier([LATER, EARLY], {"later": "SUPPORTED"}), TODAY)
    assert info is None and out is r


def test_at_most_two_advisories_are_judged():
    third = NS(document_id="third", score=0.84, text="DYCI 3RD ADVISORY March 12, 2026 Thank you for your patience.")
    _, info, _ = _run("no classes on march 12", "INSUFFICIENT_EVIDENCE", EARLY, [EARLY],
                      [EARLY, LATER, FOLLOW, third],
                      {"later": "INSUFFICIENT_EVIDENCE", "follow": "INSUFFICIENT_EVIDENCE", "third": "SUPPORTED"})
    assert [a["document_id"] for a in info["advisories"]] == ["later", "follow"]


@pytest.mark.parametrize("label, text, expected", [
    ("SUPPORTED", "DYCI ADVISORY March 12, 2026 Classes are suspended immediately.", "SUPPORTED"),       # header only
    ("SUPPORTED", "DYCI ADVISORY March 11, 2026 Classes are suspended immediately.", "INSUFFICIENT_EVIDENCE"),
    ("SUPPORTED", "DYCI ADVISORY March 12, 2026 Classes on March 14 are suspended.", "INSUFFICIENT_EVIDENCE"),  # body about another day
    ("SUPPORTED", "DYCI ADVISORY March 12, 2026 Classes on March 12 are suspended.", "SUPPORTED"),
    ("INSUFFICIENT_EVIDENCE", "DYCI ADVISORY March 12, 2026 Thank you.", "INSUFFICIENT_EVIDENCE"),
])
def test_amendment_3_date_check(label, text, expected):
    assert sd.date_checked("no classes on march 12", label, text) == expected


# Rule S (display only; results/round2/same_day_show_plan.md)
def test_show_off_by_default(monkeypatch):
    monkeypatch.delenv(sd.SHOW_ENV, raising=False)
    assert not sd.show_enabled()


def test_show_puts_the_hidden_advisory_first_and_changes_nothing_else():
    r = _result("CONTRADICTED", EARLY, [EARLY, OTHER])
    out, info = sd.show("no classes on march 12", "no classes on march 12", r,
                        FakeVerifier([EARLY, OTHER, OLD_YEAR, LATER, FOLLOW], {}), TODAY)
    assert [d.document_id for d in out.retrieved_evidence] == ["later", "follow", "early", "other"]
    assert (out.verdict, out.used_evidence, out.explanation, out.classification_result) == \
           (r.verdict, r.used_evidence, r.explanation, r.classification_result)
    assert [a["rank"] for a in info["added"]] == [4, 5]


@pytest.mark.parametrize("message, shown, ranked", [
    ("no classes on march 12", [EARLY, LATER], [EARLY, LATER]),        # already shown
    ("no classes on march 12", [EARLY], [EARLY, LOW, OLD_YEAR]),       # below the gate / another year
    ("no classes next week", [EARLY], [EARLY, LATER]),                 # no explicit date
])
def test_show_adds_nothing(message, shown, ranked):
    r = _result("CONTRADICTED", EARLY, shown)
    out, info = sd.show(message, message, r, FakeVerifier(ranked, {}), TODAY)
    assert info is None and out is r


# Rule L (display only, found by header date; results/round2/same_day_lookup_plan.md)
def _docs(*posts):
    return [{"document_id": p.document_id, "text": p.text, "source_type": "t", "source_url": ""} for p in posts]


class LookupVerifier(FakeVerifier):
    def __init__(self, documents, ranked):
        super().__init__(ranked, {})
        self._docs = documents

    @property
    def _retriever(self):
        ranked, docs = self._ranked, self._docs
        return NS(retrieve=lambda text, top_k: ranked[:top_k], _documents=docs)


FORMAL = NS(document_id="formal", score=0.5, text="DYCI ADVISORY March 12, 2026 Academic and office operations are "
                                                  "suspended effective immediately.")
NOYEAR = NS(document_id="noyear", score=0.5, text="DYCI ADVISORY March 12 Operations are suspended.")


def test_lookup_off_by_default(monkeypatch):
    monkeypatch.delenv(sd.LOOKUP_ENV, raising=False)
    assert not sd.lookup_enabled()


def test_lookup_finds_an_advisory_dated_that_day_whatever_its_wording_or_score():
    r = _result("CONTRADICTED", EARLY, [EARLY, OTHER])
    out, info = sd.lookup("no classes on march 12", "no classes on march 12", r,
                          LookupVerifier(_docs(EARLY, OTHER, FORMAL), [EARLY, OTHER]), TODAY)     # FORMAL not ranked
    assert [d.document_id for d in out.retrieved_evidence] == ["formal", "early", "other"]
    assert (out.verdict, out.used_evidence, out.explanation) == (r.verdict, r.used_evidence, r.explanation)
    assert info["added"] == [{"document_id": "formal", "rank": None, "score": 0.0}]
    assert info["day_name"] == "March 12, 2026"


def test_lookup_prefers_the_retrievers_ranking_and_takes_at_most_two():
    sd._INDEX.update(documents=None, index={})
    r = _result("CONTRADICTED", EARLY, [EARLY])
    third = NS(document_id="third", score=0.1, text="DYCI 3RD ADVISORY March 12, 2026 Thank you.")
    out, info = sd.lookup("no classes on march 12", "no classes on march 12", r,
                          LookupVerifier(_docs(EARLY, FORMAL, LATER, third), [EARLY, third, LATER]), TODAY)
    assert [a["document_id"] for a in info["added"]] == ["third", "later"]


def test_lookup_year_comes_from_the_post_date_when_the_header_has_none():
    sources = {"noyear": NS(published_date="2025-03-12")}
    r = _result("CONTRADICTED", EARLY, [EARLY])
    out, info = sd.lookup("no classes on march 12", "no classes on march 12", r,
                          LookupVerifier(_docs(EARLY, NOYEAR), [EARLY]), TODAY, sources)
    assert info is None                                       # dated 2025, the message is about 2026


@pytest.mark.parametrize("message, docs", [
    ("no classes on march 12", (EARLY, LATER)),               # already shown
    ("no classes on march 12", (EARLY, OLD_YEAR)),            # another year
    ("no classes on march 20", (EARLY, LATER)),               # nothing dated that day
    ("no classes next week", (EARLY, LATER)),                 # no explicit date
])
def test_lookup_adds_nothing(message, docs):
    sd._INDEX.update(documents=None, index={})
    r = _result("CONTRADICTED", EARLY, [EARLY, LATER] if LATER in docs else [EARLY])
    out, info = sd.lookup(message, message, r, LookupVerifier(_docs(*docs), list(docs)), TODAY)
    assert info is None and out is r


# Rule J, Z v2 (judged after finding by header date; guard G; results/round2/same_day_judge_plan.md)
MOVED = NS(document_id="moved", score=0.5, text="DYCI ADVISORY April 15, 2026 The midterm examinations scheduled today "
                                                "are moved to April 20.")


class JudgeVerifier(LookupVerifier):
    def __init__(self, documents, ranked, labels):
        super().__init__(documents, ranked)
        by_text = {d["text"]: d["document_id"] for d in documents}
        self._classifier = NS(classify=lambda claim, premise: NS(label=labels[by_text[premise]], confidence=0.9))


def test_judge_off_by_default(monkeypatch):
    monkeypatch.delenv(sd.JUDGE_ENV, raising=False)
    assert not sd.judge_enabled()


def test_judge_finds_by_header_date_and_supports():
    sd._INDEX.update(documents=None, index={})
    r = _result("CONTRADICTED", EARLY, [EARLY])
    v = JudgeVerifier(_docs(EARLY, FORMAL), [EARLY], {"formal": "SUPPORTED", "early": "CONTRADICTED"})
    out, info = sd.judge_same_day("no classes on march 12", "no classes on march 12", r, v, TODAY)
    assert out.verdict == "SUPPORTED" and out.used_evidence.document_id == "formal" and info["action"] == "supported"
    assert out.retrieved_evidence[0].document_id == "formal"


def test_guard_g_blocks_a_moved_event_from_confirming_the_old_date():
    april = dt.date(2026, 4, 15)
    early = NS(document_id="e", score=0.9, text="COLLEGE ADVISORY April 13, 2026 Midterm exams are on April 15.")
    r = _result("SUPPORTED", early, [early])
    v = JudgeVerifier(_docs(early, MOVED), [early], {"moved": "SUPPORTED", "e": "SUPPORTED"})
    out, info = sd.judge_same_day("midterm exams are on april 15", "midterm exams are on april 15", r, v, april)
    assert info["advisories"][0]["guard_g"] is True and info["action"] == "stale_withheld"
    assert out.verdict == "INSUFFICIENT_EVIDENCE"
    out, info = sd.judge_same_day("the midterm on april 15 was moved", "the midterm on april 15 was moved",
                                  _result("INSUFFICIENT_EVIDENCE", early, [early]), v, april)
    assert out.verdict == "SUPPORTED" and not info["advisories"][0]["guard_g"]


@pytest.mark.parametrize("message, advisory, label, expected", [
    ("exams on april 15", MOVED.text, "SUPPORTED", ("INSUFFICIENT_EVIDENCE", True)),
    ("exams on april 15 were postponed", MOVED.text, "SUPPORTED", ("SUPPORTED", False)),
    ("no classes on march 12", FORMAL.text, "SUPPORTED", ("SUPPORTED", False)),            # no change word
    ("exams on april 15", MOVED.text, "CONTRADICTED", ("CONTRADICTED", False)),            # only SUPPORTED is guarded
])
def test_guard_g(message, advisory, label, expected):
    assert sd.guard_g(message, advisory, label) == expected


def test_judge_does_nothing_when_the_post_used_is_the_advisory():
    sd._INDEX.update(documents=None, index={})
    r = _result("SUPPORTED", FORMAL, [FORMAL])
    out, info = sd.judge_same_day("no classes on march 12", "no classes on march 12", r,
                                  JudgeVerifier(_docs(FORMAL), [FORMAL], {"formal": "SUPPORTED"}), TODAY)
    assert info is None and out is r


# Rule J v3 (results/round2/same_day_judge_v3_plan.md)
ABOUT_OTHER = NS(document_id="other_day_adv", score=0.5, text="COLLEGE ADVISORY March 12, 2026 Classes on March 17 "
                                                              "will be online.")


@pytest.mark.parametrize("text, expected", [
    (FORMAL.text, True),                                                    # header only
    ("DYCI ADVISORY March 12, 2026 Classes are suspended today, March 12.", True),
    ("DYCI 2ND ADVISORY March 12, 2026 Cleared today, March 12. Classes on March 13-14 are online.", True),
    (ABOUT_OTHER.text, False),                                              # dated D, about another day
])
def test_v3_about_day(text, expected):
    assert sd.about_day(text, 3, 12) is expected


def test_v3_an_advisory_dated_d_about_another_day_is_not_used():
    sd._INDEX.update(documents=None, index={})
    early = NS(document_id="e", score=0.9, text="ANNOUNCEMENT March 9, 2026 The blood drive is on March 12.")
    r = _result("SUPPORTED", early, [early])
    v = JudgeVerifier(_docs(early, ABOUT_OTHER), [early], {"other_day_adv": "INSUFFICIENT_EVIDENCE", "e": "SUPPORTED"})
    out, info = sd.judge_same_day("the blood drive is on march 12", "the blood drive is on march 12", r, v, TODAY)
    assert info is None and out is r


@pytest.mark.parametrize("advisory", [
    "DYCI ADVISORY October 11, 2028 The expo scheduled today will no longer be held.",
    "DYCI ADVISORY October 11, 2028 The fair will not push through today.",
    "DYCI ADVISORY October 11, 2028 The fair will not be held today.",
])
def test_v3_guard_g_phrases(advisory):
    assert sd.guard_g("the career fair is on october 11", advisory, "SUPPORTED") == ("INSUFFICIENT_EVIDENCE", True)


def test_v3_withdrawn_is_not_a_change_word():
    adv = "DYCI ADVISORY July 26, 2028 The strike was withdrawn. Classes proceed today."
    assert sd.guard_g("there are classes on july 26", adv, "SUPPORTED") == ("SUPPORTED", False)


def test_v3_amendment_1_today_counts_even_with_another_date_in_the_body():
    assert sd.about_day(MOVED.text, 4, 15) is True                          # "scheduled today ... moved to April 20"
    assert sd.about_day(ABOUT_OTHER.text, 3, 12) is False                   # still excluded
