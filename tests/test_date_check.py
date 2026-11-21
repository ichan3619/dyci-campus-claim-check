"""
src/verification/date_check.py and ClaimVerifier(date_check=True). The check may only WITHHOLD a verdict
(SUPPORTED / CONTRADICTED -> INSUFFICIENT_EVIDENCE), never create or upgrade one. No model downloads.
"""
import pytest

from src.classification.result import ClassificationResult
from src.retrieval.retriever import RetrievedDocument
from src.verification.date_check import (check_verdict, event_dates, extract_dates, header_dates,
                                         prefer_dated_documents)
from src.verification.verifier import ClaimVerifier

SHORTENED = ("DYCI Advisory 27 September 2026 SHORTENED CLASS HOURS Please be informed that tomorrow, September 28, "
             "2026, CLASSES at ALL LEVELS and OFFICE OPERATIONS will be until 1:00 PM only.")
MIDTERM = ("COLLEGE ADVISORY 20 September 2026 Please be informed that the Midterm Examinations for the 1st Semester of "
           "AY 2026–2027 are scheduled on September 28–30, 2026.")
SEPT10 = "DYCI ADVISORY 10 September 2026 To all DYCIans, IN-PERSON CLASSES at ALL LEVELS are SUSPENDED today, September 10, 2026."
UNDATED = "Official Notice #101: All classes are suspended tomorrow due to Typhoon Bagyo. Students should stay home."


# ---- extraction -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, dates", [
    ("sept 27 shortened class", {(9, 27)}), ("Sept. 28 to 30", {(9, 28), (9, 29), (9, 30)}),
    ("September 28–30, 2026", {(9, 28), (9, 29), (9, 30)}), ("28 to 30 September", {(9, 28), (9, 29), (9, 30)}),
    ("ika-10 ng Setyembre", {(9, 10)}), ("enrollment august 10", {(8, 10)}), ("Oct 31 to Nov 2", {(10, 31), (11, 2)}),
    ("May 10, 2026", {(5, 10)}),
    ("library 24/7 finals", set()), ("AY 2026–2027", set()), ("classes until 1:00 PM", set()),
    ("may 10 slots pa", set()),            # Tagalog existential 'may' + a count, not the month
    ("september 31", set()),               # not a real date
])
def test_extract_dates(text, dates):
    assert extract_dates(text) == dates


def test_the_post_header_date_is_not_an_event_date():
    assert header_dates(SHORTENED) == {(9, 27)} and event_dates(SHORTENED) == {(9, 28)}
    assert event_dates(MIDTERM) == {(9, 28), (9, 29), (9, 30)}
    assert event_dates(SEPT10) == {(9, 10)}          # the body repeats the header date, so it IS an event date
    assert header_dates(UNDATED) == set() and event_dates(UNDATED) == set()


# ---- the two observed failures, and what must NOT change -------------------------------------------------------------
def test_advisory_date_does_not_support_a_claim_about_that_day():
    assert check_verdict("sept 27 shortened class", "SUPPORTED", SHORTENED) == \
        ("INSUFFICIENT_EVIDENCE", "claimed_date_not_in_evidence")


def test_undated_evidence_cannot_contradict_a_dated_claim():
    assert check_verdict("online classes sept 10 dahil sa habagat", "CONTRADICTED", UNDATED) == \
        ("INSUFFICIENT_EVIDENCE", "evidence_has_no_date")


@pytest.mark.parametrize("claim, verdict, evidence", [
    ("shortened class sept 28", "SUPPORTED", SHORTENED),                                   # date matches
    ("midterm exams on sept 29", "SUPPORTED", MIDTERM),                                    # inside the range
    ("The midterm examinations are scheduled on October 5 to 7, 2026.", "CONTRADICTED", MIDTERM),   # a real date contradiction
    ("Classes will be until 1:00 PM only tomorrow.", "SUPPORTED", SHORTENED),             # no date in the claim
    ("Classes are suspended tomorrow.", "CONTRADICTED", UNDATED),                         # no date in the claim
    ("sept 27 shortened class", "INSUFFICIENT_EVIDENCE", SHORTENED),                      # never upgraded
])
def test_verdicts_the_check_must_leave_alone(claim, verdict, evidence):
    assert check_verdict(claim, verdict, evidence) == (verdict, None)


# ---- candidate preference ---------------------------------------------------------------------------------------------
def _doc(i, text, score):
    return RetrievedDocument(document_id=i, score=score, text=text, source_type="facebook", source_url="")


def test_prefers_gate_passing_documents_that_contain_the_claimed_date():
    a, b = _doc("undated", UNDATED, 0.90), _doc("sept10", SEPT10, 0.85)
    assert prefer_dated_documents("classes suspended sept 10", [a, b]) == [b]
    assert prefer_dated_documents("classes suspended sept 11", [a, b]) == [a, b]     # nobody has it: unchanged
    assert prefer_dated_documents("classes suspended tomorrow", [a, b]) == [a, b]    # no date: unchanged


# ---- ClaimVerifier integration ----------------------------------------------------------------------------------------
class _Retriever:
    default_relevance_threshold = 0.5

    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, top_k=3):
        return self.docs[:top_k]


class _Always:
    def __init__(self, label):
        self.label, self.seen = label, []

    def classify(self, claim, evidence):
        self.seen.append(evidence)
        return ClassificationResult(label=self.label, confidence=0.9, label_scores={self.label: 0.9})


def test_off_by_default_and_unchanged():
    v = ClaimVerifier(_Retriever([_doc("s", SHORTENED, 0.9)]), _Always("SUPPORTED"))
    r = v.verify_claim("sept 27 shortened class")
    assert r.verdict == "SUPPORTED" and r.date_check is None


def test_on_withholds_and_keeps_the_classifier_label_visible():
    v = ClaimVerifier(_Retriever([_doc("s", SHORTENED, 0.9)]), _Always("SUPPORTED"), date_check=True)
    r = v.verify_claim("sept 27 shortened class")
    assert r.verdict == "INSUFFICIENT_EVIDENCE" and r.date_check == "claimed_date_not_in_evidence"
    assert r.classification_result.label == "SUPPORTED" and "withheld" in r.explanation


def test_on_uses_the_dated_document_but_never_one_below_the_gate():
    clf = _Always("SUPPORTED")
    docs = [_doc("undated", UNDATED, 0.90), _doc("sept10", SEPT10, 0.85), _doc("low", SEPT10 + " x", 0.40)]
    r = ClaimVerifier(_Retriever(docs), clf, date_check=True).verify_claim("classes suspended on september 10")
    assert r.used_evidence.document_id == "sept10" and r.verdict == "SUPPORTED"
    r = ClaimVerifier(_Retriever([docs[0], docs[2]]), clf, date_check=True).verify_claim("classes suspended on september 10")
    assert r.used_evidence.document_id == "undated" and r.verdict == "INSUFFICIENT_EVIDENCE"


def test_api_reports_the_date_check(api_settings):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    for flag in (True, False):
        api_main = api_settings(date_check=flag)
        with TestClient(api_main.app) as c:
            assert c.get("/health").json()["date_check"] is flag
            assert api_main.state["manager"].current().verifier._date_check is flag
