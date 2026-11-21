"""
Round-2 value check (src/verification/value_check.py). Invented examples only: the M12.3 independent messages are
never used to build or test fixes.
"""
from src.verification.value_check import check_values, extract_values, find_conflicts

NOTICE = ("CAMPUS ADVISORY 3 March 2026. The leadership camp will run for 3 days, March 9-11, at the Main Gym. "
          "Registration closes at 5:00 PM. 48 out of 50 slots are taken, a 96% fill rate. Guests arrive from Japan.")


def test_extracts_each_kind():
    v = extract_values("tatlong araw ba yung camp sa japan, 5pm, 96%, 48 out of 50, march 10?")
    assert v["duration"] == {(3, "day")} and v["place"] == {"japan"} and v["time"] == {(17, 0)}
    assert v["percent"] == {96.0} and v["count_of"] == {(48, 50)} and v["date"] == {(3, 10)}


def test_hyphenated_duration_and_number_words():
    assert extract_values("a 2-day seminar")["duration"] == {(2, "day")}
    assert extract_values("two weeks lang daw")["duration"] == {(2, "week")}


def test_tagalog_may_is_not_a_month_and_bare_numbers_are_ignored():
    assert "date" not in extract_values("may 2 sections pa ba?")
    assert extract_values("room 104 lang") == {}


def test_header_date_is_not_an_event_date():
    assert extract_values(NOTICE, event_only=True)["date"] == {(3, 9), (3, 10), (3, 11)}


def test_a_wrong_value_in_the_judged_passage_creates_contradicted():
    premise = "The leadership camp will run for 3 days, March 9-11, at the Main Gym."
    assert check_values("4 days daw yung camp?", "INSUFFICIENT_EVIDENCE", premise, NOTICE) == ("CONTRADICTED", "value_conflict")
    assert check_values("march 12 ba yung camp?", "INSUFFICIENT_EVIDENCE", premise, NOTICE)[0] == "CONTRADICTED"


def test_a_value_inside_a_range_or_elsewhere_in_the_post_is_not_a_conflict():
    premise = "The leadership camp will run for 3 days, March 9-11, at the Main Gym."
    assert check_values("march 10 yung camp", "SUPPORTED", premise, NOTICE) == ("SUPPORTED", None)
    assert find_conflicts("registration until 5pm", premise, NOTICE) == []          # no time in the premise


def test_no_competing_value_in_the_premise_means_no_change():
    premise = "Registration closes soon."
    assert check_values("4 days daw yung camp?", "INSUFFICIENT_EVIDENCE", premise, NOTICE) == ("INSUFFICIENT_EVIDENCE", None)


def test_supported_with_a_conflict_is_withheld_and_contradicted_is_never_changed():
    premise = "48 out of 50 slots are taken, a 96% fill rate."
    assert check_values("45 out of 50 na ang puno", "SUPPORTED", premise, NOTICE) == \
           ("INSUFFICIENT_EVIDENCE", "value_conflict_withheld")
    assert check_values("45 out of 50 na ang puno", "CONTRADICTED", premise, NOTICE) == ("CONTRADICTED", None)


def test_places_compare_by_canonical_name():
    premise = "Guests arrive from Japan."
    assert check_values("taga Korea daw yung guests", "INSUFFICIENT_EVIDENCE", premise, NOTICE)[0] == "CONTRADICTED"
    assert check_values("taga hapon daw yung guests", "SUPPORTED", premise, NOTICE) == ("SUPPORTED", None)


def test_no_premise_means_no_change():
    assert check_values("4 days daw", "INSUFFICIENT_EVIDENCE", None, None) == ("INSUFFICIENT_EVIDENCE", None)


# ---- wiring: ClaimVerifier(value_check=...) and CAMPUS_VERIFY_VALUE_CHECK ------------------------------------------
from src.classification.result import ClassificationResult  # noqa: E402
from src.retrieval.retriever import RetrievedDocument  # noqa: E402
from src.verification.verifier import ClaimVerifier  # noqa: E402


class _Retriever:
    default_relevance_threshold = 0.5

    def __init__(self, text):
        self.doc = RetrievedDocument(document_id="camp", score=0.9, text=text, source_type="facebook", source_url="")

    def retrieve(self, query, top_k=3):
        return [self.doc]


class _Neutral:
    def classify(self, claim, evidence):
        return ClassificationResult(label="INSUFFICIENT_EVIDENCE", confidence=0.9, label_scores={"INSUFFICIENT_EVIDENCE": 0.9})


def test_verifier_off_by_default_changes_nothing():
    r = ClaimVerifier(_Retriever(NOTICE), _Neutral()).verify_claim("4 days daw yung camp?")
    assert r.verdict == "INSUFFICIENT_EVIDENCE" and r.value_check is None


def test_verifier_on_applies_the_committed_rule_after_the_nli():
    r = ClaimVerifier(_Retriever(NOTICE), _Neutral(), value_check=True).verify_claim("4 days daw yung camp?")
    assert (r.verdict, r.value_check) == ("CONTRADICTED", "value_conflict")
    assert r.classification_result.label == "INSUFFICIENT_EVIDENCE" and "Value check" in r.explanation


def test_api_setting_defaults_off_and_is_reported(api_settings):
    from fastapi.testclient import TestClient
    from src.api.pipeline import Settings
    assert Settings().value_check is False
    for flag in (True, False):
        api_main = api_settings(value_check=flag)
        with TestClient(api_main.app) as c:
            assert c.get("/health").json()["value_check"] is flag
            assert api_main.state["manager"].current().verifier._value_check is flag
