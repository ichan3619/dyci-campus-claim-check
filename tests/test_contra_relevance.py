"""Rule K: a "Contradicted" answer needs a post about the same thing (results/round2/contra_relevance_plan.md)."""
import pytest

from src.verification import contra_relevance as k

FLOOD = ("DYCI Advisory 29 August 2026 The DYCI Elida Campus has opened its doors to welcome families in our vicinity "
         "affected by the ongoing floods and rains.")
MIDTERMS = "COLLEGE ADVISORY The Midterm Examinations for the 1st Semester are scheduled on September 28-30, 2026."


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(k.ENV, raising=False)
    assert k.enabled() is False
    monkeypatch.setenv(k.ENV, "1")
    assert k.enabled() is True


def test_an_unrelated_post_cannot_contradict():
    out = k.check("dyci is opening a new campus in malolos next year", "CONTRADICTED", FLOOD)
    assert out["version"] == k.VERSION and out["share"] == 0.2 and out["found"] == ["campus"] and "below" not in out


@pytest.mark.parametrize("claim", [
    "The midterm examinations are scheduled on October 5 to 7, 2026.",   # a wrong date: the subject matches
    "midterm exams on oct 5",
])
def test_a_real_contradiction_stays(claim):
    assert k.check(claim, "CONTRADICTED", MIDTERMS) is None


def test_exactly_one_third_stays():
    assert k.check("eh, 11 ang fiesta diba?", "CONTRADICTED", "The Fiesta Mass is on October 12") is None
    assert k.check("eh fiesta diba sana", "CONTRADICTED", "The Fiesta Mass is on October 12") is not None   # 1 of 4


@pytest.mark.parametrize("verdict", ["SUPPORTED", "INSUFFICIENT_EVIDENCE", None])
def test_only_contradicted_is_touched(verdict):
    assert k.check("dyci is opening a new campus in malolos next year", verdict, FLOOD) is None


def test_no_post_or_no_content_words():
    assert k.check("dyci is opening a new campus", "CONTRADICTED", "") is None
    assert k.check("is it on october 12?", "CONTRADICTED", FLOOD) is None


def test_layer_is_last_and_only_turns_contradicted_into_not_confirmed():
    import inspect
    import src.api.main as api
    src = inspect.getsource(api._answer)
    assert 'contra_relevance.enabled() and model.verdict == "CONTRADICTED"' in src
    assert '"verdict": "INSUFFICIENT_EVIDENCE", "contra_relevance": found' in src
    assert src.index("reason_parts.enabled()") < src.index("contra_relevance.enabled()")
    assert src.rstrip().endswith("return model")
