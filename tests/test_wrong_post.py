"""Wrong-post rule (1c, src/verification/wrong_post.py, CAMPUS_VERIFY_WRONG_POST default off;
results/round2/wrong_post_plan.md). Synthetic posts only; the fresh test set is used once, by the plan's run."""
import os
import re
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from src.verification import wrong_post as wp
from src.verification.verifier import VerificationResult

SUSPEND = NS(document_id="sep10", text="DYCI ADVISORY 10 September 2026 In-person classes at all levels are suspended "
                                       "today, September 10, 2026.")
HOLIDAY = NS(document_id="aug21", text="DYCI ADVISORY 20 August 2026 No classes at all levels tomorrow, August 21, 2026, "
                                       "for Ninoy Aquino Day.")
SEMINAR = NS(document_id="sem", text="The DYCI community welcomed LGU representatives on September 23, 2026 for a Campus "
                                     "Safety and Security Seminar.")
MIDTERMS = NS(document_id="mid", text="The Midterm Examinations are scheduled on September 28-30, 2026.")
UNDATED = NS(document_id="und", text="Classes at all levels will push through as scheduled.")


def _result(verdict, post, shown=()):
    return VerificationResult(claim="c", verdict=verdict, evidence_status="RELEVANT_EVIDENCE_FOUND",
                              retrieved_evidence=[post, *shown], used_evidence=post, classification_result=None,
                              explanation="E.")


def _check(message, verdict, post, shown=()):
    r = _result(verdict, post, shown)
    return wp.check(message, r, r.retrieved_evidence)


@pytest.mark.parametrize("message, post", [
    ("The DYCI library will be closed for renovation from November 3 to 7, 2026.", SUSPEND),
    ("DYCI will hold its foundation day celebration on December 5, 2026.", SEMINAR),
])
def test_a_post_about_something_else_is_withheld(message, post):
    out, info = _check(message, "CONTRADICTED", post)
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["withheld"] and info["reason"] == "no_shared_topic_word"
    assert info["original_verdict"] == "CONTRADICTED" and "Wrong-post check" in out.explanation
    assert out.used_evidence is post                                   # the post stays shown


def test_a_better_dated_post_among_those_shown_withholds():
    out, info = _check("I heard classes at DYCI are suspended on August 21.", "CONTRADICTED", SUSPEND, [HOLIDAY])
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["reason"] == "better_dated_post"
    assert info["better_dated_post"] == "aug21"


@pytest.mark.parametrize("message, post", [
    ("No classes on August 22, 2026 for Ninoy Aquino Day.", HOLIDAY),                       # same event, day changed
    ("The midterm examinations are scheduled on October 5 to 7, 2026.", MIDTERMS),            # month changed
    ("DYCI held a Campus Safety and Security Seminar on September 25, 2026.", SEMINAR),
])
def test_a_fake_that_changes_a_detail_keeps_its_contradiction(message, post):
    out, info = _check(message, "CONTRADICTED", post)
    assert out.verdict == "CONTRADICTED" and info["withheld"] is False and info["shared_topic_words"]


@pytest.mark.parametrize("verdict", ["SUPPORTED", "INSUFFICIENT_EVIDENCE"])
def test_only_contradicted_is_ever_touched(verdict):
    out, info = _check("The DYCI library will be closed on November 3, 2026.", verdict, SUSPEND)
    assert out.verdict == verdict and info is None


def test_messages_without_a_date_or_posts_with_the_date_are_left_alone():
    assert _check("The DYCI library is closed for renovation.", "CONTRADICTED", SUSPEND)[1] is None
    assert _check("Classes resume on September 10, 2026.", "CONTRADICTED", SUSPEND)[1] is None


def test_an_undated_post_on_the_same_topic_keeps_its_contradiction():
    out, info = _check("There are no classes at all levels on August 21, 2026.", "CONTRADICTED", UNDATED)
    assert out.verdict == "CONTRADICTED" and "classes" in info["shared_topic_words"]


def test_the_setting_is_on_by_default_since_v1_7(monkeypatch):
    monkeypatch.delenv(wp.ENV, raising=False)
    assert wp.enabled() is True
    monkeypatch.setenv(wp.ENV, "0")
    assert wp.enabled() is False


def test_the_word_list_is_the_one_fixed_in_the_plan():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plan = open(os.path.join(root, "results", "round2", "wrong_post_plan.md"), encoding="utf-8").read()
    listed = set(plan.split("## 5. Fixed word list (not topic words)")[1].split("\n## ")[0].split())
    assert listed == set(wp.TOPIC_STOP)


# ---- the route --------------------------------------------------------------------------------------------------------
def test_the_route_withholds_only_with_the_setting_on(api_settings, monkeypatch):
    post = NS(score=0.86, source_type="facebook_post", source_url="https://www.facebook.com/x", **vars(SUSPEND))
    r = _result("CONTRADICTED", post)
    bodies = {}
    for on in ("0", "1"):
        monkeypatch.setenv(wp.ENV, on)
        app = api_settings()
        with TestClient(app.app) as c:
            monkeypatch.setattr(type(app.state["manager"].current().verifier), "verify_claim", lambda self, text: r)
            bodies[on] = c.post("/verify/text", json={"claim": "The DYCI library will be closed on November 3, 2026."}).json()
    assert bodies["0"]["verdict"] == "CONTRADICTED" and bodies["0"]["wrong_post_check"] is None
    assert bodies["1"]["verdict"] == "INSUFFICIENT_EVIDENCE" and bodies["1"]["wrong_post_check"]["withheld"] is True


def test_the_page_explains_a_withheld_verdict_and_never_sets_one():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = open(os.path.join(root, "frontend", "js", "app.js"), encoding="utf-8").read()
    assert app.count("data.wrong_post_check && data.wrong_post_check.withheld") == 2
    assert not re.search(r"wrong_post_check[^\n]*verdict\s*=(?!=)", app)
