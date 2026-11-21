"""Rule M version 2: a SUPPORTED message with a reason marker has its main point checked alone
(results/round2/main_point_v2_plan.md). Examples are made up or from set M (no longer blind); none from the v2 set."""
import inspect
from types import SimpleNamespace

import pytest

from src.verification import main_point as mp

FAKE = ("Dr. Yanga's Colleges, Inc. ANNOUNCEMENT 27 SEPTEMBER 2026 NO CLASSES SEPTEMBER 28, 2026 ALL LEVELS IN "
        "OBSERVANCE OF THE 34TH NATIONAL FAMILY WEEK PURSUANT TO CIRCULAR MEMORANDUM NO. 136")


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(mp.ENV, raising=False)
    assert mp.enabled() is False


def test_version():
    assert mp.VERSION == "main-point-v2"


def test_split_at_the_first_marker():
    main, phrase = mp.split(FAKE)
    assert main == "Dr. Yanga's Colleges, Inc. ANNOUNCEMENT 27 SEPTEMBER 2026 NO CLASSES SEPTEMBER 28, 2026 ALL LEVELS"
    assert phrase == "in observance of"


@pytest.mark.parametrize("text, phrase", [
    # version 1's list
    ("no classes on july 9 due to the storm", "due to"), ("Classes end at noon to give way to the assembly", "to give way to"),
    ("Online classes, in view of the rainfall warning", "in view of"), ("walang pasok dahil sa bagyo", "dahil sa"),
    ("closed as per memo 3", "as per"), ("held in line with the program", "in line with"),
    # added in version 2
    ("Library open until 8 PM because the books will be sprayed", "because"),
    ("College classes online since the rooms are used for testing", "since"),
    ("Classes held online following the town memorandum", "following"),
    ("No classes so that sections can hold parties", "so that"),
    ("Gym closed in order to repair the roof", "in order to"),
    ("No classes to allow all sections to attend the program", "to allow"),
    ("Morning classes end early to make way for the fair", "to make way for"),
    ("NO CLASSES ON JULY 12 THIS IS TO GIVE WAY TO THE ASSEMBLY", "this is"),
    ("Classes were suspended. This was ordered by the mayor", "this was"),
])
def test_markers(text, phrase):
    assert mp.split(text)[1] == phrase


@pytest.mark.parametrize("text, main, phrase", [
    ("Classes resumed since Monday and the gym is closed since the roof leaks",
     "Classes resumed since Monday and the gym is closed", "since"),                 # the first "since" is guarded
    ("The registrar has been open since 8:00 AM, closed since the staff are on training",
     "The registrar has been open since 8:00 AM, closed", "since"),
    ("Send the following documents; classes online following the memo",
     "Send the following documents; classes online", "following"),
    ("Exams are due to start at 8 AM, classes suspended due to the storm",
     "Exams are due to start at 8 AM, classes suspended", "due to"),
])
def test_guards_skip_to_the_next_marker(text, main, phrase):
    assert mp.split(text) == (main, phrase)


@pytest.mark.parametrize("text", [
    "Classes online since Monday", "Open since 2030", "Closed since then", "Shortened classes since early morning",
    "The following sections are online", "Exam schedule following: 8 AM", "Exam following is on Friday",
    "Exams are due to start at 8 AM", "Classes are due to resume on Monday", "The gym is due to be repainted",
    "no classes on july 9 so all sections can attend", "Classes suspended for the parade", "Online as the gym is used",
    "classes are suspended", "", "the overdue tour",
])
def test_no_split(text):
    assert mp.split(text) is None


@pytest.mark.parametrize("text, main", [
    ("REGISTRAR AUGUST 5, 2030 THE CLOSURE IS DUE TO THE ANNUAL INVENTORY", "REGISTRAR AUGUST 5, 2030"),
    ("Gym closed on Monday, it is because of repairs", "Gym closed on Monday"),
    ("Online classes on Friday; our classes are due to flooding", "Online classes on Friday"),
])
def test_dangling_clause_trimmed(text, main):
    assert mp.split(text)[0] == main


@pytest.mark.parametrize("text", [
    "Update for July 12, 2030: this is to give way to the assembly",
    "Advisory for October 25, 2030: this is due to the power interruption",
    "ANNOUNCEMENT August 1, 2030 due to the inventory",
    "Dr. Yanga's Colleges, Inc. DYCI Bocaue, Bulacan, Monday, 5 Aug: due to the storm",
    "This is due to the storm",
])
def test_no_content_word_no_split(text):
    assert mp.split(text) is None


@pytest.mark.parametrize("text, yes", [
    ("Midterms, August 29, 2030", True), ("REGISTRAR AUGUST 5, 2030", True), ("NO CLASSES ON JULY 12, 2030", True),
    ("Update for July 12, 2030:", False), ("Monday 5th, tomorrow morning", False), ("Yanga's Colleges Inc.", False),
    ("It is OK", False),
])
def test_has_content_word(text, yes):
    assert mp.has_content_word(text) is yes


@pytest.mark.parametrize("inner, outcome", [
    ("SUPPORTED", "main_supported"), ("CONTRADICTED", "main_not_supported"),
    ("INSUFFICIENT_EVIDENCE", "main_not_supported"), (None, "main_not_supported"),
])
def test_decide_never_contradicted(inner, outcome):
    assert mp.decide(inner) == outcome


def _model(verdict, post):
    from src.api.schemas import VerificationResultModel
    fields = VerificationResultModel.model_fields
    m = VerificationResultModel.model_construct(**{k: None for k in fields})
    return m.model_copy(update={"claim": "whole", "verdict": verdict, "explanation": "e.",
                                "used_evidence": SimpleNamespace(document_id=post) if post else None})


@pytest.mark.parametrize("inner_verdict, inner_post, verdict, outcome", [
    ("CONTRADICTED", "A", "INSUFFICIENT_EVIDENCE", "main_not_supported"),
    ("CONTRADICTED", "B", "INSUFFICIENT_EVIDENCE", "main_not_supported"),
    ("INSUFFICIENT_EVIDENCE", "A", "INSUFFICIENT_EVIDENCE", "main_not_supported"),
    ("SUPPORTED", "A", "SUPPORTED", "main_supported"),
])
def test_apply(monkeypatch, inner_verdict, inner_post, verdict, outcome):
    import src.api.main as api
    seen = {}

    def fake_answer(verifier, text, **kw):
        seen["text"], seen["active"] = text, api._MAIN_POINT_ACTIVE.get()
        return _model(inner_verdict, inner_post)

    monkeypatch.setattr(api, "_answer", fake_answer)
    out = api._apply_main_point(_model("SUPPORTED", "A"), FAKE, None, None, None, None, None, False)
    assert seen["text"].endswith("NO CLASSES SEPTEMBER 28, 2026 ALL LEVELS") and seen["active"] is True
    assert out.verdict == verdict and out.main_point["outcome"] == outcome and out.claim == "whole"
    assert out.main_point["version"] == "main-point-v2" and out.main_point["inner_post"] == inner_post
    assert out.used_evidence.document_id == "A"                 # the whole message's post is kept
    assert api._MAIN_POINT_ACTIVE.get() is False


@pytest.mark.parametrize("text", ["OK due to the storm", "Update for July 12: this is to give way to the assembly"])
def test_too_short_a_main_point_changes_nothing(monkeypatch, text):
    import src.api.main as api
    monkeypatch.setattr(api, "_answer", lambda *a, **k: pytest.fail("not re-checked"))
    m = _model("SUPPORTED", "A")
    assert api._apply_main_point(m, text, None, None, None, None, None, False) is m


def test_wired_after_rule_p_on_supported_only():
    import src.api.main as api
    src = inspect.getsource(api._answer)
    assert src.index("_apply_reason_parts") < src.index("_apply_main_point")
    assert 'main_point.enabled() and model.verdict == "SUPPORTED" and not _MAIN_POINT_ACTIVE.get()' in src


def test_never_contradicted_in_code():
    import src.api.main as api
    assert '"CONTRADICTED"' not in inspect.getsource(api._apply_main_point)
