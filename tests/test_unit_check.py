"""Rule N, swapped office, level, grade or venue names (results/round2/unit_check_plan.md)."""
import pytest

from src.verification import unit_check as uc

ACCOUNTING = ("DYCI ADVISORY 6 September 2030 The Accounting Office is closed today, September 6, 2030. This is for the "
              "semestral audit of financial records. Please be guided.")
SHS = ("DYCI ADVISORY 18 October 2030 There are no classes in the Senior High School department today, October 18, "
       "2030. This is to give way to the Grade 12 career assessment test.")
GYM = ("COLLEGE ADVISORY 5 December 2030 The Christmas Lantern Parade tomorrow, December 6, 2030, is moved from the Main "
       "Quadrangle to the Gymnasium. It will start at 3:00 PM.")
ALL = "DYCI ADVISORY 15 July 2030 Classes in all levels are suspended today, July 15, 2030, due to heavy rain."


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(uc.ENV, raising=False)
    assert uc.enabled() is False


def test_units():
    assert uc.units("the registrar's office is closed") == {"office": frozenset({"registrar"})}
    assert uc.units("no shs classes")["level"] == {"senior high school"}
    assert uc.units("grade 11 and Grade 7")["grade"] == {"grade 11", "grade 7"}
    assert uc.units("moved to the gym")["venue"] == {"gymnasium"}
    assert uc.units("COLLEGE ADVISORY 5 December 2030 The parade") == {}          # header label only
    assert uc.units("Dr. Yanga's Colleges, Inc.") == {}
    assert uc.units("classes end 2 hr early") == {}
    assert uc.units("no classes on july 10") == {}


def test_swapped_office_is_a_conflict():
    c = uc.conflict("registrar's office closed on september 6 2030 for the semestral audit", ACCOUNTING, ACCOUNTING)
    assert c["kind"] == "office" and c["message_unit"] == "registrar" and c["post_units"] == ["accounting"]
    assert "Accounting" not in uc.note(c) and "accounting" in uc.note(c)


@pytest.mark.parametrize("message", [
    "accounting office closed on september 6", "cashier closed on september 6",
    "no classes on september 6", "the office is closed on september 6 for the audit",
])
def test_same_or_no_unit_is_not_a_conflict(message):
    assert uc.conflict(message, ACCOUNTING, ACCOUNTING) is None


def test_levels_and_grades():
    assert uc.conflict("no college classes on october 18", SHS, SHS)["message_unit"] == "college"
    assert uc.conflict("no shs classes on october 18", SHS, SHS) is None
    assert uc.conflict("grade 11 career assessment test on october 18", SHS, SHS)["kind"] == "grade"
    assert uc.conflict("grade 12 career assessment test on october 18", SHS, SHS) is None


def test_venues_named_anywhere_in_the_post_are_not_conflicts():
    assert uc.conflict("lantern parade on december 6 moved to the auditorium", GYM, GYM)["message_unit"] == "auditorium"
    assert uc.conflict("lantern parade on december 6 moved to the gym", GYM, GYM) is None
    # the main quadrangle is in the post (where it was moved from): not a swap the rule can see
    assert uc.conflict("lantern parade on december 6 at the quadrangle", GYM, GYM) is None


def test_covering_words():
    assert uc.conflict("no college classes on july 15", ALL, ALL) is None
    assert uc.conflict("grade 9 has no classes on july 15", ALL, ALL) is None
    closed = "DYCI ADVISORY The campus will be closed tomorrow, September 4, 2030. Senior High School students stay home."
    assert uc.conflict("no college classes on september 4", closed, closed) is None
    offices = "DYCI ADVISORY All offices are closed on December 10, 2030, starting with the Registrar."
    assert uc.conflict("accounting office closed on december 10", offices, offices) is None


def test_premise_without_a_unit_of_that_kind_is_not_a_conflict():
    post = "DYCI ADVISORY 9 July 2030 There will be no classes tomorrow, July 10, 2030, due to the power interruption."
    assert uc.conflict("no college classes on july 10", post, post) is None


def test_route_withholds_only_supported(monkeypatch):
    """The layer in _answer: SUPPORTED -> INSUFFICIENT_EVIDENCE with unit_check; other verdicts untouched."""
    import inspect
    import src.api.main as api
    src = inspect.getsource(api._answer)
    assert 'unit_check.enabled() and model.verdict == "SUPPORTED"' in src
    assert '"verdict": "INSUFFICIENT_EVIDENCE", "unit_check": found' in src
