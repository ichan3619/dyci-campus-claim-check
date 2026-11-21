"""Rule C, class wording written the standard way (results/round2/class_wording_plan.md)."""
import inspect

import pytest

from src.normalization import class_wording as cw


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(cw.ENV, raising=False)
    assert cw.enabled() is False


@pytest.mark.parametrize("message,rewritten,rules", [
    ("no class on november 12", "no classes on november 12", ["plural"]),
    ("november 12 no class", "no classes on november 12", ["date_first", "plural"]),
    ("nov 12, 2026 - classes suspended", "classes suspended on nov 12, 2026", ["date_first"]),
    ("sept 9: no class", "no classes on sept 9", ["date_first", "plural"]),
    ("dec 5 no f2f class dyci", "no f2f classes dyci on dec 5", ["date_first", "plural"]),
    ("the class is suspended on july 3", "the classes are suspended on july 3", ["plural"]),
    ("No Class on Nov 12", "No Classes on Nov 12", ["plural"]),
    ("NO CLASS NOV 12", "NO CLASSES NOV 12", ["plural"]),
])
def test_rewrites(message, rewritten, rules):
    r = cw.rewrite(message)
    assert r["rewritten"] == rewritten and r["rules"] == rules and r["original"] == message


@pytest.mark.parametrize("message", [
    "no classes on november 12", "classes are suspended on september 9", "november 12", "no classmates on nov 12",
    "nov 12 seminar at the avr", "the program is on nov 12", "", "walang pasok sa november 12",
])
def test_unchanged(message):
    assert cw.rewrite(message) is None


def test_wired_after_translation_and_dates():
    import src.api.main as api
    src = inspect.getsource(api._answer)
    assert src.index("relative_resolve.resolve(message)") < src.index("class_wording.rewrite(message)") \
        < src.index("holidays.note(message")
    assert '"class_wording": wording' in src
