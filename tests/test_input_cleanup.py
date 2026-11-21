"""Input hardening (results/round2/input_hardening_plan.md): academic year and lookalike letters. Examples are made up;
none from the red-team or phrasing sets."""
import inspect
import os

import pytest

from src.normalization import input_cleanup as ic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(ic.ENV_YEAR, raising=False)
    monkeypatch.delenv(ic.ENV_LOOKALIKE, raising=False)
    assert ic.year_enabled() is False and ic.lookalike_enabled() is False
    assert ic.clean("Finals for AY 2026-2027 are on May 4") == ("Finals for AY 2026-2027 are on May 4", None)


@pytest.mark.parametrize("text, out", [
    ("Finals for AY 2026-2027 are on May 4", "Finals for A.Y. 2026-2027 are on May 4"),
    ("Finals for AY 2026–2027", "Finals for A.Y. 2026–2027"),
    ("Finals for AY2026-27", "Finals for A.Y.2026-27"),
    ("Finals for AY 2026 - 27 start", "Finals for A.Y. 2026 - 27 start"),
])
def test_academic_year(text, out):
    assert ic.academic_year(text)[0] == out


@pytest.mark.parametrize("text", [
    "Ang finals ay 2026-2027 pa",            # lowercase Tagalog "ay" is never touched
    "The AY committee meets on May 4",        # no year range
    "SAY 2026-2027",                          # part of a word
    "A.Y. 2026-2027",                         # already written that way
    "AY 2026 only",
])
def test_academic_year_untouched(text):
    assert ic.academic_year(text) == (text, 0)


def test_academic_year_passes_the_language_gate():
    from src.api.verdict_policy import withhold_for_language
    text = "The finals for the 1st Semester of AY 2026-2027 are on December 1-3, 2026."
    assert withhold_for_language(text) is True                      # the frozen gate, as it is
    assert withhold_for_language(ic.academic_year(text)[0]) is False
    assert withhold_for_language("Ang finals ay sa December 1") is True


@pytest.mark.parametrize("text, out", [
    ("nо clаsses оn Mоnday", "no classes on Monday"),           # Cyrillic inside Latin words
    ("ΑUGUST EXΑMINΑTIΟN", "AUGUST EXAMINATION"),             # Greek capitals
    ("Аnd аt", "And at"),
    ("Classes ае today", "Classes ae today"),                              # all table letters: whole message
])
def test_lookalike(text, out):
    got, n = ic.lookalike(text)
    assert got == out and n > 0


@pytest.mark.parametrize("text", [
    "Привет мир",                      # real Russian, no Latin letters
    "Ελλάδα",                                          # real Greek
    "No classes on Monday",                                                           # plain Latin
    "",
])
def test_lookalike_untouched(text):
    assert ic.lookalike(text) == (text, 0)


def test_russian_word_kept_next_to_a_mixed_word():
    text = "мир clаsses"                     # Russian word (not all in the table) + a mixed word
    assert ic.lookalike(text)[0] == "мир classes"


def test_clean_reports_counts(monkeypatch):
    monkeypatch.setenv(ic.ENV_YEAR, "1")
    monkeypatch.setenv(ic.ENV_LOOKALIKE, "1")
    out, info = ic.clean("Midterms аre for AY 2026-2027")
    assert out == "Midterms are for A.Y. 2026-2027"
    assert info == {"version": "input-cleanup-v1", "academic_year": 1, "lookalike": 1}


def test_wired_after_fold_before_the_language_gate():
    import src.api.main as api
    src = inspect.getsource(api._answer)
    assert src.index("fold(text)") < src.index("input_cleanup.clean(message)") < src.index("tagalog_phrases.enabled()")
    assert '"input_cleanup": cleanup' in src


def test_text_box_limit_matches_the_api():
    from src.api.schemas import MAX_CLAIM_CHARS
    html = open(os.path.join(ROOT, "frontend", "index.html"), encoding="utf-8").read()
    assert f'maxlength="{MAX_CLAIM_CHARS}"' in html
