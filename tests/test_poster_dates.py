"""Rule D: picture text, letter-spaced dates are rejoined before checking (results/round2/poster_dates_plan.md)."""
import inspect

import pytest

from src.image import poster_dates as pd


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(pd.ENV, raising=False)
    assert pd.enabled() is False
    monkeypatch.setenv(pd.ENV, "1")
    assert pd.enabled() is True


@pytest.mark.parametrize("before, after", [
    ("ANNOUNCEMENT 16 N 0 VE M B E R 2 0 2 6 FINAL EXAMINATION NOVEMBER 3 0, 20 26 footer",
     "ANNOUNCEMENT 16 NOVEMBER 2026 FINAL EXAMINATION NOVEMBER 30, 2026 footer"),
    ("D E C E M B E R 1 2, 2 0 3 0", "DECEMBER 12, 2030"),
    ("1 5 J U L Y 2 0 3 0 CLASSES", "15 JULY 2030 CLASSES"),
    ("SEP T EMBER 2 2 , 2030", "SEPTEMBER 22, 2030"),
    ("MAY 2 0 2 6 ok", "MAY 2026 ok"),
    ("J U N E 5 classes", "JUNE 5 classes"),
    ("from 2 to 4 D E C 2030", "from 2 to 4 DEC 2030"),
])
def test_spaced_dates_are_rejoined(before, after):
    text, changes = pd.rejoin(before)
    assert text == after and changes


@pytest.mark.parametrize("text", [
    "NOVEMBER 30, 2030", "no classes on may 3", "the march of 1 2 people", "Dr. Ya nga s 2 0 2 6",
    "on November 12, 2026 at 3 PM", "Room 2 0 4 is closed", "",
])
def test_other_text_is_left_exactly_as_it_was(text):
    assert pd.rejoin(text) == (text, [])


def test_wired_into_the_picture_check_only():
    import src.api.main as api
    src = inspect.getsource(api.verify_image)
    assert "poster_dates.enabled()" in src and "poster_dates.rejoin(text)" in src
    assert "poster_dates" not in inspect.getsource(api.verify_text)
