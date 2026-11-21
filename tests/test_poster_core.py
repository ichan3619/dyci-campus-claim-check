"""Rule E: picture text, the poster's core is checked without its seal, header and footer
(results/round2/poster_core_plan.md)."""
import inspect

import pytest

from src.image import poster_core as pc
from src.image.poster_dates import rejoin

FINAL = ("YANGAS COLLECES x < Jq)C * * * 5 HOCAUE) BULA 2 Dr - Ya nga' s colle g e $ Inc. ANNOUNCEMENT 16 N 0 VE M B E R "
         "2 0 2 6 FINAL EXAMINATION FOR IST SEMESTER, A.Y 2026-2027 NOVEMBER 3 0, 20 26 Dr. Yanga's colleges, Inc "
         "dyci.edu.ph")


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(pc.ENV, raising=False)
    assert pc.enabled() is False


def test_the_final_examination_poster_core():
    text, info = pc.core(rejoin(FINAL)[0])
    assert text == "FINAL EXAMINATION FOR IST SEMESTER, A.Y 2026-2027 NOVEMBER 30, 2026"
    assert info["poster_date"] == "16 NOVEMBER 2026" and "dyci.edu.ph" in info["removed"]


@pytest.mark.parametrize("name, ok", [
    ("Dr. Yanga's Colleges, Inc.", True), ("Dr - Ya nga' s colle g e $ Inc.", True), ("Dr. Yanga's Colleqes, lnc.", True),
    ("DR. YANGA S COLLEGES INC", True), ("Dr. Yanga Hospital Inc.", False), ("Bulacan State University", False),
])
def test_school_name_tolerance(name, ok):
    toks = (name + " ANNOUNCEMENT NO CLASSES ON JULY 18, 2030 ALL LEVELS").split()
    assert bool(pc._school_spans(toks)) is ok


@pytest.mark.parametrize("text", [
    "classes suspended on november 12 per dyci advisory",       # nothing left after the "header": unchanged
    "no classes on november 12",
    "",
])
def test_text_without_a_poster_layout_is_unchanged(text):
    assert pc.core(text) == (text, None)


def test_wired_into_the_picture_check_only():
    import src.api.main as api
    src = inspect.getsource(api.verify_image)
    assert "poster_core.enabled()" in src and src.index("poster_dates.rejoin") < src.index("poster_core.core")
    assert "poster_core" not in inspect.getsource(api.verify_text)
