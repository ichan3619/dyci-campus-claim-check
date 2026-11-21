"""Rule G: picture text, the school's own name and one stray misread word do not withhold the verdict
(results/round2/poster_language_plan.md)."""
from types import SimpleNamespace

import pytest

from src.image import poster_language as pl

POSTER = ("NANGA S COLLEGES ~ 95i 3 BOCAUE> BULA Dr. Ya nga' s Colleges, Inc. ANNOUNCEMENT 1 0 NO VE M B E R 2 0 2 6 "
          "NO CLASSES IN ALL LEVELS & MO OFFICE OPERATIONS NOVEMBER 11, 2026 IN OBSERVANCE OF THE CELEBRATION OF THE "
          "FEAST OF ST. MARTIN OF TOURS Dr. Yanga's Colleges, Inc_ dyci.edu.ph")


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(pl.ENV, raising=False)
    assert pl.enabled() is False
    monkeypatch.setenv(pl.ENV, "1")
    assert pl.enabled() is True


@pytest.mark.parametrize("tokens, names", [
    (["dr", "ya", "nga", "s", "colleges"], {0, 1, 2, 3}),
    (["ya", "ngas", "colleges"], {0, 1}),
    (["dryangas", "colleges"], {0}),
    (["yanga", "s", "colleges"], {0, 1}),
    (["ya", "nga", "ang", "klase"], {0, 1}),         # the plan's rule: any run spelling "yanga" is the name
    (["oo", "nga", "ang", "klase"], set()),
    (["nga", "po"], set()),
])
def test_school_name_positions(tokens, names):
    assert pl.school_name_positions(tokens) == names


def test_the_10_november_poster_is_released():
    from src.api.verdict_policy import withhold_for_language
    assert withhold_for_language(POSTER)                    # the gate withholds it today ("nga", "mo")
    found = pl.decide(POSTER)
    assert found is not None and found["stray"] == ["mo"] and "nga" in found["ignored"]


@pytest.mark.parametrize("text", [
    "WALANG PASOK SA LAHAT NG ANTAS 10 SETYEMBRE 2030 dahil sa bagyo, mag-ingat po ang lahat",   # router fires
    "SHS ONLINE CLASSES MUNA SA 12 AT 13 NOBYEMBRE kaya ingat din kayo sa biyahe ha",           # several list words
    "MO CLASSES 31 OCTOBER",                                                                    # too short
    "Dr. Ya nga' s Colleges MO CLASSES ON 31 OCTOBER",                                          # 6 words left
])
def test_not_released(text):
    assert pl.decide(text) is None


def test_the_wrapper_acts_only_while_the_picture_check_runs(monkeypatch):
    import src.api.main as api
    assert api.withhold_for_language(POSTER) is True        # typed messages: unchanged
    notes = []
    a, r = pl.ACTIVE.set(True), pl.RELEASED.set(notes)
    try:
        assert api.withhold_for_language(POSTER) is False
        assert api.withhold_for_language("walang pasok bukas sa lahat ng antas dahil sa bagyo") is True
    finally:
        pl.ACTIVE.reset(a)
        pl.RELEASED.reset(r)
    assert notes and notes[0]["version"] == pl.VERSION and notes[0]["stray"] == ["mo"]
    assert api.withhold_for_language(POSTER) is True


def test_english_text_is_never_touched(monkeypatch):
    import src.api.main as api
    a = pl.ACTIVE.set(True)
    try:
        assert api.withhold_for_language("no classes in all levels on november 11 2026") is False
    finally:
        pl.ACTIVE.reset(a)


def test_language_verify_text_marks_the_answer():
    import src.api.main as api

    class R(SimpleNamespace):
        def model_copy(self, update):
            return R(**{**self.__dict__, **update})

    def check(req):
        api.withhold_for_language(req.claim)
        return R(verdict="SUPPORTED")
    out = api._language_verify_text(check)(SimpleNamespace(claim=POSTER))
    assert out.poster_language["released"] is True and out.poster_language["stray"] == ["mo"]
    plain = api._language_verify_text(check)(SimpleNamespace(claim="no classes on november 11 2026 in all levels"))
    assert not hasattr(plain, "poster_language")
    assert pl.ACTIVE.get() is False                          # reset after the check
