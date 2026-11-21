"""Option B: a picture whose whole text conflicts with a DYCI post it closely matches is put at Level 1
(results/round2/text_conflict_level_plan.md). Examples are made up; none from the option B set."""
import inspect
from types import SimpleNamespace

import pytest

from src.image import text_conflict_level as b

POST = ("DYCI ADVISORY 27 SEPTEMBER 2026 Tomorrow, September 28, 2026, classes in all levels will be held until "
        "1:00 PM only in observance of the National Family Week.")
PICTURE = ("YANGAS COLLECES x < Jq)C * * 5 HOCAUE) BULA Dr. Yanga's Colleges, Inc. ANNOUNCEMENT 27 SEPTEMBER 2026 "
           "CLASSES IN ALL LEVELS UNTIL 10:00 AM ONLY SEPTEMBER 28, 2026 NATIONAL FAMILY WEEK dyci.edu.ph")
OTHER = "ANNOUNCEMENT BLOOD DONATION DRIVE GYMNASIUM VOLUNTEERS WELCOME OCTOBER 11"


def _check(verdict="CONTRADICTED", withheld=None, text=POST):
    ev = SimpleNamespace(document_id="P1", text=text, published_date="2026-09-29T06:31:23+0000", original_date=None)
    return SimpleNamespace(verdict=verdict, verdict_withheld=withheld, used_evidence=ev)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(b.ENV, raising=False)
    assert b.enabled() is False


@pytest.mark.parametrize("level", [2, 3, 4])
def test_fires(level):
    info = b.check(PICTURE, _check(), level)
    assert info["from_level"] == level and info["post"] == "P1" and info["post_dated"] == "27 September"
    assert info["version"] == "text-conflict-level-v1" and info["post_name"] == "DYCI's announcement dated 27 September"


@pytest.mark.parametrize("level", [1, 5, None])
def test_only_from_levels_2_to_4(level):
    assert b.check(PICTURE, _check(), level) is None


@pytest.mark.parametrize("verdict", ["SUPPORTED", "INSUFFICIENT_EVIDENCE", None])
def test_only_contradicted(verdict):
    assert b.check(PICTURE, _check(verdict), 2) is None


def test_withheld_or_missing():
    assert b.check(PICTURE, _check(withheld="language"), 2) is None
    assert b.check(PICTURE, None, 2) is None
    assert b.check(PICTURE, _check(text=""), 2) is None


def test_unrelated_picture_not_moved():
    assert b.check(OTHER, _check(), 3) is None


def test_overlap_boundary(monkeypatch):
    import src.image.changed_date as cd
    monkeypatch.setattr(cd, "overlap", lambda a, p: (2, 4))
    assert b.check(PICTURE, _check(), 3)["overlap"] == "2 of 4"
    monkeypatch.setattr(cd, "overlap", lambda a, p: (1, 3))
    assert b.check(PICTURE, _check(), 3) is None
    monkeypatch.setattr(cd, "overlap", lambda a, p: (0, 0))
    assert b.check(PICTURE, _check(), 3) is None


def test_apply():
    card = {"level": 2, "name": "x", "reasons": ["old"], "why_not_sure": "w", "closest": [1], "check": None, "explain": "e"}
    out = b.apply(card, b.check(PICTURE, _check(), 2))
    assert out["level"] == 1 and out["name"] == "Conflicts with DYCI" and out["explain"] == "e"
    assert out["reasons"] == ["Checked as one piece of text, the words read from this picture conflict with DYCI's "
                              "announcement dated 27 September.", "old"]
    assert out["why_not_sure"] is None and out["closest"] == []


def test_wired_after_rule_f_before_option_a_picture_only():
    import src.api.main as api
    src = inspect.getsource(api.verify_image)
    assert "text_conflict_level.enabled()" in src
    assert src.index("changed_date.check") < src.index("text_conflict_level.check") < src.index("text_line.build")
    assert "text_conflict_level" not in inspect.getsource(api.verify_text)
