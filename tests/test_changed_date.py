"""Rule F: a picture that closely matches a DYCI post but gives a different date -> Level 2
(results/round2/changed_date_plan.md)."""
import inspect

import pytest

from src.image import changed_date as cd

POST = {"document_id": "ingested:x", "published_date": "2026-09-29T06:31:23+0000",
        "text": "DYCI Advisory 27 September 2026 SHORTENED CLASS HOURS Please be informed that tomorrow, September 28, "
                "2026, CLASSES at ALL LEVELS and OFFICE OPERATIONS will be until 1:00 PM only. This is in observance of "
                "the 34th National Family Week, pursuant to Circular Memorandum No. 136 issued by the Malacañang Office. "
                "Regular operations will resume on the next working day. Thank you."}
PICTURE = ("YANGA S = COLLEGES 950] 5 ROCAUE) BULA Dr. Yanga' s College s_ Inc. ANNOUNCEMENT 27 SEPTEMBER 2026 "
           "SHORTENED CLASS HOURS SEPTEMBER 29, 2026, UNTIL 1:00 PM ONLY IN OBSERVANCE OF THE 34Th NATIONAL FAMILY WEEK "
           "PURSUANT To CIRCULAR MEMORANDUM NO. 136 ISSUED BY THE MALACANANG OFFICE Dr. Yanga's Colleges, Inc. "
           "dyci.edu.ph")


def whole(verdict="INSUFFICIENT_EVIDENCE", post=POST):
    return {"verdict": verdict, "used_evidence": post}


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(cd.ENV, raising=False)
    assert cd.enabled() is False


def test_the_owners_september_29_picture():
    info = cd.check(PICTURE, whole(), 3)
    assert info["picture_date"] == "29 September" and info["post_date"] == "28 September"
    assert info["overlap"] == "13 of 16"
    card = cd.apply({"level": 3, "name": "Can't confirm", "reasons": ["x"], "why_not_sure": "y", "closest": [1]}, info)
    assert card["level"] == 2 and card["reasons"][0] == (
        "This picture closely matches DYCI's announcement dated 27 September, but it gives 29 September where that "
        "announcement says 28 September. It may have been edited.")
    assert card["reasons"][1] == "x"


def test_a_post_without_a_header_date_is_named_by_its_posting_date():
    post = {**POST, "text": POST["text"].replace("DYCI Advisory 27 September 2026 ", "")}
    card = cd.apply({"level": 3, "reasons": []}, cd.check(PICTURE, whole(post=post), 3))
    assert card["reasons"][0].startswith("This picture closely matches DYCI's post of 29 September 2026, but")
    assert card["why_not_sure"] is None and card["closest"] == []


def test_the_right_date_does_not_fire():
    assert cd.check(PICTURE.replace("SEPTEMBER 29", "SEPTEMBER 28"), whole(), 3) is None


def test_the_issue_date_alone_is_not_a_changed_date():
    assert cd.check("ANNOUNCEMENT 27 SEPTEMBER 2026 SHORTENED CLASS HOURS UNTIL 1:00 PM ONLY NATIONAL FAMILY WEEK",
                    whole(), 3) is None


def test_more_than_7_days_off_does_not_fire():
    assert cd.check(PICTURE.replace("SEPTEMBER 29", "OCTOBER 6"), whole(), 3) is None      # 8 days
    assert cd.check(PICTURE.replace("SEPTEMBER 29", "OCTOBER 5"), whole(), 3) is not None  # 7 days


def test_a_loose_match_does_not_fire():
    assert cd.check("ANNOUNCEMENT BLOOD DONATION DRIVE GYMNASIUM DONORS SEPTEMBER 29, 2026", whole(), 3) is None


@pytest.mark.parametrize("level, verdict", [(1, "INSUFFICIENT_EVIDENCE"), (2, "INSUFFICIENT_EVIDENCE"),
                                            (5, "SUPPORTED"), (3, "CONTRADICTED")])
def test_only_levels_3_and_4_and_never_over_a_contradiction(level, verdict):
    assert cd.check(PICTURE, whole(verdict), level) is None


def test_no_post_no_finding():
    assert cd.check(PICTURE, None, 3) is None
    assert cd.check(PICTURE, {"verdict": "INSUFFICIENT_EVIDENCE", "used_evidence": None}, 3) is None


def test_days_wrap_the_year():
    assert cd._days((12, 30), (1, 2)) == 3


def test_wired_into_the_picture_check_only():
    import src.api.main as api
    src = inspect.getsource(api.verify_image)
    assert "changed_date.enabled()" in src and src.index("assess(response.model_dump()") < src.index("changed_date.check")
    assert "changed_date" not in inspect.getsource(api.verify_text)
