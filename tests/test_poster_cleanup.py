"""Rule R, poster clean-up (results/round2/poster_cleanup_plan.md)."""
import pytest

from src.image import poster_cleanup as pc

POSTER = ("NANGA S COLLEGES ~ 95i 3 BOCAUE> BULA Dr. Ya nga' s Colleges, Inc. ANNOUNCEMENT 1 0 NO VE M B E R 2 0 2 6 "
          "NO CLASSES IN ALL LEVELS & MO OFFICE OPERATIONS NOVEMBER 11, 2026 IN OBSERVANCE OF THE CELEBRATION OF THE "
          "FEAST OF ST. MARTIN OF TOURS Dr. Yanga's Colleges, Inc_ dyci.edu.ph")


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(pc.ENV, raising=False)
    assert pc.enabled() is False
    monkeypatch.setenv(pc.ENV, "1")
    assert pc.enabled() is True


def test_full_poster():
    r = pc.clean(POSTER)
    assert r.text == ("NO CLASSES IN ALL LEVELS & MO OFFICE OPERATIONS NOVEMBER 11, 2026 IN OBSERVANCE OF THE "
                      "CELEBRATION OF THE FEAST OF ST. MARTIN OF TOURS")
    assert r.poster_date == "10 NOVEMBER 2026"
    assert r.header_found
    assert r.changes == ["website", "spaced_letters", "seal_and_header", "footer"]


@pytest.mark.parametrize("raw,out", [
    ("1 0 NO VE M B E R 2 0 2 6", "10 NOVEMBER 2026"),
    ("3 O C T O B E R 2 0 3 0", "3 OCTOBER 2030"),
    ("AT 1 PM IN THE GYM", "AT 1 PM IN THE GYM"),           # fewer than half single characters: unchanged
    ("NO CLASSES IN ALL LEVELS", "NO CLASSES IN ALL LEVELS"),
    ("a b c d", "a b c d"),                                  # lowercase: unchanged
])
def test_rejoin_spaced(raw, out):
    assert pc.rejoin_spaced(raw) == out


def test_letter_spaced_school_name_is_found():
    r = pc.clean("D r . Y a n g a ' s C o l l e g e s , I n c . ADVISORY NO CLASSES ON JULY 10, 2030 DUE TO RAIN")
    assert r.header_found and r.text == "NO CLASSES ON JULY 10, 2030 DUE TO RAIN"


@pytest.mark.parametrize("label", ["ANNOUNCEMENT", "ADVISORY", "PUBLIC ADVISORY", "DYCI ADVISORY", "REMINDER", "NOTICE"])
def test_labels(label):
    r = pc.clean(f"Dr. Yanga's Colleges, Inc. {label} 9 JULY 2030 NO CLASSES IN ALL LEVELS JULY 10, 2030")
    assert r.text == "NO CLASSES IN ALL LEVELS JULY 10, 2030"
    assert r.poster_date == "9 JULY 2030"


def test_month_first_issue_date():
    r = pc.clean("Dr. Yanga's Colleges, Inc. ADVISORY JULY 9, 2030 NO CLASSES IN ALL LEVELS TOMORROW")
    assert r.poster_date == "JULY 9, 2030" and r.text == "NO CLASSES IN ALL LEVELS TOMORROW"


def test_no_issue_date_keeps_event_date_line():
    r = pc.clean("Dr. Yanga's Colleges, Inc. REMINDER LIBRARY CLOSED AUGUST 30, 2030 FOR THE INVENTORY")
    assert r.poster_date is None
    assert r.text == "LIBRARY CLOSED AUGUST 30, 2030 FOR THE INVENTORY"


def test_school_name_late_in_text_is_not_a_header():
    words = " ".join(["word"] * 20)
    r = pc.clean(f"{words} Dr. Yanga's Colleges, Inc. will hold a seminar for all students in the gymnasium next week")
    assert not r.header_found
    assert r.text.startswith("word")


def test_school_name_in_the_body_is_not_a_footer():
    text = ("Dr. Yanga's Colleges, Inc. ADVISORY The DYCI Foundation Day program will be held in the gymnasium on "
            "October 11, 2030 with all students")
    r = pc.clean(text)
    assert "Foundation Day program" in r.text and "with all students" in r.text
    assert "footer" not in r.changes


def test_plain_message_unchanged():
    r = pc.clean("Classes are suspended on July 10 due to heavy rain.")
    assert r.text == "Classes are suspended on July 10 due to heavy rain." and r.changes == []


def test_website_removed_anywhere():
    r = pc.clean("See https://www.dyci.edu.ph/news for the schedule of the enrollment period")
    assert "dyci.edu.ph" not in r.text and "website" in r.changes


def test_language_ok():
    assert pc.language_ok("NO CLASSES IN ALL LEVELS & MO OFFICE OPERATIONS NOVEMBER 11, 2026")
    assert not pc.language_ok("MO OFFICE OPERATIONS")                       # fewer than 8 words
    assert not pc.language_ok("sabi nga nila no classes on november 11 in all levels")   # 2 extra-list words
    assert not pc.language_ok("walang pasok sa lahat ng levels sa november 11 2026")     # router check


def test_language_gate_only_while_active(monkeypatch):
    import src.api.main as api
    text = "NO CLASSES IN ALL LEVELS & MO OFFICE OPERATIONS NOVEMBER 11, 2026"
    assert api.withhold_for_language(text) is True
    token = pc.POSTER_ACTIVE.set(True)
    try:
        assert api.withhold_for_language(text) is False
        assert api.withhold_for_language("walang pasok sa lahat ng levels sa november 11 2026") is True
    finally:
        pc.POSTER_ACTIVE.reset(token)
    assert api.withhold_for_language(text) is True
