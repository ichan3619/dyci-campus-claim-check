"""Tagalog/Taglish fixed-phrase reader (src/normalization/tagalog_phrases.py, CAMPUS_VERIFY_TAGALOG_PHRASES default off;
results/round2/tagalog_phrases_plan.md). The plan's fresh team set is used once, by its run; these are synthetic."""
import datetime as dt
import os
import re

import pytest
from fastapi.testclient import TestClient

from src.normalization import tagalog_phrases as tp
from src.verification import relative_resolve as rr

VOCAB = tp.build_vocabulary(["The Nursing pinning ceremony and the novena mass were held at the Elida Convention Center.",
                             "Teacher's Day event, foundation day and the midterm examinations."])


@pytest.mark.parametrize("text, frame, reading", [
    ("may pasok ba bukas?", "T1", "There are classes tomorrow."),
    ("Walang pasok bukas", "T2", "There are no classes tomorrow."),
    ("walang ftf sa 21?", "T2", "There are no face-to-face classes on the 21st."),
    ("may klase po ba sa lunes?", "T1", "There are classes on Monday."),
    ("suspended ba ang klase ngayon?", "T3", "Classes are suspended today."),
    ("pasok suspended bukas", "T3", "Classes are suspended tomorrow."),
    ("online class ba lahat bukas?", "T4", "Classes are online for all levels tomorrow."),
    ("may novena mass ngayong october?", "T5", "There is novena mass in October."),
    ("may misa sa sept 8?", "T5", "There is mass on September 8."),
    ("walang midterm examinations sa 30 oktubre", "T6", "There is no midterm examinations on October 30."),
    ("nagkaroon ng Teacher's day event?", "T7", "There was teacher's day event."),
    ("may pinning ba ang nursing?", "T8", "Nursing has pinning."),
    ("may pinning ceremony sa Elida Convention?", "T5", "There is pinning ceremony at Elida Convention."),
    ("may pasok bukas diba", "T1", "There are classes tomorrow."),
])
def test_messages_that_fit_a_frame_get_an_english_reading(text, frame, reading):
    got = tp.read(text, VOCAB)
    assert got["frame"] == frame and got["reading"] == reading and got["vocabulary_sha256"] == VOCAB[1]


@pytest.mark.parametrize("text", [
    "Diba chinese daw sila?",                                 # a reaction word
    "akala ko sa 2 pa gaganapin?",
    "hindi ba may pasok bukas?",                              # a negation outside the frames
    "may pasok bukas at walang exam",                         # a second clause
    "may pasok bukas, diba?",
    "anong oras ang misa?",                                   # a question word
    "may mga pasalubong ba?",                                 # an unknown word
    "binaha ba ang campus?",                                  # no frame
    "may pasok bukas sa lunes",                               # two WHEN slots
    "may", "", "There are classes tomorrow.",
])
def test_anything_else_is_refused(text):
    assert tp.read(text, VOCAB) is None


def test_the_vocabulary_drops_tagalog_listed_words_and_short_words():
    vocab, _ = tp.build_vocabulary(["Walang pasok ngayon sa DYCI at sa lahat ng campus."])
    assert "campus" in vocab and "dyci" in vocab
    assert not {"walang", "pasok", "ngayon", "lahat", "sa", "ng"} & vocab


def test_the_setting_is_off_by_default(monkeypatch):
    monkeypatch.delenv(tp.ENV, raising=False)
    assert tp.enabled() is False
    monkeypatch.setenv(tp.ENV, "1")
    assert tp.enabled() is True


def test_the_route_checks_the_reading_and_keeps_the_message_as_typed(api_settings, monkeypatch):
    monkeypatch.setattr(rr, "today", lambda: dt.date(2026, 10, 27))
    bodies = {}
    for on in ("0", "1"):
        monkeypatch.setenv(tp.ENV, on)
        app = api_settings()
        with TestClient(app.app) as c:
            bodies[on] = c.post("/verify/text", json={"claim": "may pasok daw ba bukas?"}).json()
    assert "tagalog_reading" not in bodies["0"] and "relative_resolution" not in bodies["0"]
    on = bodies["1"]
    assert on["claim"] == "may pasok daw ba bukas?" and on["verdict"] is not None
    assert on["tagalog_reading"]["reading"] == "There are classes tomorrow."
    assert on["relative_resolution"]["checked_as"] == "There are classes on October 28, 2026."
    assert on.get("hearsay_notice")                                   # "daw" is read from the message as typed


def test_the_page_shows_the_reading_and_never_sets_a_verdict():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = open(os.path.join(root, "frontend", "js", "app.js"), encoding="utf-8").read()
    block = app[app.index("function tagalogNote(data)"):app.index("// Display-only relative-date notice")]
    assert "Read as" in block and "limited" in block and not re.search(r"verdict\s*=(?!=)", block)
