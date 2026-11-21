"""Rule P, check the parts of a message (src/verification/reason_parts.py, CAMPUS_VERIFY_REASON_PARTS default off;
results/round2/reason_parts_plan.md). Synthetic posts and a fake classifier only; test set P is used once, by the run."""
from types import SimpleNamespace as NS

import pytest

from src.verification import reason_parts as rp

POST = ("DYCI ADVISORY November 12, 2026 Classes and office operations are suspended immediately. A security threat to "
        "the Elida Campus has been reported. Further updates will follow.")


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(rp.ENV, raising=False)
    assert not rp.enabled()


@pytest.mark.parametrize("message, expected", [
    ("no classes on november 12 due to security threat", ("no classes on november 12", "security threat")),
    ("no classes on nov 12 because of the bomb threat", ("no classes on nov 12", "the bomb threat")),
    ("classes are suspended on march 3 because it is raining.", ("classes are suspended on march 3", "it is raining")),
    ("library closed on may 2 owing to pest control", ("library closed on may 2", "pest control")),
    ("no classes on november 12", None),                       # no marker
    ("due to rain no classes", None),                          # marker at the start: no main claim
    ("classes off due to rain", None),                         # main claim under 3 words
])
def test_split(message, expected):
    assert rp.split(message) == expected


def test_pieces_are_sentences_and_neighbour_pairs():
    p = rp.pieces(POST)
    assert len(p) == 3 + 2
    assert "A security threat to the Elida Campus has been reported." in p


def _classify(table):
    def classify(hyp, premise):
        for (h, needle), label in table.items():
            if h in hyp and needle in premise:
                return NS(label=label, confidence=0.9)
        return NS(label="INSUFFICIENT_EVIDENCE", confidence=0.9)
    return classify


def test_reason_backed_by_another_sentence():
    label, piece = rp.check_reason(_classify({("security threat", "security threat to"): "SUPPORTED"}),
                                   "security threat", POST)
    assert label == "backed" and "security threat" in piece


def test_reason_different_and_not_stated():
    assert rp.check_reason(_classify({("typhoon", "security threat"): "CONTRADICTED"}), "a typhoon", POST)[0] == "different"
    assert rp.check_reason(_classify({}), "bomb threat", POST) == ("not_stated", None)


@pytest.mark.parametrize("whole, main, reason, expected", [
    ("SUPPORTED", "SUPPORTED", "not_stated", ("SUPPORTED", "kept_supported")),
    ("INSUFFICIENT_EVIDENCE", "SUPPORTED", "backed", ("SUPPORTED", "both_backed")),
    ("CONTRADICTED", "SUPPORTED", "backed", ("SUPPORTED", "both_backed")),
    ("INSUFFICIENT_EVIDENCE", "SUPPORTED", "not_stated", ("INSUFFICIENT_EVIDENCE", "reason_not_stated")),
    ("CONTRADICTED", "SUPPORTED", "different", ("INSUFFICIENT_EVIDENCE", "reason_different")),
    ("CONTRADICTED", "CONTRADICTED", None, ("CONTRADICTED", "main_not_supported")),
    ("INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE", None, ("INSUFFICIENT_EVIDENCE", "main_not_supported")),
])
def test_decide(whole, main, reason, expected):
    assert rp.decide(whole, main, reason) == expected


def test_never_creates_contradicted():
    for whole in ("SUPPORTED", "INSUFFICIENT_EVIDENCE"):
        for main in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"):
            for reason in ("backed", "different", "not_stated", None):
                assert rp.decide(whole, main, reason)[0] != "CONTRADICTED"
