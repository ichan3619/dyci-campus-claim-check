"""Rule H, header-only advisories (src/verification/header_date.py, CAMPUS_VERIFY_HEADER_DATE default off;
results/round2/header_date_plan.md). Synthetic posts only; test set H is used once, by the plan's run."""
import datetime as dt
from types import SimpleNamespace as NS

import pytest

from src.verification import header_date as hd
from src.verification.verifier import VerificationResult

TODAY = dt.date(2026, 11, 17)
HEADER_ONLY = NS(document_id="nov12", text="DYCI ADVISORY November 12, 2026 Classes and office operations are "
                                           "suspended immediately.")
NO_YEAR = NS(document_id="noyear", text="DYCI ADVISORY November 12 Classes are suspended immediately.")
BODY_OTHER = NS(document_id="nov11", text="DYCI ADVISORY November 11, 2026 No classes tomorrow, November 12, 2026.")
OLD = NS(document_id="old", text="DYCI ADVISORY November 12, 2025 Classes are suspended immediately.")


def _result(post, verdict="INSUFFICIENT_EVIDENCE", label="SUPPORTED", date_check="claimed_date_not_in_evidence",
            value_check=None):
    r = VerificationResult(claim="c", verdict=verdict, evidence_status="RELEVANT_EVIDENCE_FOUND",
                           retrieved_evidence=[post], used_evidence=post,
                           classification_result=NS(label=label, confidence=0.9), explanation="E.")
    r.date_check, r.value_check = date_check, value_check
    return r


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(hd.ENV, raising=False)
    assert not hd.enabled()


@pytest.mark.parametrize("message", ["no classes on november 12",
                                     "November 12, 2026 Classes are suspended immediately."])
def test_a_header_only_advisory_confirms_its_own_day(message):
    out, info = hd.check(message, _result(HEADER_ONLY), TODAY)
    assert out.verdict == "SUPPORTED" and info["header_date"] == "November 12, 2026"
    assert out.used_evidence is HEADER_ONLY and "Header date" in out.explanation


def test_year_from_the_post_date_when_the_header_has_none():
    sources = {"noyear": NS(published_date="2025-11-12")}
    out, info = hd.check("no classes on november 12", _result(NO_YEAR), TODAY, sources)
    assert info is None and out.verdict == "INSUFFICIENT_EVIDENCE"
    out, info = hd.check("no classes on november 12", _result(NO_YEAR), TODAY, {"noyear": NS(published_date="2026-11-12")})
    assert out.verdict == "SUPPORTED"


@pytest.mark.parametrize("message, kwargs, post", [
    ("no classes on november 13", {}, HEADER_ONLY),                                     # another day
    ("no classes on november 12", {}, BODY_OTHER),                                      # body names a date
    ("no classes on november 12", {}, OLD),                                             # another year
    ("no classes on november 12 and november 13", {}, HEADER_ONLY),                     # two dates
    ("no classes on november 12", {"label": "CONTRADICTED"}, HEADER_ONLY),              # classifier said no
    ("no classes on november 12", {"date_check": None}, HEADER_ONLY),                   # not withheld by the date check
    ("no classes on november 12", {"date_check": "evidence_has_no_date"}, HEADER_ONLY),
    ("no classes on november 12", {"value_check": "value_conflict_withheld"}, HEADER_ONLY),
    ("no classes on november 12", {"verdict": "CONTRADICTED"}, HEADER_ONLY),            # never touches a decisive verdict
])
def test_does_not_apply(message, kwargs, post):
    r = _result(post, **kwargs)
    out, info = hd.check(message, r, TODAY)
    assert info is None and out is r
