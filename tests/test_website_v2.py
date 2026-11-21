"""Rule W v2: agency names (rule N kind) and the rule-based website confirmer (results/round2/website_source_v2_plan.md)."""
import json
from types import SimpleNamespace

import pytest

from src.verification import unit_check as uc

POST = ("DYCI ADVISORY Classes in all levels are suspended on August 26, 2025 following the directive of the DILG. "
        "Junior and senior high students will receive reviewers for the QAT.")


def test_agencies_off_by_default(monkeypatch):
    monkeypatch.delenv(uc.AGENCIES_ENV, raising=False)
    assert "agency" not in uc.units(POST)
    assert uc.conflict("classes suspended on august 26 2025 per deped directive", POST, POST) is None


@pytest.mark.parametrize("message,mine", [
    ("classes suspended on august 26 2025 per deped directive", "DepEd"),
    ("classes suspended on august 26 2025 per ched memo", "CHED"),
    ("classes suspended august 26 2025 as per pagasa", "PAGASA"),
])
def test_agency_swaps(monkeypatch, message, mine):
    monkeypatch.setenv(uc.AGENCIES_ENV, "1")
    c = uc.conflict(message, POST, POST)
    assert c["kind"] == "agency" and c["message_unit"] == mine and c["post_units"] == ["DILG"]


def test_agency_same_or_none(monkeypatch):
    monkeypatch.setenv(uc.AGENCIES_ENV, "1")
    assert uc.conflict("classes suspended august 26 2025 per dilg", POST, POST) is None
    assert uc.conflict("classes suspended on august 26 2025", POST, POST) is None
    assert uc.units("per the provincial government of bulacan")["agency"] == {"provincial government"}
    assert uc.units("memorandum from the office of the president")["agency"] == {"Malacanang"}


def _result(verdict, used_text=""):
    class R(SimpleNamespace):
        def model_copy(self, update):
            return R(**{**self.__dict__, **update})
    return R(verdict=verdict, used_evidence=SimpleNamespace(text=used_text) if used_text else None,
             verdict_withheld=None, explanation="")


def test_confirmer(monkeypatch, tmp_path):
    import src.api.main as api
    from src.ingestion import dyci_website as w
    item = {"document_id": "website:announcement:10", "text": POST, "source_type": "official_site",
            "source_url": "https://dyci.edu.ph/announcements/x", "published_date": "2025-08-25"}
    p = tmp_path / "web.jsonl"
    p.write_text(json.dumps(item) + "\n", encoding="utf-8")
    monkeypatch.setenv(w.ENV, "2")
    monkeypatch.setenv(w.FILE_ENV, str(p))
    monkeypatch.setenv(uc.AGENCIES_ENV, "1")
    ok = "classes in all levels suspended on august 26 2025 dilg directive"
    out = api._apply_website_confirmer(_result("INSUFFICIENT_EVIDENCE"), ok)
    assert out.verdict == "SUPPORTED" and out.website_match["why"] == "confirmer"
    assert out.used_evidence.provenance == "dyci_website"
    wrong = api._apply_website_confirmer(_result("INSUFFICIENT_EVIDENCE"),
                                         "classes in all levels suspended on august 26 2025 deped directive")
    assert wrong.verdict == "INSUFFICIENT_EVIDENCE"
    other_day = _result("CONTRADICTED", used_text="Classes resume on August 27, 2025.")
    assert api._apply_website_confirmer(other_day, ok).verdict == "SUPPORTED"
    same_day = _result("CONTRADICTED", used_text="Classes push through on August 26, 2025.")
    assert api._apply_website_confirmer(same_day, ok) is same_day
    assert api._apply_website_confirmer(_result("SUPPORTED"), ok).verdict == "SUPPORTED"
    no_date = _result("INSUFFICIENT_EVIDENCE")
    assert api._apply_website_confirmer(no_date, "classes suspended dilg directive") is no_date


def test_mode(monkeypatch):
    from src.ingestion import dyci_website as w
    for value, mode in (("1", "1"), ("2", "2"), ("0", "off"), ("3", "off")):
        monkeypatch.setenv(w.ENV, value)
        assert w.mode() == mode and w.enabled() is (mode != "off")
