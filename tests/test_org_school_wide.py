"""Rule O-SW: department pages may confirm school-wide messages, with a note (results/round2/org_school_wide_plan.md).
Examples are made up; none from the rule's test set."""
import datetime as dt
import inspect
import json
import os
from types import SimpleNamespace

import pytest

from src.verification import org_pages as op

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POST = ("📢 ANNOUNCEMENT To all DYCIans: there will be NO CLASSES in all levels on Friday, March 14, 2030, to give way "
        "to the general assembly. Classes resume on Monday, March 17, 2030. #DYCI")


def _result(verdict, used_text="", withheld=None):
    class R(SimpleNamespace):
        def model_copy(self, update):
            return R(**{**self.__dict__, **update})
    used = SimpleNamespace(text=used_text) if used_text else None
    return R(verdict=verdict, used_evidence=used, verdict_withheld=withheld)


@pytest.fixture
def api(monkeypatch, tmp_path):
    import src.api.main as api
    from src.verification import post_age
    (tmp_path / "dept.jsonl").write_text(json.dumps({"document_id": "ingested:d1", "text": POST,
                                                      "source_url": "https://fb/d1",
                                                      "published_date": "2030-03-10T01:00:00+0000"}) + "\n",
                                         encoding="utf-8")
    monkeypatch.setenv(op.ENV, "31")
    monkeypatch.setenv(op.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(op.SOURCES_ENV, "dept=College of Testing")
    monkeypatch.setattr(post_age, "today", lambda: dt.date(2030, 3, 12))
    return api


WIDE = "no classes in all levels on march 14 2030"


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(op.SCHOOL_WIDE_ENV, raising=False)
    assert op.school_wide_enabled() is False


def test_off_keeps_v31(api, monkeypatch):
    monkeypatch.delenv(op.SCHOOL_WIDE_ENV, raising=False)
    out = api._apply_org_pages_v3(_result("INSUFFICIENT_EVIDENCE"), WIDE)
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and out.org_related["name"] == "College of Testing"


def test_on_confirms_school_wide_with_the_note(api, monkeypatch):
    monkeypatch.setenv(op.SCHOOL_WIDE_ENV, "1")
    out = api._apply_org_pages_v3(_result("INSUFFICIENT_EVIDENCE"), WIDE)
    assert out.verdict == "SUPPORTED"
    src = out.org_source
    assert src["school_wide"] is True and src["version"] == "org-pages-v3.1-sw" and src["label"] is None
    assert src["note"] == ("This answer comes from a post by College of Testing, a DYCI department page, not from "
                           "DYCI's official page.")
    assert out.explanation == src["note"]


@pytest.mark.parametrize("message", [
    "no classes in all levels on march 13 2030",                     # another date
    "no classes in all levels on march 14 2030 online classes",      # a word not in the post
    "classes suspended at all levels on march 14 2030",              # "suspended" not in the post
])
def test_on_keeps_the_strict_match(api, monkeypatch, message):
    monkeypatch.setenv(op.SCHOOL_WIDE_ENV, "1")
    assert api._apply_org_pages_v3(_result("INSUFFICIENT_EVIDENCE"), message).verdict != "SUPPORTED"


def test_on_dyci_first(api, monkeypatch):
    monkeypatch.setenv(op.SCHOOL_WIDE_ENV, "1")
    supported = _result("SUPPORTED")
    assert api._apply_org_pages_v3(supported, WIDE) is supported                  # DYCI settled it
    same_day = _result("CONTRADICTED", used_text="Classes push through on March 14, 2030.")
    assert api._apply_org_pages_v3(same_day, WIDE) is same_day                    # DYCI's post is about this day
    withheld = _result("INSUFFICIENT_EVIDENCE", withheld="language")
    assert api._apply_org_pages_v3(withheld, WIDE) is withheld


def test_on_stale(api, monkeypatch):
    from src.verification import post_age
    monkeypatch.setenv(op.SCHOOL_WIDE_ENV, "1")
    monkeypatch.setattr(post_age, "today", lambda: dt.date(2030, 7, 20))         # 132 days after the post
    assert api._apply_org_pages_v3(_result("INSUFFICIENT_EVIDENCE"), WIDE).verdict == "INSUFFICIENT_EVIDENCE"


def test_own_event_keeps_v31_label(api, monkeypatch, tmp_path):
    monkeypatch.setenv(op.SCHOOL_WIDE_ENV, "1")
    (tmp_path / "dept.jsonl").write_text(json.dumps({
        "document_id": "ingested:d2", "text": "Web seminar on March 14, 2030 at 1:00 PM in the AVR, 50 pesos.",
        "source_url": "https://fb/d2", "published_date": "2030-03-10T01:00:00+0000"}) + "\n", encoding="utf-8")
    out = api._apply_org_pages_v3(_result("INSUFFICIENT_EVIDENCE"), "web seminar on march 14 2030 in the avr")
    assert out.verdict == "SUPPORTED" and out.org_source["label"] == op.LABEL
    assert not out.org_source.get("school_wide")


def test_page_shows_the_department_note():
    js = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
    assert "data.org_source && data.org_source.school_wide" in js and "data.org_source.note" in js


def test_wired_in_v3_only():
    import src.api.main as api
    src = inspect.getsource(api._apply_org_pages_v3)
    assert "org_pages.school_wide_enabled()" in src and "org_pages.school_wide_note" in src
