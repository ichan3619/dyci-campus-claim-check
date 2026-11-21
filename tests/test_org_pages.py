"""Rule O: a student organisation's own posts can decide about its own matters (results/round2/org_pages_plan.md)."""
import json
from types import SimpleNamespace

import pytest

from src.verification import org_pages as op


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(op.ENV, raising=False)
    assert op.enabled() is False


@pytest.mark.parametrize("value,mode", [("1", "verdict"), ("related", "related"), ("RELATED ", "related"), ("0", "off"),
                                        ("yes", "off")])
def test_mode(monkeypatch, value, mode):
    monkeypatch.setenv(op.ENV, value)
    assert op.mode() == mode and op.enabled() is (mode != "off")


def test_related_mode_never_decides(monkeypatch):
    import src.api.main as api
    monkeypatch.setenv(op.ENV, "related")
    monkeypatch.setattr(api, "_org_verifier", lambda: ("verifier", {"ingested:1_2": ("dyci_aces", "ACES")}, {}))
    monkeypatch.setattr(api, "_gated", lambda fn, *a, **k: _fake("SUPPORTED"))
    out = api._apply_org_pages(_fake("INSUFFICIENT_EVIDENCE"), "technovate on october 9", SimpleNamespace(relevance_checker=None))
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and not hasattr(out, "org_source") and out.org_related["name"] == "ACES"


def test_mode_v2(monkeypatch):
    monkeypatch.setenv(op.ENV, "2")
    assert op.mode() == "verdict2" and op.enabled()


@pytest.mark.parametrize("message", ["christmas break starts on december 15", "the holiday on november 1",
                                     "summer vacation", "finals week is december 9", "the semestral break",
                                     "second semester starts", "no classes on november 14"])
def test_school_wide_v2(message):
    assert op.school_wide_v2(message)


def test_school_wide_v2_keeps_events():
    assert not op.school_wide_v2("site coding contest final round on october 9")


def test_clean():
    assert op.clean("📢 GENERAL ASSEMBLY ALERT! Join us 🎉 on July 16. #SITE #DYCI2030") == \
        "GENERAL ASSEMBLY ALERT! Join us on July 16."
    assert op.clean("1️⃣ First ✨ prize") == "1 First prize"


def test_recent():
    import datetime as dt
    today = dt.date(2030, 10, 1)
    assert op.recent({"published_date": "2030-07-03T01:00:00+0000"}, today)           # 90 days
    assert not op.recent({"published_date": "2030-07-02T01:00:00+0000"}, today)       # 91 days
    assert not op.recent({"published_date": "2030-10-02"}, today)                     # after the check date
    assert not op.recent({"published_date": ""}, today)


def test_note_v2():
    assert op.note("ACES", "SUPPORTED", True) == (" Supported: department announcement only. From ACES, not an official "
                                                  "DYCI announcement.")


def test_v2_verifier_keeps_recent_cleaned_posts(monkeypatch, tmp_path):
    import json as _json
    import datetime as dt
    import src.api.main as api
    from src.verification import post_age
    (tmp_path / "p.jsonl").write_text("\n".join(_json.dumps(r) for r in [
        {"document_id": "a", "text": "🎉 Fair on 𝐎𝐂𝐓 9 #SITE", "published_date": "2030-09-20"},
        {"document_id": "b", "text": "Old fair", "published_date": "2029-09-20"}]) + "\n", encoding="utf-8")
    monkeypatch.setenv(op.ENV, "2")
    monkeypatch.setenv(op.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(op.SOURCES_ENV, "p=SITE")
    monkeypatch.setattr(post_age, "today", lambda: dt.date(2030, 10, 1))
    seen = []
    monkeypatch.setitem(api.state, "manager", SimpleNamespace(temp_verifier=lambda docs: seen.append(docs) or "v"))
    api.state.pop("org_key", None)
    verifier, owner, sources = api._org_verifier()
    assert [d["text"] for d in seen[0]] == ["Fair on OCT 9"] and set(sources) == {"a"}
    assert api.state["org_originals"]["a"] == "🎉 Fair on OCT 9 #SITE"
    api.state.pop("org_key", None)


def test_sources(monkeypatch):
    monkeypatch.setenv(op.SOURCES_ENV, "dyci_aces=ACES, a DYCI student organisation; other=Other Org ;bad")
    assert op.sources() == {"dyci_aces": "ACES, a DYCI student organisation", "other": "Other Org"}
    monkeypatch.delenv(op.SOURCES_ENV)
    assert op.sources() == {}


@pytest.mark.parametrize("message", [
    "no classes on november 14", "classes are suspended on november 14", "walang pasok bukas",
    "online classes on july 10", "face to face on july 10", "early dismissal at 1pm", "the registrar office is closed",
    "campus closed on july 10", "final exams moved to december 9", "midterm examination on october 7",
    "enrollment opens december 1", "enrolment extended", "tuition due on july 30", "the fee is 75 pesos",
])
def test_school_wide(message):
    assert op.school_wide(message)


@pytest.mark.parametrize("message", [
    "aces general assembly on july 16 at 1pm in the avr", "technovate tech fair on october 9 entrance is free",
    "codeclash prize is 5000 pesos", "aces bake sale at the gym", "hackathon on september 27",
])
def test_not_school_wide(message):
    assert not op.school_wide(message)


def test_load_folds_styled_letters(tmp_path, monkeypatch):
    (tmp_path / "dyci_aces.jsonl").write_text(json.dumps({"document_id": "ingested:1_2", "text": "𝐓𝐄𝐂𝐇𝐍𝐎𝐕𝐀𝐓𝐄 𝟐𝟎𝟑𝟎",
                                                          "published_date": "2030-09-27T01:00:00+0000"}) + "\n\n",
                                              encoding="utf-8")
    monkeypatch.setenv(op.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(op.SOURCES_ENV, "dyci_aces=ACES;missing=Nobody")
    docs, owner, stamp = op.load("unused")
    assert [d["text"] for d in docs] == ["TECHNOVATE 2030"]
    assert owner == {"ingested:1_2": ("dyci_aces", "ACES")} and [k for k, _ in stamp] == ["dyci_aces"]


def test_related_and_note():
    used = SimpleNamespace(document_id="ingested:1_2", text="Technovate on October 9", source_url="https://fb/x",
                           published_date="2030-09-27T01:00:00+0000")
    r = SimpleNamespace(used_evidence=used, evidence_status="RELEVANT_EVIDENCE_FOUND")
    rel = op.related(r, {"ingested:1_2": ("dyci_aces", "ACES")})
    assert rel["name"] == "ACES" and rel["post_date"] == "2030-09-27"
    assert op.related(SimpleNamespace(used_evidence=used, evidence_status="NO_RELEVANT_EVIDENCE"), {}) is None
    assert op.note("ACES", "SUPPORTED") == " Supported according to ACES, not an official DYCI announcement."


def _fake(verdict, doc_id="ingested:1_2", status="RELEVANT_EVIDENCE_FOUND", withheld=None):
    used = SimpleNamespace(document_id=doc_id, text="post", source_url="https://fb/x", published_date="2030-09-27")

    class R(SimpleNamespace):
        def model_copy(self, update):
            return R(**{**self.__dict__, **update})
    return R(verdict=verdict, used_evidence=used, evidence_status=status, verdict_withheld=withheld, explanation="e")


def test_apply_dyci_first_and_school_wide(monkeypatch):
    import src.api.main as api
    owner = {"ingested:1_2": ("dyci_aces", "ACES")}
    monkeypatch.setattr(api, "_org_verifier", lambda: ("verifier", owner, {}))
    pipeline = SimpleNamespace(relevance_checker=None)
    monkeypatch.setattr(api, "_gated", lambda fn, *a, **k: _fake("SUPPORTED"))
    dyci = _fake("SUPPORTED")
    assert api._apply_org_pages(dyci, "technovate on october 9", pipeline) is dyci          # DYCI decided
    withheld = _fake("INSUFFICIENT_EVIDENCE", withheld="language")
    assert api._apply_org_pages(withheld, "technovate", pipeline) is withheld
    out = api._apply_org_pages(_fake("INSUFFICIENT_EVIDENCE"), "technovate on october 9", pipeline)
    assert out.org_source["name"] == "ACES" and out.org_source["dyci_verdict"] == "INSUFFICIENT_EVIDENCE"
    assert out.explanation.endswith("not an official DYCI announcement.")
    wide = api._apply_org_pages(_fake("INSUFFICIENT_EVIDENCE"), "no classes on november 14", pipeline)
    assert wide.verdict == "INSUFFICIENT_EVIDENCE" and not hasattr(wide, "org_source") and wide.org_related["name"] == "ACES"
    monkeypatch.setattr(api, "_gated", lambda fn, *a, **k: _fake("INSUFFICIENT_EVIDENCE"))
    undecided = api._apply_org_pages(_fake("INSUFFICIENT_EVIDENCE"), "technovate", pipeline)
    assert undecided.verdict == "INSUFFICIENT_EVIDENCE" and undecided.org_related["name"] == "ACES"


# ---- Rule O v3 (results/round2/org_pages_v3_plan.md) ----------------------------------------------------------------
POST_V3 = ("WEB DEV SEMINAR 2030 Calling all IT students! SITE invites you to our Web Development Seminar on March 14, "
           "2030, 1:00 PM at the AVR. Free for SITE members, 50 pesos for non-members. Champion gets 3,000 pesos.")


def test_mode_v3(monkeypatch):
    monkeypatch.setenv(op.ENV, "3")
    assert op.mode() == "verdict3" and op.enabled()


def test_content_words():
    assert op.content_words("site web dev seminar is on march 14 2030 1pm at the avr") == ["site", "web", "dev",
                                                                                        "seminar", "avr"]
    assert op.content_words("codefest 730am champion gets 3,000 pesos") == ["codefest", "champion", "3000", "pesos"]


@pytest.mark.parametrize("message", [
    "site web dev seminar is on march 14 2030 1pm at the avr",
    "web development seminar march 14 at the avr 50 pesos for non-members",
    "champion gets 3000 pesos on march 14",
])
def test_match_confirms(message):
    m = op.match(message, POST_V3)
    assert m is not None and m["coverage"] >= op.COVERAGE


@pytest.mark.parametrize("message", [
    "site web dev seminar is on march 15 2030 at the avr",           # another date
    "site web dev seminar march 14 at the gymnasium",                 # venue swapped (rule N)
    "site web dev seminar march 14 starts at 3pm",                    # time swapped (rule V)
    "site web dev seminar on march 14 is cancelled",                  # word not in the post
    "site web dev seminar march 14 is not free",                      # negation kept as a content word
    "web dev seminar march 14 champion gets 5000 pesos",              # number not in the post
    "site web dev seminar",                                           # no date
])
def test_match_refuses(message):
    assert op.match(message, POST_V3) is None


def _result(verdict, used_text="", withheld=None):
    class R(SimpleNamespace):
        def model_copy(self, update):
            return R(**{**self.__dict__, **update})
    used = SimpleNamespace(text=used_text) if used_text else None
    return R(verdict=verdict, used_evidence=used, verdict_withheld=withheld)


def test_apply_v3(monkeypatch, tmp_path):
    import json as _json
    import datetime as dt
    import src.api.main as api
    from src.verification import post_age
    (tmp_path / "site.jsonl").write_text(_json.dumps({"document_id": "ingested:s1", "text": "💻 " + POST_V3 + " #SITE",
                                                       "source_url": "https://fb/s1",
                                                       "published_date": "2030-03-09T01:00:00+0000"}) + "\n",
                                          encoding="utf-8")
    monkeypatch.setenv(op.ENV, "3")
    monkeypatch.setenv(op.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(op.SOURCES_ENV, "site=SITE, a DYCI student organisation")
    monkeypatch.setattr(post_age, "today", lambda: dt.date(2030, 3, 11))
    ok = "site web dev seminar is on march 14 2030 at the avr"
    out = api._apply_org_pages_v3(_result("INSUFFICIENT_EVIDENCE"), ok)
    assert out.verdict == "SUPPORTED" and out.org_source["label"] == op.LABEL and out.used_evidence.text.startswith("💻")
    assert out.explanation.startswith("Supported: department announcement only. From SITE")
    other_day = _result("CONTRADICTED", used_text="Classes resume on March 20, 2030.")
    assert api._apply_org_pages_v3(other_day, ok).verdict == "SUPPORTED"         # DYCI's post is about another day
    same_day = _result("CONTRADICTED", used_text="No events on March 14, 2030.")
    assert api._apply_org_pages_v3(same_day, ok) is same_day                     # DYCI's post is about this day
    wrong = api._apply_org_pages_v3(_result("INSUFFICIENT_EVIDENCE"), "site web dev seminar march 14 is cancelled")
    assert wrong.verdict == "INSUFFICIENT_EVIDENCE" and wrong.org_related["name"].startswith("SITE")
    wide = api._apply_org_pages_v3(_result("INSUFFICIENT_EVIDENCE"), "no classes on march 14 because of the seminar")
    assert wide.verdict == "INSUFFICIENT_EVIDENCE"
    withheld = _result("INSUFFICIENT_EVIDENCE", withheld="language")
    assert api._apply_org_pages_v3(withheld, ok) is withheld
    monkeypatch.setattr(post_age, "today", lambda: dt.date(2030, 7, 1))          # 114 days later: stale
    assert api._apply_org_pages_v3(_result("INSUFFICIENT_EVIDENCE"), ok).verdict == "INSUFFICIENT_EVIDENCE"


# ---- Rule O v3.1 (results/round2/org_pages_v31_plan.md) -------------------------------------------------------------
M22_POST = ("ACES CLASH CUP Calling all gamers! Our Mobile Legends esports tournament will be held on September 19, 2030 "
            "at 9:00 AM in the gymnasium. Teams of 5 can register for 250 pesos per team. The champion will win 8,000 cash!")


def test_mode_v31(monkeypatch):
    monkeypatch.setenv(op.ENV, "31")
    assert op.mode() == "verdict31" and op.enabled()


def test_strict_numbers_blocks_a_wrong_amount():
    wrong = "aces mobile legends tournament on september 19 champion wins 10000"
    assert op.match(wrong, M22_POST) is not None                      # v3: 6 of 7 words (86%) passes
    assert op.match(wrong, M22_POST, strict_numbers=True) is None     # v3.1: the missing number fails it
    right = "aces mobile legends tournament on september 19 champion wins 8000"
    assert op.match(right, M22_POST, strict_numbers=True) is not None
    assert op.match("mobile legends tournament september 19 teams of 5", M22_POST, strict_numbers=True) is not None


# ---- Rule O v3.1 fallback (CAMPUS_VERIFY_ORG_RELATED_FALLBACK) --------------------------------------------------------
def _v31_setup(monkeypatch, tmp_path, fallback):
    import json as _json
    import datetime as dt
    import src.api.main as api
    from src.verification import post_age
    (tmp_path / "aces.jsonl").write_text(_json.dumps({"document_id": "ingested:a1", "text": M22_POST,
                                                       "source_url": "https://fb/a1",
                                                       "published_date": "2030-09-10T01:00:00+0000"}) + "\n",
                                          encoding="utf-8")
    monkeypatch.setenv(op.ENV, "31")
    monkeypatch.setenv(op.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(op.SOURCES_ENV, "aces=ACES, a DYCI student organisation")
    monkeypatch.setattr(post_age, "today", lambda: dt.date(2030, 9, 12))
    if fallback:
        monkeypatch.setenv(op.RELATED_FALLBACK_ENV, "1")
    else:
        monkeypatch.delenv(op.RELATED_FALLBACK_ENV, raising=False)
    calls = []
    monkeypatch.setattr(api, "_org_verifier", lambda: ("verifier", {"ingested:1_2": ("aces", "ACES")}, {}))
    monkeypatch.setattr(api, "_gated", lambda fn, *a, **k: calls.append(1) or _fake("SUPPORTED"))
    return api, calls, SimpleNamespace(relevance_checker=None)


def test_fallback_off_by_default(monkeypatch, tmp_path):
    api, calls, pipeline = _v31_setup(monkeypatch, tmp_path, fallback=False)
    assert op.related_fallback() is False
    out = api._apply_org_pages(_result("INSUFFICIENT_EVIDENCE"), "aces mobile legends tournament", pipeline)
    assert not getattr(out, "org_related", None) and calls == []


@pytest.mark.parametrize("message", ["aces mobile legends tournament",                  # no date in the message
                                     "aces mobile legends tournament on october 3 2030", # no post names this date
                                     "no classes because of the aces tournament"])        # school-wide, no date
def test_fallback_shows_related_never_decides(monkeypatch, tmp_path, message):
    api, calls, pipeline = _v31_setup(monkeypatch, tmp_path, fallback=True)
    out = api._apply_org_pages(_result("INSUFFICIENT_EVIDENCE"), message, pipeline)
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and out.org_related["name"] == "ACES" and calls == [1]
    assert not getattr(out, "org_source", None)


def test_fallback_leaves_v31_answers_alone(monkeypatch, tmp_path):
    api, calls, pipeline = _v31_setup(monkeypatch, tmp_path, fallback=True)
    ok = api._apply_org_pages(_result("INSUFFICIENT_EVIDENCE"),
                              "aces mobile legends tournament on september 19 champion wins 8000", pipeline)
    assert ok.verdict == "SUPPORTED" and ok.org_source["version"] == "org-pages-v3.1"
    dated = api._apply_org_pages(_result("INSUFFICIENT_EVIDENCE"),                        # v3.1's own related post
                                 "aces mobile legends tournament on september 19 champion wins 10000", pipeline)
    assert dated.verdict == "INSUFFICIENT_EVIDENCE" and dated.org_related["document_id"] == "ingested:a1"
    dyci = _result("SUPPORTED")
    assert api._apply_org_pages(dyci, "aces mobile legends tournament", pipeline) is dyci
    contra = _result("CONTRADICTED", used_text="Classes resume on September 19, 2030.")
    assert api._apply_org_pages(contra, "aces mobile legends tournament", pipeline) is contra
    withheld = _result("INSUFFICIENT_EVIDENCE", withheld="language")
    assert api._apply_org_pages(withheld, "aces mobile legends tournament", pipeline) is withheld
    assert calls == []


def test_fallback_nothing_relevant(monkeypatch, tmp_path):
    api, calls, pipeline = _v31_setup(monkeypatch, tmp_path, fallback=True)
    monkeypatch.setattr(api, "_gated", lambda fn, *a, **k: _fake("INSUFFICIENT_EVIDENCE", status="NO_RELEVANT_EVIDENCE"))
    out = api._apply_org_pages(_result("INSUFFICIENT_EVIDENCE"), "dyci tuition will increase", pipeline)
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and not getattr(out, "org_related", None)
