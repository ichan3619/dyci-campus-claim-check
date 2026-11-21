"""Option A version 2: a picture's level card also shows its whole-text verdict
(results/round2/picture_text_line_v2_plan.md)."""
import inspect
import os
from types import SimpleNamespace

import pytest

from src.image import text_line as tl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POST = "DYCI ADVISORY 27 SEPTEMBER 2026 Tomorrow, September 28, 2026, classes will be until 1:00 PM only."


def _check(verdict, withheld=None, text=POST, date="2026-09-29T06:31:23+0000", original=None,
           url="https://www.facebook.com/dyci/posts/1", original_url=None, rule_g=None):
    ev = SimpleNamespace(document_id="P1", text=text, published_date=date, original_date=original, source_url=url,
                         original_url=original_url)
    return SimpleNamespace(verdict=verdict, verdict_withheld=withheld, used_evidence=ev, poster_language=rule_g)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(tl.ENV, raising=False)
    assert tl.enabled() is False


def test_version():
    assert tl.VERSION == "picture-text-line-v2"


@pytest.mark.parametrize("verdict, line", [
    ("SUPPORTED", "Checked as one piece of text, the words read from this picture match DYCI's announcement dated "
                  "27 September."),
    ("CONTRADICTED", "Checked as one piece of text, the words read from this picture conflict with DYCI's announcement "
                     "dated 27 September."),
    ("INSUFFICIENT_EVIDENCE", "Checked as one piece of text, the words read from this picture could not be confirmed."),
])
def test_lines(verdict, line):
    out = tl.build(_check(verdict), 3)
    assert out["line"] == line and out["verdict"] == verdict and "typed text" not in out["line"]


def test_header_date_named_not_the_repost_date():
    out = tl.build(_check("SUPPORTED"), 5)
    assert out["post_dated"] == "27 September" and out["post_date"] == "29 September 2026"
    assert "29 September" not in out["line"]


@pytest.mark.parametrize("original, date, post", [
    ("2026-09-27", "2026-09-29", "DYCI's post of 27 September 2026"),       # original_date first
    (None, "2026-09-29T06:31:23+0000", "DYCI's post of 29 September 2026"),
    (None, None, "DYCI's post"),
])
def test_no_header_date_falls_back(original, date, post):
    out = tl.build(_check("CONTRADICTED", text="Classes are suspended tomorrow.", original=original, date=date), 2)
    assert out["post_dated"] is None and out["line"].endswith(f"conflict with {post}.")


def test_rule_g_note():
    out = tl.build(_check("SUPPORTED", rule_g={"released": True}), 5)
    assert out["rule_g"] is True and out["line"].endswith(tl.RULE_G_NOTE.strip())
    assert tl.build(_check("SUPPORTED"), 5)["rule_g"] is False
    assert "set aside" not in tl.build(_check("SUPPORTED"), 5)["line"]


def test_post_fields():
    out = tl.build(_check("CONTRADICTED"), 2)
    assert (out["post"], out["post_url"]) == ("P1", "https://www.facebook.com/dyci/posts/1")
    assert tl.build(_check("SUPPORTED", original_url="https://www.facebook.com/dyci/posts/9"), 5)["post_url"].endswith("/9")
    assert tl.build(_check("SUPPORTED", url=""), 5)["post_url"] is None
    assert tl.build(_check("INSUFFICIENT_EVIDENCE"), 3)["post"] is None     # no post named for "could not be confirmed"


@pytest.mark.parametrize("verdict, level, differs", [
    ("CONTRADICTED", 1, False), ("CONTRADICTED", 2, True), ("CONTRADICTED", 3, True),
    ("SUPPORTED", 5, False), ("SUPPORTED", 4, False), ("SUPPORTED", 3, True), ("SUPPORTED", 2, True),
    ("INSUFFICIENT_EVIDENCE", 3, False), ("INSUFFICIENT_EVIDENCE", 2, True), ("INSUFFICIENT_EVIDENCE", 4, True),
])
def test_differs(verdict, level, differs):
    assert tl.build(_check(verdict), level)["differs"] is differs


@pytest.mark.parametrize("check", [None, _check("SUPPORTED", withheld="language"), _check(None), _check("UNKNOWN")])
def test_no_line(check):
    assert tl.build(check, 3) is None


def test_wired_after_level_and_rule_f_picture_only():
    import src.api.main as api
    src = inspect.getsource(api.verify_image)
    assert "text_line.enabled()" in src
    assert src.index("assess(response.model_dump()") < src.index("changed_date.check") < src.index("text_line.build")
    assert "text_line" not in inspect.getsource(api.verify_text)


def test_page_shows_the_line_and_the_note():
    js = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
    card = js[js.index("function renderLevelCard"):js.index("imageForm.addEventListener")]
    assert "data.text_line" in card and "tl.line" in card and "safeLink(tl.post_url" in card
    assert tl.DIFFERS_NOTE in card and "tl.differs" in card
