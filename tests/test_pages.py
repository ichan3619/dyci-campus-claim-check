"""About, Terms of Use and Privacy Notice pages (panel request 2026-10-28): served with the page, linked from the menu and
the sign-in screen, plain HTML with no scripts."""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def page(name):
    return " ".join(open(os.path.join(ROOT, "frontend", name), encoding="utf-8").read().split())


@pytest.mark.parametrize("name, words", [
    ("about.html", ["How it works", "Not confirmed yet", "not an official DYCI service", "terms.html", "privacy.html"]),
    ("terms.html", ["Acceptable use", "not an official DYCI service", "@dyci.edu.ph", "privacy.html",
                    "Republic Act No. 10173"]),
])
def test_the_pages_say_what_they_must(name, words):
    text = page(name)
    for w in words:
        assert w in text, w
    assert "<script" not in text


def test_the_menu_and_sign_in_link_to_them_and_possible_outcomes_is_no_longer_a_tab():
    index = page("index.html")
    assert 'href="about.html"' in index and 'href="terms.html"' in index
    assert "By signing in you agree to the" in index
    assert 'data-view="guide"' not in index


def test_the_landing_page_explains_the_checker_before_sign_in():
    index = page("index.html")
    landing = index[index.index('id="view-signin"'):index.index('id="view-account"')]
    for words in ("Check it before you pass it on", "Type it", "Screenshot it", "Link it", "google-btn", "terms.html"):
        assert words in landing, words


def test_dates_on_the_answer_are_written_for_people():
    app = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
    assert "function readableDate" in app and "readableDate(data.used_evidence.original_date" in app


def test_the_outcomes_card_says_exactly_what_the_tour_says():
    import re
    index = page("index.html")
    app = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
    rows = re.findall(r'\["(?:supported|contradicted|insufficient)", "([^"]+)", "([^"]+)"\]', app)
    assert len(rows) == 3
    card = index[index.index('id="guide"'):index.index("</section>", index.index('id="guide"'))]
    for label, text in rows:
        assert f"<h3>{label}</h3><p>{text}</p>" in card, label
    assert "What the answer means" in card and "Always read the evidence" in card
    assert "Under the answer you see the official post it came from, with its date and a link. Open it to be sure." in card


def test_the_picture_levels_card_says_exactly_what_the_tour_says():
    import re
    index = page("index.html")
    app = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
    block = app[app.index("function exampleLevels"):app.index("function exampleEvidence")]
    block = re.sub(r'"\s*\+\s*"', "", block).replace(chr(92) + "u2019", chr(0x2019))
    rows = re.findall(r'\[(\d), "([^"]+)", "([^"]+)"\]', block)
    assert [r[0] for r in rows] == ["5", "4", "3", "2", "1"]
    card = index[index.index('id="guide-levels"'):index.index("</ul>", index.index('id="guide-levels"'))]
    for level, label, text in rows:
        assert f"<h3>Level {level}: {label}</h3><p>{text}</p>" in card, level
    assert 'id="guide-levels" class="guide-levels" hidden' in index        # only when the Image Check is on
    assert '$("guide-levels").hidden = false' in app
