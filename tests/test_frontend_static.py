"""
Static-asset sanity checks for the M11.1 frontend (frontend/).

These are deliberately shallow, file-level checks -- there is no JavaScript
runtime in the Python suite, so they do NOT test UI behavior (that was
verified separately in a real browser; see the M11.1 report). What they
DO guard, cheaply and permanently, are the project decisions that are easy
to break by accident in a later edit:

  - the agreed file structure exists and is wired together in order
  - plain HTML/CSS/JS only: no build step, no framework, no CDN/external
    assets
  - API base URL is configurable in config.js and defaults to the agreed
    local address
  - the frontend makes only the existing backend calls (/health, /verify/text, /verify/url, and /verify/image for the
    gated Image Check section, owner-approved 2026-10-13)
  - backend-supplied text is never inserted as HTML (evidence from
    /verify/url is arbitrary third-party content)
  - canonical verdicts are preserved; scores are never shown as percentages
  - no Facebook UI; an image upload exists ONLY inside the hidden-unless-enabled Image Check section (Phase 1, owner-
    approved 2026-10-13), sent as the raw file body: still no FileReader or FormData anywhere
  - the backend CORS allow-list contains the two agreed origins and never "*"
"""

import os
import re

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(PROJECT_ROOT, "frontend")

EXPECTED_FILES = [
    "index.html",
    os.path.join("css", "styles.css"),
    os.path.join("js", "config.js"),
    os.path.join("js", "api.js"),
    os.path.join("js", "app.js"),
]


def read(*parts):
    with open(os.path.join(FRONTEND, *parts), encoding="utf-8") as f:
        return f.read()


def all_js():
    return {name: read("js", name) for name in ("config.js", "api.js", "app.js")}


# ---- structure ------------------------------------------------------------

@pytest.mark.parametrize("relative_path", EXPECTED_FILES)
def test_expected_frontend_file_exists_and_is_not_empty(relative_path):
    path = os.path.join(FRONTEND, relative_path)
    assert os.path.isfile(path), f"missing frontend file: {relative_path}"
    assert os.path.getsize(path) > 0


def test_index_html_wires_stylesheet_and_scripts_in_dependency_order():
    html = read("index.html")
    assert 'href="css/styles.css"' in html
    order = [html.index(f'src="js/{name}"') for name in ("config.js", "api.js", "app.js")]
    assert order == sorted(order), "scripts must load config.js -> api.js -> app.js"


def test_no_build_step_or_framework_files():
    for forbidden in ("package.json", "webpack.config.js", "vite.config.js", "node_modules"):
        assert not os.path.exists(os.path.join(FRONTEND, forbidden))


def test_index_html_uses_no_external_assets():
    html = read("index.html")
    urls = re.findall(r'(?:src|href)\s*=\s*"([^"]+)"', html)
    external = [u for u in urls if re.match(r"^(https?:)?//", u)]
    assert external == [], f"external assets found: {external}"
    css = read("css", "styles.css")
    assert "@import" not in css and "url(http" not in css


# ---- configuration ---------------------------------------------------------

def test_config_default_api_base_url_is_the_agreed_local_address():
    config = read("js", "config.js")
    assert re.search(r'API_BASE_URL:\s*"http://127\.0\.0\.1:8000"', config)


def test_api_client_reads_base_url_from_config_not_a_literal():
    api = read("js", "api.js")
    assert "API_BASE_URL" in api
    assert "127.0.0.1" not in api and "localhost" not in api


# ---- API surface -----------------------------------------------------------

def test_frontend_calls_only_the_known_endpoints():
    # /verify/image was added for the gated Image Check section (Phase 1, owner-approved 2026-10-13); /stats for the
    # dashboard (CAMPUS_VERIFY_STATS, counts only; owner request 2026-10-26); /auth/*, /account/history and /admin/users
    # for Sign in with Google, DYCI accounts only (CAMPUS_VERIFY_AUTH=google; owner and panel request 2026-10-28);
    # /account/checks* for My checks (CAMPUS_VERIFY_HISTORY; panel request, owner approval 2026-10-29); /stats.csv for
    # the dashboard's spreadsheet export (same access rule as /stats; panel request 2026-10-29); /admin/fetch-status for
    # the Sources page's live-fetch box (fix B, CAMPUS_VERIFY_FETCH_STATUS; owner 2026-11-17).
    api = read("js", "api.js")
    called = set(re.findall(r'(?:request|postJson|download)\(\s*"(/[^"]*)"', api))
    assert called == {"/health", "/verify/text", "/verify/url", "/verify/image", "/stats?days=", "/auth/config", "/auth/me",
                      "/auth/google", "/auth/logout", "/account/history", "/account/delete", "/account/tour-seen", "/admin/users",
                      "/admin/users/", "/account/checks", "/account/checks/", "/account/checks/clear", "/stats.csv?days=",
                      "/admin/sources", "/admin/sources/", "/admin/sources/cutoff", "/admin/fetch-status"}


# ---- safety ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["api.js", "app.js"])
def test_no_html_injection_apis_or_dynamic_code_execution(name):
    source = all_js()[name]
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function("):
        assert forbidden not in source, f"{name} uses {forbidden}"


def test_no_browser_storage_or_third_party_calls():
    for name, source in all_js().items():
        assert "localStorage" not in source and "sessionStorage" not in source
        # Only the configured API base may be contacted.
        assert not re.search(r'fetch\(\s*["\']https?://', source), name


# ---- project decisions preserved -----------------------------------------------

def test_canonical_verdicts_are_present_and_unrenamed():
    app = read("js", "app.js")
    for verdict in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"):
        assert verdict in app
    # Insufficient evidence must never be described as false.
    assert "does not mean the claim is false" in app


def test_scores_use_agreed_terminology_and_are_never_percentages():
    app = read("js", "app.js")
    assert "Similarity" in app
    assert "Model score" in app
    assert "not the probability that the claim is true" in app
    assert "* 100" not in app and '"%"' not in app and "'%'" not in app
    assert "toFixed(" in app


def test_backend_evidence_status_values_are_not_reinterpreted():
    app = read("js", "app.js")
    # The raw backend value is displayed alongside any readable form.
    assert "evidence_status" in app
    # No frontend-side verdict decision logic.
    assert 'verdict = "' not in app and "verdict = '" not in app


def _image_section(html):
    m = re.search(r'<section id="image-check"[^>]*>.*?</section>', html, re.S)
    return m.group(0) if m else ""


def test_out_of_scope_ui_is_absent():
    # No Facebook integration in the browser; the WORD "Facebook" is allowed (T2 provenance labels). Image upload
    # (Phase 1, owner-approved 2026-10-13) is allowed ONLY inside the Image Check section, which is hidden by default
    # and shown only when /health reports image_check true; it accepts JPEG/PNG/WebP and sends the raw file body.
    raw = read("index.html")
    section = _image_section(raw)
    assert section and re.search(r'<section id="image-check"[^>]*\bhidden\b', section)
    outside = raw.replace(section, "").lower()
    assert 'type="file"' not in outside and "image" not in outside
    assert section.count('type="file"') == 1 and 'accept="image/jpeg,image/png,image/webp"' in section
    app = read("js", "app.js")
    assert 'if (h.image_check === true && imageCheck) { imageCheck.hidden = false; }' in app
    assert 'var imageCheck = $("image-check");' in app
    html = raw.lower()
    for name, source in [("index.html", html)] + list(all_js().items()):
        low = source.lower()
        assert "connect.facebook.net" not in low and "fb.init" not in low and "fb.login" not in low, name
        assert "FileReader" not in source and "FormData" not in source, name


def test_test_page_copies_are_never_labelled_as_dycis_page():
    app = read("js", "app.js")
    assert "test_page_mirror" in app and "own test Facebook Page (not DYCI" in app
    assert '"Original DYCI post"' in app and '"Not recorded"' in app and '"Copy on the test Page"' in app
    for form in ("DYCI's Facebook Page", "DYCI’s Facebook Page", "DYCI\\u2019s Facebook Page"):
        assert form not in app, form


def test_no_hidden_leftover_sections_and_the_verdict_footer_stays():
    # Owner decision 2026-10-25 (pre-deployment review H2): no "About the data" prototype notice, since the checker
    # is meant to work with DYCI pages later. The commented-out copy is removed; the footer's explanation of the
    # verdicts stays.
    html = read("index.html")
    assert "<section" not in "".join(re.findall(r"<!--.*?-->", html, flags=re.S))
    assert "example campus notices" not in html
    assert "Insufficient evidence does not mean a" in html


# ---- backend CORS contract the frontend depends on --------------------------------

def test_backend_cors_allows_the_agreed_frontend_origins_and_never_wildcard():
    with open(os.path.join(PROJECT_ROOT, "src", "api", "main.py"), encoding="utf-8") as f:
        main_py = f.read()
    assert "http://127.0.0.1:5173" in main_py
    assert "http://localhost:5173" in main_py
    assert "allow_origins=ALLOWED_ORIGINS" in main_py
    assert 'allow_origins=["*"]' not in main_py and "allow_origins=['*']" not in main_py


def test_verdict_cards_stay_readable_in_both_themes():
    css = read("css", "styles.css")
    if "@media (prefers-color-scheme: dark)" in css:
        dark = css.split("@media (prefers-color-scheme: dark)")[1].split("}")[0]
        # Every card colour used for text backgrounds is redefined for dark mode (the info card was white on white).
        for var in ("--info:", "--info-bg:", "--supported-bg:", "--contradicted-bg:", "--insufficient-bg:"):
            assert var in dark, var
    else:
        # Light-only page (owner decision 2026-10-12): it must say so, so form controls are not drawn dark.
        assert "color-scheme: light" in css
    # The verdict icon must not take its background from its own (white) text colour.
    mark = css.split(".verdict-mark {")[1].split("}")[0]
    assert "currentColor" not in mark and "var(--tone)" in mark


# ---- Image Check Part B Stage 2: sentence-level results on the page ------------------------------------------------------
def _image_section_js():
    app = read("js", "app.js")
    start = app.index("var SENTENCE_HEADLINES")
    return app[start:app.index("imageForm.addEventListener")]


def test_page_uses_sentence_results_when_the_api_says_so_and_labels_the_fallback():
    js = _image_section_js()
    assert 'data.check_basis === "sentences"' in js and "sentence_summary" in js
    assert "Checked as one piece of text" in js                                  # B1: the fallback is labelled
    assert "sum.label" in js                                                      # the headline comes from the API


def test_page_never_calls_a_picture_true_or_fake_and_keeps_not_confirmed_wording():
    js = _image_section_js().lower()
    for word in ("is true", "fake", "authentic", "genuine", "is real"):
        assert word not in js
    assert "does not mean the claim is false" in js


def test_page_shows_every_statement_outcome_and_skipped_parts():
    js = _image_section_js()
    for outcome in ("Matches an announcement", "Conflicts with an announcement", "Not confirmed",
                    "Couldn\u2019t check reliably", "limit of 12"):
        assert outcome in js
    assert "isHttpUrl(used.source_url)" in js                                    # post links are web links only


def test_a_contradiction_from_a_post_without_the_messages_day_gets_a_display_only_note():
    app = read("js", "app.js")
    assert "function contradictionDayNote(data)" in app
    note = app[app.index("function contradictionDayNote(data)"):app.index("function oldPostNote(data)")]
    assert 'data.verdict === "CONTRADICTED"' in note and "this post may not apply" in note
    assert not re.search(r"verdict\s*=(?!=)", note)                       # display only: never assigns a verdict


def test_an_old_post_gets_a_highlighted_warning_under_the_verdict_and_never_changes_it():
    app = read("js", "app.js")
    note = app[app.index("function oldPostNote(data)"):app.index("// Display-only relative-date notice")]
    assert "data.post_age_notice" in note and "old-post" in note
    assert not re.search(r"verdict\s*=(?!=)", note)                       # display only: never assigns a verdict
    shown = app[app.index("resultBox.appendChild(renderVerdict(data));"):]
    assert shown.index("oldPostNote(data)") < shown.index("renderClosest(data, \"claim\")")
    assert ".date-note.old-post" in read("css", "styles.css")


def test_the_connection_status_line_exists_is_optional_and_speaks_to_students():
    """Pre-deployment review H2: the status line used to sit inside a commented-out block, so the startup code threw
    on every page load. It is now a real element, the code tolerates its absence, and it never shows developer text."""
    html, app = read("index.html"), read("js", "app.js")
    visible = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    assert 'id="api-status"' in visible and 'role="status"' in visible
    startup = app[app.index("// ---- startup: GET /health"):]
    assert "if (!statusEl) return;" in startup and "&& imageCheck)" in startup
    assert "setTimeout(checkHealth" in startup                       # keeps retrying while a sleeping host starts
    for developer_text in ("uvicorn", "Classifier:", "data_status", "Data status"):
        assert developer_text not in startup
    assert ".api-status.pending" in read("css", "styles.css")


def test_error_messages_are_written_for_students_with_the_technical_text_kept_aside():
    """Pre-deployment review L4: titles, hints and replacement sentences contain no API/server/framework terms; the
    server's text is kept under "Technical details"; server wording written for people (image upload, too short, too
    many requests) is still shown as it is."""
    app, html = read("js", "app.js"), read("index.html")
    block = app[app.index("var ERROR_TITLES = {"):app.index("function plainMessage(err)")]
    for term in ("API", "server logs", "config.js", "uvicorn", "origin", "HTTP"):
        assert term not in block, term
    plain = app[app.index("function plainMessage(err)"):app.index("// ---- state helpers")]
    assert '!err.fieldReport' in plain and "Refusing to fetch" in plain
    assert 'id="error-tech"' in html and 'id="error-tech-text"' in html
    shown = app[app.index("function showError(err)"):]
    assert "plainMessage(err)" in shown and "tech.hidden = !raw || raw === message" in shown
    assert "error.fieldReport = Array.isArray(body && body.detail);" in read("js", "api.js")
    assert "Unrecognized verdict from the API" not in app



def test_the_tour_explains_the_five_picture_levels_with_the_checks_own_names():
    from src.image.levels import NAMES
    app = read("js", "app.js").replace("\\u2019", "'")
    assert "function exampleLevels" in app and "pictureTabAdded ? { title: \"Pictures get a level from 1 to 5\"" in app
    for level in (5, 4, 3, 1):
        assert f'[{level}, "{NAMES[level]}"' in app
    assert NAMES[2].startswith("Sketchy") and '[2, "Sketchy"' in app
    assert "not a percentage" in app
