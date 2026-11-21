"""Hosting options (docs/deploy.md): framing by named sites, privacy-safe request log, the page served by the API, and
the Docker allow-list. All default off; the image turns them on."""
import logging
import os
import re
import subprocess
import sys

from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_framing_by_default_and_named_sites_only_when_set(api_settings, monkeypatch):
    app = api_settings()
    with TestClient(app.app) as c:
        monkeypatch.delenv("CAMPUS_VERIFY_FRAME_ANCESTORS", raising=False)
        plain = c.get("/health").headers
        monkeypatch.setenv("CAMPUS_VERIFY_FRAME_ANCESTORS", "https://huggingface.co")
        framed = c.get("/health").headers
    assert plain["x-frame-options"] == "DENY" and "content-security-policy" not in plain
    assert "x-frame-options" not in framed
    assert framed["content-security-policy"] == "frame-ancestors 'self' https://huggingface.co"


def test_the_request_log_has_no_address_message_or_query(api_settings, monkeypatch, caplog):
    monkeypatch.setenv("CAMPUS_VERIFY_REQUEST_LOG", "1")
    app = api_settings()
    log = app._request_logger()
    log.propagate = True                                       # let pytest capture it for this test
    try:
        with TestClient(app.app) as c, caplog.at_level(logging.INFO, logger="campus_verify.requests"):
            c.post("/verify/text?secret=q", json={"claim": "Zyxwv classes are suspended tomorrow."},
                   headers={"X-Forwarded-For": "203.0.113.77"})
    finally:
        log.propagate = False
    lines = [r.getMessage() for r in caplog.records if r.name == "campus_verify.requests"]
    assert lines and re.match(r"^POST /verify/text 200 \d+ms$", lines[-1])
    text = " ".join(lines)
    assert "Zyxwv" not in text and "203.0.113" not in text and "secret" not in text and "testclient" not in text


def test_the_api_serves_the_page_with_a_same_origin_config_only_when_asked():
    code = ("import os, sys; sys.path.insert(0, os.getcwd()); from fastapi.testclient import TestClient; "
            "import src.api.main as m; c = TestClient(m.app); "
            "p = c.get('/'); k = c.get('/js/config.js'); "
            "print(p.status_code, 'verify-form' in p.text, k.status_code, repr(k.text.strip()))")
    out = {}
    for on in ("0", "1"):
        env = dict(os.environ, CAMPUS_VERIFY_SERVE_FRONTEND=on, HF_HUB_OFFLINE="1")
        out[on] = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env,
                                 timeout=300).stdout.strip().splitlines()[-1]
    assert out["0"].startswith("404 False 404")
    assert out["1"].startswith("200 True 200") and 'API_BASE_URL: ""' in out["1"]


def test_the_docker_allow_list_keeps_private_data_out():
    ignore = open(os.path.join(ROOT, ".dockerignore"), encoding="utf-8").read().splitlines()
    allowed = [line[1:] for line in ignore if line.startswith("!")]
    assert ignore[ignore.index("*")] == "*"                     # everything excluded first
    for private in ("data/dev2", "data/validation_independent", "data/image_eval", "data/photo_eval", "results",
                    "data/acceptance", "HANDOFF.md", ".venv312"):
        assert not any(a.rstrip("/") == private or a.startswith(private + "/") for a in allowed), private
    docker = open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8").read()
    copied = re.findall(r"^COPY --chown=user (.+?) \S+$", docker, re.M)
    sources = [s for line in copied for s in line.split()]
    for s in sources:
        assert any(s == a.rstrip("/") or s.startswith(a.rstrip("/") + "/") or a.rstrip("/").startswith(s + "/")
                   for a in allowed), f"{s} is copied but not allowed by .dockerignore"
