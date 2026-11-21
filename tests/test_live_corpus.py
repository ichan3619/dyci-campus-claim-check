"""
src/api/pipeline.py: new posts become evidence without restarting the API, and the optional Facebook refresh.
Temporary corpus files and fake models only; no network.
"""
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.api import pipeline as P
from src.api.pipeline import PipelineManager, Settings
from src.classification.result import ClassificationResult
from src.ingestion.corpus_storage import append_to_corpus
from tests.test_m11_2_corpus import rec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(ROOT, "data", "dev", "sample_claims.csv")


class _Clf:
    def classify(self, claim, evidence):
        return ClassificationResult(label="SUPPORTED", confidence=0.9, label_scores={"SUPPORTED": 0.9})


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture()
def corpus(tmp_path):
    path = tmp_path / "fb.jsonl"
    append_to_corpus(rec("Library closed October 12 for inventory.", "post-1"), str(path))
    return path


def _manager(corpus, clock=None, encoder=None, **settings):
    s = replace(Settings(), **{"classifier": "tfidf_lr", "retriever": "tfidf", "corpus": str(corpus),
                               "claim_rewrite": "none", "reload_seconds": 0, **settings})
    return PipelineManager(s, _Clf(), encoder, project_root=ROOT, dev_dataset_path=DEV, clock=clock or _Clock())


def _grow(path, text, ident):
    append_to_corpus(rec(text, ident), str(path))
    os.utime(path, ns=(os.stat(path).st_atime_ns, os.stat(path).st_mtime_ns + 10_000_000))   # coarse-clock filesystems


# ---- live reload ------------------------------------------------------------------------------------------------------
def test_a_new_post_becomes_evidence_without_a_restart(corpus):
    m = _manager(corpus)
    assert len(m.current().documents) == 1
    _grow(corpus, "Enrollment for the second semester extended until August 30.", "post-2")
    p = m.current()
    assert len(p.documents) == 2 and p.sources[0].documents == 2
    r = p.verifier.verify_claim("enrollment extended until August 30")
    assert r.used_evidence is not None and "August 30" in r.used_evidence.text


def test_unchanged_files_do_not_rebuild(corpus):
    m = _manager(corpus)
    first = m.current()
    assert m.current() is first and m.current() is first


def test_checks_are_throttled_by_reload_seconds(corpus):
    clock = _Clock()
    m = _manager(corpus, clock=clock, fb_corpus=str(corpus))
    m.settings = replace(m.settings, reload_seconds=10)
    m.current()                                          # first check happens now
    _grow(corpus, "A second post.", "post-2")
    clock.t = 5
    assert len(m.current().documents) == 1               # within 10 s: not checked yet
    clock.t = 11
    assert len(m.current().documents) == 2


def test_negative_reload_seconds_never_reloads(corpus):
    m = _manager(corpus, reload_seconds=-1)
    _grow(corpus, "A second post.", "post-2")
    assert len(m.current().documents) == 1


def test_a_broken_file_keeps_the_previous_corpus_and_recovers(corpus):
    m = _manager(corpus)
    good = m.current()
    with open(corpus, "a", encoding="utf-8") as f:
        f.write('{"half-written": ')                     # e.g. read while the refresh is still writing
    os.utime(corpus, ns=(os.stat(corpus).st_atime_ns, os.stat(corpus).st_mtime_ns + 10_000_000))
    assert m.current() is good and m.reload_error
    lines = open(corpus, encoding="utf-8").read().splitlines()[:-1]
    open(corpus, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    _grow(corpus, "A second post.", "post-2")
    assert len(m.current().documents) == 2 and m.reload_error is None


def test_the_first_build_must_succeed(tmp_path):
    with pytest.raises(RuntimeError):
        _manager(tmp_path / "missing.jsonl")


def test_rebuild_embeds_only_the_new_passages(corpus):
    from src.retrieval.e5_retriever import E5Encoder
    from tests.test_m12_2_e5_and_passages import FakeModel
    enc = E5Encoder(model=FakeModel())
    m = _manager(corpus, encoder=enc)
    before = len(enc._model.seen)
    _grow(corpus, "Enrollment extended until August 30.", "post-2")
    m.current()
    new = enc._model.seen[before:]
    assert new and all("August 30" in t for t in new)    # the old post was not embedded again


# ---- Facebook refresh -------------------------------------------------------------------------------------------------
def test_refresh_without_credentials_reports_and_does_nothing(corpus, monkeypatch):
    monkeypatch.delenv("META_PAGE_ID", raising=False)
    monkeypatch.delenv("META_ACCESS_TOKEN", raising=False)
    m = _manager(corpus, fb_corpus=str(corpus))
    status = m.refresh_facebook()
    assert status.last_error.startswith("META_PAGE_ID") and status.last_added is None


def test_refresh_appends_new_posts_and_reloads(corpus, monkeypatch):
    from src.ingestion import facebook_page
    monkeypatch.setenv("META_PAGE_ID", "123")
    monkeypatch.setenv("META_ACCESS_TOKEN", "secret-token")

    def fake_refresh(path, page_id, token, **_):
        append_to_corpus(rec("Classes resume on Monday, October 13.", "post-9"), path)
        return SimpleNamespace(added=1), None

    monkeypatch.setattr(facebook_page, "refresh_from_mock_page", fake_refresh)
    m = _manager(corpus, fb_corpus=str(corpus), reload_seconds=-1)        # proves refresh reloads by itself
    status = m.refresh_facebook()
    assert status.last_added == 1 and status.last_error is None and len(m.current().documents) == 2


def test_refresh_errors_never_expose_the_token(corpus, monkeypatch):
    from src.ingestion import facebook_page
    monkeypatch.setenv("META_PAGE_ID", "123")
    monkeypatch.setenv("META_ACCESS_TOKEN", "secret-token")

    def failing(*_a, **_k):
        raise RuntimeError("GET https://graph.facebook.com/123/posts?access_token=secret-token failed: 400")

    monkeypatch.setattr(facebook_page, "refresh_from_mock_page", failing)
    status = _manager(corpus, fb_corpus=str(corpus)).refresh_facebook()
    assert "secret-token" not in status.last_error and "[REDACTED]" in status.last_error


def test_background_refresh_runs_and_stops(corpus, monkeypatch):
    calls = []
    m = _manager(corpus, fb_corpus=str(corpus), fb_refresh_minutes=60)
    monkeypatch.setattr(m, "refresh_facebook", lambda: calls.append(1))
    m.start_facebook_refresh()
    m._thread.join(timeout=0.5) if not calls else None
    m.stop()
    assert calls and not m._thread.is_alive()


def test_refresh_target_must_be_a_corpus_source(corpus):
    s = replace(Settings(), classifier="tfidf_lr", retriever="tfidf", corpus="dev", claim_rewrite="none",
                fb_refresh_minutes=5, fb_corpus=str(corpus))
    with pytest.raises(RuntimeError, match="not one of the corpus sources"):
        s.validate()


# ---- settings ---------------------------------------------------------------------------------------------------------
def test_defaults_are_the_configuration_chosen_on_the_development_data(monkeypatch, tmp_path):
    for k in [k for k in os.environ if k.startswith("CAMPUS_VERIFY_")]:
        monkeypatch.delenv(k)
    monkeypatch.setattr(P, "DEFAULT_TAGALOG_MODEL_DIR", str(tmp_path))          # 'downloaded'
    s = Settings.from_env()
    assert (s.classifier, s.retriever, s.corpus, s.router, s.claim_rewrite, s.date_check) == \
           ("nli", "e5", P.DEFAULT_CORPUS_SPEC, True, "optionB", True)
    assert s.tagalog_nli_model == str(tmp_path) and s.fb_refresh_minutes == 0


def test_tagalog_model_is_skipped_when_not_downloaded_or_switched_off(monkeypatch, tmp_path):
    monkeypatch.delenv("CAMPUS_VERIFY_CLASSIFIER", raising=False)
    monkeypatch.setattr(P, "DEFAULT_TAGALOG_MODEL_DIR", str(tmp_path / "absent"))
    monkeypatch.delenv("CAMPUS_VERIFY_TAGALOG_NLI_MODEL", raising=False)
    assert Settings.from_env().tagalog_nli_model == ""
    monkeypatch.setattr(P, "DEFAULT_TAGALOG_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("CAMPUS_VERIFY_TAGALOG_NLI_MODEL", "none")
    assert Settings.from_env().tagalog_nli_model == ""
