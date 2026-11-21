"""
Background corpus rebuilds and the main.py wiring (src/api/background_reload.py; scale plan S1). Synthetic posts only.
"""
import os
import threading
import time
from dataclasses import replace

import pytest

from src.api import pipeline as P
from src.api.background_reload import ScalablePipelineManager
from src.api.pipeline import PipelineManager, Settings
from src.retrieval import vector_store as vs
from tests.test_live_corpus import DEV, ROOT, _Clf, _Clock, _grow, corpus  # noqa: F401  (corpus is a fixture)
from tests.test_vector_store import BatchModel


def _manager(corpus, background=True, encoder=None, clock=None):
    s = replace(Settings(), classifier="tfidf_lr", retriever="tfidf", corpus=str(corpus), claim_rewrite="none",
                reload_seconds=0)
    return ScalablePipelineManager(s, _Clf(), encoder, project_root=ROOT, dev_dataset_path=DEV,
                                   clock=clock or _Clock(), background=background)


@pytest.fixture()
def gate(monkeypatch):
    """build_verifier waits for `release` once armed, so a rebuild can be held in progress."""
    armed, release, calls = threading.Event(), threading.Event(), []
    real = P.build_verifier

    def held(*args, **kwargs):
        calls.append(1)
        if armed.is_set():
            assert release.wait(10)
        return real(*args, **kwargs)
    monkeypatch.setattr(P, "build_verifier", held)
    return armed, release, calls


def test_with_background_off_it_behaves_like_the_parent(corpus):
    m = _manager(corpus, background=False)
    _grow(corpus, "Enrollment extended until August 30.", "post-2")
    assert len(m.current().documents) == 2                           # rebuilt inside the request, as before


def test_a_background_rebuild_never_makes_a_request_wait(corpus, gate):
    armed, release, _ = gate
    m = _manager(corpus)
    old = m.current()
    armed.set()
    _grow(corpus, "Enrollment extended until August 30.", "post-2")
    t = time.perf_counter()
    during = m.current()
    assert time.perf_counter() - t < 0.5 and during is old            # served at once, from the old index
    assert m.current() is old
    release.set()
    assert m.wait_for_rebuild(10)
    new = m.current()
    assert new is not old and len(new.documents) == 2 and m.reload_error is None


def test_only_one_rebuild_runs_at_a_time(corpus, gate):
    armed, release, calls = gate
    m = _manager(corpus)
    before = len(calls)
    armed.set()
    _grow(corpus, "Enrollment extended until August 30.", "post-2")
    for _ in range(20):
        m.current()
    release.set()
    assert m.wait_for_rebuild(10)
    assert len(calls) - before == before                             # exactly one rebuild: as many calls as the first build


def test_a_failed_background_rebuild_keeps_the_previous_corpus(corpus, monkeypatch):
    m = _manager(corpus)
    old = m.current()

    def broken(*args, **kwargs):
        raise RuntimeError("half-written file")
    monkeypatch.setattr(P, "build_verifier", broken)
    _grow(corpus, "Enrollment extended until August 30.", "post-2")
    m.current()
    assert m.wait_for_rebuild(10)
    assert m.current() is old and "half-written file" in m.reload_error


def test_reloads_save_vectors_and_a_restart_reads_them(corpus, tmp_path):
    store = tmp_path / "vectors"
    _manager(corpus, background=False, encoder=vs.PersistentE5Encoder(str(store), model=BatchModel(), revision="r1"))
    segments = [n for root, _, files in os.walk(store) for n in files if n.startswith("seg-")]
    assert len(segments) == 1
    model = BatchModel()
    _manager(corpus, background=False, encoder=vs.PersistentE5Encoder(str(store), model=model, revision="r1"))
    assert not [c for c in model.calls if all(t.startswith("passage: ") for t in c)]   # no passage embedded again


def test_a_new_post_found_by_a_background_rebuild_is_saved_too(corpus, tmp_path):
    store = tmp_path / "vectors"
    m = _manager(corpus, encoder=vs.PersistentE5Encoder(str(store), model=BatchModel(), revision="r1"))
    _grow(corpus, "Enrollment extended until August 30.", "post-2")
    m.current()
    assert m.wait_for_rebuild(10)
    segments = [n for root, _, files in os.walk(store) for n in files if n.startswith("seg-")]
    assert len(segments) == 2 and len(m.current().documents) == 2


# ---- main.py wiring --------------------------------------------------------------------------------------------------
def test_main_builds_exactly_the_v1_6_manager_when_both_settings_are_off(monkeypatch):
    from src.api import main
    monkeypatch.delenv(vs.ENV, raising=False)
    monkeypatch.delenv("CAMPUS_VERIFY_BACKGROUND_RELOAD", raising=False)
    m = main._build_manager(main.SETTINGS, main.build_classifier(main.SETTINGS))
    assert type(m) is PipelineManager


def test_main_selects_the_scalable_manager_from_the_settings(monkeypatch, tmp_path):
    from src.api import main
    monkeypatch.setenv(vs.ENV, str(tmp_path / "vectors"))
    monkeypatch.setenv("CAMPUS_VERIFY_BACKGROUND_RELOAD", "1")
    m = main._build_manager(main.SETTINGS, main.build_classifier(main.SETTINGS))
    assert type(m) is ScalablePipelineManager and m._background is True
    assert m.encoder is None                                         # the test configuration uses TF-IDF: nothing to save
    m.stop()


def test_the_rebuild_switches_threads_more_often_and_restores_the_setting(corpus, monkeypatch):
    import sys
    from src.api import background_reload as br
    seen, before = [], sys.getswitchinterval()
    real = P.build_verifier

    def watch(*args, **kwargs):
        seen.append(sys.getswitchinterval())
        return real(*args, **kwargs)
    m = _manager(corpus)
    monkeypatch.setattr(P, "build_verifier", watch)
    _grow(corpus, "Enrollment extended until August 30.", "post-2")
    m.current()
    assert m.wait_for_rebuild(10)
    assert seen and all(v == br.REBUILD_SWITCH_INTERVAL for v in seen)
    assert sys.getswitchinterval() == before


def test_the_setting_is_restored_even_when_the_rebuild_fails(corpus, monkeypatch):
    import sys
    before = sys.getswitchinterval()
    m = _manager(corpus)
    monkeypatch.setattr(P, "build_verifier", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken")))
    _grow(corpus, "Enrollment extended until August 30.", "post-2")
    m.current()
    assert m.wait_for_rebuild(10)
    assert sys.getswitchinterval() == before and "broken" in m.reload_error

