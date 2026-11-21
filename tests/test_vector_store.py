"""
Saved passage vectors (src/retrieval/vector_store.py; scale plan S1). Synthetic text and a fake model only: no
download, no real posts.
"""
import os
import re
import threading
import zlib

import numpy as np
import pytest

from src.retrieval import vector_store as vs
from src.retrieval.e5_retriever import E5Encoder, E5Retriever

DIM = 32


class BatchModel:
    """Hashed bag of words whose vectors also depend on the batch size, like the real model's padding (measured: up to
    6.7e-8). Records every batch it is asked to encode."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=True):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % DIM] += 1.0
            out[i, 0] += np.float32(1e-6) * len(texts)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


DOCS = [{"document_id": f"d{i}", "text": f"Notice {i}. Classes in room {i} move online. Shared closing line."}
        for i in range(6)]


def _vectors(retriever):
    return {d["text"]: retriever.cached_passages(d["text"])[1] for d in DOCS}


def _fresh():
    return _vectors(E5Retriever(DOCS, encoder=E5Encoder(model=BatchModel()), score_mode="passage"))


def _persistent(tmp_path, revision="r1", model=None):
    return vs.PersistentE5Encoder(str(tmp_path), model=model or BatchModel(), revision=revision)


def test_the_fake_model_really_depends_on_the_batch():
    m = BatchModel()
    assert not np.array_equal(m.encode(["passage: a b"])[0], m.encode(["passage: a b", "passage: c"])[0])


def test_a_restart_reproduces_a_fresh_build_bit_for_bit_without_embedding_again(tmp_path):
    cold = _persistent(tmp_path)
    with vs.persisting():
        E5Retriever(DOCS, encoder=cold, score_mode="passage")
    assert cold.flush() is None
    warm_model = BatchModel()
    warm = _persistent(tmp_path, model=warm_model)
    got = _vectors(E5Retriever(DOCS, encoder=warm, score_mode="passage"))
    assert warm_model.calls == []                                      # every passage batch came from the store
    expected = _fresh()
    assert all(np.array_equal(got[t], expected[t]) for t in expected)


def test_vectors_are_saved_only_during_a_reload_and_queries_never(tmp_path):
    enc = _persistent(tmp_path)
    enc.encode_passages(["page fetched by /verify/url"])               # outside persisting(): memory only
    with vs.persisting():
        enc.encode_queries(["is it open?"])
    assert enc.flush() is None and len(enc.store) == 0
    assert not [n for n in os.listdir(enc.store.folder) if n.startswith("seg-")]


def test_a_different_model_revision_uses_a_different_folder(tmp_path):
    a, b = _persistent(tmp_path, "r1"), _persistent(tmp_path, "r2")
    assert a.store.folder != b.store.folder
    assert all(os.path.isfile(os.path.join(f, "signature.json")) for f in (a.store.folder, b.store.folder))


def test_an_unreadable_segment_is_skipped_and_its_vectors_computed_again(tmp_path):
    cold = _persistent(tmp_path)
    with vs.persisting():
        E5Retriever(DOCS, encoder=cold, score_mode="passage")
    cold.flush()
    with open(os.path.join(cold.store.folder, "seg-broken.npz"), "wb") as f:
        f.write(b"not a numpy file")
    warm = _persistent(tmp_path)
    assert "unreadable segment seg-broken.npz" in warm.store_error
    got, expected = _vectors(E5Retriever(DOCS, encoder=warm, score_mode="passage")), _fresh()
    assert all(np.array_equal(got[t], expected[t]) for t in expected)


def test_an_unknown_model_revision_falls_back_to_memory(tmp_path):
    enc = vs.PersistentE5Encoder(str(tmp_path), model_name="nobody/no-such-model", model=BatchModel())
    assert enc.store is None and "revision" in enc.store_error
    got, expected = _vectors(E5Retriever(DOCS, encoder=enc, score_mode="passage")), _fresh()
    assert all(np.array_equal(got[t], expected[t]) for t in expected)
    assert enc.flush() is None


def test_a_failed_write_keeps_working_and_reports_it(tmp_path, monkeypatch):
    enc = _persistent(tmp_path)

    def refuse(path, data):
        raise PermissionError("locked")
    monkeypatch.setattr(vs.SegmentStore, "_atomic_write", staticmethod(refuse))
    with vs.persisting():
        E5Retriever(DOCS, encoder=enc, score_mode="passage")
    assert "write failed" in enc.flush() and "write failed" in enc.store_error
    got, expected = _vectors(E5Retriever(DOCS, encoder=enc, score_mode="passage")), _fresh()
    assert all(np.array_equal(got[t], expected[t]) for t in expected)


def test_concurrent_queries_and_rebuilds_are_safe_and_give_the_same_vectors(tmp_path):
    enc, errors = _persistent(tmp_path), []
    expected_q = E5Encoder(model=BatchModel()).encode_queries(["are classes online?"])[0]

    def rebuild():
        try:
            with vs.persisting():
                for _ in range(20):
                    E5Retriever(DOCS, encoder=enc, score_mode="passage")
        except Exception as e:
            errors.append(e)

    def ask():
        try:
            for _ in range(200):
                assert np.array_equal(enc.encode_queries(["are classes online?"])[0], expected_q)
                enc.encode_passages([f"passage {threading.get_ident()} {_}"])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=rebuild)] + [threading.Thread(target=ask) for _ in range(3)]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert errors == []
    got, expected = _vectors(E5Retriever(DOCS, encoder=enc, score_mode="passage")), _fresh()
    assert all(np.array_equal(got[t], expected[t]) for t in expected)


def test_the_signature_records_what_changes_the_vectors(tmp_path):
    sig = vs.model_signature(E5Encoder(model=BatchModel()), revision="r1")
    for field in ("model", "revision", "prefix", "normalize_embeddings", "max_seq_length", "modules", "precision",
                  "truncate_dim", "device", "versions"):
        assert field in sig
    assert vs.model_signature(E5Encoder(model_name="nobody/no-such-model", model=BatchModel())) is None
