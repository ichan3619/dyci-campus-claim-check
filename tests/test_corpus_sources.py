"""
M12.3: CAMPUS_VERIFY_CORPUS accepts several sources ('dev;<file>;...'). The combined pool must be the pool the M12
experiments use (scripts/m12_retrieval/eval_retrieval.py), and the 15-document pool recorded in results/m12_3 must stay
reproducible from the repository. Synthetic files for the error cases; no network.
"""
import hashlib
import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest

from src.api.corpus_config import (CORPUS_ENV_VAR, DEV_CORPUS_LABEL, CorpusSource, load_configured_corpus,
                                   load_configured_sources)
from src.ingestion.corpus_storage import append_to_corpus
from tests.test_m11_2_corpus import rec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(ROOT, "data", "dev", "sample_claims.csv")
MOCK = "data/corpus/mock_facebook_corpus.jsonl"
EVALUATED_POOL_SHA256 = "076e179e9d50954c3014f3aa423d859311f032695826b134f5bdd5d28b4dde94"   # results/m12_3 configs
EVALUATED_POOL_SIZE = 15


def load(value):
    with patch.dict(os.environ, {CORPUS_ENV_VAR: value}):
        return load_configured_sources(ROOT, DEV)


def eval_pool():
    sys.path.insert(0, os.path.join(ROOT, "scripts", "m12_retrieval"))
    import eval_retrieval
    return eval_retrieval.load_corpora(ROOT)["combined"]


# ---- the evaluated pool ---------------------------------------------------------------------------------------------
def test_combined_pool_is_the_pool_the_m12_experiments_use():
    docs, sources = load(f"dev;{MOCK}")
    expected = eval_pool()
    assert [(d["text"], d["source_type"], d["source_url"]) for d in docs] == \
           [(d["text"], d["source_type"], d["source_url"]) for d in expected]
    # only the Facebook IDs differ: the experiments alias them (FB_MIDTERM, ...), the API keeps the stored IDs
    renamed = [(a["document_id"], b["document_id"]) for a, b in zip(docs, expected) if a["document_id"] != b["document_id"]]
    assert all(a.startswith("ingested:") for a, _ in renamed)
    assert [s.label for s in sources] == [DEV_CORPUS_LABEL, "mock_facebook_corpus.jsonl"]
    assert sum(s.documents for s in sources) == len(docs)


def test_the_evaluated_15_document_pool_is_still_reproducible():
    """New posts may be appended to the corpus file; the first 15 documents must stay the evaluated ones."""
    pool = eval_pool()[:EVALUATED_POOL_SIZE]
    digest = hashlib.sha256(json.dumps(pool, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert digest == EVALUATED_POOL_SHA256


# ---- parsing ---------------------------------------------------------------------------------------------------------
def test_default_is_unchanged():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(CORPUS_ENV_VAR, None)
        docs, sources = load_configured_sources(ROOT, DEV)
        _, label = load_configured_corpus(ROOT, DEV)
    assert sources == [CorpusSource(DEV_CORPUS_LABEL, len(docs))] and label == DEV_CORPUS_LABEL


def test_single_file_behaves_as_before():
    docs, sources = load(MOCK)
    assert [s.label for s in sources] == ["mock_facebook_corpus.jsonl"] and all(d["source_type"] == "facebook" for d in docs)


def test_dev_keyword_alone_and_trailing_separator():
    a, _ = load("dev")
    b, _ = load(" DEV ; ")
    assert a == b and len(a) > 0


def test_label_joins_sources_in_order():
    with patch.dict(os.environ, {CORPUS_ENV_VAR: f"{MOCK};dev"}):
        _, label = load_configured_corpus(ROOT, DEV)
    assert label == "mock_facebook_corpus.jsonl+dev_dataset"


@pytest.mark.parametrize("value", [";", " ; ; "])
def test_a_value_that_names_no_source_fails(value):
    with pytest.raises(RuntimeError, match="names no source"):
        load(value)


@pytest.mark.parametrize("value", ["dev;DEV", f"{MOCK};{MOCK}", f"{MOCK};./{MOCK}"])
def test_a_source_listed_twice_fails(value):
    with pytest.raises(RuntimeError, match="more than once"):
        load(value)


def test_a_missing_file_in_the_list_fails_and_never_falls_back():
    with pytest.raises(RuntimeError, match="no such file"):
        load("dev;no/such/file.jsonl")


def test_the_same_document_id_in_two_sources_fails():
    with tempfile.TemporaryDirectory() as t:
        a, b = os.path.join(t, "a.jsonl"), os.path.join(t, "b.jsonl")
        append_to_corpus(rec("Library closed October 12.", "same-post"), a)
        append_to_corpus(rec("Library closed October 12.", "same-post"), b)
        with pytest.raises(RuntimeError, match="appears in both"):
            load(f"{a};{b}")


# ---- API -------------------------------------------------------------------------------------------------------------
def test_development_pool_is_still_the_dev_notices_plus_the_facebook_posts():
    from src.api.pipeline import DEVELOPMENT_CORPUS_SPEC
    assert DEVELOPMENT_CORPUS_SPEC == f"dev;{MOCK}"                   # what the M12 tools and experiments use


def test_api_serves_real_announcements_only_by_default():
    from src.api.corpus_config import load_sources
    from src.api.pipeline import DEFAULT_CORPUS_SPEC, DEV_DATASET_PATH, PROJECT_ROOT, SERVED_CORPUS_SPEC, Settings
    assert DEFAULT_CORPUS_SPEC == SERVED_CORPUS_SPEC == Settings().corpus
    assert "dev" not in DEFAULT_CORPUS_SPEC.split(";")
    documents, sources = load_sources(DEFAULT_CORPUS_SPEC, PROJECT_ROOT, DEV_DATASET_PATH)
    assert [s.label for s in sources] == ["mock_facebook_corpus.jsonl", "independent_announcements.jsonl",
                                          "dev2_announcements.jsonl"]
    assert documents and all(d["document_id"].startswith("ingested:") for d in documents)   # no fictional evt_* notice
    assert all(d["source_type"] in ("facebook", "manual_entry") for d in documents)


def test_health_lists_every_source(api_settings):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    api_main = api_settings(corpus=f"dev;{MOCK}")
    with TestClient(api_main.app) as c:
        h = c.get("/health").json()
    assert h["corpus_source"] == "dev_dataset+mock_facebook_corpus.jsonl"
    assert [s["label"] for s in h["corpus_sources"]] == ["dev_dataset", "mock_facebook_corpus.jsonl"]
    assert sum(s["documents"] for s in h["corpus_sources"]) == h["corpus_size"]
