"""
Chooses the evidence corpus the API runs on (M11.2; several sources since M12.3). Kept separate from
main.py, with no FastAPI imports, so it can be unit-tested on its own.

DEFAULT (nothing configured): real DYCI announcements only -- src/api/pipeline.py SERVED_CORPUS_SPEC, the posts
mirrored on the project's test Page plus the announcements entered by hand. The fictional development notices are
not served by default.

OTHER POOLS: set CAMPUS_VERIFY_CORPUS to one source, or several separated by ';'.
A source is a corpus JSONL file (for example the one written by
scripts/ingest_facebook_page.py) or the keyword `dev` for the development
dataset. PowerShell:

    # the test-Page posts only
    $env:CAMPUS_VERIFY_CORPUS = "data/corpus/mock_facebook_corpus.jsonl"
    # the pool the M12 experiments were run on (and the 0.833 e5 gate was calibrated on)
    $env:CAMPUS_VERIFY_CORPUS = "dev;data/corpus/mock_facebook_corpus.jsonl"
    uvicorn src.api.main:app --reload

Startup FAILS LOUDLY if a listed file is missing or empty, if a source is listed
twice, or if two sources contain the same document_id. It never silently falls
back to the development dataset or drops a document, because that would let you
believe claims are being checked against evidence they are not.

The combined pool `dev;<mock facebook corpus>` contains the same 15 documents
(same text, source type and URL, in the same order) that
scripts/m12_retrieval/eval_retrieval.py builds for the M12 experiments;
tests/test_corpus_sources.py checks this. Only the IDs of the two Facebook
posts differ: the experiments rename them FB_MIDTERM / FB_SHORTENED, the API
keeps their stored IDs. Retrieval scores depend on the text only.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

from src.data.loader import load_dataset
from src.retrieval.retriever import build_corpus
from src.ingestion.corpus_storage import load_corpus_for_retrieval

CORPUS_ENV_VAR = "CAMPUS_VERIFY_CORPUS"
DEV_CORPUS_LABEL = "dev_dataset"
DEV_KEYWORD = "dev"
SOURCE_SEPARATOR = ";"


@dataclass(frozen=True)
class CorpusSource:
    label: str        # 'dev_dataset' or the file name
    documents: int


def _load_dev(dev_dataset_path: str) -> List[dict]:
    load_result = load_dataset(dev_dataset_path)
    if load_result.errors:
        raise RuntimeError(f"Dev dataset failed to load cleanly at API startup: {load_result.errors}")
    return build_corpus(load_result.records)


def _load_file(project_root: str, configured: str) -> Tuple[List[dict], str]:
    path = configured if os.path.isabs(configured) else os.path.join(project_root, configured)
    if not os.path.isfile(path):
        raise RuntimeError(
            f"{CORPUS_ENV_VAR} lists {configured!r} but no such file exists. "
            f"Run scripts/ingest_facebook_page.py first, or remove it from {CORPUS_ENV_VAR} "
            f"(use '{DEV_KEYWORD}' for the development dataset)."
        )
    documents = load_corpus_for_retrieval(path)
    if not documents:
        raise RuntimeError(f"Corpus file {configured!r} contains no documents; refusing to start on an empty evidence pool.")
    return documents, os.path.basename(path)


def _entries(spec: str, project_root: str) -> List[str]:
    entries = [e.strip() for e in spec.split(SOURCE_SEPARATOR) if e.strip()]
    if not entries:
        raise RuntimeError(f"{CORPUS_ENV_VAR} is set to {spec!r} but names no source.")
    keys = [DEV_KEYWORD if e.lower() == DEV_KEYWORD else os.path.normcase(os.path.abspath(os.path.join(project_root, e)))
            for e in entries]
    repeated = sorted({e for e, k in zip(entries, keys) if keys.count(k) > 1})
    if repeated:
        raise RuntimeError(f"{CORPUS_ENV_VAR} lists the same source more than once: {repeated}")
    return entries


def source_paths(spec: str, project_root: str, dev_dataset_path: str) -> List[str]:
    """Absolute paths of every file a corpus spec reads (the dev keyword -> the dev CSV). Used to detect changes."""
    if not spec.strip():
        return [os.path.abspath(dev_dataset_path)]
    return [os.path.abspath(dev_dataset_path) if e.lower() == DEV_KEYWORD else os.path.abspath(os.path.join(project_root, e))
            for e in _entries(spec, project_root)]


def load_configured_sources(project_root: str, dev_dataset_path: str) -> Tuple[List[dict], List[CorpusSource]]:
    """Return (documents for the retriever, one CorpusSource per configured source), from CAMPUS_VERIFY_CORPUS."""
    return load_sources(os.environ.get(CORPUS_ENV_VAR, ""), project_root, dev_dataset_path)


def load_sources(spec: str, project_root: str, dev_dataset_path: str) -> Tuple[List[dict], List[CorpusSource]]:
    """Return (documents for the retriever, one CorpusSource per source in `spec`, in the order given)."""
    if not spec.strip():
        docs = _load_dev(dev_dataset_path)
        return docs, [CorpusSource(DEV_CORPUS_LABEL, len(docs))]
    entries = _entries(spec, project_root)

    documents: List[dict] = []
    sources: List[CorpusSource] = []
    origin = {}                                   # document_id -> label of the source it came from
    for entry in entries:
        if entry.lower() == DEV_KEYWORD:
            docs, label = _load_dev(dev_dataset_path), DEV_CORPUS_LABEL
        else:
            docs, label = _load_file(project_root, entry)
        for d in docs:
            if d["document_id"] in origin:
                raise RuntimeError(
                    f"document_id {d['document_id']!r} appears in both {origin[d['document_id']]!r} and {label!r}; "
                    "refusing to start rather than let one silently replace the other.")
            origin[d["document_id"]] = label
        documents.extend(docs)
        sources.append(CorpusSource(label, len(docs)))
    return documents, sources


def load_configured_corpus(project_root: str, dev_dataset_path: str) -> Tuple[List[dict], str]:
    """Return (documents for Retriever, a short label naming the source(s), joined with '+')."""
    documents, sources = load_configured_sources(project_root, dev_dataset_path)
    return documents, "+".join(s.label for s in sources)
