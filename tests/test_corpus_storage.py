"""
Tests for src/ingestion/corpus_storage.py (Milestone 8.3).

Covers: save/load round-trip, multiple records, empty corpus, metadata
preservation, malformed/invalid stored data (loud failure, not silent
skipping), duplicate handling per the chosen conservative design, and
deterministic serialization.
"""

import json
import os
import tempfile

import pytest

from src.ingestion.evidence import normalize
from src.ingestion.corpus_storage import (
    save_corpus, load_corpus, append_to_corpus, load_corpus_for_retrieval,
    CorpusStorageError,
)


def make_evidence(text="Some evidence.", source_type="official_site", **kwargs):
    return normalize(text=text, source_type=source_type, **kwargs)


def temp_path():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(path)  # start with a genuinely nonexistent path for most tests
    return path


# --------------------------------------------------------------------
# Save/load round-trip
# --------------------------------------------------------------------

def test_save_then_load_round_trip_single_record():
    path = temp_path()
    original = make_evidence(text="Classes are suspended tomorrow.", source_url="https://x.edu/1")
    save_corpus([original], path)

    loaded = load_corpus(path)

    assert len(loaded) == 1
    assert loaded[0] == original  # dataclass equality -- every field must match exactly
    os.remove(path)


def test_save_then_load_round_trip_multiple_records():
    path = temp_path()
    records = [
        make_evidence(text="First evidence.", source_url="https://x.edu/1"),
        make_evidence(text="Second evidence.", source_url="https://x.edu/2"),
        make_evidence(text="Third evidence.", source_url="https://x.edu/3"),
    ]
    save_corpus(records, path)

    loaded = load_corpus(path)

    assert len(loaded) == 3
    assert [r.document_id for r in loaded] == [r.document_id for r in records]
    os.remove(path)


# --------------------------------------------------------------------
# Empty corpus
# --------------------------------------------------------------------

def test_save_empty_list_produces_empty_file():
    path = temp_path()
    save_corpus([], path)

    assert os.path.exists(path)
    assert os.path.getsize(path) == 0
    os.remove(path)


def test_load_empty_file_returns_empty_list():
    path = temp_path()
    save_corpus([], path)

    loaded = load_corpus(path)

    assert loaded == []
    os.remove(path)


def test_load_nonexistent_file_raises_corpus_storage_error():
    with pytest.raises(CorpusStorageError, match="does not exist"):
        load_corpus("/definitely/does/not/exist.jsonl")


# --------------------------------------------------------------------
# Metadata preservation
# --------------------------------------------------------------------

def test_all_metadata_fields_survive_round_trip():
    path = temp_path()
    original = make_evidence(
        text="Classes are suspended tomorrow.",
        source_type="official_site",
        source_url="https://x.edu/notice/1",
        published_date="2026-01-10",
        fetched_at="2026-01-15T10:00:00+00:00",
        credibility_tier="official",
        source_identifier="notice-1",
    )
    save_corpus([original], path)
    loaded = load_corpus(path)[0]

    assert loaded.source_url == "https://x.edu/notice/1"
    assert loaded.published_date == "2026-01-10"
    assert loaded.fetched_at == "2026-01-15T10:00:00+00:00"
    assert loaded.credibility_tier == "official"
    assert loaded.source_identifier == "notice-1"
    os.remove(path)


def test_missing_optional_metadata_survives_as_empty_strings():
    path = temp_path()
    original = make_evidence(text="Minimal evidence.", source_type="manual_entry")
    save_corpus([original], path)
    loaded = load_corpus(path)[0]

    assert loaded.source_url == ""
    assert loaded.published_date == ""
    os.remove(path)


# --------------------------------------------------------------------
# Malformed / invalid stored data -- must fail loudly, not silently
# --------------------------------------------------------------------

def test_load_rejects_invalid_json_line():
    path = temp_path()
    with open(path, "w") as f:
        f.write("{not valid json\n")

    with pytest.raises(CorpusStorageError, match="invalid JSON"):
        load_corpus(path)
    os.remove(path)


def test_load_rejects_record_missing_required_field():
    path = temp_path()
    with open(path, "w") as f:
        f.write(json.dumps({"document_id": "x", "source_type": "official_site"}) + "\n")  # missing "text"

    with pytest.raises(CorpusStorageError, match="missing or empty required field 'text'"):
        load_corpus(path)
    os.remove(path)


def test_load_rejects_invalid_source_type():
    path = temp_path()
    with open(path, "w") as f:
        f.write(json.dumps({
            "document_id": "x", "text": "Some text.", "source_type": "not_a_real_type"
        }) + "\n")

    with pytest.raises(CorpusStorageError, match="not one of"):
        load_corpus(path)
    os.remove(path)


def test_load_rejects_invalid_credibility_tier():
    path = temp_path()
    with open(path, "w") as f:
        f.write(json.dumps({
            "document_id": "x", "text": "Some text.", "source_type": "official_site",
            "credibility_tier": "extremely_trustworthy",
        }) + "\n")

    with pytest.raises(CorpusStorageError, match="not one of"):
        load_corpus(path)
    os.remove(path)


def test_one_bad_line_fails_the_whole_load_not_silently_skipped():
    """Per the explicit 'avoid accidental silent corruption' requirement
    -- a corpus with 2 good records and 1 bad one must NOT silently
    return just the 2 good ones."""
    path = temp_path()
    good = make_evidence(text="Good record.")
    with open(path, "w") as f:
        f.write(json.dumps({"document_id": good.document_id, "text": good.text,
                             "source_type": good.source_type}) + "\n")
        f.write("{this is not json\n")

    with pytest.raises(CorpusStorageError):
        load_corpus(path)
    os.remove(path)


def test_blank_lines_are_tolerated_not_treated_as_corruption():
    path = temp_path()
    good = make_evidence(text="Good record.")
    with open(path, "w") as f:
        f.write(json.dumps({"document_id": good.document_id, "text": good.text,
                             "source_type": good.source_type}) + "\n")
        f.write("\n")  # a blank line, e.g. trailing newline
        f.write("   \n")

    loaded = load_corpus(path)
    assert len(loaded) == 1
    os.remove(path)


# --------------------------------------------------------------------
# Duplicate handling (append_to_corpus)
# --------------------------------------------------------------------

def test_append_to_new_file_creates_it():
    path = temp_path()
    record = make_evidence(text="First record.", source_url="https://x.edu/1")

    was_appended = append_to_corpus(record, path)

    assert was_appended is True
    assert len(load_corpus(path)) == 1
    os.remove(path)


def test_append_duplicate_document_id_is_skipped_by_default():
    path = temp_path()
    record = make_evidence(text="Same source.", source_url="https://x.edu/1")
    append_to_corpus(record, path)

    # Re-ingesting the SAME source produces the SAME document_id
    # (deterministic derivation in evidence.py) -- default behavior
    # must skip it, not duplicate it.
    was_appended_again = append_to_corpus(record, path)

    assert was_appended_again is False
    assert len(load_corpus(path)) == 1  # still just one record
    os.remove(path)


def test_append_allow_duplicates_true_appends_anyway():
    path = temp_path()
    record = make_evidence(text="Same source.", source_url="https://x.edu/1")
    append_to_corpus(record, path)

    was_appended_again = append_to_corpus(record, path, allow_duplicates=True)

    assert was_appended_again is True
    assert len(load_corpus(path)) == 2  # explicitly allowed to duplicate
    os.remove(path)


def test_append_different_sources_both_persist():
    path = temp_path()
    record_a = make_evidence(text="Source A.", source_url="https://x.edu/a")
    record_b = make_evidence(text="Source B.", source_url="https://x.edu/b")

    append_to_corpus(record_a, path)
    append_to_corpus(record_b, path)

    assert len(load_corpus(path)) == 2
    os.remove(path)


# --------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------

def test_save_is_deterministic_given_same_input():
    path_a = temp_path()
    path_b = temp_path()
    records = [
        make_evidence(text="Evidence one.", source_url="https://x.edu/1"),
        make_evidence(text="Evidence two.", source_url="https://x.edu/2"),
    ]

    save_corpus(records, path_a)
    save_corpus(records, path_b)

    with open(path_a) as fa, open(path_b) as fb:
        assert fa.read() == fb.read()  # byte-identical output
    os.remove(path_a)
    os.remove(path_b)


# --------------------------------------------------------------------
# Integration with retrieval
# --------------------------------------------------------------------

def test_load_corpus_for_retrieval_produces_retriever_ready_dicts():
    path = temp_path()
    original = make_evidence(text="Library open 24/7 during finals.", source_url="https://x.edu/1")
    save_corpus([original], path)

    documents = load_corpus_for_retrieval(path)

    assert len(documents) == 1
    assert set(documents[0].keys()) == {"document_id", "text", "source_type", "source_url"}
    os.remove(path)


def test_stored_corpus_works_with_a_real_retriever():
    """End-to-end: save a record, load it back, feed it to a REAL
    Retriever, confirm it's searchable -- proving the full
    ingestion -> storage -> retrieval chain, not just storage alone."""
    from src.retrieval.retriever import Retriever

    path = temp_path()
    evidence = make_evidence(
        text="Official notice: the library will be open 24/7 during finals week.",
        source_url="https://x.edu/library-hours",
    )
    save_corpus([evidence], path)

    documents = load_corpus_for_retrieval(path)
    retriever = Retriever(documents)
    results = retriever.retrieve("Is the library open late during finals?", top_k=1)

    assert len(results) == 1
    assert results[0].document_id == evidence.document_id
    os.remove(path)


# --------------------------------------------------------------------
# Structural independence
# --------------------------------------------------------------------

def test_corpus_storage_does_not_import_retrieval_internals():
    import src.ingestion.corpus_storage as module
    import inspect

    import_lines = [
        line for line in inspect.getsource(module).splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    source_of_imports = "\n".join(import_lines)
    assert "TfidfVectorizer" not in source_of_imports
    assert "cosine_similarity" not in source_of_imports
    assert "src.classification" not in source_of_imports
    assert "src.verification" not in source_of_imports
