"""
Corpus storage (Milestone 8.3).

Pipeline position:

    Source Adapter -> normalize() -> NormalizedEvidence -> CORPUS STORAGE -> Retriever

Responsibility of this module, and ONLY this module: persist
NormalizedEvidence records to disk, and load them back. It does not
fetch (that's src/ingestion/adapters.py), does not normalize/validate
raw content (that's src/ingestion/evidence.py), does not search or rank
anything (that's src/retrieval/retriever.py), and knows nothing about
classification or verification. It receives already-normalized records
and persists them, or reads persisted records back into memory --
nothing more.

============================================================================
FORMAT: JSONL (one JSON object per line), not a single JSON array or CSV
============================================================================
- NOT a single JSON array: appending one record to a JSON array requires
  reading the entire existing array, adding to it, and rewriting the
  whole file -- expensive and riskier (a crash mid-rewrite can corrupt
  the whole file) for something meant to accumulate over repeated
  ingestion runs.
- NOT CSV: the existing data/dev/sample_claims.csv is a different kind
  of artifact -- a fixed-schema, hand-editable LABELED TRAINING table
  (claim, evidence, label). A corpus store is a programmatically
  accumulating collection of standalone evidence documents with no
  claim/label attached at all; forcing it into the same CSV shape would
  conflate two genuinely different representations (see the "distinct
  from the manual CSV workflow" note already established in Milestone
  8.1's methodology section).
- JSONL: each line is independently parseable and independently
  appendable. A malformed line only affects that one record, not the
  whole file, and is easy to detect and report precisely (see
  load_corpus() below, which reports the exact line number of any
  problem rather than silently skipping or failing the whole file).

============================================================================
DUPLICATE HANDLING -- see append_to_corpus() for full reasoning
============================================================================
document_id (derived deterministically in evidence.py from
source_identifier/source_url/text-hash) is the ONLY signal currently
available to detect a duplicate. There is currently no reliable way to
distinguish "this source was re-ingested unchanged" from "this source
was edited since we last ingested it" -- both would produce records
that either share or don't share a document_id in ways this module
cannot yet interpret confidently. The conservative default (skip an
exact document_id collision, don't silently accumulate duplicates) is
used rather than inventing a more elaborate content-diffing policy this
project does not yet have enough information to build correctly.
"""

import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional

from src.ingestion.evidence import NormalizedEvidence
from src.data.schema import VALID_SOURCE_TYPES, VALID_CREDIBILITY_TIERS


class CorpusStorageError(Exception):
    """
    Raised when stored corpus data cannot be safely loaded -- e.g.
    malformed JSON, a missing required field, or an invalid
    source_type/credibility_tier value on a stored record.

    A distinct exception type (rather than reusing ValueError, which
    this project already uses extensively for invalid FUNCTION
    ARGUMENTS elsewhere -- e.g. classifier.py, verifier.py) so a caller
    can specifically catch "the corpus FILE ON DISK is broken" as its
    own category of problem, separate from "I passed a bad argument."
    """
    pass


def save_corpus(records: List[NormalizedEvidence], path: str) -> None:
    """
    Write `records` to `path` as JSONL, overwriting any existing file.

    This always writes the FULL current set of records -- it does not
    append. See append_to_corpus() for the incremental, single-record
    case.

    Deterministic: dataclasses.asdict() preserves NormalizedEvidence's
    declared field order on every call, so given the same records in
    the same order, this function produces byte-identical output every
    time -- useful for diffing corpus files across ingestion runs (e.g.
    under version control).

    An empty `records` list produces a valid, empty (0-byte) file, not
    an error -- a corpus with nothing ingested yet is a normal state.
    """
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")


def save_corpus_atomic(records: List[NormalizedEvidence], path: str) -> None:
    """
    save_corpus(), but the file is replaced in one step: written to a temporary file next to it, flushed to disk, then
    swapped in with os.replace(). A reader (e.g. the API reloading its evidence) sees either the old corpus or the new
    one, never a half-written file. Same bytes as save_corpus() for the same records.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rewrite_corpus(path: str, changes: Dict[str, Optional[NormalizedEvidence]],
                   additions: List[NormalizedEvidence] = ()) -> None:
    """
    Apply edits and removals to a corpus file atomically. `changes` maps document_id -> the replacement record, or
    None to remove it; `additions` are appended. Every other line is kept BYTE-FOR-BYTE (records written by older code
    keep their exact form), so the file only differs where something actually changed.
    """
    raw = open(path, "rb").read()
    eol = b"\r\n" if b"\r\n" in raw else b"\n"          # keep the file's own line ending (append_to_corpus writes
    out = []                                            # CRLF on Windows, LF elsewhere)
    for line in raw.splitlines(keepends=True):
        if not line.strip():
            continue
        doc_id = json.loads(line.decode("utf-8")).get("document_id")
        if doc_id in changes:
            if changes[doc_id] is not None:
                out.append(json.dumps(asdict(changes[doc_id])).encode("utf-8") + eol)
        else:
            out.append(line if line.endswith((b"\n", b"\r")) else line + eol)
    out.extend(json.dumps(asdict(r)).encode("utf-8") + eol for r in additions)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"".join(out))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_corpus(path: str) -> List[NormalizedEvidence]:
    """
    Read a JSONL corpus file back into a list of NormalizedEvidence.

    Raises CorpusStorageError -- loudly, not silently skipping -- if:
        - `path` does not exist
        - any line contains invalid JSON
        - any record is missing a required field (document_id, text, source_type)
        - any record's source_type/credibility_tier isn't a recognized value

    Per the explicit "avoid accidental silent corruption" requirement,
    a malformed line does NOT get quietly skipped while the rest of the
    file loads -- the whole load fails, with the exact line number
    reported, so a corrupted corpus is impossible to miss.

    An EMPTY file (0 bytes, or only blank lines) is valid and returns an
    empty list, matching how Retriever itself already treats an empty
    document list as a normal (if unhelpful) state, not an error.
    """
    if not os.path.exists(path):
        raise CorpusStorageError(f"corpus file does not exist: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue  # tolerate blank lines (e.g. trailing newline) -- not corruption

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise CorpusStorageError(
                    f"line {line_number} in '{path}': invalid JSON ({e})"
                )

            records.append(_dict_to_evidence(data, line_number, path))

    return records


def _dict_to_evidence(data: dict, line_number: int, path: str) -> NormalizedEvidence:
    """Validate and convert one parsed JSON object into a NormalizedEvidence."""
    for required_field in ("document_id", "text", "source_type"):
        if required_field not in data or not str(data[required_field]).strip():
            raise CorpusStorageError(
                f"line {line_number} in '{path}': missing or empty required "
                f"field '{required_field}'"
            )

    if data["source_type"] not in VALID_SOURCE_TYPES:
        raise CorpusStorageError(
            f"line {line_number} in '{path}': source_type '{data['source_type']}' "
            f"is not one of {sorted(VALID_SOURCE_TYPES)}"
        )

    credibility_tier = data.get("credibility_tier", "")
    if credibility_tier and credibility_tier not in VALID_CREDIBILITY_TIERS:
        raise CorpusStorageError(
            f"line {line_number} in '{path}': credibility_tier "
            f"'{credibility_tier}' is not one of {sorted(VALID_CREDIBILITY_TIERS)}"
        )

    return NormalizedEvidence(
        document_id=data["document_id"],
        text=data["text"],
        source_type=data["source_type"],
        source_url=data.get("source_url", ""),
        published_date=data.get("published_date", ""),
        fetched_at=data.get("fetched_at", ""),
        credibility_tier=credibility_tier,
        source_identifier=data.get("source_identifier", ""),
        # M8.5-C optional provenance fields: absent in older corpus files.
        origin_url=data.get("origin_url", ""),
        origin_date=data.get("origin_date", ""),
        acquisition_note=data.get("acquisition_note", ""),
    )


def append_to_corpus(record: NormalizedEvidence, path: str, *, allow_duplicates: bool = False) -> bool:
    """
    Append ONE record to a corpus file (created if it doesn't exist yet).

    Returns True if the record was written, False if it was skipped as
    a duplicate.

    DUPLICATE HANDLING (the design question this milestone specifically
    asked about): a record is treated as a duplicate if its document_id
    already exists in the corpus file. By default (allow_duplicates=
    False), a duplicate is SKIPPED, not appended and not an error -- see
    this module's top docstring for why document_id is the only signal
    available, and why "skip" (not "overwrite", not "error") is the
    conservative default: since document_id is already derived
    deterministically from the source itself (see evidence.py), an
    unchanged re-ingested source naturally produces the same ID, and
    skipping it is exactly the expected behavior for "nothing changed."

    Setting allow_duplicates=True permits appending anyway (producing
    two lines sharing a document_id) -- kept possible, not forbidden,
    because this module cannot yet distinguish "the source content was
    genuinely updated" from "this is an accidental duplicate"; that
    requires a content-versioning policy this project does not have yet
    (see docs/methodology.md's M8 corpus storage limitations).

    Implementation note: this loads the full existing corpus to check
    for a duplicate ID before appending. At thesis scale (hundreds to
    low thousands of documents) this is fast and simple; it is a
    deliberate simplicity tradeoff, not a performance oversight, and
    matches this project's "prefer the simplest appropriate solution"
    principle over building indexing infrastructure prematurely.
    """
    existing_ids = set()
    if os.path.exists(path):
        existing_ids = {r.document_id for r in load_corpus(path)}

    if record.document_id in existing_ids and not allow_duplicates:
        return False

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")
    return True


def load_corpus_for_retrieval(path: str) -> List[dict]:
    """
    Convenience function: load a stored corpus and convert every record
    directly into the dict shape src.retrieval.retriever.Retriever's
    constructor expects.

    This is the ONE function that shows how "corpus storage" and
    "existing retrieval" connect -- everything upstream of this line is
    ingestion/storage, everything from this return value onward is
    unmodified, pre-existing Milestone 2 retrieval code:

        documents = load_corpus_for_retrieval("corpus.jsonl")
        retriever = Retriever(documents)   # Retriever itself: ZERO changes
    """
    records = load_corpus(path)
    return [r.to_retriever_document() for r in records]
