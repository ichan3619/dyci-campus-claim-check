"""
Normalized evidence representation (Milestone 8).

Responsibility of this module, and ONLY this module: define the ONE
common shape that raw content from ANY source adapter (local fixture
today; website fetch, Facebook Graph API later) gets converted into,
and the validation/normalization logic that produces it.

============================================================================
WHY THIS IS SEPARATE FROM RETRIEVAL AND FROM THE CLASSIFIER TRAINING CSV
============================================================================
This is NOT the same thing as a row in data/dev/sample_claims.csv. That
CSV format is (claim, evidence, label, ...) -- a LABELED TRAINING
EXAMPLE for the classifier, built by a human per the manual workflow in
docs/data_collection_plan.md. A NormalizedEvidence here has no claim and
no label at all -- it is just a piece of campus evidence text with
provenance metadata, on its way into the RETRIEVAL corpus (see
src/retrieval/retriever.py's Retriever, which is what actually consumes
these documents, NOT src/retrieval/retriever.py's build_corpus(), which
is specifically for turning labeled CSV rows into a corpus).

Put simply, there are now two independent paths into the retrieval
corpus:
    1. Labeled CSV rows -> build_corpus() -> Retriever  (existing, M1-M7)
    2. Raw source content -> ingest+normalize -> Retriever  (NEW, M8)
Both paths produce the SAME kind of dict shape
({"document_id", "text", "source_type", "source_url"}) that Retriever's
constructor expects -- see to_retriever_document() below -- so
Retriever itself needed ZERO changes to accept ingested evidence.

============================================================================
REUSE, NOT DUPLICATION
============================================================================
Validation reuses the EXACT SAME enums already defined in
src/data/schema.py (VALID_SOURCE_TYPES, VALID_CREDIBILITY_TIERS) rather
than inventing a parallel set of allowed values -- an ingested document
tagged "official_site" must mean the same thing here as it already means
in the classifier training CSV. Whitespace normalization reuses
src/data/preprocessing.normalize_whitespace() rather than reimplementing
text cleaning.
"""

import hashlib
from dataclasses import dataclass
from typing import Optional

from src.data.schema import VALID_SOURCE_TYPES, VALID_CREDIBILITY_TIERS
from src.data.preprocessing import normalize_whitespace


@dataclass
class NormalizedEvidence:
    """
    The common representation every source adapter's raw output gets
    converted into, regardless of where it came from.

    Fields:
        document_id: a stable identifier for this document. Derived
            (see normalize() below) from an adapter-provided identifier
            when available, otherwise from a hash of the source URL/text
            -- stable specifically so re-ingesting the SAME page twice
            produces the SAME document_id, rather than creating
            duplicate corpus entries every time ingestion re-runs.
        text: the cleaned evidence text (whitespace-normalized only --
            same conservative stance as src/data/preprocessing.py:
            dates, numbers, and names are NEVER stripped, since they
            carry meaning this project depends on).
        source_type: one of src.data.schema.VALID_SOURCE_TYPES. Required
            -- an ingested document with an unrecognized or missing
            source type is rejected (see normalize()), not silently
            defaulted, because retrieval/classification-adjacent future
            logic depends on this being trustworthy.
        source_url: the URL content was fetched from, if applicable. May
            be empty (e.g. manually transcribed content has no URL).
        published_date: when the SOURCE content was published/posted,
            if known. May be empty -- most real sources won't reliably
            expose this. Distinct from fetched_at below.
        fetched_at: an ISO-8601 timestamp of when THIS ingestion pipeline
            retrieved the content -- NOT when it was published. This is
            NEW metadata beyond what data/schema.py's CSV format
            captures (that schema only has a publication "date" column,
            with no fetch-time concept, because until now all data entry
            was manual with no distinct "fetch" step). Kept here rather
            than added to schema.py in this milestone -- see the M8
            architectural-concerns note in the final report.
        credibility_tier: one of src.data.schema.VALID_CREDIBILITY_TIERS,
            if the adapter/caller supplied one. Optional, exactly as it
            already is in the CSV schema (Milestone 5) -- not every
            source is credibility-reviewed at ingestion time.
        source_identifier: an adapter-specific raw identifier (e.g. a
            filename, or a future API object ID), kept for traceability/
            debugging even after document_id has been derived from it.

        M8.5-C provenance fields (all OPTIONAL, default empty, so every
        record and corpus file written before M8.5-C loads unchanged).
        They separate two things that are easy to conflate:

          ACQUISITION SOURCE -- where THIS system obtained the data:
              source_type / source_url / source_identifier / published_date
              / fetched_at. For a mirrored post, source_url is the MIRROR
              Page's permalink and source_identifier is the mirror post's
              Graph API ID.
          CONTENT ORIGIN -- where the announcement originally came from:
              origin_url / origin_date (from the source map, when known).

        origin_url: URL of the original announcement, if known. Never
            overwrites source_url.
        origin_date: date of the original announcement, if known. Never
            overwrites published_date.
        acquisition_note: free-text statement of how this record was
            obtained, e.g. that it came from a MOCK/TEST mirror Page and
            NOT from an official DYCI source. Descriptive only; retrieval,
            classification and verification do not read it.
    """
    document_id: str
    text: str
    source_type: str
    source_url: str = ""
    published_date: str = ""
    fetched_at: str = ""
    credibility_tier: str = ""
    source_identifier: str = ""
    origin_url: str = ""
    origin_date: str = ""
    acquisition_note: str = ""

    def to_retriever_document(self) -> dict:
        """
        Convert to the exact dict shape src.retrieval.retriever.Retriever
        expects: {"document_id", "text", "source_type", "source_url"}.

        This is the ONE integration point between ingestion and
        retrieval. Retrieval itself was NOT modified to accommodate
        ingestion -- ingestion instead produces output already shaped to
        fit retrieval's existing, unmodified input contract.
        """
        return {
            "document_id": self.document_id,
            "text": self.text,
            "source_type": self.source_type,
            "source_url": self.source_url,
        }


def _derive_document_id(source_identifier: str, source_url: str, text: str) -> str:
    """
    Produce a stable document_id.

    Preference order: an explicit adapter-provided source_identifier
    (most meaningful, e.g. a real announcement ID) > the source_url
    (also meaningful and stable) > a hash of the text itself (last
    resort, for content with neither -- e.g. a manually pasted fixture
    with no URL). Hashing (not a random UUID) is deliberate: it makes
    document_id DETERMINISTIC, so re-ingesting identical content twice
    produces the identical ID rather than creating a duplicate corpus
    entry each run.
    """
    if source_identifier:
        return f"ingested:{source_identifier}"
    if source_url:
        return f"ingested:{hashlib.sha256(source_url.encode('utf-8')).hexdigest()[:16]}"
    return f"ingested:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def normalize(*, text: str, source_type: str, source_url: str = "",
              published_date: str = "", fetched_at: str = "",
              credibility_tier: str = "", source_identifier: str = "",
              origin_url: str = "", origin_date: str = "",
              acquisition_note: str = "") -> NormalizedEvidence:
    """
    Validate and normalize raw source content into a NormalizedEvidence.

    Raises ValueError for:
        - empty/whitespace-only text (there is no evidence document here)
        - source_type not in VALID_SOURCE_TYPES (see src/data/schema.py)
        - a non-empty credibility_tier that isn't in VALID_CREDIBILITY_TIERS

    This function is the single place normalization rules live -- every
    source adapter funnels its raw output through this same function
    (see src/ingestion/adapters.py), so a rule change here applies
    uniformly regardless of which adapter produced the content.
    """
    cleaned_text = normalize_whitespace(text)
    if not cleaned_text:
        raise ValueError("cannot normalize empty or whitespace-only source content")

    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(
            f"source_type '{source_type}' is not one of {sorted(VALID_SOURCE_TYPES)} "
            "(see src/data/schema.py VALID_SOURCE_TYPES -- reused here, not redefined)"
        )

    if credibility_tier and credibility_tier not in VALID_CREDIBILITY_TIERS:
        raise ValueError(
            f"credibility_tier '{credibility_tier}' is not one of "
            f"{sorted(VALID_CREDIBILITY_TIERS)}"
        )

    document_id = _derive_document_id(source_identifier, source_url, cleaned_text)

    return NormalizedEvidence(
        document_id=document_id,
        text=cleaned_text,
        source_type=source_type,
        source_url=source_url,
        published_date=published_date,
        fetched_at=fetched_at,
        credibility_tier=credibility_tier,
        source_identifier=source_identifier,
        origin_url=origin_url,
        origin_date=origin_date,
        acquisition_note=acquisition_note,
    )
