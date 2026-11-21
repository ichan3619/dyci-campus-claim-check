"""
Source map (M11.2): record where each MIRRORED post originally came from.

When posts are re-posted by hand to a mock page, the fetched record carries
the MIRROR's link and date. A CSV maps each mock post to its original:

    mock_post,original_url,original_date
    123456_789012,https://www.facebook.com/<original page>/posts/111,2026-09-15

`mock_post` matches a fetched record's post ID (source_identifier) or its
permalink (source_url). `original_url` / `original_date` are optional per
row; a blank cell leaves that field as fetched. The record's
`source_identifier` (the mock post's ID) is never changed, so the mirror
copy stays traceable, and `document_id` does not change either.
"""

import csv
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple

from src.ingestion.evidence import NormalizedEvidence

REQUIRED_COLUMNS = ("mock_post", "original_url", "original_date")


def load_source_map(path: str) -> Dict[str, Tuple[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"source map {path!r} is missing column(s): {missing}")
        mapping: Dict[str, Tuple[str, str]] = {}
        for line_no, row in enumerate(reader, start=2):
            key = (row["mock_post"] or "").strip()
            if not key:
                raise ValueError(f"source map {path!r} line {line_no}: empty mock_post")
            if key in mapping:
                raise ValueError(f"source map {path!r} line {line_no}: duplicate mock_post {key!r}")
            mapping[key] = ((row["original_url"] or "").strip(), (row["original_date"] or "").strip())
    return mapping


def apply_source_map(
    records: Iterable[NormalizedEvidence], mapping: Dict[str, Tuple[str, str]]
) -> Tuple[List[NormalizedEvidence], List[NormalizedEvidence]]:
    """Return (records with originals applied, records that had no map entry)."""
    out: List[NormalizedEvidence] = []
    unmapped: List[NormalizedEvidence] = []
    for r in records:
        entry = mapping.get(r.source_identifier) or mapping.get(r.source_url)
        if entry is None:
            out.append(r)
            unmapped.append(r)
            continue
        url, date = entry
        out.append(replace(r, source_url=url or r.source_url, published_date=date or r.published_date))
    return out, unmapped


def annotate_origin(
    records: Iterable[NormalizedEvidence], mapping: Dict[str, Tuple[str, str]]
) -> Tuple[List[NormalizedEvidence], List[NormalizedEvidence]]:
    """
    M8.5-C: record each mirrored post's ORIGINAL url/date WITHOUT losing the
    mirror's own.

    Unlike apply_source_map() above (kept unchanged; it OVERWRITES
    source_url / published_date with the originals), this puts the
    original into the separate `origin_url` / `origin_date` fields and
    leaves source_url (the mirror Page's permalink) and published_date
    (the mirror post's time) exactly as fetched. That keeps both facts on
    the record: where the content originally came from, and where this
    system actually acquired it.

    Matching is the same as apply_source_map(): the map's `mock_post` key
    may be the record's source_identifier (post ID) or its source_url. A
    blank cell in the map leaves that origin field empty. Returns
    (records with origins attached, records that had no map entry).
    """
    out: List[NormalizedEvidence] = []
    unmapped: List[NormalizedEvidence] = []
    for r in records:
        entry = mapping.get(r.source_identifier) or mapping.get(r.source_url)
        if entry is None:
            out.append(r)
            unmapped.append(r)
            continue
        url, date = entry
        out.append(replace(r, origin_url=url or r.origin_url, origin_date=date or r.origin_date))
    return out, unmapped
