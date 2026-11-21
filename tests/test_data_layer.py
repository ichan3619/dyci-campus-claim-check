"""
Tests for the data layer (schema, validation, loader).

These are the highest-value tests for this milestone: if data loading and
validation can't be trusted, nothing built on top of it (retrieval,
classifiers, evaluation) can be trusted either. We are NOT testing ML
here -- there is no model yet, on purpose.
"""

import csv
import os
import tempfile

from src.data.loader import load_dataset
from src.data.validation import validate_record, find_duplicate_claims
from src.data.preprocessing import normalize_whitespace
from src.data.schema import REQUIRED_COLUMNS


def write_temp_csv(rows, fieldnames):
    """Helper: write rows to a temporary CSV file and return its path.

    csv.DictWriter fills any column missing from a row dict with an
    empty string by default (restval=""), so individual test rows below
    don't need to mention every column -- only the ones relevant to
    what that test is checking.
    """
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# Derived from the schema itself (not hand-typed) so this list can never
# silently drift out of sync with REQUIRED_COLUMNS the way it did when
# Milestone 5 added "credibility_tier" -- that change broke every test
# below until this line was updated to match.
FIELDNAMES = sorted(REQUIRED_COLUMNS)


def test_valid_dataset_loads():
    rows = [
        {
            "claim": "Classes are suspended tomorrow.",
            "evidence": "Official notice: classes are suspended tomorrow.",
            "label": "SUPPORTED",
            "source_type": "synthetic",
            "source_url": "",
            "date": "",
            "group_id": "evt_test_1",
        }
    ]
    path = write_temp_csv(rows, FIELDNAMES)
    result = load_dataset(path)

    assert len(result.records) == 1
    assert len(result.errors) == 0
    os.remove(path)


def test_invalid_label_is_rejected():
    rows = [
        {
            "claim": "Classes are suspended tomorrow.",
            "evidence": "Some evidence.",
            "label": "MAYBE_TRUE",  # not a valid label
            "source_type": "synthetic",
            "source_url": "",
            "date": "",
            "group_id": "evt_test_2",
        }
    ]
    path = write_temp_csv(rows, FIELDNAMES)
    result = load_dataset(path)

    assert len(result.records) == 0
    assert len(result.errors) == 1
    assert "not one of" in result.errors[0]
    os.remove(path)


def test_missing_required_field_is_detected():
    rows = [
        {
            "claim": "Classes are suspended tomorrow.",
            "evidence": "Some evidence.",
            "label": "SUPPORTED",
            "source_type": "",  # required field, left empty
            "source_url": "",
            "date": "",
            "group_id": "evt_test_3",
        }
    ]
    path = write_temp_csv(rows, FIELDNAMES)
    result = load_dataset(path)

    assert len(result.records) == 0
    assert any("source_type" in e for e in result.errors)
    os.remove(path)


def test_empty_claim_is_rejected():
    rows = [
        {
            "claim": "   ",  # whitespace only -> should be treated as empty
            "evidence": "Some evidence.",
            "label": "SUPPORTED",
            "source_type": "synthetic",
            "source_url": "",
            "date": "",
            "group_id": "evt_test_4",
        }
    ]
    path = write_temp_csv(rows, FIELDNAMES)
    result = load_dataset(path)

    assert len(result.records) == 0
    assert any("claim" in e for e in result.errors)
    os.remove(path)


def test_insufficient_evidence_allows_empty_evidence():
    rows = [
        {
            "claim": "The university will have a surprise holiday.",
            "evidence": "",  # empty is VALID for this label
            "label": "INSUFFICIENT_EVIDENCE",
            "source_type": "synthetic",
            "source_url": "",
            "date": "",
            "group_id": "evt_test_5",
        }
    ]
    path = write_temp_csv(rows, FIELDNAMES)
    result = load_dataset(path)

    assert len(result.records) == 1
    assert len(result.errors) == 0
    os.remove(path)


def test_supported_requires_non_empty_evidence():
    rows = [
        {
            "claim": "Enrollment starts August 10.",
            "evidence": "",  # invalid: SUPPORTED needs evidence
            "label": "SUPPORTED",
            "source_type": "synthetic",
            "source_url": "",
            "date": "",
            "group_id": "evt_test_6",
        }
    ]
    path = write_temp_csv(rows, FIELDNAMES)
    result = load_dataset(path)

    assert len(result.records) == 0
    assert any("requires non-empty evidence" in e for e in result.errors)
    os.remove(path)


def test_duplicate_detection_works():
    row = {
        "claim": "Classes are suspended tomorrow.",
        "evidence": "Official notice: classes are suspended tomorrow.",
        "label": "SUPPORTED",
        "source_type": "synthetic",
        "source_url": "",
        "date": "",
        "group_id": "evt_test_7",
    }
    rows = [row, dict(row)]  # exact duplicate row
    path = write_temp_csv(rows, FIELDNAMES)
    result = load_dataset(path)

    assert len(result.records) == 2  # duplicates are flagged, not removed
    assert len(result.duplicate_warnings) == 1
    os.remove(path)


def test_normalize_whitespace_collapses_and_strips():
    messy = "  Classes  are\tsuspended   tomorrow.  "
    assert normalize_whitespace(messy) == "Classes are suspended tomorrow."


def test_missing_column_in_header_is_caught():
    # Write a CSV missing the 'group_id' column entirely.
    fieldnames_missing = ["claim", "evidence", "label", "source_type", "source_url", "date"]
    rows = [
        {
            "claim": "Test claim.",
            "evidence": "Test evidence.",
            "label": "SUPPORTED",
            "source_type": "synthetic",
            "source_url": "",
            "date": "",
        }
    ]
    path = write_temp_csv(rows, fieldnames_missing)
    result = load_dataset(path)

    assert len(result.records) == 0
    assert any("missing required column" in e for e in result.errors)
    os.remove(path)


def test_invalid_source_type_is_rejected():
    """Milestone 5: source_type must be one of the recognized categories
    (synthetic/official_site/facebook/manual_entry) -- a typo or an
    invented category from a future ingestion source should be caught,
    not silently accepted."""
    rows = [
        {
            "claim": "A claim.",
            "evidence": "Some evidence.",
            "label": "SUPPORTED",
            "source_type": "university_website",  # not in VALID_SOURCE_TYPES
            "source_url": "",
            "date": "",
            "group_id": "evt_src_1",
        }
    ]
    path = write_temp_csv(rows, FIELDNAMES)
    result = load_dataset(path)

    assert len(result.records) == 0
    assert any("source_type" in e and "not one of" in e for e in result.errors)
    os.remove(path)


def test_valid_source_types_are_accepted():
    """Each of the four recognized source types should load without error."""
    for source_type in ["synthetic", "official_site", "facebook", "manual_entry"]:
        rows = [
            {
                "claim": "A claim.",
                "evidence": "Some evidence.",
                "label": "SUPPORTED",
                "source_type": source_type,
                "source_url": "",
                "date": "",
                "group_id": f"evt_{source_type}",
            }
        ]
        path = write_temp_csv(rows, FIELDNAMES)
        result = load_dataset(path)

        assert len(result.records) == 1, f"source_type '{source_type}' should be valid"
        os.remove(path)


def test_empty_credibility_tier_is_valid():
    """credibility_tier is optional -- an empty value must NOT be rejected,
    unlike source_type which is required."""
    rows = [
        {
            "claim": "A claim.",
            "evidence": "Some evidence.",
            "label": "SUPPORTED",
            "source_type": "synthetic",
            "source_url": "",
            "date": "",
            "group_id": "evt_cred_1",
            "credibility_tier": "",
        }
    ]
    path = write_temp_csv(rows, FIELDNAMES)
    result = load_dataset(path)

    assert len(result.records) == 1
    os.remove(path)


def test_invalid_credibility_tier_is_rejected():
    rows = [
        {
            "claim": "A claim.",
            "evidence": "Some evidence.",
            "label": "SUPPORTED",
            "source_type": "official_site",
            "source_url": "",
            "date": "",
            "group_id": "evt_cred_2",
            "credibility_tier": "very_trustworthy",  # not a recognized tier
        }
    ]
    path = write_temp_csv(rows, FIELDNAMES)
    result = load_dataset(path)

    assert len(result.records) == 0
    assert any("credibility_tier" in e and "not one of" in e for e in result.errors)
    os.remove(path)
