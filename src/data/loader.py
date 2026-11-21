"""
Loads a campus claim/evidence CSV file into validated Python records.

This is the one place that knows how to go from "a CSV file on disk" to
"a list of clean, checked dictionaries." Nothing downstream (retrieval,
classification, tests) should need to know about CSV parsing at all --
they just receive the output of load_dataset().
"""

import csv
from dataclasses import dataclass, field
from typing import List

from src.data.schema import REQUIRED_COLUMNS
from src.data.preprocessing import normalize_whitespace
from src.data.validation import validate_record, find_duplicate_claims


@dataclass
class DatasetLoadResult:
    """
    Everything you get back from loading a dataset file:
      - records: the valid rows, ready to use
      - errors: per-row problems that caused a row to be rejected
      - duplicate_warnings: rows that are exact duplicates of earlier rows
        (not rejected automatically -- flagged for human review)
    """
    records: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duplicate_warnings: List[str] = field(default_factory=list)


def load_dataset(csv_path: str) -> DatasetLoadResult:
    """
    Read a CSV file, normalize whitespace in text fields, validate every
    row, and return the valid records plus a report of anything wrong.

    Invalid rows are excluded from `records` but reported in `errors` --
    we don't want one bad row to crash the whole load, but we also don't
    want bad data silently entering the pipeline.
    """
    result = DatasetLoadResult()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Check the header has every column we expect before reading rows.
        # This catches a completely wrong/misnamed CSV early, with one
        # clear error, instead of many confusing "missing field" errors
        # per row.
        actual_columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - actual_columns
        if missing_columns:
            result.errors.append(
                f"CSV is missing required column(s): {sorted(missing_columns)}"
            )
            return result

        all_rows = []
        for row_number, row in enumerate(reader, start=2):  # header is row 1
            # Normalize whitespace on the text fields we care about most.
            # We do this before validation so that a claim like "  "
            # (only whitespace) is correctly caught as empty.
            row["claim"] = normalize_whitespace(row.get("claim", ""))
            row["evidence"] = normalize_whitespace(row.get("evidence", ""))

            validation = validate_record(row, row_number)
            if validation.is_valid:
                result.records.append(row)
                all_rows.append(row)
            else:
                result.errors.extend(validation.errors)

    # Duplicate check runs across the valid rows only.
    result.duplicate_warnings = find_duplicate_claims(all_rows)

    return result
