"""
Validation rules for campus claim/evidence records.

Two levels of validation:
  1. Per-record checks (does THIS row make sense on its own?)
  2. Dataset-level checks (are there duplicates across ALL rows?)

We separate these because they answer different questions and are useful
independently -- e.g. you might want to check one new row before adding it
to the dataset, without re-scanning the whole file for duplicates.
"""

from dataclasses import dataclass, field
from typing import List

from src.data.schema import VALID_LABELS, REQUIRED_NON_EMPTY_FIELDS, VALID_SOURCE_TYPES, VALID_CREDIBILITY_TIERS


@dataclass
class ValidationResult:
    """
    Holds the outcome of validating one record.

    is_valid is True only if `errors` is empty. We keep both instead of
    just a boolean so that when something is wrong, we know exactly why
    (useful for debugging a bad CSV row, and useful later for a data
    collection team to fix source data).
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)


def validate_record(row: dict, row_number: int) -> ValidationResult:
    """
    Validate a single record (one CSV row, as a dict).

    row_number is only used to make error messages point to a specific
    line in the source file, e.g. "row 7: label 'MAYBE' is not valid".
    """
    errors = []

    # Check required fields are present and non-empty.
    for required_field in REQUIRED_NON_EMPTY_FIELDS:
        value = row.get(required_field, "")
        if value is None or str(value).strip() == "":
            errors.append(
                f"row {row_number}: required field '{required_field}' is missing or empty"
            )

    # Check the label, if present, is one of the three allowed values.
    label = row.get("label", "")
    if label and label not in VALID_LABELS:
        errors.append(
            f"row {row_number}: label '{label}' is not one of {sorted(VALID_LABELS)}"
        )

    # Check source_type, if present, is one of the recognized categories.
    # Added in Milestone 5 to prevent future ingestion sources (website
    # scraping, Facebook Graph API, manual entry) from each inventing
    # their own inconsistent labels for the same source category -- see
    # src/data/schema.py for the full rationale.
    source_type = row.get("source_type", "")
    if source_type and source_type not in VALID_SOURCE_TYPES:
        errors.append(
            f"row {row_number}: source_type '{source_type}' is not one of "
            f"{sorted(VALID_SOURCE_TYPES)}"
        )

    # Check credibility_tier, if present and non-empty, is one of the
    # recognized tiers. Unlike source_type, an EMPTY credibility_tier is
    # valid (it's an optional field -- see schema.py) -- this only
    # rejects a NON-empty value that isn't one of the recognized tiers.
    credibility_tier = row.get("credibility_tier", "")
    if credibility_tier and credibility_tier not in VALID_CREDIBILITY_TIERS:
        errors.append(
            f"row {row_number}: credibility_tier '{credibility_tier}' is not one of "
            f"{sorted(VALID_CREDIBILITY_TIERS)}"
        )

    # Special case: INSUFFICIENT_EVIDENCE rows are expected to have empty
    # evidence (no matching campus document exists). SUPPORTED and
    # CONTRADICTED rows, on the other hand, must have non-empty evidence,
    # since the whole point of those labels is a relationship to a
    # specific piece of evidence text.
    evidence = row.get("evidence", "")
    if label in ("SUPPORTED", "CONTRADICTED") and (evidence is None or str(evidence).strip() == ""):
        errors.append(
            f"row {row_number}: label '{label}' requires non-empty evidence"
        )

    return ValidationResult(is_valid=(len(errors) == 0), errors=errors)


def find_duplicate_claims(rows: List[dict]) -> List[str]:
    """
    Return warnings for any (claim, evidence) pair that appears more than
    once in the dataset.

    This is a duplicate CHECK, not an automatic remover -- exact duplicate
    rows usually indicate a copy-paste mistake during manual data entry,
    and should be reviewed by a human rather than silently dropped.
    """
    seen = {}
    warnings = []

    for i, row in enumerate(rows, start=1):
        key = (row.get("claim", "").strip(), row.get("evidence", "").strip())
        if key in seen:
            warnings.append(
                f"row {i}: duplicate of row {seen[key]} (identical claim + evidence)"
            )
        else:
            seen[key] = i

    return warnings
