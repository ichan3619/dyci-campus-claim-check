"""
JSON-safe summary of a NormalizationResult, for API responses and evaluation logs.

This is the contract the Sprint 3 ClaimRouter reads: short_claim, has_number, code_switched (plus counts). Keep the key
list below in sync with NormalizationInfoModel in src/api/schemas.py -- tests/test_normalization_report.py enforces it.
Stdlib only, so it is testable without FastAPI/pydantic.
"""
from typing import Dict

from src.normalization.taglish import NormalizationResult

NORMALIZATION_SUMMARY_KEYS = (
    "original_user_input", "normalized_claim", "changed", "edit_rules", "flags",
    "short_claim", "has_number", "code_switched", "token_count", "content_token_count",
    "normalizer_version", "lexicon_sha256",
)


def normalization_summary(result: NormalizationResult) -> Dict[str, object]:
    f = result.features
    return {
        "original_user_input": result.original,
        "normalized_claim": result.normalized,
        "changed": result.changed,
        "edit_rules": [e.rule for e in result.edits],
        "flags": list(result.flags),
        "short_claim": bool(f["short_claim"]),
        "has_number": bool(f["has_number"]),
        "code_switched": bool(f["code_switched"]),
        "token_count": int(f["token_count"]),
        "content_token_count": int(f["content_token_count"]),
        "normalizer_version": result.version,
        "lexicon_sha256": result.lexicon_sha256,
    }
