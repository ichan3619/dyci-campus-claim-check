"""
Optional claim rewrite for the verification path: M12.3 Option B (yes/no question -> statement).

WHAT IT DOES
    Deletes question markers and nothing else: a terminal '?' becomes '.', the Tagalog particle 'ba' / 'bang' is removed
    (re-attaching the linker), and a few English auxiliary-first questions are reordered. It never adds a content word,
    never translates, and refuses (returns the input unchanged) when the structure is not safe to rewrite.
        'may exam ba sa sept 28?'  ->  'may exam sa sept 28.'
        'wala bang pasok bukas?'   ->  'walang pasok bukas.'

WHY IT IS LOADED FROM scripts/
    The rules live in scripts/m12_retrieval/m12_3_normalizer.py, the exact file whose outputs are recorded in
    results/m12_3/. Copying them into src/ would create a second implementation that could drift from the evaluated
    one. Instead this module loads that file by path and refuses to run unless its SHA-256 equals FROZEN_SHA256, so
    whatever the API applies is provably the evaluated Option B. (Verified on 2026-10-07: this version reproduces all
    53 normalized claims recorded in results/m12_3/m12_3_optionB_dev.json, and it is the version the C1/C2 runs used.)

STATUS
    OFF by default (CAMPUS_VERIFY_CLAIM_REWRITE=optionB turns it on). On the development set it improved verdicts on
    2 of 27 checkable cases with 0 regressions, and changed nothing on the 26 canonical claims. M12.3 decision 2
    (raw vs normalized) is not taken until the independent set has been run; until then this is an opt-in.
    Apply it ONLY to inputs the router classified as CHECKABLE_CLAIM (wh-questions are refused anyway, guard G1).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from dataclasses import dataclass
from typing import Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPTION_B_PATH = os.path.join(PROJECT_ROOT, "scripts", "m12_retrieval", "m12_3_normalizer.py")
FROZEN_SHA256 = "133265f88dc312bea1d10dad690773f4b3018ddd794ed1ee425796a928539930"
REWRITE_NAME = "optionB"

_module = None


class FrozenRulesChanged(RuntimeError):
    """The Option B file no longer matches the evaluated version."""


@dataclass(frozen=True)
class ClaimRewrite:
    original: str
    rewritten: str
    rules_applied: Tuple[str, ...]
    refused: Optional[str]
    version: str
    rules_sha256: str

    @property
    def changed(self) -> bool:
        return self.rewritten != self.original

    def to_dict(self) -> dict:
        """Keys must stay in sync with ClaimRewriteModel in src/api/schemas.py (tests enforce it)."""
        return {"original": self.original, "rewritten": self.rewritten, "changed": self.changed,
                "rules_applied": list(self.rules_applied), "refused": self.refused, "version": self.version,
                "rules_sha256": self.rules_sha256}


CLAIM_REWRITE_KEYS = ("original", "rewritten", "changed", "rules_applied", "refused", "version", "rules_sha256")


def _load(path: str = OPTION_B_PATH, expected_sha256: str = FROZEN_SHA256):
    global _module
    if _module is not None and path == OPTION_B_PATH:
        return _module
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest != expected_sha256:
        raise FrozenRulesChanged(
            f"{path} has SHA-256 {digest}, expected the evaluated version {expected_sha256}. Re-run the M12.3 "
            "Option B evaluation before changing FROZEN_SHA256.")
    spec = importlib.util.spec_from_file_location("_m12_3_option_b_frozen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if path == OPTION_B_PATH:
        _module = module
    return module


def rewrite_question(text: str) -> ClaimRewrite:
    """Option B applied to one claim. The original is never modified; both strings are returned."""
    m = _load()
    r = m.normalize(text)
    return ClaimRewrite(original=text, rewritten=r.normalized_claim, rules_applied=tuple(r.rules_applied),
                        refused=r.refused, version=m.NORMALIZER_VERSION, rules_sha256=FROZEN_SHA256)
