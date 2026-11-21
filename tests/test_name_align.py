"""
School-name alignment (src/verification/name_align.py), on development sentences only (the probe's kind of sentence and
synthetic posts); the frozen test claims are never used here.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from src.verification import name_align as na

LONG_POST = "Dr. Yanga’s Colleges, Inc. hosted the first-ever Teachers’ Day Celebration on October 1."
SHORT_POST = "For DYCI, exchanges such as this are ultimately about people."
BOTH_POST = "DYCI ADVISORY. Dr. Yanga’s Colleges, Inc. announces the schedule."


@dataclass
class _Doc:
    text: str
    document_id: str = "ingested:X"


@dataclass
class _Result:
    verdict: str
    explanation: str = "First check."
    used_evidence: Optional[_Doc] = None
    retrieved_evidence: List[_Doc] = field(default_factory=list)


def test_short_claim_against_a_long_form_post_uses_the_posts_exact_spelling():
    assert na.aligned_claim("DYCI hosted the Teachers' Day Celebration.", LONG_POST) == \
        "Dr. Yanga’s Colleges, Inc. hosted the Teachers' Day Celebration."
    assert na.aligned_claim("DYCI's delegation went to Surigao.", "The Dr. Yanga Colleges Inc. team went.") == \
        "Dr. Yanga Colleges Inc.'s delegation went to Surigao."


def test_long_claim_against_a_short_form_post_becomes_dyci():
    assert na.aligned_claim("For Dr. Yanga's Colleges, Inc., exchanges are about people.", SHORT_POST) == \
        "For DYCI, exchanges are about people."


def test_the_rule_does_not_apply_when_names_already_agree_or_are_mixed_or_absent():
    assert na.aligned_claim("DYCI hosted it.", SHORT_POST) is None                 # same form
    assert na.aligned_claim("DYCI hosted it.", BOTH_POST) is None                  # post uses both forms
    assert na.aligned_claim("DYCI and Dr. Yanga's Colleges hosted it.", LONG_POST) is None   # claim uses both
    assert na.aligned_claim("The school hosted it.", LONG_POST) is None            # no name
    assert na.aligned_claim("All DYCIans are invited.", LONG_POST) is None         # DYCIans is not the name


def test_apply_rechecks_with_the_aligned_claim_and_says_so():
    calls = []
    first = _Result("CONTRADICTED", used_evidence=_Doc(LONG_POST))
    result, info = na.apply("DYCI hosted the celebration.", first,
                            lambda t: calls.append(t) or _Result("SUPPORTED", explanation="Second check."))
    assert calls == ["Dr. Yanga’s Colleges, Inc. hosted the celebration."]
    assert result.verdict == "SUPPORTED" and "Name check" in result.explanation and "first result: CONTRADICTED" in result.explanation
    assert info == {"original": "DYCI hosted the celebration.", "aligned": calls[0], "first_verdict": "CONTRADICTED",
                    "second_verdict": "SUPPORTED"}


def test_apply_falls_back_to_the_top_retrieved_post_and_leaves_other_results_alone():
    below_gate = _Result("INSUFFICIENT_EVIDENCE", retrieved_evidence=[_Doc(SHORT_POST)])
    result, info = na.apply("Dr. Yanga's Colleges, Inc. values people.", below_gate, lambda t: _Result("SUPPORTED"))
    assert result.verdict == "SUPPORTED" and info["aligned"] == "DYCI values people."
    untouched = _Result("SUPPORTED", used_evidence=_Doc(SHORT_POST))
    assert na.apply("DYCI values people.", untouched, lambda t: 1 / 0) == (untouched, None)
    empty = _Result("INSUFFICIENT_EVIDENCE")
    assert na.apply("DYCI values people.", empty, lambda t: 1 / 0) == (empty, None)


def test_the_setting_is_on_by_default_since_v1_4(monkeypatch):
    monkeypatch.delenv(na.ENV, raising=False)
    assert na.enabled() is True
    monkeypatch.setenv(na.ENV, "0")
    assert na.enabled() is False
