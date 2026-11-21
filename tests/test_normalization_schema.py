"""Sprint 3 prep: the API result can carry normalization features (short_claim, has_number, code_switched). Needs pydantic.

Layout contract: the API payload is the FLAT dict produced by src.normalization.report.normalization_summary()
(short_claim / has_number / code_switched sit at the top level of `normalization`). NormalizationResult.to_dict() is
a different, NESTED shape (features / edits) meant for evaluation logs; it is not what the API schema accepts.
"""
import pytest
from pydantic import ValidationError

from src.api.schemas import NormalizationInfoModel, VerificationResultModel
from src.normalization import TaglishNormalizer
from src.normalization.report import normalization_summary


def _base(**kw):
    return dict(claim="wp bukas", verdict="INSUFFICIENT_EVIDENCE", evidence_status="NO_RELEVANT_EVIDENCE",
                retrieved_evidence=[], used_evidence=None, classification_result=None, explanation="x",
                data_status="development", **kw)


def test_normalization_is_optional_and_defaults_to_none():
    assert VerificationResultModel(**_base()).normalization is None


def test_normalizer_output_round_trips_through_the_schema():
    d = normalization_summary(TaglishNormalizer().normalize("exam sept 28"))
    m = VerificationResultModel(**_base(normalization=d))
    n = m.normalization
    assert (n.short_claim, n.has_number, n.code_switched) == (True, True, False)
    assert n.original_user_input == "exam sept 28" and n.normalized_claim == "exam September 28"
    dumped = m.model_dump()["normalization"]
    assert dumped["has_number"] is True
    assert "features" not in dumped          # flat layout: no nested features object


def test_missing_required_feature_is_rejected():
    d = normalization_summary(TaglishNormalizer().normalize("exam sept 28"))
    NormalizationInfoModel(**d)              # control: the complete flat payload is accepted
    del d["short_claim"]
    with pytest.raises(ValidationError) as exc:
        NormalizationInfoModel(**d)          # fails because of THIS field, not for an unrelated reason
    assert "short_claim" in str(exc.value)


def test_nested_to_dict_shape_is_not_accepted_by_the_api_schema():
    nested = TaglishNormalizer().normalize("exam sept 28").to_dict()
    assert "features" in nested
    with pytest.raises(ValidationError):
        NormalizationInfoModel(**nested)
