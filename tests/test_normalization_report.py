"""
Sprint 3 contract tests: the normalization features the ClaimRouter needs must reach the API schema.
Stdlib only (the schema file is checked with `ast`, because pydantic/FastAPI may not be installed where this runs).
Runs under pytest or:  python tests/test_normalization_report.py  (from repo root, PYTHONPATH=.)
"""
import ast
import json
import os

from src.normalization import TaglishNormalizer
from src.normalization.report import NORMALIZATION_SUMMARY_KEYS, normalization_summary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _class_fields(path, class_name):
    tree = ast.parse(open(os.path.join(ROOT, path), encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [n.target.id for n in node.body if isinstance(n, ast.AnnAssign)]
    raise AssertionError(f"{class_name} not found in {path}")


def test_summary_has_exactly_the_contract_keys_and_is_json_safe():
    s = normalization_summary(TaglishNormalizer().normalize("wp bukas, hnd daw 2morrow sept 28"))
    assert tuple(s) == NORMALIZATION_SUMMARY_KEYS
    assert json.loads(json.dumps(s)) == s


def test_router_features_are_present_and_correct():
    n = TaglishNormalizer()
    short = normalization_summary(n.normalize("wp bukas"))
    assert short["short_claim"] is True and short["has_number"] is False
    longer = normalization_summary(n.normalize("Classes are suspended on September 28 because of the typhoon warning"))
    assert longer["short_claim"] is False and longer["has_number"] is True
    mixed = normalization_summary(n.normalize("the exam ba is sa gym"))
    assert mixed["code_switched"] is True


def test_original_input_is_carried_through_unmodified():
    raw = "  hnd po   pde "
    s = normalization_summary(TaglishNormalizer().normalize(raw))
    assert s["original_user_input"] == raw and s["changed"] is True


def test_schema_model_fields_match_the_summary_keys():
    assert tuple(_class_fields("src/api/schemas.py", "NormalizationInfoModel")) == NORMALIZATION_SUMMARY_KEYS


def test_verification_result_model_exposes_normalization_as_optional():
    assert "normalization" in _class_fields("src/api/schemas.py", "VerificationResultModel")
    src = open(os.path.join(ROOT, "src/api/schemas.py"), encoding="utf-8").read()
    assert "normalization: Optional[NormalizationInfoModel]" in src and "default=None" in src


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as exc:      # noqa: BLE001
                failed += 1; print("FAIL", name, repr(exc))
    sys.exit(1 if failed else 0)
