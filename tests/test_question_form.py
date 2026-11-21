"""
src/normalization/question_form.py: the opt-in Option B rewrite must be EXACTLY the evaluated M12.3 Option B.
"""
import json
import os
import shutil

import pytest

from src.normalization import question_form as qf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_rewrites_yes_no_questions_and_keeps_the_original():
    r = qf.rewrite_question("wala bang pasok bukas dahil sa bagyo?")
    assert (r.original, r.rewritten, r.changed) == ("wala bang pasok bukas dahil sa bagyo?",
                                                    "walang pasok bukas dahil sa bagyo.", True)
    assert r.version == "m12_3-optionB-v1" and r.rules_sha256 == qf.FROZEN_SHA256


@pytest.mark.parametrize("text", ["kailan ang enrollment?", "Classes are suspended tomorrow."])
def test_wh_questions_and_statements_are_left_unchanged(text):
    assert qf.rewrite_question(text).rewritten == text


def test_reproduces_every_recorded_option_b_output():
    rows = json.load(open(os.path.join(ROOT, "results", "m12_3", "m12_3_optionB_dev.json"), encoding="utf-8"))["rows"]
    assert len(rows) == 53
    for r in rows:
        assert qf.rewrite_question(r["original_user_input"]).rewritten == r["normalized_claim"], r["case_id"]


def test_refuses_to_load_a_changed_rules_file(tmp_path):
    changed = tmp_path / "m12_3_normalizer.py"
    shutil.copy(qf.OPTION_B_PATH, changed)
    changed.write_bytes(changed.read_bytes() + b"\n# edited\n")
    with pytest.raises(qf.FrozenRulesChanged):
        qf._load(str(changed))


def test_to_dict_keys_match_the_api_schema():
    pytest.importorskip("pydantic")
    from src.api.schemas import ClaimRewriteModel
    d = qf.rewrite_question("may exam ba sa sept 28?").to_dict()
    assert tuple(d) == qf.CLAIM_REWRITE_KEYS == tuple(ClaimRewriteModel.model_fields)
    assert ClaimRewriteModel(**d).rewritten == "may exam sa sept 28."
