"""Relevance check (src/verification/relevance_check.py, CAMPUS_VERIFY_RELEVANCE_CHECK, default off): features, the JSON
model, withhold-only behaviour, the API wiring, and the pre-registered split rules (plan section 5.3, checks 1-7)."""
import csv
import hashlib
import json
import os
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from src.retrieval.retriever import RetrievedDocument
from src.verification import relevance_check as rc
from src.verification.relevance_check import FoldError, Message, audit_folds, make_folds, make_groups
from src.verification.verifier import VerificationResult

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = [
    {"document_id": "exam", "text": "The midterm examinations are scheduled on September 28 to 30 at the DYCI Elida campus."},
    {"document_id": "mass", "text": "The Fiesta Mass in honor of Saint Carlo Acutis will be held on October 11."},
    {"document_id": "camp", "text": "Students may claim their new ID cards at the Registrar starting Monday."},
]


def _doc(i, score):
    return RetrievedDocument(document_id=i, score=score, text="", source_type="facebook", source_url="")


# ---- features ---------------------------------------------------------------------------------------------------
def test_features_follow_the_plan_definitions():
    rows = rc.FeatureBuilder(DOCS).pair_features("Are the midterm examinations at Elida on September 28?",
                                                 [_doc("exam", 0.90), _doc("mass", 0.84), _doc("camp", 0.80)], 0.833)
    a, b, c = rows
    assert list(a) == list(rc.FEATURES)
    assert (a["rank"], b["rank"], c["rank"]) == (1.0, 2.0, 3.0)
    assert a["e5_margin"] == pytest.approx(0.06) and b["e5_margin"] == pytest.approx(-0.06)
    assert (a["above_gate"], b["above_gate"], c["above_gate"]) == (1.0, 1.0, 0.0)
    assert a["bm25_rel"] == 1.0 and 0.0 <= b["bm25_rel"] < 1.0
    assert a["content_overlap"] == 1.0                        # midterm, examinations, elida, september all in the post
    assert (a["has_numbers"], a["number_match"]) == (1.0, 1.0)  # "28" and "september"
    assert (a["has_names"], a["name_match"]) == (1.0, 1.0)      # Elida, September (the first word is not a name)
    assert b["number_match"] == 0.0 and b["content_overlap"] < a["content_overlap"]


def test_terms_use_the_existing_lists_and_absent_terms_give_zero():
    assert rc.content_words("Is the canteen open?") == {"canteen", "open"}
    assert rc.number_terms("may exam ba sa sept 28") == {"sept", "28"}          # 'may' is not a month (router list)
    assert rc.name_terms("Classes at Bocaue resume") == {"bocaue"}
    row = rc.FeatureBuilder(DOCS).pair_features("hello there", [_doc("exam", 0.5)], 0.833)[0]
    assert (row["has_numbers"], row["number_match"], row["has_names"], row["name_match"], row["e5_margin"]) == (0, 0, 0, 0, 0)


# ---- model --------------------------------------------------------------------------------------------------------
def _toy_xy():
    X = [[0.9, 0.05, 1, 1, 1.0, 0.8, 1, 1.0, 1, 1.0], [0.85, -0.05, 2, 1, 0.2, 0.1, 1, 0.0, 1, 0.0],
         [0.88, 0.02, 1, 1, 0.9, 0.7, 0, 0.0, 1, 1.0], [0.80, -0.08, 3, 0, 0.1, 0.0, 0, 0.0, 0, 0.0],
         [0.91, 0.04, 1, 1, 1.0, 0.9, 1, 0.5, 0, 0.0], [0.83, -0.08, 2, 0, 0.3, 0.2, 1, 0.0, 0, 0.0]]
    return X, [1, 0, 1, 0, 1, 0]


def test_json_prediction_equals_sklearn():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y = _toy_xy()
    params = rc.fit(X, y)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(scaler.transform(X), y)
    for x, p in zip(X, model.predict_proba(scaler.transform(X))[:, 1]):
        assert rc.predict(params, x) == pytest.approx(p, abs=1e-9)
    assert params["model_params"] == {"C": 1.0, "class_weight": "balanced", "max_iter": 1000}
    assert params["features"] == list(rc.FEATURES) and params["threshold"] == 0.5


def test_load_model_checks_sha_and_features(tmp_path):
    path = str(tmp_path / "m.json")
    sha = rc.save_model(rc.fit(*_toy_xy()), path)
    assert rc.load_model(path, sha)["threshold"] == 0.5
    with pytest.raises(RuntimeError, match="SHA-256"):
        rc.load_model(path, "0" * 64)
    with pytest.raises(RuntimeError, match="no adopted model"):
        rc.load_model(path, "")
    bad = json.load(open(path)); bad["features"] = bad["features"][:-1]
    bad_path = str(tmp_path / "bad.json"); bad_sha = rc.save_model(bad, bad_path)
    with pytest.raises(RuntimeError, match="features"):
        rc.load_model(bad_path, bad_sha)


# ---- withhold only ------------------------------------------------------------------------------------------------
def _params(intercept):
    return {"intercept": intercept, "coef": [0.0] * 10, "scaler_mean": [0.0] * 10, "scaler_scale": [1.0] * 10}


def _result(verdict, used="exam"):
    retrieved = [_doc("exam", 0.9), _doc("mass", 0.85), _doc("camp", 0.8)]
    used_doc = next((d for d in retrieved if d.document_id == used), None) if used else None
    return VerificationResult(claim="c", verdict=verdict, evidence_status="RELEVANT_EVIDENCE_FOUND",
                              retrieved_evidence=retrieved, used_evidence=used_doc, classification_result=None,
                              explanation="E.")


@pytest.mark.parametrize("verdict", ["SUPPORTED", "CONTRADICTED"])
def test_low_relevance_withholds_to_insufficient_only(verdict):
    out, info = rc.RelevanceChecker(DOCS, _params(-5.0)).apply(_result(verdict), "midterm exam", 0.833)
    assert out.verdict == "INSUFFICIENT_EVIDENCE" and info["withheld"] and info["original_verdict"] == verdict
    assert "closest post may not address this claim" in out.explanation
    assert out.used_evidence.document_id == "exam" and out.retrieved_evidence == _result(verdict).retrieved_evidence


@pytest.mark.parametrize("verdict", ["SUPPORTED", "CONTRADICTED"])
def test_high_relevance_keeps_the_verdict_unchanged(verdict):
    original = _result(verdict)
    out, info = rc.RelevanceChecker(DOCS, _params(5.0)).apply(original, "midterm exam", 0.833)
    assert out is original and info["withheld"] is False


def test_never_creates_or_flips_a_verdict():
    for intercept in (-5.0, 5.0):
        checker = rc.RelevanceChecker(DOCS, _params(intercept))
        insufficient = _result("INSUFFICIENT_EVIDENCE")
        assert checker.apply(insufficient, "x", 0.833) == (insufficient, None)
        no_evidence = _result("INSUFFICIENT_EVIDENCE", used=None)
        assert checker.apply(no_evidence, "x", 0.833) == (no_evidence, None)
        for v in ("SUPPORTED", "CONTRADICTED"):
            assert checker.apply(_result(v), "x", 0.833)[0].verdict in (v, "INSUFFICIENT_EVIDENCE")


# ---- the split (plan section 5.3) -----------------------------------------------------------------------------------
def _msgs():
    return [
        Message("acceptance:S01", ("A", "B", "C"), frozenset({"A"})),
        Message("questions:EQ01", ("A", "D", "B"), frozenset({"A"})),            # question form of S01: same group
        Message("acceptance:C07", ("E", "F", "B"), frozenset({"E", "F", "G"})),  # multi-announcement case
        Message("acceptance:Q04", ("G", "E", "C"), frozenset({"G"})),            # linked to C07 through G
        Message("round2:R1", ("H", "A", "C"), frozenset({"H"}), anchor="H"),
        Message("round2:R2", ("B", "C", "D"), frozenset(), anchor="H"),          # no answer, same announcement shown
        Message("acceptance:I04", ("C", "D", "B")),
        Message("questions:EQ28", ("D", "C", "E")),
        Message("acceptance:I01", ("B", "C", "D")),
    ]


def test_check3_shared_expected_announcements_form_one_group_even_through_a_multi_announcement_case():
    g = make_groups(_msgs())
    assert g["acceptance:S01"] == g["questions:EQ01"]
    assert g["acceptance:C07"] == g["acceptance:Q04"]
    assert g["acceptance:S01"] != g["acceptance:C07"]


def test_check4_round2_messages_with_the_same_announcement_shown_form_one_group():
    g = make_groups(_msgs())
    assert g["round2:R1"] == g["round2:R2"]


def test_check5_listed_merges_are_applied_and_nothing_else_is_merged():
    plain = make_groups(_msgs())
    merged = make_groups(_msgs(), [("acceptance:I04", "questions:EQ28")])
    assert plain["acceptance:I04"] != plain["questions:EQ28"] and merged["acceptance:I04"] == merged["questions:EQ28"]
    assert merged["acceptance:I01"] not in (merged["acceptance:I04"], merged["acceptance:S01"])
    with pytest.raises(FoldError, match="unknown message"):
        make_groups(_msgs(), [("acceptance:I04", "questions:NOPE")])


def test_groups_never_depend_on_candidates():
    moved = [replace(m, candidates=tuple(reversed(m.candidates))) for m in _msgs()]
    assert make_groups(moved) == make_groups(_msgs())


def test_checks_1_2_6_no_held_document_or_message_in_training_and_each_message_held_out_once():
    folds = make_folds(_msgs(), [("acceptance:I04", "questions:EQ28")], min_positive=0, min_negative=0)
    for f in folds:
        held = set(f.held_docs)
        assert not {d for _, d in f.train_pairs} & held                                 # check 1
        assert not {k for k, _ in f.train_pairs} & set(f.test_keys)                     # check 2
        assert {d for _, d in f.test_pairs} <= held
    assert sorted(k for f in folds for k in f.test_keys) == sorted(m.key for m in _msgs())   # check 6
    s01 = next(f for f in folds if "acceptance:S01" in f.test_keys)
    assert {"A", "B", "C", "D"} <= set(s01.held_docs) and s01.dropped_pairs > 0


def test_held_docs_include_expected_and_announcement_shown_even_when_not_candidates():
    msgs = [Message("round2:R1", ("X",), frozenset({"H"}), anchor="Z"), Message("acceptance:S1", ("H", "Z", "Y"), frozenset({"Y"}))]
    f = next(f for f in make_folds(msgs, min_positive=0, min_negative=0) if f.test_keys == ["round2:R1"])
    assert set(f.held_docs) == {"X", "H", "Z"} and f.train_pairs == [("acceptance:S1", "Y")]


def test_check7_a_fold_below_the_minimum_refuses():
    with pytest.raises(FoldError, match="minimum 10/10"):
        make_folds(_msgs())


def test_audit_catches_a_leak():
    folds = make_folds(_msgs(), min_positive=0, min_negative=0)
    f = folds[0]
    f.train_pairs.append(("acceptance:I01", f.held_docs[0]))
    with pytest.raises(FoldError, match="held-out documents"):
        audit_folds(folds, _msgs())


# ---- settings and API -----------------------------------------------------------------------------------------------
def test_setting_defaults_off_and_flag_on_without_a_model_sha_refuses():
    from src.api.pipeline import Settings
    assert Settings().relevance_check is False and Settings().relevance_model_sha256 == ""
    with pytest.raises(RuntimeError, match="RELEVANCE_MODEL_SHA256"):
        replace(Settings(), relevance_check=True).validate()


def _claims(n=12):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "dev", "sample_claims.csv"), encoding="utf-8")))
    return [r["claim"] for r in rows[:n]] + ["walang pasok bukas dahil sa bagyo", "When is the enrollment?"]


def _bodies(api_main, texts):
    with TestClient(api_main.app) as c:
        health = c.get("/health").json()
        return health, {t: c.post("/verify/text", json={"claim": t}).json() for t in texts}


def _model(tmp_path, intercept):
    path = str(tmp_path / "relevance.json")
    params = rc.fit(*_toy_xy())
    params.update(intercept=intercept, coef=[0.0] * 10)
    return path, rc.save_model(params, path)


def test_default_off_reports_off_and_adds_nothing(api_settings):
    health, bodies = _bodies(api_settings(), _claims())
    assert health["relevance_check"] is False
    assert all(b.get("relevance_check") is None for b in bodies.values())
    assert any(b["verdict"] in rc.SC for b in bodies.values())          # the comparison below is not vacuous


def test_flag_on_only_withholds_english_verdicts(api_settings, tmp_path):
    texts = _claims()
    _, off = _bodies(api_settings(relevance_check=False, relative_date_check=False), texts)   # this check alone (B off)
    path, sha = _model(tmp_path, -5.0)                                  # P(relevant) ~ 0.007 for every pair
    health, on = _bodies(api_settings(relevance_check=True, relevance_model=path, relevance_model_sha256=sha,
                                      relative_date_check=False), texts)
    assert health["relevance_check"] is True
    from src.api.verdict_policy import withhold_for_language
    withheld = 0
    for t in texts:
        a, b = off[t], on[t]
        if a["verdict"] in rc.SC and not withhold_for_language(t):
            withheld += 1
            assert b["verdict"] == "INSUFFICIENT_EVIDENCE" and b["relevance_check"]["withheld"] is True
            assert b["relevance_check"]["original_verdict"] == a["verdict"]
            assert {k: v for k, v in b.items() if k not in ("verdict", "explanation", "relevance_check")} == \
                   {k: v for k, v in a.items() if k not in ("verdict", "explanation", "relevance_check")}
        else:
            assert b == a                                                # INSUFFICIENT, INFO, Tagalog/Taglish: identical
    assert withheld > 0


def test_flag_on_with_a_wrong_sha_fails_startup(api_settings, tmp_path):
    path, _ = _model(tmp_path, 0.0)
    api_main = api_settings(relevance_check=True, relevance_model=path, relevance_model_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256"):
        with TestClient(api_main.app):
            pass


def test_frontend_explains_a_withheld_verdict():
    app = open(os.path.join(ROOT, "frontend", "js", "app.js"), encoding="utf-8").read()
    assert "data.relevance_check && data.relevance_check.withheld" in app
    assert "The closest post may not address this claim" in app
    assert "That does not mean it is false" in app                      # plain wording since 2026-10-28 (owner)
