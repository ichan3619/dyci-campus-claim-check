"""Styled-text folding (src/normalization/fold.py, CAMPUS_VERIFY_UNICODE_FOLD, default off): the unit tests of
results/round2/unicode_fold_plan_draft.md revision 3, section 8 (allowlist, representations, scope, safety)."""
import hashlib
import json
import os
import shutil
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.normalization.fold import ALLOWLIST, ALLOWLIST_SHA256, fold, has_foldable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N04_ID = "ingested:N04"


# ---- allowlist ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("styled, plain", [
    ("₱\U0001D7CF,\U0001D7D0\U0001D7CF\U0001D7D6.\U0001D7D3\U0001D7CE", "₱1,218.50"),   # N04's amount
    ("\U0001D413\U0001D407\U0001D404", "THE"),
    ("ＡＢＣ１２３", "ABC123"),
    ("ａｂｃ", "abc"),
])
def test_styled_letters_and_digits_fold(styled, plain):
    assert fold(styled) == plain and len(fold(styled)) == len(styled) and has_foldable(styled)


@pytest.mark.parametrize("text", ["P1,218.50！", "①", "ﬁ", "…", "²", "₱", "ñ",
                                  "n̈", "\U0001F64F", "\U0001D6A8", "\U0001D6A4", "plain ASCII text."])
def test_everything_else_is_unchanged(text):
    assert fold(text) == text


def test_allowlist_is_exactly_the_frozen_764():
    assert len(ALLOWLIST) == 764
    assert ALLOWLIST_SHA256 == "800828cdad314dc8146809e6f00d4497d95d38badf564b892fe8ba53894cdf34"
    assert all(len(v) == 1 and v.isascii() and v.isalnum() for v in ALLOWLIST.values())


def test_idempotent_and_length_preserving():
    s = "\U0001D413\U0001D407\U0001D404 caroling raised ₱\U0001D7CF,\U0001D7D0\U0001D7CF\U0001D7D6.\U0001D7D3\U0001D7CE！"
    assert fold(fold(s)) == fold(s) and len(fold(s)) == len(s)


def test_only_n04_and_n14_change_in_the_served_corpus():
    from src.api.corpus_config import load_sources
    from src.api.pipeline import DEV_DATASET_PATH, PROJECT_ROOT, SERVED_CORPUS_SPEC
    docs, _ = load_sources(SERVED_CORPUS_SPEC, PROJECT_ROOT, DEV_DATASET_PATH)
    assert sorted(d["document_id"] for d in docs if fold(d["text"]) != d["text"]) == ["ingested:N04", "ingested:N14"]


# ---- representations (offline API over a small corpus containing the real N04 post) -----------------------------------
@pytest.fixture()
def styled_corpus(tmp_path):
    src = os.path.join(ROOT, "data", "corpus", "dev2_announcements.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    keep = [r for r in rows if r["document_id"] in (N04_ID, "ingested:N06", "ingested:N20")]
    path = tmp_path / "styled.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    return str(path), {r["document_id"]: r["text"] for r in keep}


def _post(api_main, text):
    with TestClient(api_main.app) as c:
        return c.post("/verify/text", json={"claim": text}).json(), c.get("/health").json(), api_main.state["manager"].current()


def test_on_processing_is_folded_and_display_is_original(api_settings, styled_corpus):
    path, originals = styled_corpus
    before = hashlib.sha256(open(path, "rb").read()).hexdigest()
    body, health, pipeline = _post(api_settings(corpus=path, unicode_fold=True), "The caroling raised P1,218.50.")
    assert health["unicode_fold"] is True
    docs = {d["document_id"]: d["text"] for d in pipeline.documents}
    assert docs[N04_ID] == fold(originals[N04_ID]) != originals[N04_ID]          # processing copy is folded
    assert pipeline.original_texts[N04_ID] == originals[N04_ID]
    shown = [*body["retrieved_evidence"], *([body["used_evidence"]] if body.get("used_evidence") else [])]
    assert shown and all(d["text"] == originals[d["document_id"]] for d in shown)  # display is the stored original
    assert body["claim"] == "The caroling raised P1,218.50."
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == before          # corpus file unchanged on disk


def test_message_side_folding_reaches_the_checks_but_not_the_echo(api_settings, styled_corpus, monkeypatch):
    # (Note: the date reader already understood styled digits before folding, because Python's \d matches every Unicode
    # digit; so the observable here is the text the verifier itself receives.)
    path, _ = styled_corpus
    styled = "The nursing passing rate among first-time takers was 𝟗𝟗.𝟑𝟓%."
    seen = {}
    for flag in (False, True):
        api_main = api_settings(corpus=path, unicode_fold=flag)
        with TestClient(api_main.app) as c:
            verifier = api_main.state["manager"].current().verifier
            real = verifier.verify_claim
            monkeypatch.setattr(verifier, "verify_claim", lambda q, real=real, flag=flag: (seen.__setitem__(flag, q), real(q))[1])
            body = c.post("/verify/text", json={"claim": styled}).json()
        assert body["claim"] == styled                                          # the echo is always as typed
    assert seen[False] == styled
    assert seen[True] == "The nursing passing rate among first-time takers was 99.35%."  # processing copy is folded


def test_passages_are_mapped_back_to_the_original_by_position():
    from src.api.main import _restore_original_text
    original = "Intro. We raised ₱\U0001D7CF,\U0001D7D0\U0001D7CF\U0001D7D6.\U0001D7D3\U0001D7CE in donations. End."
    folded = fold(original)
    passage = "We raised ₱1,218.50 in donations."
    result = NS(retrieved_evidence=[NS(document_id="d", text=folded)], used_evidence=NS(document_id="d", text=folded),
                selected_passage=NS(document_id="d", text=passage, context_before_text="Intro."))
    _restore_original_text(result, {"d": original}, {"d": folded})
    assert result.selected_passage.text == original[original.index("We raised"):original.index(" End.")]
    assert result.retrieved_evidence[0].text == original and result.used_evidence.text == original
    assert result.selected_passage.context_before_text == "Intro."


# ---- safety ----------------------------------------------------------------------------------------------------------
def test_default_off_and_off_changes_nothing(api_settings, styled_corpus):
    from src.api.pipeline import Settings
    assert Settings().unicode_fold is False
    path, originals = styled_corpus
    body, health, pipeline = _post(api_settings(corpus=path), "The caroling raised P1,218.50.")
    assert health["unicode_fold"] is False and pipeline.original_texts is None
    assert {d["document_id"]: d["text"] for d in pipeline.documents}[N04_ID] == originals[N04_ID]


@patch("src.ingestion.adapters.requests.get")
def test_url_page_is_folded_for_processing_and_displayed_as_fetched(mock_get, api_settings):
    from tests.test_api import mock_html_response
    page = "Notice: the caroling raised ₱\U0001D7CF,\U0001D7D0\U0001D7CF\U0001D7D6.\U0001D7D3\U0001D7CE for charity."
    mock_get.return_value = mock_html_response(text=f"<html><body><p>{page}</p></body></html>")
    with TestClient(api_settings(unicode_fold=True).app) as c:
        r = c.post("/verify/url", json={"claim": "The caroling raised P1,218.50.", "url": "https://example.edu/a"}).json()
    shown = [*r["retrieved_evidence"], *([r["used_evidence"]] if r.get("used_evidence") else [])]
    assert shown and all("\U0001D7CF" in d["text"] for d in shown)                # the page as fetched is displayed
