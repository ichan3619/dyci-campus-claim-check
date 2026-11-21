"""
M12.2 tests. They use a deterministic FAKE embedding model (hashed bag-of-words), so they need
no download and no torch. They verify PLUMBING and contracts -- prefixes, ordering, thresholds,
splitting, selection, gating, backward compatibility. They say nothing about e5's real quality;
that is measured by scripts/m12_retrieval/eval_retrieval.py with the real model.
"""
import re
import zlib

import numpy as np
import pytest

from src.classification.result import ClassificationResult
from src.retrieval.e5_retriever import (E5Encoder, E5Retriever, E5_THRESHOLD_ENV_VAR,
                                        PASSAGE_PREFIX, PROVISIONAL_E5_THRESHOLD, QUERY_PREFIX)
from src.retrieval.passage_selector import PassageSelector, split_into_passages
from src.retrieval.retriever import Retriever
from src.verification.verifier import ClaimVerifier, NO_RELEVANT_EVIDENCE

DIM = 128


class FakeModel:
    """Hashed bag-of-words, L2-normalised. Records every string it is asked to encode."""
    def __init__(self):
        self.seen = []

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), DIM))
        for i, t in enumerate(texts):
            self.seen.append(t)
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w in {"query", "passage"}:
                    continue
                out[i, zlib.crc32(w.encode()) % DIM] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


def doc(i, text):
    return {"document_id": i, "text": text, "source_type": "synthetic", "source_url": ""}


DOCS = [
    doc("typhoon", "All classes are suspended tomorrow due to the typhoon. Students should stay home."),
    doc("enroll", "Second semester enrollment begins August 10 and ends August 20."),
    doc("fb", "DYCI Advisory 27 September 2026 SHORTENED CLASS HOURS Please be informed that tomorrow, "
              "September 28, 2026, classes will be until 1:00 PM only. This is in observance of the 34th "
              "National Family Week, pursuant to Circular Memorandum No. 136 issued by the Malacanang Office. "
              "Regular operations will resume on the next working day."),
]


@pytest.fixture
def enc():
    return E5Encoder(model=FakeModel())


# ---------------- Step 1: e5 retriever ----------------
def test_prefixes_are_applied_to_queries_and_documents(enc):
    r = E5Retriever(DOCS, encoder=enc, relevance_threshold=0.5)
    r.retrieve("enrollment dates")
    seen = enc._model.seen
    assert all(s.startswith(PASSAGE_PREFIX) for s in seen[:len(DOCS)])
    assert seen[-1] == QUERY_PREFIX + "enrollment dates"


def test_returns_same_type_sorted_and_respects_top_k(enc):
    r = E5Retriever(DOCS, encoder=enc, relevance_threshold=0.5)
    res = r.retrieve("semester enrollment August", top_k=2)
    assert len(res) == 2 and res[0].document_id == "enroll"
    assert res[0].score >= res[1].score
    assert type(res[0]).__name__ == "RetrievedDocument"


def test_documents_encoded_once_not_per_query(enc):
    r = E5Retriever(DOCS, encoder=enc, relevance_threshold=0.5)
    n = len(enc._model.seen)
    r.retrieve("a"); r.retrieve("b")
    assert len(enc._model.seen) == n + 2          # only the two queries were encoded


def test_empty_query_and_empty_corpus(enc):
    assert E5Retriever(DOCS, encoder=enc, relevance_threshold=0.5).retrieve("   ") == []
    assert E5Retriever([], encoder=enc, relevance_threshold=0.5).retrieve("x") == []


def test_threshold_precedence_arg_over_env_over_provisional(enc, monkeypatch):
    monkeypatch.delenv(E5_THRESHOLD_ENV_VAR, raising=False)
    assert E5Retriever(DOCS, encoder=enc).default_relevance_threshold == PROVISIONAL_E5_THRESHOLD
    monkeypatch.setenv(E5_THRESHOLD_ENV_VAR, "0.9")
    assert E5Retriever(DOCS, encoder=enc).default_relevance_threshold == 0.9
    assert E5Retriever(DOCS, encoder=enc, relevance_threshold=0.7).default_relevance_threshold == 0.7


# ---------------- Step 4: passage splitting ----------------
def test_split_keeps_circular_memorandum_number_together():
    ps = split_into_passages(DOCS[2]["text"])
    assert len(ps) == 3
    assert any("Memorandum No. 136 issued" in p for p in ps)
    assert not any(p.startswith("136") for p in ps)


def test_split_handles_titles_initials_and_month_abbreviations():
    ps = split_into_passages("See Dr. J. Santos at Rm. 204. Classes resume Sept. 28, 2026. Thank you very much!")
    assert ps == ["See Dr. J. Santos at Rm. 204.", "Classes resume Sept. 28, 2026.", "Thank you very much!"]


def test_split_never_loses_a_document():
    assert split_into_passages("Hi.") == ["Hi."]          # all fragments too short -> whole text
    assert split_into_passages("   ") == []


# ---------------- Step 4: passage selection ----------------
def test_selector_picks_the_relevant_sentence(enc):
    from src.retrieval.retriever import RetrievedDocument
    d = RetrievedDocument("fb", 0.9, DOCS[2]["text"], "synthetic", "")
    sel = PassageSelector(enc).select("Classes will be until 1:00 PM only tomorrow.", [d])
    assert len(sel) == 1 and sel[0].passage_index == 0 and "1:00 PM" in sel[0].text
    assert sel[0].document_id == "fb" and sel[0].document_score == 0.9


def test_selector_top_n_and_validation(enc):
    from src.retrieval.retriever import RetrievedDocument
    d = RetrievedDocument("fb", 0.9, DOCS[2]["text"], "synthetic", "")
    assert len(PassageSelector(enc, top_n=2).select("classes until 1:00 PM", [d])) == 2
    with pytest.raises(ValueError):
        PassageSelector(enc, top_n=0)
    assert PassageSelector(enc).select("x", []) == []


def test_selector_caches_document_embeddings(enc):
    from src.retrieval.retriever import RetrievedDocument
    d = RetrievedDocument("fb", 0.9, DOCS[2]["text"], "synthetic", "")
    s = PassageSelector(enc)
    s.select("classes until 1:00 PM", [d]); n = len(enc._model.seen)
    s.select("classes until 1:00 PM", [d])
    assert len(enc._model.seen) == n              # nothing re-encoded: passages cached, repeated query reused (M12.3 query cache)


# ---------------- verifier integration ----------------
class RecordingClassifier:
    def __init__(self):
        self.calls = []

    def classify(self, claim, evidence):
        self.calls.append((claim, evidence))
        return ClassificationResult(label="SUPPORTED", confidence=0.9, label_scores={"SUPPORTED": 0.9})


def test_verifier_passes_selected_passage_not_whole_post_to_classifier(enc):
    retr = E5Retriever(DOCS, encoder=enc, relevance_threshold=0.05)
    clf = RecordingClassifier()
    v = ClaimVerifier(retr, clf, passage_selector=PassageSelector(enc))
    res = v.verify_claim("Classes will be until 1:00 PM only tomorrow.")
    premise = clf.calls[0][1]
    assert premise != DOCS[2]["text"] and "1:00 PM" in premise and len(premise) < len(DOCS[2]["text"])
    assert res.selected_passage is not None and res.used_evidence.document_id == "fb"
    assert "selected passage" in res.explanation


def test_verifier_gate_still_blocks_before_selection_and_nli(enc):
    retr = E5Retriever(DOCS, encoder=enc, relevance_threshold=0.99)
    clf = RecordingClassifier()
    res = ClaimVerifier(retr, clf, passage_selector=PassageSelector(enc)).verify_claim("weather in paris")
    assert res.evidence_status == NO_RELEVANT_EVIDENCE and clf.calls == [] and res.selected_passage is None


def test_verifier_uses_retriever_default_threshold_when_not_given(enc):
    v = ClaimVerifier(E5Retriever(DOCS, encoder=enc, relevance_threshold=0.42), RecordingClassifier())
    assert v._relevance_threshold == 0.42


def test_backward_compat_tfidf_default_threshold_and_whole_document_premise():
    retr = Retriever([doc("typhoon", DOCS[0]["text"])])
    clf = RecordingClassifier()
    v = ClaimVerifier(retr, clf)                       # no selector, no threshold -> exactly the old behaviour
    assert v._relevance_threshold == 0.15
    res = v.verify_claim("Classes are suspended tomorrow.")
    assert clf.calls[0][1] == DOCS[0]["text"] and res.selected_passage is None


# ---------------- M12.2b: passage-level document scoring ----------------
def test_invalid_score_mode_rejected(enc):
    with pytest.raises(ValueError):
        E5Retriever(DOCS, encoder=enc, relevance_threshold=0.5, score_mode="sentence")


def test_passage_mode_scores_a_buried_detail_higher_than_document_mode(enc):
    claim = "Regular operations will resume on the next working day."       # one sentence inside the long FB post
    doc_score = {r.document_id: r.score for r in E5Retriever(DOCS, encoder=E5Encoder(model=FakeModel()), relevance_threshold=0.1).retrieve(claim, top_k=3)}
    psg = E5Retriever(DOCS, encoder=E5Encoder(model=FakeModel()), relevance_threshold=0.1, score_mode="passage")
    psg_score = {r.document_id: r.score for r in psg.retrieve(claim, top_k=3)}
    assert psg_score["fb"] > doc_score["fb"] and psg_score["fb"] == pytest.approx(1.0)
    assert type(psg.retrieve(claim)[0]).__name__ == "RetrievedDocument"


def test_passage_mode_default_threshold_differs_and_env_still_wins(enc, monkeypatch):
    from src.retrieval.e5_retriever import PROVISIONAL_E5_PASSAGE_THRESHOLD
    monkeypatch.delenv(E5_THRESHOLD_ENV_VAR, raising=False)
    assert E5Retriever(DOCS, encoder=enc, score_mode="passage").default_relevance_threshold == PROVISIONAL_E5_PASSAGE_THRESHOLD
    monkeypatch.setenv(E5_THRESHOLD_ENV_VAR, "0.5")
    assert E5Retriever(DOCS, encoder=enc, score_mode="passage").default_relevance_threshold == 0.5


def test_selector_reuses_retriever_passage_embeddings_without_re_encoding(enc):
    retr = E5Retriever(DOCS, encoder=enc, relevance_threshold=0.1, score_mode="passage")
    sel = PassageSelector(enc, passage_source=retr)
    docs = retr.retrieve("Regular operations will resume on the next working day.", top_k=3)
    n = len(enc._model.seen)
    chosen = sel.select("Regular operations will resume on the next working day.", docs)
    assert chosen[0].document_id == "fb" and "Regular operations" in chosen[0].text
    assert len(enc._model.seen) == n              # no passage re-embedded; the claim's vector is reused from retrieval (M12.3 query cache)


def test_query_cache_encodes_each_new_claim_once_and_stays_bounded(enc):
    enc.encode_queries(["a claim"]); n = len(enc._model.seen)
    enc.encode_queries(["a claim", "another claim"])
    assert len(enc._model.seen) == n + 1                      # only the new claim
    for i in range(enc.QUERY_CACHE_SIZE + 10):
        enc.encode_queries([f"claim {i}"])
    assert len(enc._query_cache) == enc.QUERY_CACHE_SIZE
    assert enc.encode_queries([]).shape[0] == 0


def test_verifier_end_to_end_in_passage_mode_reaches_a_buried_detail(enc):
    retr = E5Retriever(DOCS, encoder=enc, relevance_threshold=0.6, score_mode="passage")
    clf = RecordingClassifier()
    res = ClaimVerifier(retr, clf, passage_selector=PassageSelector(enc, passage_source=retr)).verify_claim(
        "Regular operations will resume on the next working day.")
    assert res.used_evidence.document_id == "fb" and clf.calls[0][1] == "Regular operations will resume on the next working day."


# ---------------- M12.2c: optional context window ----------------
def _fb_doc():
    from src.retrieval.retriever import RetrievedDocument
    return RetrievedDocument("fb", 0.9, DOCS[2]["text"], "synthetic", "")


def test_context_before_zero_is_unchanged_behaviour(enc):
    sel = PassageSelector(enc).select("observance of the National Family Week", [_fb_doc()])[0]
    assert sel.context_before_text == "" and sel.premise == sel.text


def test_context_before_prepends_previous_sentence_but_keeps_selected_text_separate(enc):
    sel = PassageSelector(enc, context_before=1).select("observance of the National Family Week Circular Memorandum", [_fb_doc()])[0]
    assert sel.passage_index == 1 and sel.text.startswith("This is in observance")
    assert "SHORTENED CLASS HOURS" in sel.context_before_text and sel.premise.endswith(sel.text)
    assert PassageSelector.premise_from([sel]) == sel.premise


def test_context_before_first_sentence_has_no_context_and_negative_rejected(enc):
    sel = PassageSelector(enc, context_before=2).select("classes until 1:00 PM only", [_fb_doc()])[0]
    assert sel.passage_index == 0 and sel.context_before_text == ""
    with pytest.raises(ValueError):
        PassageSelector(enc, context_before=-1)
