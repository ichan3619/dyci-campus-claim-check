"""
Sprint 2 tests: TagalogBM25 + HybridRetriever.

They verify PLUMBING and CONTRACTS with a deterministic fake embedding model and stub retrievers: tokenization,
n-gram overlap, RRF arithmetic, score preservation, the verifier's threshold gate, and backward compatibility.
They say NOTHING about E5's real retrieval quality (that needs the real model; see scripts/m12_retrieval/m12_4_hybrid_eval.py).
Runs under pytest or standalone:  python tests/test_hybrid_retriever.py   (from the repo root, with PYTHONPATH=.)
"""
import re
import zlib

import numpy as np

from src.classification.result import ClassificationResult
from src.normalization import TaglishNormalizer
from src.retrieval.bm25 import TagalogBM25, char_ngrams, word_tokens
from src.retrieval.e5_retriever import E5Encoder, E5Retriever
from src.retrieval.hybrid_retriever import HybridRetrievedDocument, HybridRetriever
from src.retrieval.passage_selector import PassageSelector
from src.retrieval.retriever import RetrievedDocument, Retriever
from src.verification.verifier import ClaimVerifier, NO_RELEVANT_EVIDENCE, RELEVANT_EVIDENCE_FOUND

THRESHOLD = 0.84


# ------------------------------------------------------------------ helpers
def doc(i, text):
    return {"document_id": i, "text": text, "source_type": "synthetic", "source_url": ""}


class FakeModel:
    """Hashed bag-of-words, L2-normalised (same idea as tests/test_m12_2_e5_and_passages.py)."""
    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w not in {"query", "passage"}:
                    out[i, zlib.crc32(w.encode()) % 256] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


class StubDense:
    """Dense retriever with hand-set cosines: table[query][doc_id] = cosine. Unknown queries score 0."""
    def __init__(self, docs, table, threshold=THRESHOLD):
        self.docs, self.table, self.default_relevance_threshold = docs, table, threshold

    def retrieve(self, query, top_k=3):
        row = self.table.get(query, {})
        scored = sorted(((row.get(d["document_id"], 0.0), n) for n, d in enumerate(self.docs)), key=lambda p: (-p[0], p[1]))
        return [RetrievedDocument(self.docs[n]["document_id"], s, self.docs[n]["text"], "synthetic", "") for s, n in scored[:top_k]]


class StubBM25:
    """BM25 with hand-set ranking: ranks[query] = [doc_index, ...] best first."""
    def __init__(self, ranks):
        self.ranks = ranks

    def rank(self, query):
        return [(i, 1.0 / (r + 1)) for r, i in enumerate(self.ranks.get(query, []))]


class CountingClassifier:
    def __init__(self):
        self.calls = 0

    def classify(self, claim, evidence):
        self.calls += 1
        return ClassificationResult(label="SUPPORTED", confidence=0.9, label_scores={"SUPPORTED": 0.9})


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


ABC = [doc("A", "alpha text"), doc("B", "beta text"), doc("C", "gamma text")]

# ------------------------------------------------------------------ BM25: tokenization
def test_bm25_keeps_short_tagalog_function_words():
    assert word_tokens("May exam at sa gym ba?") == ["may", "exam", "at", "sa", "gym", "ba"]


def test_bm25_stopwords_are_opt_in():
    assert word_tokens("may at", stopwords=frozenset({"at"})) == ["may"]


def test_bm25_finds_may_where_the_tfidf_retriever_cannot():
    docs = [doc("exam", "May exam sa September 28 sa gym."), doc("typhoon", "Walang pasok bukas dahil sa bagyo."),
            doc("enroll", "Enrollment begins August 10.")]
    bm = TagalogBM25([d["text"] for d in docs])
    top = bm.rank("may")
    assert [i for i, _ in top] == [0]                                        # 'may' is a real, retrievable term
    tfidf = Retriever(docs).retrieve("may", top_k=3)                         # stop_words='english' deletes 'may'
    assert all(d.score == 0.0 for d in tfidf)


def test_bm25_idf_is_non_negative_even_for_words_in_every_document():
    bm = TagalogBM25(["sa bahay", "sa school", "sa gym"])
    assert all(s >= 0 for s in bm.score("sa")) and all(v >= 0 for v in bm._word.idf.values())


# ------------------------------------------------------------------ BM25: character n-grams
def test_informal_spelling_shares_char_ngrams_with_standard_spelling():
    shared = set(char_ngrams("bukas")) & set(char_ngrams("bkas"))
    assert {"kas", "as>", "kas>"} <= shared


def test_char_channel_recovers_bkas_but_word_channel_alone_cannot():
    texts = ["Walang pasok bukas.", "Enrollment begins August 10."]
    with_chars = TagalogBM25(texts, char_guard="off")                       # unguarded: a lone unseen spelling is recovered
    word_only = TagalogBM25(texts, char_weight=0.0)
    assert [i for i, _ in with_chars.rank("bkas")] == [0]
    assert word_only.rank("bkas") == []
    assert [i for i, _ in word_only.rank("bukas")] == [0]                   # exact spelling still works word-only


def test_guard_still_recovers_an_informal_spelling_inside_a_sentence_with_a_real_word_hit():
    bm = TagalogBM25(["Walang pasok bukas.", "Enrollment begins August 10."])   # default char_guard="query"
    assert [i for i, _ in bm.rank("walang bkas")] == [0]                    # 'walang' is a word hit, so char credit is allowed
    assert bm.char_suppressed("walang bkas") is False


def test_guard_suppresses_the_char_channel_when_the_query_has_no_word_hit_at_all():
    bm = TagalogBM25(["Walang pasok bukas.", "Enrollment begins August 10."])
    assert bm.char_guard == "query"                                         # the default
    assert bm.rank("bkas") == [] and bm.char_suppressed("bkas") is True
    assert [i for i, _ in TagalogBM25(["Walang pasok bukas."], char_guard="off").rank("bkas")] == [0]


def test_invalid_char_guard_is_rejected():
    try:
        TagalogBM25(["x"], char_guard="maybe")
    except ValueError:
        return
    raise AssertionError("bad char_guard accepted")


_XLING_DOCS = [doc("exp", "All classes are suspended tomorrow due to the typhoon."),
               doc("noise", "Passport renewal while walking distance to the bookstore."),
               doc("other", "Enrollment begins August 10.")]
_XLING_Q = "walang pasok bukas"                                              # Tagalog query, English corpus: zero word hits


def _xling_hybrid(char_guard):
    dense = StubDense(_XLING_DOCS, {_XLING_Q: {"exp": 0.80, "noise": 0.60, "other": 0.50}})      # dense ranks the right doc first
    bm = TagalogBM25([d["text"] for d in _XLING_DOCS], char_guard=char_guard)
    return HybridRetriever(_XLING_DOCS, dense, bm25=bm, gate_protect=False), bm


def test_unguarded_char_channel_reproduces_the_m12_4_cross_lingual_regression():
    hy, bm = _xling_hybrid("off")
    assert [_XLING_DOCS[i]["document_id"] for i, _ in bm.rank(_XLING_Q)][:1] == ["noise"]   # incidental n-grams (wal-, pas-) match noise
    assert hy.retrieve(_XLING_Q, top_k=3)[0].document_id == "noise"          # RRF demotes the dense top-1


def test_guard_restores_the_dense_order_for_a_query_with_no_word_hits():
    hy, bm = _xling_hybrid("query")
    assert bm.rank(_XLING_Q) == []
    top = hy.retrieve(_XLING_Q, top_k=3)
    assert [d.document_id for d in top] == ["exp", "noise", "other"] and top[0].bm25_rank is None


def test_numbers_match_as_words_but_get_no_char_grams():
    assert char_ngrams("28") == []
    bm = TagalogBM25(["Exam on September 28", "Exam on September 2028"])
    assert [i for i, _ in bm.rank("28")] == [0]                             # '28' must not half-match '2028'


def test_bm25_rank_omits_zero_overlap_and_breaks_ties_by_index():
    bm = TagalogBM25(["alpha beta", "alpha beta", "zzz"])
    assert [i for i, _ in bm.rank("alpha")] == [0, 1]
    assert bm.rank("") == [] and TagalogBM25([]).rank("alpha") == []


# ------------------------------------------------------------------ Hybrid: RRF arithmetic
def test_rrf_matches_hand_calculation():
    dense = StubDense(ABC, {"q": {"A": 0.90, "B": 0.80, "C": 0.70}})
    bm25 = StubBM25({"q": [1, 2]})                                          # B first, then C; A has no overlap
    hy = HybridRetriever(ABC, dense, bm25=bm25, rrf_k=60, gate_protect=False)
    got = {d.document_id: d for d in hy.retrieve("q", top_k=3)}
    k = 60
    assert close(got["A"].rrf_score, 1 / (k + 1))
    assert close(got["B"].rrf_score, 1 / (k + 2) + 1 / (k + 1))
    assert close(got["C"].rrf_score, 1 / (k + 3) + 1 / (k + 2))
    assert [d.document_id for d in hy.retrieve("q", top_k=3)] == ["B", "C", "A"]
    assert got["A"].bm25_rank is None and got["B"].bm25_rank == 1 and got["C"].dense_rank == 3


def test_merge_of_real_e5_scores_and_real_bm25_ranks():
    docs = [doc("typhoon", "All classes are suspended tomorrow due to the typhoon."),
            doc("enroll", "Second semester enrollment begins August 10 and ends August 20."),
            doc("exam", "The midterm exam is on September 28 in the gym.")]
    dense = E5Retriever(docs, encoder=E5Encoder(model=FakeModel()), relevance_threshold=0.3)
    hy = HybridRetriever(docs, dense)
    res = hy.retrieve("midterm exam september 28", top_k=3)
    assert res[0].document_id == "exam"
    assert res[0].dense_rank == 1 and res[0].bm25_rank == 1 and res[0].rrf_score > res[1].rrf_score


# ------------------------------------------------------------------ Hybrid: dense cosine preserved on doc.score
def test_score_is_the_dense_cosine_and_never_the_rrf_value():
    dense = StubDense(ABC, {"q": {"A": 0.91, "B": 0.55, "C": 0.20}})
    hy = HybridRetriever(ABC, dense, bm25=StubBM25({"q": [0, 1]}))
    for d in hy.retrieve("q", top_k=3):
        assert close(d.score, {"A": 0.91, "B": 0.55, "C": 0.20}[d.document_id])
        assert d.score != d.rrf_score and d.rrf_score < 0.05             # RRF values are ~0.03, cosines are not


def test_score_equals_the_real_dense_retriever_score_for_the_same_query():
    docs = [doc("a", "classes suspended tomorrow typhoon"), doc("b", "enrollment begins august"), doc("c", "exam september gym")]
    dense = E5Retriever(docs, encoder=E5Encoder(model=FakeModel()), relevance_threshold=0.2)
    hy = HybridRetriever(docs, dense)
    want = {d.document_id: d.score for d in dense.retrieve("exam gym september", top_k=3)}
    got = {d.document_id: d.score for d in hy.retrieve("exam gym september", top_k=3)}
    assert all(close(want[k], got[k]) for k in want)


def test_with_variants_the_score_is_the_max_cosine_over_variants():
    dense = StubDense(ABC, {"raw": {"A": 0.30}, "norm": {"A": 0.88}})
    hy = HybridRetriever(ABC, dense, bm25=StubBM25({}))
    top = hy.retrieve("raw", top_k=1, variants=["raw", "norm"])[0]
    assert top.document_id == "A" and close(top.score, 0.88)


# ------------------------------------------------------------------ Hybrid: the gate
def test_gate_protect_makes_the_gate_decision_identical_to_dense_only():
    """BM25 puts the weak document first; protect keeps the document that clears the gate on top."""
    dense = StubDense(ABC, {"q": {"A": 0.90, "B": 0.50, "C": 0.10}})
    bm25 = StubBM25({"q": [1, 2, 0]})
    protected = HybridRetriever(ABC, dense, bm25=bm25, gate_protect=True).retrieve("q", top_k=3)
    assert protected[0].document_id == "A" and protected[0].score >= THRESHOLD


def test_without_gate_protect_bm25_can_demote_a_document_that_dense_would_have_verified():
    """The gate exception, made explicit: this is why gate_protect defaults to True."""
    dense = StubDense(ABC, {"q": {"A": 0.90, "B": 0.50, "C": 0.10}})
    bm25 = StubBM25({"q": [1, 2, 0]})
    free = HybridRetriever(ABC, dense, bm25=bm25, gate_protect=False).retrieve("q", top_k=3)
    assert free[0].document_id == "B" and free[0].score < THRESHOLD


_GATE_DOCS = [doc(f"d{i}", t) for i, t in enumerate([
    "classes suspended tomorrow due to typhoon", "second semester enrollment begins august 10",
    "midterm exam september 28 gym", "library open 24 hours during finals", "walang pasok bukas dahil sa bagyo",
    "id validation at the registrar office", "tuition payment deadline october 15", "graduation rehearsal friday"])]
_GATE_QUERIES = ["classes suspended tomorrow", "wlang pasok bkas", "may exam ba sa sept 28", "library 24 hours", "weather in paris",
                 "tuition deadline", "hnd bukas ang enrollment", "graduation friday", "zzz qqq", "pls wp bukas"]


def test_without_a_normalizer_the_gate_decision_is_identical_to_dense_only():
    dense = E5Retriever(_GATE_DOCS, encoder=E5Encoder(model=FakeModel()), relevance_threshold=0.45)
    hy = HybridRetriever(_GATE_DOCS, dense, gate_protect=True)
    for q in _GATE_QUERIES:
        d_top, h_top = dense.retrieve(q, top_k=1)[0], hy.retrieve(q, top_k=1)[0]
        assert (d_top.score >= 0.45) == (h_top.score >= 0.45), q
        assert close(h_top.score, d_top.score) or h_top.score <= d_top.score + 1e-12, q   # never inflated above the dense max


def test_with_a_normalizer_the_gate_decision_equals_dense_only_over_the_same_variants():
    """Hybrid must not move the gate by itself. Any change vs raw-only dense must come from the variants (normalization)."""
    norm = TaglishNormalizer()
    dense = E5Retriever(_GATE_DOCS, encoder=E5Encoder(model=FakeModel()), relevance_threshold=0.45)
    hy = HybridRetriever(_GATE_DOCS, dense, normalizer=norm, gate_protect=True)
    for q in _GATE_QUERIES:
        best_dense = max(dense.retrieve(v, top_k=1)[0].score for v in norm.variants(q))
        h_top = hy.retrieve(q, top_k=1)[0]
        assert (best_dense >= 0.45) == (h_top.score >= 0.45), q
        assert close(h_top.score, best_dense), q


def test_normalization_can_open_the_gate_for_a_slang_spelling_and_hybrid_reports_it_honestly():
    dense = E5Retriever(_GATE_DOCS, encoder=E5Encoder(model=FakeModel()), relevance_threshold=0.45)
    raw_top = dense.retrieve("wlang pasok bkas", top_k=1)[0]
    hy_top = HybridRetriever(_GATE_DOCS, dense, normalizer=TaglishNormalizer()).retrieve("wlang pasok bkas", top_k=1)[0]
    assert raw_top.score < 0.45 <= hy_top.score and hy_top.document_id == "d4"


def test_claim_verifier_gate_sees_the_cosine_and_still_blocks_below_threshold():
    high = StubDense(ABC, {"strong": {"A": 0.91}, "weak": {"A": 0.80}})
    hy = HybridRetriever(ABC, high, bm25=StubBM25({"strong": [0], "weak": [0]}))
    clf = CountingClassifier()
    ver = ClaimVerifier(hy, clf)
    assert ver._relevance_threshold == THRESHOLD                       # read from the dense retriever through the hybrid
    ok = ver.verify_claim("strong")
    assert ok.evidence_status == RELEVANT_EVIDENCE_FOUND and clf.calls == 1 and close(ok.used_evidence.score, 0.91)
    blocked = ver.verify_claim("weak")
    assert blocked.evidence_status == NO_RELEVANT_EVIDENCE and clf.calls == 1   # classifier NOT called; strong BM25 match cannot override


def test_a_strong_lexical_match_cannot_open_the_cosine_gate():
    """Documents the limitation stated in the module docstring: hybrid repairs ranking, not gate errors."""
    dense = StubDense(ABC, {"q": {"A": 0.60}})
    hy = HybridRetriever(ABC, dense, bm25=StubBM25({"q": [0]}))
    top = hy.retrieve("q", top_k=1)[0]
    assert top.bm25_rank == 1 and top.score < THRESHOLD


# ------------------------------------------------------------------ Hybrid: variants, interface, compatibility
def test_normalizer_variants_recover_a_spelling_the_raw_query_cannot_reach():
    docs = [doc("typhoon", "Walang pasok bukas dahil sa bagyo."), doc("enroll", "Second semester enrollment begins August 10.")]
    dense = E5Retriever(docs, encoder=E5Encoder(model=FakeModel()), relevance_threshold=0.3)
    plain = HybridRetriever(docs, dense)
    with_norm = HybridRetriever(docs, dense, normalizer=TaglishNormalizer())
    raw_top = plain.retrieve("wlang bkas", top_k=1)[0]
    norm_top = with_norm.retrieve("wlang bkas", top_k=1)[0]
    assert raw_top.score == 0.0                                        # fake dense sees no shared word in the raw spelling
    assert norm_top.document_id == "typhoon" and norm_top.score > 0.5  # the normalized variant's cosine is preserved


def test_result_type_and_diagnostics():
    hy = HybridRetriever(ABC, StubDense(ABC, {"q": {"A": 0.9}}), bm25=StubBM25({"q": [0]}))
    top = hy.retrieve("q", top_k=1)[0]
    assert isinstance(top, HybridRetrievedDocument) and isinstance(top, RetrievedDocument)
    assert (top.document_id, top.source_type, top.text) == ("A", "synthetic", "alpha text")


def test_top_k_empty_query_and_empty_corpus():
    hy = HybridRetriever(ABC, StubDense(ABC, {}), bm25=StubBM25({}))
    assert len(hy.retrieve("anything", top_k=2)) == 2
    assert hy.retrieve("   ") == [] and hy.retrieve("") == []
    assert HybridRetriever([], StubDense([], {})).retrieve("q") == []


def test_duplicate_document_ids_are_rejected():
    try:
        HybridRetriever([doc("A", "x"), doc("A", "y")], StubDense([], {}))
    except ValueError:
        return
    raise AssertionError("duplicate ids accepted")


def test_passage_selector_can_use_the_hybrid_as_its_passage_source():
    docs = [doc("a", "Classes are suspended tomorrow. Students should stay home."), doc("b", "Enrollment begins August 10 and ends August 20.")]
    enc = E5Encoder(model=FakeModel())
    dense = E5Retriever(docs, encoder=enc, relevance_threshold=0.2, score_mode="passage")
    hy = HybridRetriever(docs, dense)
    assert hy.score_mode == "passage" and hy.cached_passages(docs[0]["text"]) is not None
    sel = PassageSelector(enc, passage_source=hy)
    chosen = sel.select("classes suspended tomorrow", hy.retrieve("classes suspended tomorrow", top_k=2))
    assert chosen and chosen[0].document_id == "a"


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as exc:      # noqa: BLE001
                failed += 1; print("FAIL", name, repr(exc))
    print(f"\n{'FAILED' if failed else 'ALL PASSED'}: {failed} failure(s)")
    sys.exit(1 if failed else 0)
