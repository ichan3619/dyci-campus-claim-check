"""src/retrieval/reranker.py and CAMPUS_VERIFY_RERANKER. Fakes only: no model is downloaded or loaded."""
from src.retrieval.reranker import RERANK_CANDIDATES, CrossEncoderReranker, RerankingRetriever
from src.retrieval.retriever import RetrievedDocument


def _doc(i, score):
    return RetrievedDocument(document_id=i, score=score, text=f"text of {i}", source_type="facebook", source_url="")


class _E5:
    default_relevance_threshold = 0.833

    def __init__(self, n=12):
        self.docs = [_doc(f"d{k}", round(0.9 - k * 0.01, 3)) for k in range(n)]
        self.asked = None

    def retrieve(self, query, top_k=3):
        self.asked = top_k
        return self.docs[:top_k]

    def cached_passages(self, text):
        return ("passages", text)


class _Model:
    """Scores 'text of dK' as K for the listed favourites, else 0."""
    def __init__(self, favourite):
        self.favourite = favourite

    def predict(self, pairs, batch_size=8):
        return [self.favourite.get(t.split()[-1], 0.0) for _, t in pairs]


def _rr(favourite, base=None):
    return RerankingRetriever(base or _E5(), CrossEncoderReranker(model=_Model(favourite)))


def test_reranks_e5s_top_ten_and_returns_top_k_in_reranker_order():
    base = _E5()
    out = _rr({"d7": 3.0, "d2": 2.0, "d9": 1.0}, base).retrieve("q", top_k=3)
    assert base.asked == RERANK_CANDIDATES == 10
    assert [d.document_id for d in out] == ["d7", "d2", "d9"]


def test_scores_stay_e5_scores_so_the_gate_rule_is_unchanged():
    out = _rr({"d7": 3.0}).retrieve("q", top_k=3)
    assert out[0].document_id == "d7" and out[0].score == 0.83          # d7's own E5 score, not the reranker's


def test_documents_beyond_e5s_top_ten_are_never_returned_and_ties_keep_e5_order():
    out = _rr({"d11": 9.0}).retrieve("q", top_k=3)                       # d11 is E5 rank 12
    assert [d.document_id for d in out] == ["d0", "d1", "d2"]


def test_passages_and_threshold_come_from_e5_and_tiny_lists_pass_through():
    rr = _rr({})
    assert rr.default_relevance_threshold == 0.833 and rr.cached_passages("t") == ("passages", "t")
    assert [d.document_id for d in _rr({}, _E5(n=1)).retrieve("q")] == ["d0"]


def test_setting_defaults_off_and_build_verifier_wraps_only_when_given_a_reranker():
    from src.api.pipeline import Settings, build_reranker, build_verifier
    assert Settings().reranker is False and build_reranker(Settings()) is None

    class _Enc:
        def encode_passages(self, texts):
            import numpy as np
            return np.ones((len(texts), 2))
    docs = [{"document_id": "a", "text": "Classes are suspended on Monday.", "source_type": "facebook", "source_url": ""}]
    plain, _ = build_verifier(docs, object(), _Enc(), Settings())
    wrapped, _ = build_verifier(docs, object(), _Enc(), Settings(), reranker=CrossEncoderReranker(model=_Model({})))
    assert type(plain).__name__ == "E5Retriever" and isinstance(wrapped, RerankingRetriever)


def test_health_reports_the_reranker_off_by_default(api_settings):
    from fastapi.testclient import TestClient
    api_main = api_settings()
    with TestClient(api_main.app) as c:
        assert c.get("/health").json()["reranker"] == "off"
