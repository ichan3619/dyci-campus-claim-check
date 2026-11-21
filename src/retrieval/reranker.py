"""
Retrieval reranking (round 2, T6; opt-in via CAMPUS_VERIFY_RERANKER, default off).

RerankingRetriever wraps the E5 retriever and changes ONE thing: the ORDER of the documents it returns. It takes E5's
top RERANK_CANDIDATES documents, scores each (query, whole announcement) pair with a multilingual cross-encoder
(bge-reranker-v2-m3, pinned revision), and returns the top_k in reranker order. Everything else is passed through:

    * each returned RetrievedDocument keeps its E5 score (the reranker score is not written into it), so the
      relevance gate, which tests the FIRST returned document's score, applies the same rule to the reranked list;
    * cached_passages() and default_relevance_threshold come from the E5 retriever, so passage selection runs
      unchanged on the returned documents (it picks the best E5 passage among the documents that pass the gate, so
      it can still choose a document other than the reranker's first).
Ties in reranker score keep E5 order. Fewer than two candidates are returned as E5 returned them.
"""
from typing import List, Optional, Sequence

from src.retrieval.retriever import RetrievedDocument

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
RERANK_CANDIDATES = 10
RERANKER_MAX_LENGTH = 512


class CrossEncoderReranker:
    """Scores (query, text) pairs; higher = more relevant. `model` may be injected (tests)."""

    def __init__(self, model_name: str = RERANKER_MODEL, revision: str = RERANKER_REVISION, model=None):
        if model is None:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_name, revision=revision, max_length=RERANKER_MAX_LENGTH)
        self._model = model
        self.label = f"{model_name.split('/')[-1]}@{revision[:8]}"

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.predict([(query, t) for t in texts], batch_size=8)]


class RerankingRetriever:
    def __init__(self, base, reranker: CrossEncoderReranker, candidates: int = RERANK_CANDIDATES):
        self._base, self._reranker, self._candidates = base, reranker, candidates
        self.default_relevance_threshold: Optional[float] = getattr(base, "default_relevance_threshold", None)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        candidates = self._base.retrieve(query, top_k=max(top_k, self._candidates))
        if len(candidates) < 2:
            return candidates[:top_k]
        scores = self._reranker.score(query, [d.text for d in candidates])
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return [candidates[i] for i in order][:top_k]

    def cached_passages(self, text: str):
        return self._base.cached_passages(text)
