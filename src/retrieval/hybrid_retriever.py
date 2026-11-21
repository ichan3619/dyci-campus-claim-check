"""
HybridRetriever -- dense (E5) + sparse (TagalogBM25) retrieval fused with Reciprocal Rank Fusion.

THE ONE RULE THIS MODULE EXISTS TO KEEP
    `RetrievedDocument.score` is ALWAYS the DENSE COSINE (the largest cosine over the query variants), never an RRF or
    BM25 value. ClaimVerifier compares retrieved[0].score with the relevance threshold and builds its NLI candidate
    list from `d.score >= threshold`; both keep their M12.2 meaning. Fusion changes only the ORDER of the documents.
    RRF / BM25 numbers are reported on separate attributes (HybridRetrievedDocument.rrf_score, .bm25_score ...).

CONSEQUENCE YOU SHOULD KNOW ABOUT
    With a cosine gate, a document below the threshold is blocked no matter how well BM25 matches it. Hybrid retrieval
    can therefore repair RANKING errors among documents that clear the gate, but it cannot repair GATE errors
    (a correct document whose cosine is below the threshold). Repairing those needs a different gate, for example
    "dense >= t OR strong lexical match"; bm25_score / bm25_rank are exposed so that can be evaluated, but it is NOT
    done here, because it would change the gate semantics validated in M12.2.

gate_protect (default True)
    Documents are ordered by RRF, then the ones whose dense cosine clears the gate are moved ahead of the ones that do
    not (relative RRF order kept inside each group). Result: for the SAME query variants, hybrid's top-1 clears the gate
    if and only if the pure dense top-1 does, so BM25/RRF cannot move the gate; only the document chosen among those
    that clear can change. (A normalizer can still open the gate for a slang query, because the dense cosine is taken as
    the max over the raw and normalized variants. That is a normalization effect, and the eval script reports it as one.) Set False to measure what BM25 does when it is allowed to promote a sub-threshold
    document (then the verifier will block a claim that dense-only would have verified).

QUERY VARIANTS
    `normalizer` is any object with .variants(text) -> list[str] (TaglishNormalizer). Each variant is sent to the dense
    retriever and to BM25 and contributes its own ranked lists to the RRF sum; a document's dense score is the max over
    variants. The verifier's NLI step still receives the claim exactly as the caller passed it.

COST
    The dense retriever is asked for ALL documents (top_k = corpus size) so every document has a real cosine. That is no
    extra work for E5Retriever (it already scores every document) and is what makes score preservation possible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.retrieval.bm25 import TagalogBM25
from src.retrieval.retriever import RetrievedDocument

DEFAULT_RRF_K = 60   # the value from the original RRF paper. With a corpus of ~15 documents a smaller k (5-20) is worth trying.


@dataclass
class HybridRetrievedDocument(RetrievedDocument):
    """A RetrievedDocument (so every existing consumer works) plus fusion diagnostics. `score` stays the dense cosine."""
    rrf_score: float = 0.0
    dense_rank: Optional[int] = None      # best dense rank over the query variants (1 = best)
    bm25_rank: Optional[int] = None       # best BM25 rank over the variants; None = no lexical overlap
    bm25_score: float = 0.0               # best combined BM25 score over the variants (a ranking aid, not a cosine)


class HybridRetriever:
    def __init__(self, documents: List[dict], dense, bm25=None, normalizer=None, rrf_k: int = DEFAULT_RRF_K,
                 gate_protect: bool = True, relevance_threshold: Optional[float] = None):
        """
        dense: an E5Retriever (or anything with .retrieve(query, top_k) returning RetrievedDocument with cosine scores).
        bm25: object with .rank(query) -> [(doc_index, score)]; defaults to TagalogBM25 over the same documents.
        relevance_threshold: used only by gate_protect and exposed as default_relevance_threshold. Defaults to the dense
            retriever's own threshold so the verifier's gate is unchanged.
        """
        ids = [d["document_id"] for d in documents]
        if len(set(ids)) != len(ids):
            raise ValueError("document_id values must be unique")
        if rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")
        self._documents = documents
        self._index_by_id = {i: n for n, i in enumerate(ids)}
        self._dense = dense
        self._bm25 = bm25 if bm25 is not None else TagalogBM25([d["text"] for d in documents])
        self._normalizer = normalizer
        self.rrf_k = rrf_k
        self.gate_protect = gate_protect
        if relevance_threshold is not None:
            self.default_relevance_threshold = float(relevance_threshold)
        else:
            self.default_relevance_threshold = float(getattr(dense, "default_relevance_threshold", 0.0))
        # Pass-throughs so PassageSelector / ClaimVerifier treat this like the dense retriever it wraps.
        self.score_mode = getattr(dense, "score_mode", "document")
        self.encoder = getattr(dense, "encoder", None)

    def cached_passages(self, text: str):
        fn = getattr(self._dense, "cached_passages", None)
        return fn(text) if fn else None

    def _queries(self, query: str, variants: Optional[Sequence[str]]) -> List[str]:
        raw = list(variants) if variants else (self._normalizer.variants(query) if self._normalizer else [query])
        out: List[str] = []
        for q in raw:
            if q and q.strip() and q not in out:
                out.append(q)
        return out or [query]

    def retrieve(self, query: str, top_k: int = 3, variants: Optional[Sequence[str]] = None) -> List[HybridRetrievedDocument]:
        if not query or not query.strip() or not self._documents:
            return []
        n = len(self._documents)
        rrf: Dict[str, float] = {}
        dense_cos: Dict[str, float] = {}
        dense_rank: Dict[str, int] = {}
        bm25_rank: Dict[str, int] = {}
        bm25_score: Dict[str, float] = {}

        for q in self._queries(query, variants):
            for pos, hit in enumerate(self._dense.retrieve(q, top_k=n), start=1):
                rrf[hit.document_id] = rrf.get(hit.document_id, 0.0) + 1.0 / (self.rrf_k + pos)
                dense_cos[hit.document_id] = max(dense_cos.get(hit.document_id, 0.0), float(hit.score))
                dense_rank[hit.document_id] = min(dense_rank.get(hit.document_id, pos), pos)
            for pos, (idx, s) in enumerate(self._bm25.rank(q), start=1):
                did = self._documents[idx]["document_id"]
                rrf[did] = rrf.get(did, 0.0) + 1.0 / (self.rrf_k + pos)
                bm25_rank[did] = min(bm25_rank.get(did, pos), pos)
                bm25_score[did] = max(bm25_score.get(did, 0.0), s)

        order = sorted(rrf, key=lambda i: (-rrf[i], -dense_cos.get(i, 0.0), self._index_by_id[i]))
        if self.gate_protect:
            t = self.default_relevance_threshold
            order = [i for i in order if dense_cos.get(i, 0.0) >= t] + [i for i in order if dense_cos.get(i, 0.0) < t]

        out: List[HybridRetrievedDocument] = []
        for did in order[:top_k]:
            d = self._documents[self._index_by_id[did]]
            out.append(HybridRetrievedDocument(
                document_id=did, score=dense_cos.get(did, 0.0), text=d["text"], source_type=d["source_type"],
                source_url=d.get("source_url", ""), rrf_score=rrf[did], dense_rank=dense_rank.get(did),
                bm25_rank=bm25_rank.get(did), bm25_score=bm25_score.get(did, 0.0)))
        return out
