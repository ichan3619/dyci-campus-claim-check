"""
Dense document retrieval with multilingual-e5-small (Milestone 12.2).

Responsibility: given a claim and a corpus, rank documents by semantic relevance
and return the top-k as the SAME `RetrievedDocument` objects the TF-IDF `Retriever`
returns. It knows nothing about NLI, verdicts, FastAPI or the frontend.

WHY e5-small (and what that claim is NOT)
    Selected in M12.1 as an ENGINEERING choice from small measurements (34 + 24 positive
    queries, 4 + 8 negatives, authored in-house). It was best or tied-best on those sets.
    It is not shown to be universally better than TF-IDF, and Tagalog support was only
    demonstrated empirically on those few queries, not documented on the model card.

THE SCORE SCALE IS DIFFERENT -- READ THIS BEFORE TOUCHING THE THRESHOLD
    e5 cosine scores are compressed: in M12.1 correct documents scored ~0.86 (median) while
    unrelated queries still scored ~0.78. TF-IDF's 0.15 gate therefore blocks NOTHING here.
    A relevance threshold is required, it is model- and corpus-specific, and it must be
    recalibrated whenever the corpus changes. The only value ever measured is 0.837, picked
    on a 34-positive/4-negative set and then checked on 24/8 held-out queries (usable 71%,
    2/8 negatives still leaked). It is NOT a universal e5 threshold. We expose it as a
    configurable PROVISIONAL default and warn once at runtime when it is used.
    Order of precedence: constructor arg > env CAMPUS_VERIFY_E5_THRESHOLD > provisional.

e5 usage rules honoured here: "query: " prefix on claims, "passage: " prefix on documents,
L2-normalised embeddings (so a dot product is the cosine), inference only, CPU by default,
no network call after the model files are cached locally.
"""
import logging
import os
from collections import OrderedDict
from typing import List, Optional, Sequence

import numpy as np

from src.retrieval.passages import split_into_passages
from src.retrieval.retriever import RetrievedDocument

logger = logging.getLogger(__name__)

DEFAULT_E5_MODEL_NAME = "intfloat/multilingual-e5-small"
E5_MODEL_ENV_VAR = "CAMPUS_VERIFY_E5_MODEL"            # model name OR local directory path
E5_THRESHOLD_ENV_VAR = "CAMPUS_VERIFY_E5_THRESHOLD"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Observed in M12.1 / M12.2 on very small evaluation sets. PROVISIONAL. See module docstring.
PROVISIONAL_E5_THRESHOLD = 0.837            # score_mode="document"
PROVISIONAL_E5_PASSAGE_THRESHOLD = 0.833    # score_mode="passage" (picked on the same main set, same protocol)
VALID_SCORE_MODES = ("document", "passage")
_warned_provisional = False


class E5Encoder:
    """
    Loads the embedding model ONCE and applies the e5 prefixes. Shared by the document
    retriever and the passage selector so the model is never loaded twice.

    `model` may be injected (any object with .encode(list[str], normalize_embeddings=True)
    returning a 2-D array) -- used by the unit tests so they need no download and no torch.
    """

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu", model=None):
        self.model_name = model_name or os.environ.get(E5_MODEL_ENV_VAR) or DEFAULT_E5_MODEL_NAME
        if model is not None:
            self._model = model
        else:
            try:
                from sentence_transformers import SentenceTransformer   # optional dependency
            except ImportError as exc:                                  # pragma: no cover
                raise ImportError(
                    "E5 retrieval needs the optional 'sentence-transformers' package "
                    "(see requirements-e5.txt). The TF-IDF Retriever works without it."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=device)

    def _encode(self, texts: Sequence[str]):
        return self._model.encode(list(texts), normalize_embeddings=True)

    # One request embeds the same claim twice (document retrieval, then passage selection). Embeddings are
    # deterministic, so recent query vectors are reused instead of recomputed. Bounded, so memory stays flat.
    QUERY_CACHE_SIZE = 256

    def encode_queries(self, texts: Sequence[str]):
        return self._cached(texts, "_query_cache", QUERY_PREFIX, self.QUERY_CACHE_SIZE)

    # Rebuilding the index after new posts arrive re-embeds every passage; cached vectors make that cost only the
    # new passages. Bounded so arbitrary pages fetched by /verify/url cannot grow memory without limit.
    PASSAGE_CACHE_SIZE = 4096

    def encode_passages(self, texts: Sequence[str]):
        return self._cached(texts, "_passage_cache", PASSAGE_PREFIX, self.PASSAGE_CACHE_SIZE)

    def _cached(self, texts: Sequence[str], attr: str, prefix: str, size: int):
        cache = self.__dict__.setdefault(attr, OrderedDict())
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            for t, vec in zip(missing, self._encode([prefix + t for t in missing])):
                cache[t] = vec
        for t in texts:
            cache.move_to_end(t)
        while len(cache) > size:
            cache.popitem(last=False)
        return np.stack([cache[t] for t in texts]) if texts else self._encode([])


class E5Retriever:
    """
    Drop-in alternative to `Retriever`: same constructor documents, same `retrieve()`
    signature and return type. Document embeddings are computed once at construction.
    """

    def __init__(self, documents: List[dict], encoder: Optional[E5Encoder] = None,
                 model_name: Optional[str] = None, relevance_threshold: Optional[float] = None,
                 score_mode: str = "document"):
        """
        score_mode:
          "document" (default, the configuration measured in M12.1): a document's score is the cosine
              between the claim and the WHOLE-document embedding.
          "passage": a document's score is the highest cosine between the claim and any one of its
              sentences. Motivated by M12.2: claims about a detail inside a long post scored below the
              gate at document level (6 of 9 deep-fact claims blocked) but not at passage level in the
              probe. Candidate design, validated only on small sets; thresholds are NOT interchangeable
              between modes.
        """
        if score_mode not in VALID_SCORE_MODES:
            raise ValueError(f"score_mode must be one of {VALID_SCORE_MODES}")
        self._documents = documents
        self.score_mode = score_mode
        self.encoder = encoder or E5Encoder(model_name)
        self._doc_vectors = None
        self._passage_cache = {}          # document text -> (passages, embeddings); passage mode only
        if documents:
            if score_mode == "document":
                self._doc_vectors = self.encoder.encode_passages([d["text"] for d in documents])
            else:
                for d in documents:
                    ps = split_into_passages(d["text"])
                    self._passage_cache[d["text"]] = (ps, self.encoder.encode_passages(ps) if ps else None)
        self.default_relevance_threshold = self._resolve_threshold(relevance_threshold, score_mode)

    def cached_passages(self, text: str):
        """(passages, embeddings) for a document text, or None (document mode / unknown text)."""
        return self._passage_cache.get(text)

    @staticmethod
    def _resolve_threshold(explicit: Optional[float], score_mode: str = "document") -> float:
        global _warned_provisional
        if explicit is not None:
            return float(explicit)
        env = os.environ.get(E5_THRESHOLD_ENV_VAR)
        if env:
            return float(env)
        provisional = PROVISIONAL_E5_PASSAGE_THRESHOLD if score_mode == "passage" else PROVISIONAL_E5_THRESHOLD
        if not _warned_provisional:
            logger.warning(
                "E5Retriever is using the PROVISIONAL relevance threshold %.3f (score_mode=%s) measured on a "
                "very small evaluation set. Recalibrate it on a larger corpus and negative set; set "
                "%s or pass relevance_threshold to override.", provisional, score_mode, E5_THRESHOLD_ENV_VAR)
            _warned_provisional = True
        return provisional

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        if not query or not query.strip() or not self._documents:
            return []
        if self.score_mode == "document" and self._doc_vectors is None:
            return []
        q = self.encoder.encode_queries([query])[0]
        if self.score_mode == "document":
            sims = self._doc_vectors @ q
        else:   # best single passage per document (documents with no passages score 0)
            sims = []
            for d in self._documents:
                ps, vecs = self._passage_cache[d["text"]]
                sims.append(float((vecs @ q).max()) if ps else 0.0)
        # Negative cosine is clamped to 0, matching the M12.1 evaluation harness exactly.
        scored = sorted(zip(self._documents, (float(max(0.0, s)) for s in sims)),
                        key=lambda p: p[1], reverse=True)[:top_k]
        return [RetrievedDocument(document_id=d["document_id"], score=s, text=d["text"],
                                  source_type=d["source_type"], source_url=d.get("source_url", ""))
                for d, s in scored]
