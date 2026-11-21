"""
Structural type for anything that can act as the document-retrieval stage.

The existing TF-IDF `Retriever` and the M12.2 `E5Retriever` both satisfy this
Protocol WITHOUT inheriting from it (duck typing), so nothing about the
original Retriever had to change. ClaimVerifier only ever calls `retrieve()`,
plus (optionally) reads `default_relevance_threshold` -- see verifier.py.
"""
from typing import List, Optional, Protocol

from src.retrieval.retriever import RetrievedDocument


class DocumentRetriever(Protocol):
    # Optional: a retriever whose score scale differs from TF-IDF's can publish a
    # starting-point gate here. ClaimVerifier uses it only when the caller did not
    # pass an explicit relevance_threshold. None / missing -> TF-IDF dev constant.
    default_relevance_threshold: Optional[float]

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]: ...
