"""
Passage (sentence) selection between document retrieval and NLI (Milestone 12.2).

Why this exists -- M12.1 evidence, not an assumption
    The NLI model labelled supported claims INSUFFICIENT_EVIDENCE at 0.99+ confidence when
    handed a whole multi-sentence Facebook post (3 of 3 supported pairs), but gave the right
    label on the right sentence alone (5 of 5 pairs). Dev documents are single sentences, which
    hid the problem. That probe covered five claim/evidence pairs on two posts, all with the key
    fact in the FIRST sentence; picking the right sentence in general is exactly what this layer
    must now be tested on.

Responsibility: split candidate documents into short passages, score each passage against the
claim with the e5 representation, return the best passage(s). It does NOT decide any verdict and
does NOT rerank documents (the document stage stays separate and unchanged).

Deliberately NOT done (smallest testable layer): no cross-encoder reranking, no learned
selection, no multi-hop, no claim rewriting. top_n=1 by default because joining several sentences
rebuilds the multi-sentence premise that M12.1 showed NLI struggles with.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.retrieval.e5_retriever import E5Encoder
from src.retrieval.passages import split_into_passages   # noqa: F401  (re-exported for callers/tests)
from src.retrieval.retriever import RetrievedDocument

@dataclass
class SelectedPassage:
    document_id: str
    text: str
    score: float              # e5 cosine between claim and THIS passage (ranking use only)
    passage_index: int        # position of the passage inside its document
    document_score: float     # the document-retrieval score of the parent document
    source_type: str
    source_url: str
    # M12.2c: the `context_before` preceding sentences of the same document, kept SEPARATE from `text`
    # so the ranking/provenance of the selected passage is unchanged. Empty unless context_before > 0.
    context_before_text: str = ""

    @property
    def premise(self) -> str:
        """What NLI receives: optional preceding context + the selected passage."""
        return (self.context_before_text + " " + self.text).strip()


class PassageSelector:
    def __init__(self, encoder: E5Encoder, top_n: int = 1, min_words: int = 3, passage_source=None,
                 context_before: int = 0):
        """
        context_before (opt-in, default 0 = unchanged behaviour): prepend this many preceding sentences
        of the same document to the premise. Motivation (M12.2 premise probe, 1 case): the sentence
        "This is in observance of ..." was INSUFFICIENT on its own because "This" has no antecedent, and
        SUPPORTED once the previous sentence was included, on both deberta-v3-small and -base, with three
        control claims unaffected. It does NOT help person mismatch (X02), and every added sentence moves the
        premise back toward the multi-sentence form M12.1 showed NLI handles poorly. Evaluate before enabling.

        passage_source: optional object with .cached_passages(text) -> (passages, embeddings) | None
        (an E5Retriever in score_mode="passage"). When given, passages are not embedded a second time.
        """
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if context_before < 0:
            raise ValueError("context_before must be >= 0")
        self._context_before = context_before
        self._encoder, self._top_n, self._min_words = encoder, top_n, min_words
        self._source = passage_source
        self._cache: Dict[str, tuple] = {}     # document text -> (passages, embeddings)

    def _passages_for(self, doc: RetrievedDocument):
        hit = self._cache.get(doc.text)
        if hit is None and self._source is not None:
            hit = self._source.cached_passages(doc.text)
            if hit is not None:
                self._cache[doc.text] = hit
        if hit is None:
            passages = split_into_passages(doc.text, self._min_words)
            hit = (passages, self._encoder.encode_passages(passages) if passages else None)
            self._cache[doc.text] = hit
        return hit

    def select(self, claim: str, candidates: Sequence[RetrievedDocument]) -> List[SelectedPassage]:
        """Best `top_n` passages across ALL candidate documents, best first. [] if none."""
        if not claim or not claim.strip() or not candidates:
            return []
        q = self._encoder.encode_queries([claim])[0]
        scored: List[SelectedPassage] = []
        for doc in candidates:
            passages, vecs = self._passages_for(doc)
            if not passages:
                continue
            for i, (p, s) in enumerate(zip(passages, vecs @ q)):
                ctx = " ".join(passages[max(0, i - self._context_before):i]) if self._context_before else ""
                scored.append(SelectedPassage(doc.document_id, p, float(s), i, doc.score,
                                              doc.source_type, doc.source_url, ctx))
        scored.sort(key=lambda sp: sp.score, reverse=True)
        return scored[: self._top_n]

    @property
    def context_before(self) -> int:
        return self._context_before

    def windows_of(self, doc: RetrievedDocument) -> List[str]:
        """Every premise window of `doc`, in document order, built exactly like SelectedPassage.premise
        (the `context_before` preceding sentences + the sentence). Read-only; used by the sliding premise mode."""
        passages, _ = self._passages_for(doc)
        return [(" ".join(passages[max(0, i - self._context_before):i]) + " " + p).strip() if self._context_before else p
                for i, p in enumerate(passages)]

    @staticmethod
    def premise_from(selected: Sequence[SelectedPassage]) -> Optional[str]:
        """NLI premise. One passage -> that passage. Several -> joined in (doc, position) order."""
        if not selected:
            return None
        ordered = sorted(selected, key=lambda sp: (sp.document_id, sp.passage_index))
        return " ".join(sp.premise for sp in ordered)
