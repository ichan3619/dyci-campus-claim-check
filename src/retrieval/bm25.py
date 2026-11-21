"""
TagalogBM25 -- sparse retrieval for short, informal Tagalog / Taglish claims.  Stdlib only.

WHY NOT THE EXISTING TF-IDF RETRIEVER FOR THIS
    src/retrieval/retriever.py builds TfidfVectorizer(stop_words="english"). sklearn's English stop list contains
    'may', 'no', 'at', 'to', 'so' -- all common Tagalog / Taglish words ('may' = there is / has). A query such as
    "may exam ba sa sept 28" loses its existential before scoring. This module removes NO words by default; BM25's
    IDF is what down-weights function words, and only as far as the corpus statistics justify.

TWO CHANNELS, ONE SCORE
    word channel  exact word match. Keeps every token, including short function words and numbers.
    char channel  character 3-5-grams of each word, padded with boundary markers (<bukas>). Gives partial credit to
                  informal spellings that share sub-word pieces: 'bkas' and 'bukas' share 'kas', 'as>' and 'kas>'.
                  Purely numeric tokens get no char-grams (so '28' cannot half-match '2028').
    Each channel is scored with Okapi BM25 (non-negative Lucene-style IDF), divided by its own maximum over the
    corpus for that query, then combined as word_weight * word + char_weight * char. The default weights
    (1.0 / 0.5) are UNTUNED starting values, not measured optima.

CHAR GUARD (char_guard="query", the default)
    M12.4 found that for a query with NO whole-word match anywhere in the corpus (e.g. a Tagalog query against English
    announcements), the char channel still "matches" unrelated documents through incidental n-gram overlap, and RRF then
    pushed the correct dense top-1 from rank 1 to rank 12. With the guard, if the query has zero word-channel hits the char
    channel is switched off for that query, rank() returns [] and a hybrid falls back to the dense order. A query with at
    least one word hit still gets char credit, so an informal spelling inside a sentence ("walang bkas") is still recovered.
    A query that is ONLY an unseen spelling ("bkas") is not; that is what TaglishNormalizer variants are for.
    Set char_guard="off" for the unguarded Sprint 2 behaviour. The guard is motivated by one dev case; it has not been
    validated on an independent set.

WHAT THIS SCORE IS (AND IS NOT)
    The combined score is only meaningful as a ranking. It is NOT a cosine, is not comparable across queries, and must
    never be compared with a relevance threshold. HybridRetriever uses only the RANK from this class.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)


def word_tokens(text: str, stopwords: FrozenSet[str] = frozenset()) -> List[str]:
    """Lowercased NFKC word tokens. No stop-word removal unless `stopwords` is passed explicitly."""
    folded = unicodedata.normalize("NFKC", text or "").lower()
    return [w for w in _TOKEN_RE.findall(folded) if w not in stopwords]


def char_ngrams(word: str, sizes: Sequence[int] = (3, 4, 5)) -> List[str]:
    """Boundary-padded character n-grams of ONE word. Purely numeric words yield none."""
    if not word or word.isdigit():
        return []
    padded = f"<{word}>"
    grams: List[str] = []
    for n in sizes:
        if len(padded) >= n:
            grams.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
    return grams


class _Channel:
    """Okapi BM25 over one kind of term. Postings are built once at construction."""

    def __init__(self, doc_terms: List[List[str]], k1: float, b: float):
        self.k1, self.b = k1, b
        self.n_docs = len(doc_terms)
        self.doc_len = [len(t) for t in doc_terms]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        for i, terms in enumerate(doc_terms):
            for term, tf in Counter(terms).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.idf = {t: math.log(1.0 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def score(self, query_terms: Iterable[str]) -> List[float]:
        scores = [0.0] * self.n_docs
        if not self.avgdl:
            return scores
        for term in set(query_terms):
            for i, tf in self.postings.get(term, ()):
                norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += self.idf[term] * tf * (self.k1 + 1.0) / (tf + norm)
        return scores


class TagalogBM25:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75,
                 ngram_sizes: Sequence[int] = (3, 4, 5), word_weight: float = 1.0, char_weight: float = 0.5,
                 stopwords: FrozenSet[str] = frozenset(), char_guard: str = "query"):
        if char_guard not in ("query", "off"):
            raise ValueError('char_guard must be "query" or "off"')
        self.char_guard = char_guard
        if word_weight < 0 or char_weight < 0 or (word_weight == 0 and char_weight == 0):
            raise ValueError("weights must be >= 0 and not both zero")
        self.ngram_sizes = tuple(ngram_sizes)
        self.word_weight, self.char_weight = word_weight, char_weight
        self.stopwords = frozenset(stopwords)
        word_docs = [word_tokens(t, self.stopwords) for t in texts]
        self._word = _Channel(word_docs, k1, b)
        self._char = _Channel([self._grams(ws) for ws in word_docs], k1, b)
        self.n_docs = len(word_docs)

    def _grams(self, words: Iterable[str]) -> List[str]:
        return [g for w in words for g in char_ngrams(w, self.ngram_sizes)]

    def scores_by_channel(self, query: str) -> Tuple[List[float], List[float]]:
        words = word_tokens(query, self.stopwords)
        return self._word.score(words), self._char.score(self._grams(words))

    def char_suppressed(self, query: str) -> bool:
        """True when the guard switches the char channel off for this query (no word-channel hit in the whole corpus)."""
        if self.char_guard != "query":
            return False
        w, _ = self.scores_by_channel(query)
        return not any(s > 0.0 for s in w)

    def score(self, query: str) -> List[float]:
        """Combined score per document, in corpus order. 0.0 means no lexical overlap at all."""
        w, c = self.scores_by_channel(query)
        if self.char_guard == "query" and not any(s > 0.0 for s in w):
            c = [0.0] * len(c)
        wmax, cmax = max(w, default=0.0), max(c, default=0.0)
        return [(self.word_weight * (ws / wmax if wmax else 0.0)) + (self.char_weight * (cs / cmax if cmax else 0.0))
                for ws, cs in zip(w, c)]

    def rank(self, query: str, top_n: int = 0) -> List[Tuple[int, float]]:
        """[(doc_index, score)] best first. Documents with no overlap are NOT listed (they have no rank).
        Ties break by lower index so results are deterministic."""
        scored = [(i, s) for i, s in enumerate(self.score(query)) if s > 0.0]
        scored.sort(key=lambda p: (-p[1], p[0]))
        return scored[:top_n] if top_n else scored
