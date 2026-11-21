"""
Standalone information retrieval for the campus corpus.

Responsibility of this module, and ONLY this module:

    Given a claim (the query) and a set of campus evidence documents
    (the corpus), rank the documents by how RELEVANT they are to the
    claim, and return the top-k with similarity scores.

This module does NOT decide whether a claim is true, false, or anything
else. It has no concept of "SUPPORTED" or "CONTRADICTED" -- that
vocabulary belongs to the classification stage, built in a later
milestone. This module only ever answers "how similar/related is this
text to that text?"

It knows nothing about FastAPI, the frontend, HTTP, or any ML classifier.
It can be imported and used from a plain Python script, which is exactly
what scripts/test_retrieval.py does below.
"""

from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------------------
# DEVELOPMENT-ONLY THRESHOLD
# --------------------------------------------------------------------------
# This value is NOT scientifically justified. It exists only so we can show
# a human-readable hint in the CLI demonstration (Step 6) that a score is
# probably too low to mean anything. It is intentionally NOT used inside
# the Retriever class itself, and it is NOT used to filter or hide results.
#
# The real threshold -- the one used to decide "retrieval found nothing
# useful, go straight to INSUFFICIENT EVIDENCE" -- must eventually be
# chosen by sweeping candidate threshold values against a labeled
# retrieval-evaluation set (claim -> correct evidence document) and
# picking the value that best balances finding real matches against
# correctly recognizing when there is no match. That experiment happens
# in a later milestone, once we have more labeled data to sweep against.
DEVELOPMENT_ONLY_RELEVANCE_HINT_THRESHOLD = 0.15


@dataclass
class RetrievedDocument:
    """
    One ranked search result.

    Fields:
        document_id: a stable identifier for this evidence document.
            We use the record's `group_id` from Milestone 1's schema,
            because group_id already represents "one real-world
            announcement/event" -- exactly what one retrievable document
            should be (see corpus-building note below).
        score: the cosine similarity between the query and this
            document's TF-IDF vector. Range is 0.0 to 1.0 in practice
            for TF-IDF vectors (both are non-negative). HIGHER MEANS MORE
            TOPICALLY SIMILAR. IT DOES NOT MEAN "MORE TRUE."
        text: the evidence text itself, so it can be shown to the user
            or passed to a future classifier.
        source_type: where this evidence came from (e.g. "synthetic",
            "official_site"), carried over from Milestone 1's schema.
        source_url: link to the original source, if any.
    """
    document_id: str
    score: float
    text: str
    source_type: str
    source_url: str


class Retriever:
    """
    A TF-IDF + cosine-similarity retriever over a fixed corpus of campus
    evidence documents.

    Usage:
        retriever = Retriever(documents)   # documents: see build_corpus()
        results = retriever.retrieve("Are classes suspended tomorrow?", top_k=3)

    Why a class, not just a function? Because building the TF-IDF
    vectorizer and vectorizing every document in the corpus is
    comparatively expensive and only needs to happen ONCE, not on every
    single query. The class stores that pre-computed state so `retrieve()`
    can stay cheap and be called many times.
    """

    def __init__(self, documents: List[dict]):
        """
        documents: a list of dicts, each with at least
            {"document_id": str, "text": str, "source_type": str, "source_url": str}
            -- see build_corpus() below for how these are produced from
            Milestone 1 records.

        What happens here, conceptually:
            TEXT (every document) -> TOKENS -> TF-IDF VECTORS
        This is the "index building" step -- the same idea as a search
        engine crawling and indexing pages before you ever type a query.
        """
        self._documents = documents

        if len(documents) == 0:
            # An empty corpus is a valid (if useless) state -- we don't
            # crash, we just remember there's nothing to search, and
            # retrieve() will always return an empty list.
            self._vectorizer = None
            self._doc_vectors = None
            return

        texts = [doc["text"] for doc in documents]

        # TfidfVectorizer does three things internally, matching the
        # conceptual pipeline explained above:
        #   1. Tokenizes each text (splits into words, lowercases them).
        #   2. Computes Term Frequency for each word in each document.
        #   3. Computes Inverse Document Frequency for each word across
        #      the whole corpus, and multiplies TF x IDF to get the
        #      final vector for each document.
        # lowercase=True (the default) is a normalization choice made
        # HERE, at retrieval time, not baked into the stored data --
        # consistent with Milestone 1's conservative preprocessing
        # stance (see src/data/preprocessing.py).
        #
        # stop_words="english" removes common function words (the, is,
        # in, a, ...) before building the vocabulary. This was added
        # after testing showed that on a SMALL corpus, IDF alone does
        # not sufficiently down-weight these words -- an unrelated query
        # like "What is the weather like today in Paris?" scored 0.27
        # against a campus hackathon announcement purely because both
        # texts contained "the" and "in". Removing stopwords is a
        # standard, well-justified IR practice specifically because
        # these words carry no topical/content meaning -- unlike the
        # conservative stance in preprocessing.py (which keeps dates,
        # numbers, and names because THOSE carry meaning), stopword
        # removal at the retrieval-vectorization step is safe because it
        # targets words that are meaningless for topic comparison by
        # definition.
        self._vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self._doc_vectors = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        """
        Rank the corpus by relevance to `query` and return the top_k.

        Input:
            query: the user's claim, as raw text.
            top_k: how many ranked results to return. See Step "Top-K"
                explanation below for why this is configurable.

        Output:
            A list of RetrievedDocument, sorted by score descending
            (most relevant first). May contain fewer than top_k items if
            the corpus itself has fewer documents than top_k.

        What this does NOT do: it never says a document "proves" or
        "disproves" the claim. It only ranks by topical similarity.
        """
        if not query or not query.strip():
            # An empty query has no meaningful vector to compare against
            # anything -- there's nothing to search for. Returning an
            # empty list (rather than raising an error) keeps this a
            # predictable, easy-to-handle case for callers.
            return []

        if self._vectorizer is None:
            # Empty corpus -- nothing to search, regardless of the query.
            return []

        # Convert the query into a TF-IDF vector using the SAME
        # vocabulary/IDF weights learned from the corpus in __init__.
        # This is important: the query must be projected into the same
        # vector space as the documents, or comparing them would be
        # meaningless.
        query_vector = self._vectorizer.transform([query])

        # cosine_similarity returns a 2D array; we only compared one
        # query against every document, so we take row 0.
        similarities = cosine_similarity(query_vector, self._doc_vectors)[0]

        # Pair each document with its similarity score, sort by score
        # descending, and take the top_k.
        scored_documents = list(zip(self._documents, similarities))
        scored_documents.sort(key=lambda pair: pair[1], reverse=True)
        top_results = scored_documents[:top_k]

        return [
            RetrievedDocument(
                document_id=doc["document_id"],
                score=float(score),
                text=doc["text"],
                source_type=doc["source_type"],
                source_url=doc.get("source_url", ""),
            )
            for doc, score in top_results
        ]


def build_corpus(records: List[dict]) -> List[dict]:
    """
    Convert Milestone 1's validated records into a retrieval corpus.

    Why this function exists: Milestone 1's records are (claim, evidence,
    label, ...) rows -- several rows can share the SAME evidence text
    (e.g. one real announcement has both a SUPPORTED claim and a
    CONTRADICTED claim pointing at it, as in the dev dataset's evt_001).
    The retrieval corpus should contain each real evidence document only
    ONCE, not once per claim that references it -- otherwise the same
    document would just compete with duplicate copies of itself in the
    ranking, which is not meaningful.

    We use `group_id` (introduced in Milestone 1 specifically to identify
    "rows about the same real-world event") as the natural document
    identifier and de-duplication key: one group_id -> one evidence
    document.

    Records with empty evidence (INSUFFICIENT_EVIDENCE rows) are
    excluded -- there is no document there to retrieve.
    """
    seen_group_ids = set()
    documents = []

    for record in records:
        group_id = record.get("group_id", "")
        evidence_text = record.get("evidence", "").strip()

        if not evidence_text:
            # INSUFFICIENT_EVIDENCE rows -- nothing to index.
            continue

        if group_id in seen_group_ids:
            # Already added this document from an earlier row
            # (e.g. the CONTRADICTED row for the same event).
            continue

        seen_group_ids.add(group_id)
        documents.append({
            "document_id": group_id,
            "text": evidence_text,
            "source_type": record.get("source_type", ""),
            "source_url": record.get("source_url", ""),
        })

    return documents
