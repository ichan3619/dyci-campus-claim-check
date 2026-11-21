"""
Tests for the retrieval module.

These test RANKING BEHAVIOR (does the obviously relevant document come
out on top?) and BASIC ROBUSTNESS (empty query, empty corpus, top_k),
not classification correctness -- there is no classifier in this
milestone.
"""

from src.retrieval.retriever import Retriever, build_corpus


def make_corpus():
    """
    A small, hand-built corpus for testing -- independent of the CSV
    dataset, so these tests don't silently break if the dev CSV changes.
    """
    return [
        {
            "document_id": "DOC_TYPHOON",
            "text": "Due to the typhoon, classes are suspended tomorrow.",
            "source_type": "synthetic",
            "source_url": "",
        },
        {
            "document_id": "DOC_ENROLLMENT",
            "text": "Enrollment for first-year students begins August 15.",
            "source_type": "synthetic",
            "source_url": "",
        },
        {
            "document_id": "DOC_BASKETBALL",
            "text": "The university basketball team won yesterday.",
            "source_type": "synthetic",
            "source_url": "",
        },
    ]


def test_relevant_document_is_retrieved():
    """A claim about class suspension should surface the typhoon document."""
    retriever = Retriever(make_corpus())
    results = retriever.retrieve("Are classes suspended tomorrow?", top_k=3)

    assert len(results) > 0
    assert results[0].document_id == "DOC_TYPHOON"


def test_relevant_ranks_above_irrelevant():
    """The basketball document should score lower than the typhoon document
    for a claim about class suspension."""
    retriever = Retriever(make_corpus())
    results = retriever.retrieve("Are classes suspended tomorrow?", top_k=3)

    scores_by_id = {r.document_id: r.score for r in results}
    assert scores_by_id["DOC_TYPHOON"] > scores_by_id["DOC_BASKETBALL"]


def test_top_k_limits_results():
    retriever = Retriever(make_corpus())
    results = retriever.retrieve("classes suspended", top_k=1)

    assert len(results) == 1


def test_top_k_larger_than_corpus_returns_all():
    retriever = Retriever(make_corpus())
    results = retriever.retrieve("classes suspended", top_k=10)

    assert len(results) == 3  # corpus only has 3 documents


def test_empty_query_returns_empty_list():
    retriever = Retriever(make_corpus())
    results = retriever.retrieve("", top_k=3)

    assert results == []


def test_whitespace_only_query_returns_empty_list():
    retriever = Retriever(make_corpus())
    results = retriever.retrieve("   ", top_k=3)

    assert results == []


def test_empty_corpus_returns_empty_list():
    retriever = Retriever([])
    results = retriever.retrieve("Are classes suspended?", top_k=3)

    assert results == []


def test_multiple_documents_can_be_returned():
    retriever = Retriever(make_corpus())
    results = retriever.retrieve("university announcement", top_k=3)

    assert len(results) == 3


def test_scores_are_consistent_across_repeated_calls():
    """The same query against the same corpus should produce the same
    scores every time -- retrieval should be deterministic."""
    retriever = Retriever(make_corpus())
    results_1 = retriever.retrieve("Are classes suspended tomorrow?", top_k=3)
    results_2 = retriever.retrieve("Are classes suspended tomorrow?", top_k=3)

    scores_1 = [r.score for r in results_1]
    scores_2 = [r.score for r in results_2]
    assert scores_1 == scores_2


def test_build_corpus_deduplicates_by_group_id():
    """Two records sharing a group_id (e.g. a SUPPORTED row and a
    CONTRADICTED row about the same real announcement) should produce
    only ONE corpus document, not two."""
    records = [
        {
            "claim": "Classes are suspended tomorrow.",
            "evidence": "Official notice: classes are suspended tomorrow.",
            "label": "SUPPORTED",
            "source_type": "synthetic",
            "source_url": "",
            "group_id": "evt_shared",
        },
        {
            "claim": "Classes will continue as normal.",
            "evidence": "Official notice: classes are suspended tomorrow.",
            "label": "CONTRADICTED",
            "source_type": "synthetic",
            "source_url": "",
            "group_id": "evt_shared",
        },
    ]
    corpus = build_corpus(records)

    assert len(corpus) == 1
    assert corpus[0]["document_id"] == "evt_shared"


def test_build_corpus_excludes_insufficient_evidence_rows():
    """Rows with empty evidence (INSUFFICIENT_EVIDENCE) have nothing to
    index and should not appear in the corpus."""
    records = [
        {
            "claim": "The university will have a surprise holiday.",
            "evidence": "",
            "label": "INSUFFICIENT_EVIDENCE",
            "source_type": "synthetic",
            "source_url": "",
            "group_id": "evt_no_evidence",
        },
    ]
    corpus = build_corpus(records)

    assert len(corpus) == 0
