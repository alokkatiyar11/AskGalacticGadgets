"""
Unit tests for hybrid search.

@author: Kevin Lundeen
Seattle University, ARIN 5360
@version: 4.0.0+w26
"""

from src.retrieval.hybrid import BM25Searcher, HybridSearcher


def test_bm25_searcher_initialization():
    """Test BM25 searcher initializes correctly."""
    searcher = BM25Searcher()
    assert searcher.bm25 is None
    assert searcher.documents == []


def test_bm25_indexing():
    """Test BM25 document indexing."""
    searcher = BM25Searcher()
    documents = [
        {"id": "doc1", "text": "Machine learning algorithms"},
        {"id": "doc2", "text": "Python programming language"},
    ]

    searcher.index_documents(documents)

    assert searcher.bm25 is not None
    assert len(searcher.documents) == 2


def test_bm25_empty_query():
    """Test BM25 handles an empty query gracefully."""
    searcher = BM25Searcher()
    documents = [{"id": "doc1", "text": "Some text"}]
    searcher.index_documents(documents)

    results = searcher.search("", n_results=5)
    # Pytest will fail: Empty query should return empty results
    assert len(results) == 0  # This will fail - BM25 may return results for empty query


def test_hybrid_searcher_initialization():
    """Test hybrid searcher initialization."""
    searcher = HybridSearcher()
    assert searcher.k == 60
    assert searcher.bm25_searcher is None


def test_bm25_search():
    """Test BM25 keyword search."""
    searcher = BM25Searcher()
    documents = [
        {"id": "doc1", "text": "Machine learning and artificial intelligence"},
        {"id": "doc2", "text": "Python is a programming language"},
        {"id": "doc3", "text": "Machine learning uses algorithms"},
    ]
    searcher.index_documents(documents)

    results = searcher.search("machine learning", n_results=2)

    assert len(results) == 2
    assert all("bm25_score" in doc for doc in results)
    # Documents with "machine learning" should score higher
    assert any("machine" in doc["text"].lower() for doc in results)


def test_hybrid_searcher_rrf():
    """Test reciprocal rank fusion."""
    hybrid = HybridSearcher(k=60)

    semantic_results = [
        {"id": "doc1", "text": "First document"},
        {"id": "doc2", "text": "Second document"},
    ]

    keyword_results = [
        {"id": "doc2", "text": "Second document"},
        {"id": "doc3", "text": "Third document"},
    ]

    fused = hybrid.reciprocal_rank_fusion(semantic_results, keyword_results)

    assert len(fused) == 3  # All unique documents
    assert all("rrf_score" in doc for doc in fused)
    # doc2 appears in both, should have highest score
    assert fused[0]["id"] == "doc2"


def test_hybrid_search_without_bm25():
    """Test hybrid search falls back to semantic when BM25 unavailable."""
    hybrid = HybridSearcher()

    semantic_results = [{"id": "doc1", "text": "Document 1"}, {"id": "doc2", "text": "Document 2"}]

    results = hybrid.search("test", semantic_results, n_results=2)

    # Should just return semantic results
    assert len(results) == 2
    assert results[0]["id"] == "doc1"
