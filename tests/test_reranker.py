"""
Unit tests for cross-encoder reranker.

@author: Kevin Lundeen
Seattle University, ARIN 5360
@version: 3.0.0+w26
"""

import pytest

from retrieval.reranker import CrossEncoderReranker


def test_reranker_initialization():
    """Test reranker initializes correctly."""
    reranker = CrossEncoderReranker()
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_reranker_empty_documents():
    """Test that reranker handles an empty document list."""
    reranker = CrossEncoderReranker()
    results = reranker.rerank("test query", [], top_k=5)
    assert results == []


def test_reranker_adds_scores():
    """Test that reranker adds rerank_score to documents."""
    reranker = CrossEncoderReranker()
    documents = [
        {"id": "doc1", "text": "Machine learning is a branch of AI"},
        {"id": "doc2", "text": "Python is a programming language"},
        {"id": "doc3", "text": "AI and machine learning are related"},
    ]

    results = reranker.rerank("machine learning", documents, top_k=3)

    assert len(results) == 3
    assert all("rerank_score" in doc for doc in results)
    assert all(isinstance(doc["rerank_score"], float) for doc in results)


def test_reranker_returns_top_k():
    """Test that reranker returns only top_k results."""
    reranker = CrossEncoderReranker()
    documents = [{"id": f"doc{i}", "text": f"Document {i}"} for i in range(10)]

    results = reranker.rerank("test", documents, top_k=3)
    assert len(results) == 3


def test_reranker_sorts_by_relevance():
    """Test that reranker sorts results by relevance."""
    reranker = CrossEncoderReranker()
    documents = [
        {"id": "doc1", "text": "Machine learning algorithms"},
        {"id": "doc2", "text": "Weather forecast"},
        {"id": "doc3", "text": "Machine learning and AI"},
    ]

    results = reranker.rerank("machine learning", documents, top_k=3)

    # Scores should be in descending order
    scores = [doc["rerank_score"] for doc in results]
    assert scores == sorted(scores, reverse=True)
