# tests/test_rag.py

"""
Unit tests for the RAGSystem.

We mock both the retriever and the LLM client so:
- No real document retrieval happens
- No real LLM calls are made
- Tests remain fast and CI-safe
"""

from unittest.mock import Mock

import pytest

from retrieval.rag import RAGSystem


def test_build_context_formats_documents():
    """_build_context should format retrieved documents clearly."""

    retriever = Mock()
    llm = Mock()
    rag = RAGSystem(retriever=retriever, llm_client=llm)

    docs = [
        {"doc_id": "doc1.txt", "text": "First document"},
        {"doc_id": "doc2.txt", "text": "Second document"},
    ]

    context = rag._build_context(docs)

    assert "[1] doc1.txt" in context
    assert "First document" in context
    assert "[2] doc2.txt" in context
    assert "Second document" in context


def test_create_prompt_contains_context_and_question():
    """Prompt should include both context and question."""

    rag = RAGSystem(retriever=Mock(), llm_client=Mock())

    context = "[1] doc1.txt\nFirst document"
    question = "What is this about?"

    prompt = rag._create_prompt(question, context)

    assert context in prompt
    assert f"Question: {question}" in prompt
    assert "Answer:" in prompt


def test_query_runs_full_pipeline():
    """
    query() should:
      - call retriever
      - call LLM
      - return answer and sources
    """

    retriever = Mock()
    retriever.search.return_value = [{"doc_id": "doc1.txt", "text": "Hello world", "score": 0.9}]

    llm = Mock()
    llm.generate.return_value = "This document says hello."

    rag = RAGSystem(retriever=retriever, llm_client=llm)

    result = rag.query("What does it say?", temperature=0.2)

    assert result["answer"] == "This document says hello."
    assert result["question"] == "What does it say?"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["doc_id"] == "doc1.txt"

    retriever.search.assert_called_once()
    llm.generate.assert_called_once()


def test_query_without_llm_returns_fallback():
    """If no LLM is configured, system should still return sources."""

    retriever = Mock()
    retriever.search.return_value = [{"doc_id": "doc1.txt", "text": "Hello world"}]

    rag = RAGSystem(retriever=retriever, llm_client=None)

    result = rag.query("Test question")

    assert "LLM not configured" in result["answer"]
    assert len(result["sources"]) == 1


def test_query_rejects_empty_question():
    """Empty question should raise ValueError."""

    rag = RAGSystem(retriever=Mock(), llm_client=Mock())

    with pytest.raises(ValueError):
        rag.query("")
