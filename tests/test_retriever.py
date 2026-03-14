"""
Simple unit tests for the DocumentRetriever class.

@author:  Aarti Dashore, Alok Katiyar
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 1.0.0+w26
"""

from pathlib import Path

import pytest

from retrieval.retriever import DocumentRetriever


@pytest.fixture
def retriever():
    """Create a DocumentRetriever for testing."""
    return DocumentRetriever()


@pytest.fixture
def sample_directory(tmp_path):
    """Create a temporary directory with sample text files."""
    #  Create some test files
    (tmp_path / "doc1.txt").write_text("Python is a programming language")
    (tmp_path / "doc2.txt").write_text("Machine learning uses neural networks")
    (tmp_path / "doc3.txt").write_text("Vector databases store embeddings")
    return str(tmp_path)


def test_search_semantics(retriever, sample_directory):
    """
    Test the basic functionality we want--search!
    """
    # FIXME
    retriever.index_documents(sample_directory)
    results = retriever.search("Are vectors vicious!?", n_results=1)
    assert results[0]["metadata"]["filename"] == "doc3.txt"
    results = retriever.search("How about Python?", n_results=1)
    assert results[0]["metadata"]["filename"] == "doc1.txt"
    results = retriever.search("ML Engineering", n_results=1)
    assert results[0]["metadata"]["filename"] == "doc2.txt"


def test_init_creates_components(retriever):
    """Test that initialization creates all components."""
    assert retriever.loader is not None
    assert retriever.store is not None
    assert retriever._indexed is False


def test_document_count_before_indexing(retriever):
    """Test document count is 0 before indexing."""
    assert retriever.document_count == 0


def test_index_documents(retriever, sample_directory):
    """Test indexing documents from a directory."""
    retriever.index_documents(sample_directory)

    assert retriever._indexed is True
    assert retriever.document_count > 0


def test_index_documents_changes_count(retriever, sample_directory):
    """Test that indexing increases document count."""
    initial_count = retriever.document_count
    retriever.index_documents(sample_directory)

    assert retriever.document_count > initial_count
    assert retriever.document_count == 3


def test_search_returns_list(retriever, sample_directory):
    """Test that search returns a list."""
    retriever.index_documents(sample_directory)
    results = retriever.search("programming")

    assert isinstance(results, list)


def test_search_before_indexing_raises_error(retriever):
    """Test that search returns empty list before indexing."""
    with pytest.raises(ValueError, match="No documents indexed"):
        retriever.search("test query")


def test_search_with_results(retriever, sample_directory):
    """Test that search returns results after indexing."""
    retriever.index_documents(sample_directory)
    results = retriever.search("Python")

    assert len(results) > 0


def test_search_respects_n_results(retriever, sample_directory):
    """Test that search respects n_results parameter."""
    retriever.index_documents(sample_directory)
    results = retriever.search("test", n_results=2)

    assert len(results) <= 2


def test_search_default_n_results(retriever, sample_directory):
    """Test that search uses default n_results=5."""
    retriever.index_documents(sample_directory)
    results = retriever.search("test")

    # Should return at most 5, but we only have 3 docs
    assert len(results) <= 5


def test_search_result_structure(retriever, sample_directory):
    """Test that search results have expected structure."""
    retriever.index_documents(sample_directory)
    results = retriever.search("machine learning")

    if results:
        result = results[0]
        assert isinstance(result, dict)
        assert "id" in result or "text" in result


def test_index_empty_directory(retriever, tmp_path):
    """Test indexing an empty directory."""
    empty_dir = str(tmp_path / "empty")
    Path(empty_dir).mkdir()

    retriever.index_documents(empty_dir)

    assert retriever.document_count == 0
