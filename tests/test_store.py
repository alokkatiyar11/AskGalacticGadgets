"""
Unit tests of vector store.

@author:  Aarti Dashore, Alok Katiyar
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 1.0.0+w26
"""

#  import our DocumentEmbedder and VectorStore
import chromadb
import pytest
from chromadb import Settings

from src.retrieval.embeddings import DocumentEmbedder
from src.retrieval.store import VectorStore


@pytest.fixture
def document_embedder():
    """
    Just use our own document embedder (at least for now)--any mock one
    would be no less succinct! Having this real one allows us to do some
    integration testing with it, too. See test_search_semantics below.
    """
    #  return DocumentEmbedder()
    return DocumentEmbedder()


@pytest.fixture
def vector_store(document_embedder):
    """Create a VectorStore for testing."""
    #  return VectorStore(document_embedder)
    return VectorStore(document_embedder)


@pytest.fixture
def sample_docs():
    """Sample documents for testing."""
    return [
        {"id": "1", "text": "Python programming", "metadata": {"filename": "file1.txt"}},
        {"id": "2", "text": "Vector databases", "metadata": {"filename": "file2.txt"}},
        {"id": "3", "text": "Semantic search", "metadata": {"filename": "file3.txt"}},
    ]


def test_search_semantics(vector_store, sample_docs):
    """
    Test the basic functionality we want--search!
    Technically, this is more of an integration test since it relies on a
    working embedder below, but that's okay since we used our real one
    instead of mocking it up.
    """
    #  make some tests with the semantic search functionality!
    vector_store.add_documents(sample_docs)
    results = vector_store.search("Are vectors vicious!?", n_results=1)
    assert results[0]["metadata"]["filename"] == "file2.txt"
    results = vector_store.search("How about Python?", n_results=1)
    assert results[0]["metadata"]["filename"] == "file1.txt"
    results = vector_store.search("Searching...", n_results=1)
    assert results[0]["metadata"]["filename"] == "file3.txt"


def test_count_empty_store(vector_store):
    """Test edge case: count returns 0 for the empty store."""
    assert vector_store.count() == 0


def test_add_documents(vector_store, sample_docs):
    """Test adding documents increases the count."""
    vector_store.add_documents(sample_docs)
    assert vector_store.count() == 3


def test_add_empty_list(vector_store):
    """Test adding an empty list doesn't change the count."""
    vector_store.add_documents([])
    assert vector_store.count() == 0


def test_search_empty_store(vector_store):
    """Test searching empty store returns the empty list."""
    results = vector_store.search("test")
    #  make an assertion!
    assert results == []


def test_search_with_n_results(vector_store, sample_docs):
    """Test search respects n_results parameter."""
    vector_store.add_documents(sample_docs)
    results = vector_store.search("some query", n_results=2)
    #  make an assertion!
    assert len(results) == 2
