"""
Unit tests for embedder.

@author: Kevin Lundeen
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 2.0.0+w26
"""

import numpy as np
import pytest

from retrieval.embeddings import DocumentEmbedder


@pytest.fixture
def embedder():
    """Create a DocumentEmbedder for testing."""
    return DocumentEmbedder()


def test_embed_documents_returns_numpy_array(embedder):
    """Test that embed_documents returns a numpy array."""
    texts = ["Python programming", "Machine learning"]
    embeddings = embedder.embed_documents(texts)

    assert isinstance(embeddings, np.ndarray)


def test_embed_documents_correct_shape(embedder):
    """Test that the embeddings have the correct shape."""
    texts = ["Hello world", "Test document", "Another text"]
    embeddings = embedder.embed_documents(texts)

    # Should have one embedding per text
    assert embeddings.shape[0] == 3
    # The default model has a dimension of 384
    assert embeddings.shape[1] == 384


def test_embed_query_returns_numpy_array(embedder):
    """Test that embed_query returns a numpy array."""
    embedding = embedder.embed_query("test query")

    assert isinstance(embedding, np.ndarray)


def test_embed_query_correct_shape(embedder):
    """Test that query embedding has the correct shape."""
    embedding = embedder.embed_query("test query")

    # Should be a 1D array with a dimension of 384
    assert embedding.shape == (384,)


def test_embed_query_with_list_input(embedder):
    """Test that embed_query works with list input."""
    embedding = embedder.embed_query(["query1", "query2"])

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (2, 384)


def test_similar_texts_have_similar_embeddings(embedder):
    """Test that similar texts produce similar embeddings."""
    # Embed similar texts
    emb1 = embedder.embed_query("Python programming")
    emb2 = embedder.embed_query("coding in Python")
    emb3 = embedder.embed_query("eating pizza")

    # Calculate cosine similarity
    similarity_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    similarity_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))

    # Similar texts should have higher similarity
    assert similarity_12 > similarity_13


def test_embed_empty_list(embedder):
    """Test embedding an empty list."""
    embeddings = embedder.embed_documents([])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 0


def test_custom_model_name():
    """Test creating embedder with a custom model name."""
    embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")

    assert embedder.model_name == "all-MiniLM-L6-v2"
