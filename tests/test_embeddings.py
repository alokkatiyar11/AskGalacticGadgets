"""
Unit tests for embedder.

@author: Aarti Dashore, Alok Katiyar
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 1.0.0+w26
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.embeddings import DocumentEmbedder

EMBED_DIM = 384  # all-MiniLM-L6-v2 embedding size


@pytest.fixture(scope="module")
def embedder():
    """Create a real DocumentEmbedder for testing."""
    return DocumentEmbedder(model_name="all-MiniLM-L6-v2")


def test_embed_documents_returns_numpy_array(embedder):
    """Test that embed_documents returns a numpy array."""
    texts = ["Python programming", "Machine learning"]
    embeddings = embedder.embed_documents(texts)

    assert isinstance(embeddings, np.ndarray)


def test_embed_documents_correct_shape(embedder):
    """Test that the embeddings have the correct shape."""
    texts = ["Hello world", "Test document", "Another text"]
    embeddings = embedder.embed_documents(texts)

    assert embeddings.shape == (3, EMBED_DIM)


def test_embed_query_returns_numpy_array(embedder):
    """Test that embed_query returns a numpy array."""
    embedding = embedder.embed_query("test query")

    assert isinstance(embedding, np.ndarray)


def test_embed_query_correct_shape(embedder):
    """Test that query embedding has the correct shape."""
    embedding = embedder.embed_query("test query")

    assert embedding.shape == (EMBED_DIM,)


def test_embed_query_with_list_input(embedder):
    """Test that embed_query works with list input."""
    embeddings = embedder.embed_query(["query1", "query2"])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, EMBED_DIM)


def test_similar_texts_have_similar_embeddings(embedder):
    """Test that similar texts produce more similar embeddings than unrelated ones."""
    emb1 = embedder.embed_query("Python programming")
    emb2 = embedder.embed_query("coding in Python")
    emb3 = embedder.embed_query("eating pizza")

    similarity_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    similarity_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))

    assert similarity_12 > similarity_13


def test_embed_empty_list(embedder):
    """Test embedding an empty list."""
    embeddings = embedder.embed_documents([])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.size == 0


def test_custom_model_name():
    """Test creating embedder with custom model name."""
    embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")

    assert embedder.model_name == "all-MiniLM-L6-v2"
