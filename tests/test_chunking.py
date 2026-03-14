"""
Unit tests for DocumentChunker.

@author: Aarti Dashore, Alok Katiyar
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 2.0.0+w26
"""

from pathlib import Path

import pytest

from retrieval.loader import DocumentChunker


def test_chunker_small_text():
    """Test that small text isn't chunked."""
    chunker = DocumentChunker(chunk_size=100, overlap=10)
    text = "Short document"

    chunks = chunker.chunk_text(text, "doc1")

    # make some assertions!
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["id"] == "doc1_0"


def test_chunker_large_text():
    """Test that large text is chunked correctly."""
    chunker = DocumentChunker(chunk_size=10, overlap=2)
    text = "A B C D E F G H I J " * 1000 + "X Y Z"  # 10003 words

    chunks = chunker.chunk_text(text, "doc1")

    # make some assertions!
    assert len(chunks) == 1251
    assert chunks[0]["text"] == "A B C D E F G H I J"
    assert chunks[1]["text"] == "I J A B C D E F G H"
    assert chunks[-1]["text"] == "X Y Z"
    assert all("id" in chunk for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)


def test_bad_overlap():
    """Test that an overlap value of 0 raises an error."""
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=10, overlap=-1)
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=10, overlap=10)


def test_init_bad_chunk_size_raises():
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=-5, overlap=0)


def test_init_valid_edge_cases():
    # overlap can be 0 as long as overlap < chunk_size
    c = DocumentChunker(chunk_size=1, overlap=0)
    assert c.chunk_size == 1
    assert c.overlap == 0


def test_chunker_metadata():
    """Test that chunk metadata is correct."""
    chunker = DocumentChunker(chunk_size=50, overlap=10)
    text = "A " * 120

    chunks = chunker.chunk_text(text, "test_doc")

    # make some assertions!
    for i, chunk in enumerate(chunks):
        assert chunk["metadata"]["chunk"] == i
        assert chunk["metadata"]["doc_id"] == "test_doc"
        assert chunk["id"] == f"test_doc_{i}"


def test_sample():
    chunker = DocumentChunker()
    test_dir = Path(__file__).parent
    sample = "dracula_by_bram_stoker"
    sample_file = test_dir / "data" / (sample + ".txt")
    with open(sample_file, encoding="utf-8") as f:
        text = f.read()
    chunks = chunker.chunk_text(text, sample)

    # make some assertions!
    assert len(chunks) == 609
    assert chunks[500]["text"][:24] == "thin mist began to creep"
    assert chunks[500]["id"] == "dracula_by_bram_stoker_500"
    assert chunks[500]["metadata"] == {"chunk": 500, "doc_id": "dracula_by_bram_stoker"}
