"""
Unit tests for DocumentLoader.

@author:  Aarti Dashore, Alok Katiyar
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 1.0.0+w26

These tests validate that DocumentLoader.load_documents:
- Loads .txt files in a directory
- Skips empty/whitespace-only files
- Returns the expected schema for each document
- Raises correct exceptions for bad directories
- Covers exception-handling paths in _load_text_file and _load_pdf_file (for 100% coverage)

"""

from __future__ import annotations

from pathlib import Path

import pytest

from retrieval.loader import DocumentLoader


@pytest.fixture
def loader() -> DocumentLoader:
    """Provide a fresh DocumentLoader instance for each test."""
    return DocumentLoader()


def _write_file(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Helper to create a text file with given content."""
    path.write_text(content, encoding=encoding)


def test_loader_loads_documents(tmp_path: Path) -> None:
    """Test loading multiple documents from a directory."""
    (tmp_path / "file1.txt").write_text("This is a test file", encoding="utf-8")
    (tmp_path / "file2.txt").write_text("This is another test file", encoding="utf-8")

    docs = DocumentLoader().load_documents(str(tmp_path))

    assert len(docs) == 2
    assert all("id" in doc and "text" in doc and "metadata" in doc for doc in docs)


def test_returns_empty_list_for_empty_directory(tmp_path: Path, loader: DocumentLoader) -> None:
    """If directory contains no loadable files, return empty list."""
    docs = loader.load_documents(str(tmp_path))
    assert docs == []


def test_loads_single_text_file(tmp_path: Path, loader: DocumentLoader) -> None:
    """A single .txt file should be loaded with correct schema."""
    _write_file(tmp_path / "a.txt", "hello world")

    docs = loader.load_documents(str(tmp_path))

    assert len(docs) == 1
    doc = docs[0]
    assert doc["id"] == "a"
    assert doc["text"] == "hello world"
    assert doc["metadata"]["filename"] == "a.txt"
    assert doc["metadata"]["type"] == "txt"


def test_ignores_non_txt_files(tmp_path: Path, loader: DocumentLoader) -> None:
    """Non-.txt and non-.pdf files should be ignored."""
    _write_file(tmp_path / "a.txt", "hello")
    _write_file(tmp_path / "b.md", "should not load")
    _write_file(tmp_path / "c.json", '{"x": 1}')

    docs = loader.load_documents(str(tmp_path))

    assert len(docs) == 1
    assert docs[0]["id"] == "a"


def test_skips_empty_files(tmp_path: Path, loader: DocumentLoader) -> None:
    """Empty and whitespace-only files should be skipped."""
    _write_file(tmp_path / "empty.txt", "")
    _write_file(tmp_path / "spaces.txt", "   \n\t  ")
    _write_file(tmp_path / "valid.txt", "content")

    docs = loader.load_documents(str(tmp_path))

    assert len(docs) == 1
    assert docs[0]["id"] == "valid"
    assert docs[0]["text"] == "content"


def test_loads_multiple_text_files(tmp_path: Path, loader: DocumentLoader) -> None:
    """Multiple .txt files should be loaded; order not assumed."""
    _write_file(tmp_path / "one.txt", "first")
    _write_file(tmp_path / "two.txt", "second")
    _write_file(tmp_path / "three.txt", "third")

    docs = loader.load_documents(str(tmp_path))
    ids = {d["id"] for d in docs}
    assert ids == {"one", "two", "three"}

    for d in docs:
        assert set(d.keys()) == {"id", "text", "metadata"}
        assert d["metadata"]["filename"] == f"{d['id']}.txt"
        assert d["metadata"]["type"] == "txt"
        assert isinstance(d["text"], str) and d["text"].strip() != ""


def test_raises_for_nonexistent_directory(loader: DocumentLoader) -> None:
    """Non-existent directory should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        loader.load_documents("/this/path/should/not/exist")


def test_raises_not_a_directory_error(tmp_path: Path, loader: DocumentLoader) -> None:
    """Passing a file path instead of directory should raise NotADirectoryError."""
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("I am a file, not a directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        loader.load_documents(str(file_path))


def test_load_text_file_logs_warning_on_exception(
    monkeypatch, tmp_path: Path, caplog, loader: DocumentLoader
) -> None:
    """Force _load_text_file() into its exception handler and ensure warning is logged."""
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("hello", encoding="utf-8")

    real_open = open  # keep reference to real open

    def fake_open(*args, **kwargs):
        if args and str(args[0]) == str(bad_file):
            raise OSError("boom")
        return real_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    with caplog.at_level("WARNING"):
        docs = loader._load_text_file(bad_file)

    assert docs == []
    assert "Failed to load" in caplog.text


def test_load_pdf_file_logs_warning_on_exception(
    monkeypatch, tmp_path: Path, caplog, loader: DocumentLoader
) -> None:
    """Force _load_pdf_file() into its exception handler and ensure warning is logged."""
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"%PDF-1.4 broken")

    class BadPdfReader:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("pdf boom")

    monkeypatch.setattr("pypdf.PdfReader", BadPdfReader)

    with caplog.at_level("WARNING"):
        docs = loader._load_pdf_file(bad_pdf)

    assert docs == []
    assert "Failed to load" in caplog.text
