"""
Unit tests for document loader.

@author: Alok Katiyar
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 2.0.0+w26
"""

import pytest

from retrieval.loader import DocumentLoader


def test_loader_loads_documents(tmp_path):
    """Test loading documents from a directory."""
    # Create some test files
    (tmp_path / "file1.txt").write_text("This is a test file.")
    (tmp_path / "file2.txt").write_text("This is another test file.")

    loader = DocumentLoader()
    documents = loader.load_documents(str(tmp_path))

    assert len(documents) == 2
    assert all("id" in doc for doc in documents)
    assert all("text" in doc for doc in documents)
    assert all("metadata" in doc for doc in documents)


def test_loader_skips_empty_files(tmp_path):
    """Test that empty files are skipped."""
    (tmp_path / "not_empty.txt").write_text("This is a test file.")
    (tmp_path / "empty.txt").write_text("")
    loader = DocumentLoader()
    documents = loader.load_documents(str(tmp_path))
    assert len(documents) == 1
    assert documents[0]["text"] == "This is a test file."
    assert documents[0]["metadata"]["filename"] == "not_empty.txt"


def test_loader_skips_nonexistent_directory():
    """Test that loading from a nonexistent directory returns an empty list."""
    loader = DocumentLoader()
    with pytest.raises(ValueError, match="Directory 'garbage' does not exist."):
        loader.load_documents("garbage")


def test_loader_handles_bad_utf8_text_file(tmp_path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_bytes(b"\xff\xfe\xfa")
    loader = DocumentLoader()
    documents = loader.load_documents(str(tmp_path))
    assert documents == []


def test_loader_pdf_empty_text_returns_empty_list(tmp_path, monkeypatch):
    # Create a dummy PDF file path
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")

    class _Page:
        def extract_text(self):
            return ""

    class _Reader:
        def __init__(self, _filepath):
            self.pages = [_Page(), _Page()]

    monkeypatch.setattr("retrieval.loader.pypdf.PdfReader", _Reader)

    loader = DocumentLoader()
    documents = loader.load_documents(str(tmp_path))
    assert documents == []


def test_loader_pdf_reader_failure_is_caught(tmp_path):
    pdf_path = tmp_path / "bad.pdf"
    pdf_path.write_bytes(b"not a real pdf")
    loader = DocumentLoader()
    documents = loader.load_documents(str(tmp_path))
    assert documents == []
