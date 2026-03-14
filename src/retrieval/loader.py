"""
DocumentLoader.

@author: Aarti Dashore, Alok Katiyar
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 0.1.0+w26
"""

from __future__ import annotations

import logging
from pathlib import Path

import pypdf

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunk documents into smaller pieces for better retrieval."""

    def __init__(self, chunk_size: int = 300, overlap: int = 30):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, doc_id: str) -> list[dict]:
        """Split the given text into overlapping chunks."""
        words = text.split()

        if len(words) <= self.chunk_size:
            return [
                {
                    "id": f"{doc_id}_0",
                    "text": text,
                    "metadata": {"chunk": 0, "doc_id": doc_id},
                }
            ]

        chunks: list[dict] = []
        start, chunk_num = 0, 0

        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])

            chunks.append(
                {
                    "id": f"{doc_id}_{chunk_num}",
                    "text": chunk,
                    "metadata": {"chunk": chunk_num, "doc_id": doc_id},
                }
            )

            start = end - self.overlap
            chunk_num += 1

        return chunks


class DocumentLoader:
    """Load and parse documents from the file system."""

    def __init__(self, chunker: DocumentChunker | None = None):
        """Initialize loader with optional chunker."""
        self.chunker = chunker

    def load_documents(self, directory: str) -> list[dict]:
        """Load all text documents from a directory."""
        documents: list[dict] = []
        path = Path(directory)

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # load text files
        for filepath in path.glob("*.txt"):
            logger.info(f"Loading document: {filepath}")
            docs = self._load_text_file(filepath)
            documents.extend(docs)

        # load PDF files
        for filepath in path.glob("*.pdf"):
            logger.info(f"Loading document: {filepath}")
            docs = self._load_pdf_file(filepath)
            documents.extend(docs)

        return documents

    def _load_text_file(self, filepath: Path) -> list[dict]:
        """Load a single text file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                return []

            doc_id = filepath.stem
            metadata = {"filename": filepath.name, "type": "txt"}

            # Chunk if chunker exists
            if self.chunker:
                chunks = self.chunker.chunk_text(text, doc_id)
                # Add filename/type to each chunk's metadata
                for chunk in chunks:
                    chunk["metadata"].update(metadata)
                return chunks

            # No chunking
            return [{"id": doc_id, "text": text, "metadata": metadata}]

        except Exception as e:
            logger.warning(f"Warning: Failed to load {filepath}: {e}")
            return []

    def _load_pdf_file(self, filepath: Path) -> list[dict]:
        """Load a single PDF file."""
        try:
            reader = pypdf.PdfReader(str(filepath))

            # Extract text from all pages
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")

            text = "\n\n".join(text_parts).strip()

            if not text:
                return []

            doc_id = filepath.stem
            metadata = {"filename": filepath.name, "type": "pdf", "num_pages": len(reader.pages)}

            # Chunk if chunker exists
            if self.chunker:
                chunks = self.chunker.chunk_text(text, doc_id)
                # Add pdf metadata to each chunk's metadata
                for chunk in chunks:
                    chunk["metadata"].update(metadata)
                return chunks
            else:
                # No chunking
                return [{"id": doc_id, "text": text, "metadata": metadata}]

        except Exception as e:
            logger.warning(f"Warning: Failed to load {filepath}: {e}")
            return []
