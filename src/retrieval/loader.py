"""
Document loading utilities.

@author: Kevin Lundeen
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 2.0.0+w26
"""

import logging
from pathlib import Path
from typing import Optional

import pypdf

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunk documents into smaller pieces for better retrieval."""

    def __init__(self, chunk_size: int = 300, overlap: int = 30):
        """
        Initialize chunker with size and overlap parameters.

        Args:
            chunk_size: Maximum words per chunk
            overlap: Number of words to overlap between chunks
        """
        if not 0 <= overlap <= chunk_size / 2:
            raise ValueError("Overlap must be between 0 and half the chunk size.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, doc_id: str) -> list[dict]:
        """
        Split the given text into overlapping chunks.

        Args:
            text: Text to chunk
            doc_id: Document identifier

        Returns:
            List of chunk dicts with id, text, and metadata
        """
        words = text.split()  # pull out words from text
        if len(words) <= self.chunk_size:
            return [{"id": f"{doc_id}_0", "text": text, "metadata": {"chunk": 0, "doc_id": doc_id}}]

        chunks, start, chunk_num = [], 0, 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_text = " ".join(words[start:end])

            chunks.append(
                {
                    "id": f"{doc_id}_{chunk_num}",
                    "text": chunk_text,
                    "metadata": {"chunk": chunk_num, "doc_id": doc_id},
                }
            )

            start = end - self.overlap
            chunk_num += 1
        return chunks


class DocumentLoader:
    """Load and parse documents from the file system."""

    def __init__(self, chunker: Optional[DocumentChunker] = None):
        """Initialize loader with optional chunker."""
        self.chunker = chunker

    def load_documents(self, directory: str) -> list[dict]:
        """
        Load all text documents from a directory.

        Args:
            directory: Path to a directory containing documents

        Returns:
            List of documents, each with 'id', 'text', and 'metadata'
        """
        documents = []
        path = Path(directory)
        if not path.is_dir() or not path.exists():
            raise ValueError(f"Directory '{directory}' does not exist.")

        # Load text files
        for filepath in path.glob("*.txt"):
            logger.info(f"Loading document: {filepath}")
            docs = self._load_text_file(filepath)
            documents.extend(docs)

        # Load PDF files
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

            if self.chunker:
                chunks = self.chunker.chunk_text(text, doc_id)
                # Add filename to each chunk's metadata
                for chunk in chunks:
                    chunk["metadata"].update(metadata)
                return chunks
            else:
                return [{"id": doc_id, "text": text, "metadata": metadata}]

        except Exception as e:
            logger.warning(f"Warning: Failed to load {filepath}: {e}")
            return []

    def _load_pdf_file(self, filepath: Path) -> list[dict]:
        """Load a single PDF file."""
        try:
            reader = pypdf.PdfReader(filepath)

            # Extract text from all pages
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text())

            text = "\n\n".join(text_parts).strip()

            if not text:
                return []

            doc_id = filepath.stem
            metadata = {"filename": filepath.name, "type": "pdf", "num_pages": len(reader.pages)}

            if self.chunker:
                chunks = self.chunker.chunk_text(text, doc_id)
                # Add PDF metadata to each chunk
                for chunk in chunks:
                    chunk["metadata"].update(metadata)
                return chunks
            else:
                return [{"id": doc_id, "text": text, "metadata": metadata}]

        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
            return []
