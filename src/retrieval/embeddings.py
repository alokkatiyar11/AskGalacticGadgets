"""
Embedding functions for document retrieval.

@author: Aarti Dashore, Alok Katiyar
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 1.0.0+w26
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class DocumentEmbedder:
    """
    Generates vector embeddings for documents and queries using a
    SentenceTransformer model.

    Args:
        model_name (str): Hugging Face model name for embeddings.
            Defaults to "all-MiniLM-L6-v2".

    Attributes:
        model (SentenceTransformer): Loaded embedding model
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of documents.

        Args:
            texts (list[str]): List of document strings

        Returns:
            np.ndarray: 2D array of shape (num_docs, embedding_dim)
        """
        if not texts:
            return np.array([])

        return np.array(self.model.encode(texts, show_progress_bar=False))

    def embed_query(self, queries: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embedding(s) for query text.

        Args:
            queries (str | list[str]): A single query string or list of queries

        Returns:
            np.ndarray:
                - 1D vector if single string input
                - 2D array if list input
        """
        if isinstance(queries, str):
            embedding = self.embed_documents([queries])
            return embedding[0]

        return self.embed_documents(queries)
