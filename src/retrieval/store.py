"""
Vector store for semantic search using ChromaDB.

@author: Kevin Lundeen
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 4.0.0+w26
"""

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings


class EmbedderAdaptor(EmbeddingFunction):
    """
    Adapts our style of embedder to ChromaDB's which wants a callable
    interface.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def is_legacy(self) -> bool:
        """Return True since we don't support build from config, etc."""
        return True

    def __call__(self, input) -> list[list[float]]:  # type: ignore[override]
        """
        Make embedder callable for ChromaDB compatibility and convert to a
        list from the numpy array returned by our embedder.
        """
        return self.embedder.embed_documents(input).tolist()


class VectorStore:
    """Manages document storage and retrieval using ChromaDB."""

    def __init__(self, embedder, collection_name: str = "documents"):
        """
        Initialize vector store with an embedder.

        Args:
            embedder: DocumentEmbedder instance for generating vectors
            collection_name: Name for the ChromaDB collection
        """
        self.embedder = EmbedderAdaptor(embedder)
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))

        # Delete any existing collection if present
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedder,  # Should use self.embedder
        )

    def add_documents(self, documents):
        """
        Add documents to the vector store.

        Args:
            documents: List of dicts with 'id', 'text', and 'metadata'
        """
        if not documents:
            return

        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def search(self, query: str, n_results: int = 5):
        """
        Search for documents similar to the query.

        Args:
            query: Search query text
            n_results: Number of results to return

        Returns:
            List of result dicts with 'id', 'text', 'distance', and 'metadata'
        """
        results = self.collection.query(query_texts=[query], n_results=n_results)

        # Add type checking before indexing
        # (then we feel safe with the type-ignores below)
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        # Format results
        formatted = []
        if len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):  # type: ignore[override]
                formatted.append(
                    {
                        "id": results["ids"][0][i],  # type: ignore[index]
                        "text": results["documents"][0][i],  # type: ignore[index]
                        "distance": results["distances"][0][i],  # type: ignore[index]
                        "metadata": results["metadatas"][0][i],  # type: ignore[index]
                    }
                )

        return formatted

    def count(self) -> int:
        """Return the number of documents in the store."""
        return self.collection.count()
