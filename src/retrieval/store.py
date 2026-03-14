"""
Vector store for semantic search using ChromaDB.

@author:  Aarti Dashore, Alok Katiyar
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 1.0.0+w26
"""

import chromadb
from chromadb import Settings
from chromadb.api.types import EmbeddingFunction


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

    #  implement the callable interface by calling the adaptor's embedder
    def __call__(self, input) -> list[list[float]]:
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
        #  use ChromaDB client
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

        #  pull out fields into separate lists like ChromaDB expects
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        #  add them to ChromaDB's collection
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Search for documents similar to the query.

        Args:
            query: Search query text
            n_results: Number of results to return

        Returns:
            List of result dicts with 'id', 'text', 'distance', and 'metadata'
        """
        #  use ChromaDB's query interface
        results = self.collection.query(query_texts=[query], n_results=n_results)

        formatted = []
        #  Format results
        if len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                formatted.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "distance": results["distances"][0][i],
                        "metadata": results["metadatas"][0][i],
                    }
                )

        return formatted

    def count(self) -> int:
        """Return the number of documents in the store."""
        #  ask the collection for its size
        return self.collection.count()
