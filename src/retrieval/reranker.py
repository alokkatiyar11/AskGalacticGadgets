"""
Reranking functionality using cross-encoder models.

@author: Kevin Lundeen
Seattle University, ARIN 5360
@version: 3.0.0+w26
"""

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank search results using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker with a cross-encoder model.

        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document dicts from initial retrieval
            top_k: Number of top results to return

        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [(query, doc["text"]) for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Combine documents with their new scores
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sort by rerank score (descending)
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]
