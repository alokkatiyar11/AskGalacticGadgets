"""
Hybrid search combining semantic and keyword-based retrieval.

@author: Kevin Lundeen
Seattle University, ARIN 5360
@version: 4.0.0+w26
"""

from collections import defaultdict
from typing import Optional

from rank_bm25 import BM25Okapi


class BM25Searcher:
    """Keyword-based search using BM25 algorithm."""

    def __init__(self):
        """Initialize BM25 searcher."""
        self.bm25 = None
        self.documents = []
        self.doc_ids = []

    def index_documents(self, documents: list[dict]):
        """
        Index documents for BM25 search.

        Args:
            documents: List of document dicts with 'id' and 'text'
        """
        self.documents = documents
        self.doc_ids = [doc["id"] for doc in documents]

        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = [doc["text"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None

    def search(self, query: str, n_results: int = 10) -> list[dict]:
        """
        Search using BM25 keyword matching.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of results with BM25 scores
        """
        if self.bm25 is None:
            return []

        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = top_indices[:n_results]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with non-zero scores
                doc = self.documents[idx].copy()
                doc["bm25_score"] = float(scores[idx])
                results.append(doc)

        return results


class HybridSearcher:
    """Combine semantic and keyword search using reciprocal rank fusion."""

    def __init__(self, k: int = 60, bm25_searcher: Optional[BM25Searcher] = None):
        """
        Initialize hybrid searcher.

        Args:
            k: RRF k parameter (default 60 is standard)
        """
        self.k = k
        self.bm25_searcher = bm25_searcher

    def reciprocal_rank_fusion(
        self, semantic_results: list[dict], keyword_results: list[dict]
    ) -> list[dict]:
        """
        Combine results using Reciprocal Rank Fusion.

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from BM25 search

        Returns:
            Fused and ranked results
        """
        # Calculate and accumulate RRF scores from semantic and keyword results
        # Use a defaultdict where the += op adds to 0.0 if not present
        rrf_scores: defaultdict[str, float] = defaultdict(float)
        for results in (semantic_results, keyword_results):
            for rank, doc in enumerate(results, start=1):
                rrf_scores[doc["id"]] += 1 / (self.k + rank)

        # Create a document map for doing easy lookup
        doc_map = {doc["id"]: doc.copy() for doc in keyword_results + semantic_results}

        # Create the final ranked list
        results = []
        for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            doc = doc_map[doc_id]
            doc["rrf_score"] = score
            results.append(doc)

        return results

    def search(self, query: str, semantic_results: list[dict], n_results: int = 5):
        """
        Perform hybrid search combining semantic and keyword results.

        Args:
            query: Search query
            semantic_results: Results from semantic search
            n_results: Number of final results to return

        Returns:
            Fused results
        """
        if self.bm25_searcher is None:
            # If no BM25, just return semantic results
            return semantic_results[:n_results]

        # Get BM25 results
        keyword_results = self.bm25_searcher.search(query, n_results=20)

        # Fuse results
        fused = self.reciprocal_rank_fusion(semantic_results, keyword_results)

        return fused[:n_results]
