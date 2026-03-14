"""
Document retrieval system.

@author: Kevin Lundeen
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 3.1.0+w26
"""

from typing import Optional

from retrieval.embeddings import DocumentEmbedder
from retrieval.hybrid import BM25Searcher, HybridSearcher
from retrieval.loader import DocumentChunker, DocumentLoader
from retrieval.reranker import CrossEncoderReranker
from retrieval.store import VectorStore


class DocumentRetriever:
    """
    Initialize retriever with optional advanced features.

    Args:
        chunk_size: Maximum characters per chunk
        overlap: Overlap between chunks
        enable_reranking: Enable cross-encoder reranking
        enable_hybrid: Enable hybrid search (BM25 + semantic)
    """

    def __init__(
        self,
        chunk_size: int = 300,
        overlap: int = 30,
        enable_reranking: bool = True,
        enable_hybrid: bool = True,
    ):
        """Initialize retriever with default components."""
        chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        self.loader = DocumentLoader(chunker=chunker)
        self.store = VectorStore(DocumentEmbedder())

        # Optional component reranker
        self.reranker: Optional[CrossEncoderReranker] = None
        if enable_reranking:
            self.reranker = CrossEncoderReranker()

        # Optional component hybrid search
        self.bm25_searcher: Optional[BM25Searcher] = None
        self.hybrid_searcher: Optional[HybridSearcher] = None
        self.use_hybrid: bool = enable_hybrid
        if self.use_hybrid:
            self.bm25_searcher = BM25Searcher()
            self.hybrid_searcher = HybridSearcher(bm25_searcher=self.bm25_searcher)

        self._indexed = False

    def index_documents(self, directory: str):
        """
        Load and index documents from a directory.

        Args:
            directory: Path to the directory containing documents

        Returns:
            Number of documents indexed
        """
        before = self.document_count
        documents = self.loader.load_documents(directory)
        self.store.add_documents(documents)

        # Store documents for BM25 if hybrid search is enabled
        if self.use_hybrid and self.bm25_searcher:
            self.bm25_searcher.index_documents(documents)

        self._indexed = True
        return self.document_count - before

    def search(
        self,
        query: str,
        n_results: int = 5,
        use_reranking: Optional[bool] = None,
        use_hybrid: Optional[bool] = None,
    ) -> list[dict]:
        """
        Search for documents relevant to the query.

        Args:
            query: Search query text
            n_results: Number of results to return
            use_reranking: Disable cross-encoder reranking by setting to False
            use_hybrid: Disable hybrid search by setting to False
        Returns:
            List of result dicts with document information
        """
        if not self._indexed:
            raise ValueError("No documents indexed. Call index_documents() first.")

        # Determine which features to use
        apply_reranking = use_reranking is not False and self.reranker is not None
        apply_hybrid = use_hybrid is not False and self.hybrid_searcher is not None

        # Get initial semantic search results
        # Retrieve more initially if we're reranking or using hybrid
        initial_k = max(20, n_results) if apply_reranking else n_results
        semantic_results = self.store.search(query, n_results=initial_k)

        # Apply fast hybrid search if enabled
        if apply_hybrid and self.hybrid_searcher:
            results = self.hybrid_searcher.search(query, semantic_results, n_results=n_results)
        else:
            results = semantic_results

        # Apply slower reranking if enabled once we have the best candidates
        if apply_reranking and self.reranker:
            results = self.reranker.rerank(query, results, top_k=n_results)
        else:
            results = results[:n_results]

        return results

    @property
    def document_count(self) -> int:
        """Return the number of indexed documents."""
        return self.store.count()
