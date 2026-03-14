"""
Simple unit tests for hybrid search in DocumentRetriever.

@author: Kevin Lundeen
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 3.1.0+w26
"""

from pathlib import Path

from retrieval.retriever import DocumentRetriever


def test_hybrid_search_effectiveness():
    """
    Demonstrate that hybrid search (semantic + BM25) improves retrieval quality.

    This test compares search results with semantic-only vs hybrid search to show:
    1. Hybrid search changes the order of results
    2. Hybrid captures both semantic meaning and exact keyword matches
    3. The rrf_score field is added by the hybrid searcher
    4. Results in both lists get better coverage with hybrid approach

    This is a key demonstration of Project 2's extra credit feature.
    """
    # Setup
    test_dir = Path(__file__).parent
    sample_dir = str(test_dir / "data")

    # Use a query with specific keywords that benefits from keyword matching
    # "Van Helsing" is a specific name that should match exactly
    query = "Van Helsing vampire hunter"

    # Test semantic-only (no hybrid)
    retriever_semantic = DocumentRetriever(enable_hybrid=False, enable_reranking=False)
    retriever_semantic.index_documents(sample_dir)
    results_semantic = retriever_semantic.search(query, n_results=5)

    # Test WITH hybrid search (semantic + BM25 + RRF)
    retriever_hybrid = DocumentRetriever(enable_hybrid=True, enable_reranking=False)
    retriever_hybrid.index_documents(sample_dir)
    results_hybrid = retriever_hybrid.search(query, n_results=5)

    # Verify both return results
    assert len(results_semantic) > 0, "Should return results with semantic-only"
    assert len(results_hybrid) > 0, "Should return results with hybrid search"

    # Verify rrf_score field presence in hybrid results
    assert all("rrf_score" in doc for doc in results_hybrid), (
        "Hybrid results should have rrf_score field"
    )
    assert not any("rrf_score" in doc for doc in results_semantic), (
        "Semantic-only results should NOT have rrf_score field"
    )

    # Get top result chunks for comparison
    top_chunks_semantic = [doc["metadata"]["chunk"] for doc in results_semantic[:3]]
    top_chunks_hybrid = [doc["metadata"]["chunk"] for doc in results_hybrid[:3]]

    # Verify hybrid search changes the order
    assert top_chunks_semantic != top_chunks_hybrid, (
        f"Hybrid search should change result order. "
        f"Semantic-only: {top_chunks_semantic}, Hybrid: {top_chunks_hybrid}"
    )

    # Print comparison for visual inspection
    print("\n" + "=" * 80)
    print("HYBRID SEARCH COMPARISON")
    print("=" * 80)
    print(f"\nQuery: '{query}'")
    print(f"\nTop 3 chunks with SEMANTIC-ONLY: {top_chunks_semantic}")
    print(f"Top 3 chunks with HYBRID:        {top_chunks_hybrid}")

    print("\n" + "-" * 80)
    print("Middle result with SEMANTIC-ONLY search:")
    print("-" * 80)
    print(f"Chunk: {results_semantic[1]['metadata']['chunk']}")
    print(f"Distance: {results_semantic[1]['distance']:.4f}")
    print(f"Text preview: {results_semantic[1]['text'][:150]}...")

    print("\n" + "-" * 80)
    print("Middle result with HYBRID search:")
    print("-" * 80)
    print(f"Chunk: {results_hybrid[1]['metadata']['chunk']}")
    print(f"RRF score: {results_hybrid[1]['rrf_score']:.4f}")
    if "distance" in results_hybrid[1]:
        print(f"Semantic distance: {results_hybrid[1]['distance']:.4f}")
    if "bm25_score" in results_hybrid[1]:
        print(f"BM25 score: {results_hybrid[1]['bm25_score']:.4f}")
    print(f"Text preview: {results_hybrid[0]['text'][:150]}...")

    print("\n" + "-" * 80)
    print("Analysis:")
    print("-" * 80)
    print("Hybrid search combines:")
    print("  • Semantic similarity (understanding meaning)")
    print("  • Keyword matching (exact term overlap via BM25)")
    print("  • Reciprocal Rank Fusion (combining both rankings)")
    print("\nThis gives better coverage by capturing documents that are either")
    print("semantically similar OR contain exact keyword matches.")

    print("\n" + "=" * 80)
    print("✓ Hybrid search successfully combined semantic and keyword results")
    print("=" * 80 + "\n")


def test_hybrid_vs_full_pipeline():
    """
    Compare hybrid search alone vs. full pipeline (hybrid + reranking).

    This demonstrates the complete P2 system with both features enabled.
    """
    # Setup
    test_dir = Path(__file__).parent
    sample_dir = str(test_dir / "data")

    query = "Mina Harker"

    # Hybrid only (no reranking)
    retriever_hybrid = DocumentRetriever(enable_hybrid=True, enable_reranking=False)
    retriever_hybrid.index_documents(sample_dir)
    results_hybrid = retriever_hybrid.search(query, n_results=5)

    # Full pipeline (hybrid + reranking)
    retriever_full = DocumentRetriever(enable_hybrid=True, enable_reranking=True)
    retriever_full.index_documents(sample_dir)
    results_full = retriever_full.search(query, n_results=5)

    # Verify both return results
    assert len(results_hybrid) > 0
    assert len(results_full) > 0

    # Verify score fields
    assert all("rrf_score" in doc for doc in results_hybrid)
    assert all("rrf_score" in doc for doc in results_full)
    assert all("rerank_score" in doc for doc in results_full)
    assert not any("rerank_score" in doc for doc in results_hybrid)

    # Get top result chunks for comparison
    top_chunks_full = [doc["metadata"]["chunk"] for doc in results_full[:3]]
    top_chunks_hybrid = [doc["metadata"]["chunk"] for doc in results_hybrid[:3]]

    # Verify hybrid search changes the order
    assert top_chunks_full != top_chunks_hybrid, (
        f"Hybrid search only should have different result order. "
        f"Full system: {top_chunks_full}, Hybrid: {top_chunks_hybrid}"
    )

    # Print comparison for visual inspection
    print("\n" + "=" * 80)
    print("HYBRID ONLY vs. FULL SEARCH COMPARISON")
    print("=" * 80)
    print(f"\nQuery: '{query}'")
    print(f"\nTop 3 chunks with FULL SYSTEM: {top_chunks_full}")
    print(f"Top 3 chunks with HYBRID ONLY: {top_chunks_hybrid}")

    print("\n" + "-" * 80)
    print("Top result with FULL SYSTEM search:")
    print("-" * 80)
    print(f"Chunk: {results_full[0]['metadata']['chunk']}")
    print(f"RRF score: {results_full[0]['rrf_score']:.4f}")
    if "distance" in results_full[0]:
        print(f"Semantic distance: {results_full[0]['distance']:.4f}")
    if "bm25_score" in results_full[0]:
        print(f"BM25 score: {results_full[0]['bm25_score']:.4f}")
    print(f"Text preview: {results_full[0]['text'][:150]}...")

    print("\n" + "-" * 80)
    print("Top result with HYBRID-ONLY search:")
    print("-" * 80)
    print(f"Chunk: {results_hybrid[0]['metadata']['chunk']}")
    print(f"RRF score: {results_hybrid[0]['rrf_score']:.4f}")
    if "distance" in results_hybrid[0]:
        print(f"Semantic distance: {results_hybrid[0]['distance']:.4f}")
    if "bm25_score" in results_hybrid[0]:
        print(f"BM25 score: {results_hybrid[0]['bm25_score']:.4f}")
    print(f"Text preview: {results_hybrid[0]['text'][:150]}...")

    print("\n" + "=" * 80)
    print("FULL PIPELINE COMPARISON")
    print("=" * 80)
    print(f"\nQuery: '{query}'")
    print("\nPipeline stages:")
    print("  Hybrid only:  Semantic + BM25 → RRF fusion")
    print("  Full system:  Semantic + BM25 → RRF fusion → Cross-encoder reranking")
    print("\n" + "=" * 80 + "\n")
