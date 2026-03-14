"""
Simple unit tests for the DocumentRetriever class.

@author: Kevin Lundeen
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 3.0.0+w26
"""

from pathlib import Path

from retrieval.retriever import DocumentRetriever


def test_reranking_effectiveness():
    """
    Demonstrate that cross-encoder reranking improves search quality.

    This test compares search results with and without reranking to show:
    1. Reranking changes the order of results
    2. Reranked results are more relevant to the query
    3. The rerank_score field is added by the reranker

    This is a key demonstration of Project 2's core feature.
    """
    # Setup
    test_dir = Path(__file__).parent
    sample_dir = str(test_dir / "data")

    # Use a query that benefits from reranking
    query = "How does Van Helsing plan to defeat Dracula?"

    # Test WITHOUT reranking
    retriever_no_rerank = DocumentRetriever(enable_reranking=False, enable_hybrid=False)
    retriever_no_rerank.index_documents(sample_dir)
    results_no_rerank = retriever_no_rerank.search(query, n_results=5)

    # Test WITH reranking
    retriever_with_rerank = DocumentRetriever(enable_reranking=True, enable_hybrid=False)
    retriever_with_rerank.index_documents(sample_dir)
    results_with_rerank = retriever_with_rerank.search(query, n_results=5)

    # Verify both return results
    assert len(results_no_rerank) > 0, "Should return results without reranking"
    assert len(results_with_rerank) > 0, "Should return results with reranking"

    # Verify rerank_score field presence
    print(results_with_rerank[0])
    assert all("rerank_score" in doc for doc in results_with_rerank), (
        "Reranked results should have rerank_score field"
    )
    assert not any("rerank_score" in doc for doc in results_no_rerank), (
        "Non-reranked results should NOT have rerank_score field"
    )

    # Get top result chunks for comparison
    top_chunks_no_rerank = [doc["metadata"]["chunk"] for doc in results_no_rerank[:3]]
    top_chunks_with_rerank = [doc["metadata"]["chunk"] for doc in results_with_rerank[:3]]

    # Verify reranking changes the order
    assert top_chunks_no_rerank != top_chunks_with_rerank, (
        f"Reranking should change result order. "
        f"Without: {top_chunks_no_rerank}, With: {top_chunks_with_rerank}"
    )

    # Print comparison for visual inspection
    print("\n" + "=" * 80)
    print("RERANKING COMPARISON")
    print("=" * 80)
    print(f"\nQuery: '{query}'")
    print(f"\nTop 3 chunks WITHOUT reranking: {top_chunks_no_rerank}")
    print(f"Top 3 chunks WITH reranking:    {top_chunks_with_rerank}")

    print("\n" + "-" * 80)
    print("Middle result WITHOUT reranking:")
    print("-" * 80)
    print(f"Chunk: {results_no_rerank[1]['metadata']['chunk']}")
    print(f"Distance: {results_no_rerank[1]['distance']:.4f}")
    print(f"Text preview: {results_no_rerank[1]['text'][:150]}...")

    print("\n" + "-" * 80)
    print("Middle result WITH reranking:")
    print("-" * 80)
    print(f"Chunk: {results_with_rerank[1]['metadata']['chunk']}")
    print(f"Rerank score: {results_with_rerank[1]['rerank_score']:.4f}")
    print(f"Text preview: {results_with_rerank[1]['text'][:150]}...")

    print("\n" + "=" * 80)
    print("✓ Reranking successfully changed result ordering")
    print("=" * 80 + "\n")
