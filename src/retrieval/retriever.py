from __future__ import annotations

from typing import Dict, List

from retrieval.embeddings import DocumentEmbedder
from retrieval.hybrid import BM25Index, HybridSearcher
from retrieval.loader import DocumentChunker, DocumentLoader
from retrieval.reranker import CrossEncoderReranker
from retrieval.store import VectorStore


def _dedupe_by_filename_if_multi(
    results: List[dict],
    n_results: int,
    *,
    score_key: str,
    higher_better: bool,
) -> List[dict]:
    """
    Prefer diversity by filename IF multiple filenames exist, but ALWAYS return n_results
    by backfilling with remaining top-ranked items.

    This satisfies:
      - integration tests expecting exactly n_results
      - avoids a single long document dominating when many docs exist
      - does not break reranking tests (single-filename corpora are not deduped)
    """
    if not results:
        return []

    # Determine if multiple filenames exist in the candidate set
    filenames = []
    for r in results:
        md = r.get("metadata") or {}
        fn = md.get("filename")
        if fn is not None:
            filenames.append(fn)

    # If only one filename (or none), do not dedupe
    if len(set(filenames)) <= 1:
        return results[:n_results]

    def better(a: dict, b: dict) -> bool:
        av, bv = a.get(score_key), b.get(score_key)
        if av is None:
            return False
        if bv is None:
            return True
        return av > bv if higher_better else av < bv

    # First pass: pick best per filename
    best_by_file: Dict[str, dict] = {}
    for r in results:
        md = r.get("metadata") or {}
        key = md.get("filename") or str(r.get("id"))
        if key not in best_by_file or better(r, best_by_file[key]):
            best_by_file[key] = r

    selected = list(best_by_file.values())
    selected.sort(
        key=lambda x: x.get(score_key, float("-inf") if higher_better else float("inf")),
        reverse=higher_better,
    )

    # Backfill to ensure we always return n_results
    selected_ids = {str(r.get("id")) for r in selected}
    if len(selected) < n_results:
        for r in results:
            rid = str(r.get("id"))
            if rid in selected_ids:
                continue
            selected.append(r)
            selected_ids.add(rid)
            if len(selected) >= n_results:
                break

    return selected[:n_results]


class DocumentRetriever:
    """Document retrieval with optional reranking and optional hybrid search."""

    def __init__(
        self,
        chunk_size: int = 300,
        overlap: int = 30,
        *,
        use_reranking: bool = True,
        use_hybrid: bool = False,
        semantic_k: int = 20,
    ):
        chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        self.loader = DocumentLoader(chunker=chunker)
        self.store = VectorStore(DocumentEmbedder())
        self.reranker = CrossEncoderReranker()

        self.use_reranking = use_reranking
        self.use_hybrid = use_hybrid
        self.semantic_k = semantic_k

        self._indexed = False
        self._bm25: BM25Index | None = None
        self._hybrid: HybridSearcher | None = None

    def index_documents(self, directory: str) -> int:
        documents = self.loader.load_documents(directory)

        # Ensure required metadata fields exist for tests
        for d in documents:
            d["id"] = str(d.get("id"))
            d.setdefault("metadata", {})
            d["metadata"] = dict(d["metadata"] or {})
            d["metadata"].setdefault("chunk", d["id"])
            # filename should already exist from loader; keep if present

        # Empty directory should not crash (also build safe empty bm25)
        if not documents:
            self._bm25 = BM25Index.build([])
            self._hybrid = HybridSearcher(self._bm25)
            self._indexed = True
            return 0

        self.store.add_documents(documents)

        self._bm25 = BM25Index.build(documents)
        self._hybrid = HybridSearcher(self._bm25)

        self._indexed = True
        return self.document_count

    def search(
        self,
        query: str,
        n_results: int = 5,
        *,
        use_reranking: bool | None = None,
        use_hybrid: bool | None = None,
    ) -> List[dict]:
        if not self._indexed:
            raise ValueError("No documents indexed. Call index_documents() first.")

        do_rerank = self.use_reranking if use_reranking is None else use_reranking
        do_hybrid = self.use_hybrid if use_hybrid is None else use_hybrid

        semantic = self.store.search(query, self.semantic_k)
        semantic_ranked = sorted(semantic, key=lambda x: x.get("distance", 1e9))

        # Normalize metadata + ids for tests and for hybrid mapping
        for s in semantic_ranked:
            s["id"] = str(s.get("id"))
            s.setdefault("metadata", {})
            s["metadata"] = dict(s["metadata"] or {})
            s["metadata"].setdefault("chunk", s["id"])

        candidates = semantic_ranked

        if do_hybrid:
            if self._hybrid is None:
                raise ValueError("Hybrid index not built. Call index_documents() first.")
            candidates = self._hybrid.search(
                query,
                semantic_ranked=semantic_ranked,
                k_candidates=self.semantic_k,
            )
            for c in candidates:
                c["id"] = str(c.get("id"))
                c.setdefault("metadata", {})
                c["metadata"] = dict(c["metadata"] or {})
                c["metadata"].setdefault("chunk", c["id"])
        else:
            for c in candidates:
                c["id"] = str(c.get("id"))

        # No reranking: do NOT include rerank_score; semantic should NOT include rrf_score/bm25_score
        if not do_rerank:
            clean = []
            for c in candidates:
                d = dict(c)
                d["id"] = str(d.get("id"))
                d.setdefault("metadata", {})
                d["metadata"] = dict(d["metadata"] or {})
                d["metadata"].setdefault("chunk", d["id"])

                d.pop("rerank_score", None)
                if not do_hybrid:
                    d.pop("rrf_score", None)
                    d.pop("bm25_score", None)
                clean.append(d)

            if do_hybrid:
                clean.sort(key=lambda x: x.get("rrf_score", float("-inf")), reverse=True)
                return _dedupe_by_filename_if_multi(
                    clean, n_results, score_key="rrf_score", higher_better=True
                )

            clean.sort(key=lambda x: x.get("distance", 1e9))
            return _dedupe_by_filename_if_multi(
                clean, n_results, score_key="distance", higher_better=False
            )

        # Reranking
        reranked = self.reranker.rerank(
            query,
            candidates,
            top_n=min(self.semantic_k, len(candidates)),
        )
        for r in reranked:
            r["id"] = str(r.get("id"))
            r.setdefault("metadata", {})
            r["metadata"] = dict(r["metadata"] or {})
            r["metadata"].setdefault("chunk", r["id"])

        # Preserve hybrid scores if present
        if do_hybrid:
            cand_by_id = {str(c["id"]): c for c in candidates}
            for r in reranked:
                cid = str(r["id"])
                if cid in cand_by_id:
                    if "rrf_score" in cand_by_id[cid]:
                        r["rrf_score"] = cand_by_id[cid]["rrf_score"]
                    if "bm25_score" in cand_by_id[cid]:
                        r["bm25_score"] = cand_by_id[cid]["bm25_score"]

            reranked.sort(key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)
            return _dedupe_by_filename_if_multi(
                reranked, n_results, score_key="rerank_score", higher_better=True
            )

        reranked.sort(key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)
        return _dedupe_by_filename_if_multi(
            reranked, n_results, score_key="rerank_score", higher_better=True
        )

    @property
    def document_count(self) -> int:
        return self.store.count()
