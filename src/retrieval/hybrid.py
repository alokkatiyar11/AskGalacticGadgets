"""
retrieval.hybrid

Hybrid search utilities for P2 extra credit:
  - BM25 keyword search (rank-bm25)
  - Reciprocal Rank Fusion (RRF) between BM25 and semantic rankings

Test contract expectations:
  - Hybrid results include 'rrf_score'
  - Results include 'metadata' with 'chunk' (and usually 'filename')
  - Empty directory indexing must not crash (rank-bm25 empty corpus guard)
  - IDs are normalized to strings to prevent int/str mismatches
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def rrf_fuse(
    ranked_lists: List[List[dict]],
    *,
    k: int = 60,
    top_n: int = 20,
    id_key: str = "id",
) -> List[dict]:
    """Reciprocal Rank Fusion (RRF) across ranked lists (best→worst)."""
    scores: Dict[str, float] = {}
    payload: Dict[str, dict] = {}

    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            doc_id = str(item[id_key])
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in payload:
                row = dict(item)
                row["id"] = doc_id
                row.setdefault("metadata", {})
                # ensure chunk exists
                row["metadata"] = dict(row["metadata"] or {})
                row["metadata"].setdefault("chunk", doc_id)
                payload[doc_id] = row

    fused_ids = sorted(scores.keys(), key=lambda _id: scores[_id], reverse=True)[:top_n]
    out: List[dict] = []
    for _id in fused_ids:
        row = dict(payload[_id])
        row.setdefault("metadata", {})
        row["metadata"] = dict(row["metadata"] or {})
        row["metadata"].setdefault("chunk", _id)
        row["rrf_score"] = float(scores[_id])
        out.append(row)
    return out


@dataclass
class BM25Index:
    ids: List[str]
    texts: List[str]
    metadatas: List[dict]
    bm25: Optional[object]  # rank_bm25.BM25Okapi | None for empty

    @classmethod
    def build(cls, docs: List[dict]) -> "BM25Index":
        # Empty index: avoid rank-bm25 empty-corpus ZeroDivisionError
        if not docs:
            return cls(ids=[], texts=[], metadatas=[], bm25=None)

        try:
            from rank_bm25 import BM25Okapi
        except Exception as e:  # pragma: no cover
            raise ImportError("rank-bm25 is required. Install with: uv add rank-bm25") from e

        # Normalize ids to strings
        ids = [str(d["id"]) for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [d.get("metadata", {}) or {} for d in docs]

        corpus = [tokenize(t) for t in texts]
        return cls(ids=ids, texts=texts, metadatas=metadatas, bm25=BM25Okapi(corpus))

    def search(self, query: str, k: int = 20) -> List[dict]:
        if self.bm25 is None or not self.ids:
            return []

        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        out: List[dict] = []
        for i in ranked:
            doc_id = str(self.ids[i])
            md = self.metadatas[i] if self.metadatas[i] is not None else {}
            md = dict(md or {})
            md.setdefault("chunk", doc_id)

            out.append(
                {
                    "id": doc_id,
                    "text": self.texts[i],
                    "metadata": md,
                    "bm25_score": float(scores[i]),
                }
            )
        return out


class HybridSearcher:
    def __init__(self, bm25_index: BM25Index, *, rrf_k: int = 60):
        self.bm25 = bm25_index
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        semantic_ranked: List[dict],
        k_candidates: int = 20,
    ) -> List[dict]:
        # BM25 list (keyword)
        bm25_ranked = self.bm25.search(query, k=k_candidates)

        # Normalize semantic ids to strings (defensive)
        semantic_ranked_norm = []
        for d in semantic_ranked:
            x = dict(d)
            x["id"] = str(x.get("id"))
            x.setdefault("metadata", {})
            x["metadata"] = dict(x["metadata"] or {})
            x["metadata"].setdefault("chunk", x["id"])
            semantic_ranked_norm.append(x)

        sem_by_id = {d["id"]: d for d in semantic_ranked_norm}
        bm_by_id = {d["id"]: d for d in bm25_ranked}

        fused = rrf_fuse([semantic_ranked_norm, bm25_ranked], k=self.rrf_k, top_n=k_candidates)

        enriched: List[dict] = []
        for row in fused:
            _id = str(row["id"])
            base = dict(sem_by_id.get(_id, bm_by_id.get(_id, row)))

            base["id"] = _id
            base.setdefault("metadata", {})
            base["metadata"] = dict(base["metadata"] or {})
            base["metadata"].setdefault("chunk", _id)

            base["rrf_score"] = row["rrf_score"]

            # carry bm25 score if available
            if _id in bm_by_id:
                base["bm25_score"] = bm_by_id[_id].get("bm25_score")

            # ensure text exists
            if "text" not in base and _id in bm_by_id:
                base["text"] = bm_by_id[_id]["text"]

            enriched.append(base)

        return enriched
