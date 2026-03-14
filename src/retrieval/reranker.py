from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Cross-encoder reranker using ms-marco MiniLM."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
        """
        Rerank candidates using a cross-encoder.

        Args:
            query: user query
            candidates: list of candidate result dicts
            top_n: number of final results

        Returns:
            Reranked list of result dicts
        """
        if not candidates:
            return []

        safe = [dict(c) for c in candidates]
        pairs = [(query, c["text"]) for c in safe]
        scores = self.model.predict(pairs)

        for c, s in zip(safe, scores):
            c["rerank_score"] = float(s)

        safe.sort(key=lambda x: x["rerank_score"], reverse=True)
        return safe[:top_n]
