# src/retrieval/rag.py

from typing import Any, Optional

from retrieval.llm import LLMClient
from retrieval.retriever import DocumentRetriever


class RAGSystem:
    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_client: Optional[LLMClient] = None,
        n_context_docs: int = 3,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.n_context_docs = n_context_docs

    def _get_system_prompt(self) -> str:
        return (
            "You are a helpful assistant. Use only the provided context to answer. "
            "If the answer is not in the context, say you don't have enough information."
        )

    def _build_context(self, results: list[dict[str, Any]]) -> str:
        parts = []
        for i, r in enumerate(results, start=1):
            doc_id = r.get("doc_id") or r.get("id") or r.get("source") or f"doc_{i}"
            text = r.get("text") or r.get("content") or r.get("chunk") or ""
            parts.append(f"[{i}] {doc_id}\n{text.strip()}\n")
        return "\n".join(parts).strip()

    def _create_prompt(self, question: str, context: str) -> str:
        return f"""Context information from relevant documents:

{context}

Based on the context above, please answer the following question.
If the context doesn't contain enough information, say so.

Question: {question}

Answer:"""

    def query(
        self,
        question: str,
        n_results: Optional[int] = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        if not question or not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        k = n_results if n_results is not None else self.n_context_docs

        results = self.retriever.search(question, k)

        context = self._build_context(results)
        prompt = self._create_prompt(question, context)

        if self.llm_client:
            answer = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                temperature=temperature,
            )
        else:
            answer = "LLM not configured. Returning retrieved documents only."

        return {
            "question": question,
            "answer": answer,
            "sources": results,
        }
