"""
RAG orchestration: retrieval + prompt assembly + LLM generation.

@author: Alok Katiyar
Seattle University, ARIN 5360
"""

# src/retrieval/rag.py

from typing import Any, Optional

from retrieval.llm import LLMClient, LLMClientError
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
        return "\n".join(
            [
                "You are a helpful AI assistant for Galactic Gadgets.",
                "Use the provided context documents to answer the user's question.",
                "Guidelines:",
                "- Base your answer primarily on the provided context.",
                "- If the context is insufficient, say so explicitly and do not invent details.",
                "- Be concise but thorough.",
                "- When you use facts from the context, cite them using the bracketed doc number like [1], [2].",
                "- Use a friendly, professional tone.",
            ]
        )

    def _format_conversation(self, conversation: list[dict[str, Any]], max_turns: int = 5) -> str:
        if not conversation:
            return ""

        turns: list[str] = []
        # Keep the prompt focused: we only include the most recent turns.
        for msg in conversation[-max_turns * 2 :]:
            role = msg.get("role")
            content = msg.get("content")
            if not role or not content:
                continue
            if role == "user":
                turns.append(f"User: {str(content).strip()}")
            elif role == "assistant":
                turns.append(f"Assistant: {str(content).strip()}")
        return "\n".join(turns).strip()

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

Previous conversation (most recent last):

{{conversation}}

Based on the context above, please answer the following question.
If the context doesn't contain enough information, say so.

Question: {question}

Answer:"""

    def query(
        self,
        question: str,
        n_results: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        pre_rerank_docs: int = 20,
        conversation: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        if not question or not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        k = n_results if n_results is not None else self.n_context_docs

        results = self.retriever.search(
            question,
            k,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
            pre_rerank_docs=pre_rerank_docs,
        )

        context = self._build_context(results)
        convo_text = self._format_conversation(conversation or [])
        # We build a single user prompt that includes:
        # - retrieved context
        # - a short conversation recap (Option A)
        # This keeps the LLM call simple and easy to test.
        prompt = self._create_prompt(question, context).format(conversation=convo_text or "(none)")

        error: Optional[str] = None
        effective_system_prompt = (
            system_prompt.strip() if system_prompt and system_prompt.strip() else None
        )

        if self.llm_client:
            try:
                answer = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=effective_system_prompt or self._get_system_prompt(),
                    temperature=temperature,
                )
            except LLMClientError as e:
                error = str(e)
                answer = (
                    "I apologize, but I couldn't generate a response right now. Please try again."
                )
        else:
            error = "LLM not configured"
            answer = "LLM not configured. Returning retrieved documents only."

        return {
            "question": question,
            "answer": answer,
            "sources": results,
            "error": error,
        }
