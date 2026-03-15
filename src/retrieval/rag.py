"""
RAG orchestration: routing + retrieval + query rewriting + prompt assembly + LLM generation.

@author: Alok Katiyar
Seattle University, ARIN 5360
"""

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
                "- If the context is insufficient, say so explicitly.",
                "- Do not invent information.",
                "- Be concise but thorough.",
                "- When using context facts, cite them like [1], [2].",
                "- Use a friendly and professional tone.",
            ]
        )

    def _get_smalltalk_system_prompt(self) -> str:
        return "\n".join(
            [
                "You are a friendly AI assistant for Galactic Gadgets.",
                "Answer casual conversation naturally and briefly.",
                "Do not mention document retrieval or sources unless asked.",
                "Be warm, professional, and helpful.",
            ]
        )

    def _classify_question(self, question: str) -> str:
        """
        Route questions into:
        - domain_qa: should use document retrieval
        - smalltalk: casual greeting or conversational message
        - out_of_scope: unrelated to Galactic Gadgets / its docs / products / support
        """

        q = question.strip().lower()

        smalltalk_patterns = {
            "hi",
            "hello",
            "hey",
            "how are you",
            "how are you?",
            "good morning",
            "good afternoon",
            "good evening",
            "thanks",
            "thank you",
            "who are you",
        }

        if q in smalltalk_patterns:
            return "smalltalk"

        domain_keywords = [
            "galactic gadgets",
            "product",
            "products",
            "headphones",
            "lamp",
            "astrolamp",
            "nebula",
            "nebulanoise",
            "faq",
            "rag",
            "reset",
            "setup",
            "troubleshoot",
            "support",
            "manual",
            "guide",
            "spec",
            "specification",
            "battery",
            "bluetooth",
            "noise cancellation",
            "transparency mode",
        ]

        if any(keyword in q for keyword in domain_keywords):
            return "domain_qa"

        # Use LLM classification if available for ambiguous cases.
        if self.llm_client:
            prompt = f"""
Classify the user's question into exactly one label:

- domain_qa -> about Galactic Gadgets products, support, setup, troubleshooting, FAQ, or knowledge base
- smalltalk -> greeting or casual conversation
- out_of_scope -> unrelated to Galactic Gadgets knowledge base

Reply with only one label.

Question: {question}
Label:
"""
            try:
                label = (
                    self.llm_client.generate(
                        prompt=prompt,
                        system_prompt="You classify user questions for a RAG assistant.",
                        temperature=0.0,
                        max_tokens=20,
                    )
                    .strip()
                    .lower()
                )

                if "domain_qa" in label:
                    return "domain_qa"
                if "smalltalk" in label:
                    return "smalltalk"
                if "out_of_scope" in label:
                    return "out_of_scope"
            except Exception:
                pass

        # Default conservative choice
        return "out_of_scope"

    def _rewrite_query(self, question: str) -> str:
        """
        Improve the user query before retrieval.
        This helps vector search find better documents.
        """

        if not self.llm_client:
            return question

        prompt = f"""
Rewrite the following user question so that it becomes a clear
search query for retrieving relevant Galactic Gadgets documents.

User question:
{question}

Rewritten search query:
"""

        try:
            rewritten = self.llm_client.generate(
                prompt=prompt,
                system_prompt="You improve search queries for document retrieval.",
                temperature=0.0,
                max_tokens=80,
            ).strip()

            if rewritten:
                print("Original query:", question)
                print("Rewritten query:", rewritten)
                return rewritten

            return question

        except Exception:
            return question

    def _format_conversation(
        self,
        conversation: list[dict[str, Any]],
        max_turns: int = 5,
    ) -> str:
        if not conversation:
            return ""

        turns: list[str] = []

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

    def _create_smalltalk_prompt(self, question: str) -> str:
        return f"""Respond naturally to the user's message.

User message: {question}

Answer:"""

    def _fallback_answer(
        self,
        results: list[dict[str, Any]],
        error: str,
    ) -> str:
        if not results:
            return (
                "I could not generate a model-based answer right now, "
                "and no relevant documents were retrieved."
            )

        snippets = []

        for i, r in enumerate(results[:3], start=1):
            text = (r.get("text") or r.get("content") or r.get("chunk") or "").strip()
            if text:
                snippets.append(f"[{i}] {text[:300]}")

        joined = "\n\n".join(snippets) if snippets else "Relevant context was found."

        return (
            "I could not generate a full model-based answer right now, "
            "so here is the most relevant retrieved information:\n\n"
            f"{joined}\n\n"
            f"Technical note: {error}"
        )

    def _answer_smalltalk(
        self,
        question: str,
        conversation: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, Optional[str]]:
        if not self.llm_client:
            return (
                "Hello! I'm doing well. I can help with Galactic Gadgets products, setup, troubleshooting, and FAQ questions.",
                "LLM not configured",
            )

        convo_text = self._format_conversation(conversation or [])
        prompt = f"""Previous conversation (most recent last):
{convo_text or "(none)"}

{self._create_smalltalk_prompt(question)}
"""

        try:
            answer = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self._get_smalltalk_system_prompt(),
                temperature=0.7,
                max_tokens=150,
            )
            return answer, None
        except LLMClientError as e:
            return (
                "Hello! I’m here and ready to help with Galactic Gadgets products, setup, troubleshooting, and FAQ questions.",
                str(e),
            )

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

        route = self._classify_question(question)
        print("Question route:", route)

        if route == "smalltalk":
            answer, smalltalk_error = self._answer_smalltalk(question, conversation)
            return {
                "question": question,
                "answer": answer,
                "sources": [],
                "error": smalltalk_error,
            }

        if route == "out_of_scope":
            return {
                "question": question,
                "answer": (
                    "I can best help with Galactic Gadgets products, setup, troubleshooting, "
                    "FAQ, and documentation. Please ask a question related to those topics."
                ),
                "sources": [],
                "error": None,
            }

        k = n_results if n_results is not None else self.n_context_docs

        search_query = self._rewrite_query(question)

        results = self.retriever.search(
            search_query,
            k,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
            pre_rerank_docs=pre_rerank_docs,
        )

        context = self._build_context(results)
        convo_text = self._format_conversation(conversation or [])
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
                answer = self._fallback_answer(results, error)
        else:
            error = "LLM not configured"
            answer = self._fallback_answer(results, error)

        return {
            "question": question,
            "answer": answer,
            "sources": results,
            "error": error,
        }
