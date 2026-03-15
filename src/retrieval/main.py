"""
Lab 4 FastAPI API.

@author: Alok Katiyar
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 2.0.0+w26
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from retrieval.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from retrieval.llm import LLMClient
from retrieval.rag import RAGSystem
from retrieval.retriever import DocumentRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global retriever instance
retriever: Optional[DocumentRetriever] = None
rag_system: Optional[RAGSystem] = None

# -----------------------------
# In-memory chat storage
# -----------------------------
chat_sessions: dict[str, dict] = {}


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=1)
    n_context_docs: int = Field(default=3, ge=1, le=10)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    system_prompt: Optional[str] = None
    use_hybrid: bool = Field(default=True)
    use_reranking: bool = Field(default=True)
    pre_rerank_docs: int = Field(default=20, ge=5, le=50)
    conversation: list[dict] = Field(default_factory=list)
    chat_id: Optional[str] = None


class RAGResponse(BaseModel):
    question: str
    answer: str
    context: list[dict]
    context_count: int
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    documents_indexed: int
    files_indexed: int
    message: str
    rag_available: bool


class SearchRequest(BaseModel):
    """Request model for search."""

    query: str
    n_results: int = 5
    use_hybrid: bool = True
    use_reranking: bool = True


class SearchResponse(BaseModel):
    """Response model for search."""

    query: str
    results: list[dict]
    count: int


class CreateChatResponse(BaseModel):
    chat_id: str
    title: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatSession(BaseModel):
    chat_id: str
    title: str
    messages: list[ChatMessage]


# Define lifespan function to load models on startup
@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        logger.info("Loading models...")

        global retriever
        retriever = DocumentRetriever()

        llm_client = LLMClient(
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
        )

        global rag_system
        rag_system = RAGSystem(retriever=retriever, llm_client=llm_client)

        num_docs = retriever.index_documents("documents")
        logger.info(f"Indexed {num_docs} chunks successfully!")

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")

    yield

    logger.info("Application shutting down (lifespan)...")


# Initialize FastAPI app
app = FastAPI(
    title="Galactic Gadgets RAG Chatbot",
    description="P3: RAG chatbot system using hybrid retrieval + reranking and LLM generation",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_chat_exists(chat_id: str) -> dict:
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat_sessions[chat_id]


@app.get("/about")
async def about():
    return {
        "app": "Galactic Gadgets RAG Assistant",
        "author": "Alok Katiyar",
        "course": "ARIN 5360",
        "version": "2.0.0+w26",
        "description": "RAG chatbot with hybrid retrieval, reranking, and conversational history.",
    }


@app.post("/chats/new", response_model=CreateChatResponse)
async def create_chat():
    chat_id = str(uuid.uuid4())

    chat_sessions[chat_id] = {
        "chat_id": chat_id,
        "title": "New Chat",
        "messages": [],
    }

    return CreateChatResponse(chat_id=chat_id, title="New Chat")


@app.get("/chats")
async def list_chats():
    return [
        {
            "chat_id": chat["chat_id"],
            "title": chat["title"],
            "message_count": len(chat["messages"]),
        }
        for chat in reversed(list(chat_sessions.values()))
    ]


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    return ensure_chat_exists(chat_id)


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    ensure_chat_exists(chat_id)
    del chat_sessions[chat_id]
    return {"detail": "Chat deleted"}


@app.post("/chats/{chat_id}/clear")
async def clear_chat(chat_id: str):
    chat = ensure_chat_exists(chat_id)
    chat["messages"] = []
    chat["title"] = "New Chat"
    return {"detail": "Chat cleared"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for documents relevant to the query.
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if request.n_results < 1 or request.n_results > 20:
        raise HTTPException(status_code=400, detail="n_results must be between 1 and 20")

    try:
        results = retriever.search(
            request.query,
            request.n_results,
            use_hybrid=request.use_hybrid,
            use_reranking=request.use_reranking,
        )
        return SearchResponse(query=request.query, results=results, count=len(results))
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    global retriever, rag_system

    documents_indexed = 0
    files_indexed = 0

    if retriever is not None:
        try:
            documents_indexed = int(getattr(retriever, "document_count", 0) or 0)
        except (TypeError, ValueError):
            documents_indexed = 0

        try:
            files_indexed = int(getattr(retriever, "file_count", 0) or 0)
        except (TypeError, ValueError):
            files_indexed = 0

    rag_available = False
    if rag_system is not None:
        llm = getattr(rag_system, "llm_client", None)
        if llm is not None:
            try:
                rag_available = bool(llm.is_available())
            except Exception:
                rag_available = False

    if retriever is None:
        return HealthResponse(
            status="unhealthy",
            documents_indexed=0,
            files_indexed=0,
            message="Retriever not initialized",
            rag_available=rag_available,
        )

    return HealthResponse(
        status="healthy",
        documents_indexed=documents_indexed,
        files_indexed=files_indexed,
        message="Service is running",
        rag_available=rag_available,
    )


@app.exception_handler(Exception)
async def general_exception_handler(_request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/test/error")
async def test_error():
    raise RuntimeError("Something went wrong")


@app.post("/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    Run full RAG pipeline:
      - retrieve docs
      - build context
      - generate answer
      - return answer + sources
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        chat_history_for_llm = request.conversation

        if request.chat_id and request.chat_id in chat_sessions:
            chat = chat_sessions[request.chat_id]
            chat_history_for_llm = chat["messages"]

        result = rag_system.query(
            question=request.question,
            n_results=request.n_context_docs,
            temperature=request.temperature,
            system_prompt=request.system_prompt,
            use_hybrid=request.use_hybrid,
            use_reranking=request.use_reranking,
            pre_rerank_docs=request.pre_rerank_docs,
            conversation=chat_history_for_llm,
        )

        if request.chat_id:
            if request.chat_id not in chat_sessions:
                chat_sessions[request.chat_id] = {
                    "chat_id": request.chat_id,
                    "title": "New Chat",
                    "messages": [],
                }

            chat_sessions[request.chat_id]["messages"].append(
                {
                    "role": "user",
                    "content": request.question,
                }
            )
            chat_sessions[request.chat_id]["messages"].append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                }
            )

            if chat_sessions[request.chat_id]["title"] == "New Chat":
                chat_sessions[request.chat_id]["title"] = request.question[:40]

        return RAGResponse(
            question=result["question"],
            answer=result["answer"],
            context=result["sources"],
            context_count=len(result["sources"]),
            error=result.get("error"),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("RAG query failed")
        raise HTTPException(
            status_code=502,
            detail=str(e) or "LLM service unavailable or error occurred",
        )


# Mount static files LAST
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("To run this application:")
    print("uv run uvicorn src.retrieval.main:app --reload --host 127.0.0.1 --port 8081")
    print("\nThen open: http://127.0.0.1:8081")
