"""
Lab 4 FastAPI API.

@author: Kevin Lundeen
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 2.0.0+w26
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from retrieval.llm import LLMClient
from retrieval.rag import RAGSystem
from retrieval.retriever import DocumentRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global retriever instance
retriever: Optional[DocumentRetriever] = None
rag_system: Optional[RAGSystem] = None


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=1)
    n_context_docs: int = Field(default=3, ge=1, le=10)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)


class RAGResponse(BaseModel):
    question: str
    answer: str
    context: list[dict]
    context_count: int


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    documents_indexed: int
    message: str
    rag_available: bool  # NEW


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


# Define lifespan function to load models on startup
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Code before the 'yield' is executed during application startup
    try:
        logger.info("Loading models...")

        # Index documents from the documents/ directory
        global retriever
        retriever = DocumentRetriever()
        llm_client = LLMClient()  # default = Ollama
        global rag_system
        rag_system = RAGSystem(retriever=retriever, llm_client=llm_client)
        num_docs = retriever.index_documents("documents")
        logger.info(f"Indexed {num_docs} chunks successfully!")
    except Exception as e:
        # Don't crash the server, but log the error
        logger.error(f"Failed to load model: {str(e)}")

    yield  # The application starts receiving requests after the yield

    # Code after the 'yield' is executed during application shutdown
    logger.info("Application shutting down (lifespan)...")


# Initialize FastAPI app
app = FastAPI(
    title="FIXME: API Title",
    description="Lab3: Semantic search system using ChromaDB and sentence transformers",
    version="1.0.0",
    lifespan=lifespan,
)

# Add cross-origin resource sharing (CORS) middleware
# (gives browser permission to call our API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for documents relevant to the query.

    Args:
        request: SearchRequest with query and optional n_results

    Returns:
        SearchResponse with results
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


# Implement health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    global retriever, rag_system

    documents_indexed = 0
    if retriever is not None:
        documents_indexed = getattr(retriever, "document_count", 0) or 0

    rag_available = False
    if rag_system is not None:
        llm = getattr(rag_system, "llm_client", None)
        if llm is not None:
            try:
                rag_available = bool(llm.is_available())  # <-- change here
            except Exception:
                rag_available = False

    if retriever is None:
        return HealthResponse(
            status="unhealthy",
            documents_indexed=0,
            message="Retriever not initialized",
            rag_available=rag_available,
        )

    return HealthResponse(
        status="healthy",
        documents_indexed=documents_indexed,
        message="Service is running",
        rag_available=rag_available,
    )


# Add error handler for general exceptions
@app.exception_handler(Exception)
async def general_exception_handler(_request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Create a test endpoint that raises exceptions (only for testing!)
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
        result = rag_system.query(
            question=request.question,
            n_results=request.n_context_docs,
            temperature=request.temperature,
        )

        return RAGResponse(
            question=result["question"],
            answer=result["answer"],
            context=result["sources"],
            context_count=len(result["sources"]),
        )

    except Exception:
        logging.exception("RAG query failed")
        raise HTTPException(
            status_code=502,
            detail="LLM service unavailable or error occurred",
        )


# Mount static files LAST - catches all remaining routes
# including / --> /static/index.html, and
#           /stlye.css --> /static/style.css
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("To run this application:")
    print("uv run uvicorn src.retrieval.main:app --reload")
    print("\nThen open: http://localhost:8000")
