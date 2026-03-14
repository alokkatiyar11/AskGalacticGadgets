"""
Lab 3 FastAPI API.

@author: Aarti Dashore, Alok Katiyar
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 1.1.0+w26
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from src.retrieval.retriever import DocumentRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

retriever: DocumentRetriever | None = None


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    message: str


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5

    # Pipeline switches (for UI / extra credit comparison)
    use_reranking: bool = True
    compare: bool = False
    use_hybrid: bool = False


class SearchResponse(BaseModel):
    query: str
    results: list[dict]
    count: int


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global retriever
    retriever = DocumentRetriever(use_reranking=True)
    try:
        logger.info("Loading models and indexing documents...")
        num_docs = retriever.index_documents("documents")
        logger.info("Indexed %s documents successfully!", num_docs)
    except Exception as e:
        logger.error("Startup failed: %s", str(e))

    yield
    logger.info("Application shutting down (lifespan)...")


app = FastAPI(
    title="Document Retrieval System API",
    description="Semantic search system using ChromaDB + sentence transformers, with optional Cross-Encoder Re-Ranking",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search")
async def search(request: SearchRequest):
    """
    Search for documents relevant to the query.

    Modes:
      - compare=False: returns {query,results,count}
      - compare=True: returns {query,semantic,reranked,hybrid,hybrid_reranked,reranking_changed}

    Notes:
      - distance is always included (vector search)
      - rerank_score is included for reranked results
      - rrf_score/bm25_score are included for hybrid results
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if request.n_results < 1 or request.n_results > 20:
        raise HTTPException(status_code=400, detail="n_results must be between 1 and 20")

    try:
        if request.compare:
            # Semantic only
            semantic = retriever.search(
                request.query,
                request.n_results,
                use_reranking=False,
                use_hybrid=False,
            )

            # Semantic + rerank
            reranked = retriever.search(
                request.query,
                request.n_results,
                use_reranking=True,
                use_hybrid=False,
            )

            # Hybrid only
            hybrid = retriever.search(
                request.query,
                request.n_results,
                use_reranking=False,
                use_hybrid=True,
            )

            # Hybrid + rerank
            hybrid_reranked = retriever.search(
                request.query,
                request.n_results,
                use_reranking=True,
                use_hybrid=True,
            )

            s_ids = [r.get("id") for r in semantic]
            r_ids = [r.get("id") for r in reranked]
            changed = s_ids != r_ids

            return {
                "query": request.query,
                "semantic": semantic,
                "reranked": reranked,
                "hybrid": hybrid,
                "hybrid_reranked": hybrid_reranked,
                "reranking_changed": changed,
            }

        results = retriever.search(
            request.query,
            request.n_results,
            use_reranking=request.use_reranking,
            use_hybrid=request.use_hybrid,  # ✅ minimal but critical
        )
        return SearchResponse(query=request.query, results=results, count=len(results))

    except Exception as e:
        logger.error("Search error: %s", str(e))
        raise HTTPException(status_code=500, detail="Search failed")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if retriever is None:
        return HealthResponse(
            status="healthy",
            message="API is running; retriever not initialized yet",
            documents_indexed=0,
        )

    return HealthResponse(
        status="healthy",
        message="API is running and ready",
        documents_indexed=retriever.document_count,
    )


@app.exception_handler(Exception)
async def general_exception_handler(_request, exc):
    logger.error("Unexpected error: %s", str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/test/error")
async def test_error():
    raise RuntimeError("Something went wrong")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def ui():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    print("uvicorn src.retrieval.main:app --reload")
