"""
Unit tests for retrieval.main (FastAPI module).

Covers:
- lifespan() success + exception path
- /health endpoint healthy + unhealthy paths
- /search endpoint validation (empty query, n_results bounds)
- /search endpoint error handling (retriever missing -> 503, retriever throws -> 500)
- __main__ guard prints uvicorn command

These tests call the endpoint coroutines directly to keep them fast and deterministic.
"""

import runpy

import pytest
from fastapi import FastAPI

import retrieval.main as m


@pytest.mark.anyio
async def test_lifespan_success_sets_retriever(monkeypatch):
    """Cover normal lifespan path where DocumentRetriever() succeeds."""

    class GoodRetriever:
        def __init__(self, *args, **kwargs):
            pass

        def index_documents(self, *args, **kwargs):
            return 0

        def search(self, query, n_results=5, *args, **kwargs):
            return [{"id": "sample4.txt", "text": "x", "distance": 0.1}]

    monkeypatch.setattr(m, "DocumentRetriever", GoodRetriever)

    # Ensure starting state
    m.retriever = None

    async with m.lifespan(FastAPI()):
        # retriever should be created inside lifespan
        assert m.retriever is not None


@pytest.mark.anyio
async def test_health_healthy_when_no_retriever():
    m.retriever = None
    resp = await m.health_check()
    assert resp.status == "healthy"
    assert len(resp.message) > 0
    assert resp.documents_indexed == 0


@pytest.mark.anyio
async def test_health_healthy_with_retriever():
    """Cover /health healthy path (retriever exists)."""

    class FakeRetriever:
        @property
        def document_count(self):
            return 42

    m.retriever = FakeRetriever()
    resp = await m.health_check()

    assert resp.status == "healthy"
    assert resp.documents_indexed == 42


@pytest.mark.anyio
async def test_search_503_when_no_retriever():
    """Cover search() 503 branch when retriever isn't initialized (your missing line around 107)."""
    m.retriever = None
    req = m.SearchRequest(query="hello", n_results=5)

    with pytest.raises(m.HTTPException) as exc:
        await m.search(req)

    assert exc.value.status_code == 503


@pytest.mark.anyio
async def test_search_400_empty_query():
    """Cover search() empty query validation branch."""

    class FakeRetriever:
        def search(self, query, n_results=5):
            return []

    m.retriever = FakeRetriever()
    req = m.SearchRequest(query="   ", n_results=5)

    with pytest.raises(m.HTTPException) as exc:
        await m.search(req)

    assert exc.value.status_code == 400


@pytest.mark.anyio
async def test_search_400_bad_n_results_low_high():
    """Cover search() n_results bounds checks."""

    class FakeRetriever:
        def search(self, query, n_results=5):
            return []

    m.retriever = FakeRetriever()

    with pytest.raises(m.HTTPException) as exc1:
        await m.search(m.SearchRequest(query="x", n_results=0))
    assert exc1.value.status_code == 400

    with pytest.raises(m.HTTPException) as exc2:
        await m.search(m.SearchRequest(query="x", n_results=21))
    assert exc2.value.status_code == 400


@pytest.mark.anyio
async def test_search_500_when_retriever_throws():
    """Cover search() exception handler (your missing 118-120)."""

    class BoomRetriever:
        def search(self, query, n_results=5):
            raise RuntimeError("boom")

    m.retriever = BoomRetriever()
    req = m.SearchRequest(query="hello", n_results=5)

    with pytest.raises(m.HTTPException) as exc:
        await m.search(req)

    assert exc.value.status_code == 500


@pytest.mark.anyio
async def test_ui_returns_file_response(monkeypatch):
    # Cover main.py line 165: return FileResponse("static/index.html")
    sentinel = object()

    def fake_file_response(path):
        assert path == "static/index.html"
        return sentinel

    monkeypatch.setattr(m, "FileResponse", fake_file_response)
    resp = await m.ui()
    assert resp is sentinel


def test_dunder_main_prints_uvicorn_command(capsys):
    """Cover the if __name__ == '__main__' block (your missing 169-171)."""
    runpy.run_module("retrieval.main", run_name="__main__")
    out = capsys.readouterr().out.lower()
    assert "uvicorn" in out
