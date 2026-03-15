# tests/test_main.py

import runpy
from contextlib import contextmanager
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import retrieval.main as main


@contextmanager
def _client_with_mocks(*, search_results=None, rag_result=None, rag_raises=False):
    if search_results is None:
        search_results = [{"doc_id": "d1", "text": "hello", "score": 1.0}]

    if rag_result is None:
        rag_result = {
            "question": "q",
            "answer": "a",
            "sources": [{"doc_id": "d1", "text": "ctx"}],
        }

    mock_retriever = Mock()
    mock_retriever.index_documents.return_value = 1
    mock_retriever.search.return_value = search_results
    mock_retriever.document_count = 1

    mock_rag = Mock()

    mock_llm = Mock()
    mock_llm.is_available.return_value = True
    mock_rag.llm_client = mock_llm  # <-- IMPORTANT (health uses rag_system.llm_client)

    if rag_raises:
        mock_rag.query.side_effect = RuntimeError("llm failed")
    else:
        mock_rag.query.return_value = rag_result

    with (
        patch("retrieval.main.DocumentRetriever", return_value=mock_retriever),
        patch("retrieval.main.LLMClient", return_value=mock_llm),  # <-- better
        patch("retrieval.main.RAGSystem", return_value=mock_rag),
    ):
        with TestClient(main.app) as client:
            client._mock_retriever = mock_retriever  # type: ignore[attr-defined]
            client._mock_rag = mock_rag  # type: ignore[attr-defined]
            yield client


def test_health_ok():
    with _client_with_mocks() as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "rag_available" in data
        assert "files_indexed" in data


def test_health_llm_is_available_exception_sets_false():
    with _client_with_mocks() as client:
        mock_rag = client._mock_rag  # type: ignore[attr-defined]
        mock_rag.llm_client.is_available.side_effect = RuntimeError("boom")

        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["rag_available"] is False


def test_search_empty_query_400():
    with _client_with_mocks() as client:
        r = client.post("/search", json={"query": "   ", "n_results": 3})
        assert r.status_code == 400
        assert "Query cannot be empty" in r.text


def test_search_success_calls_retriever():
    search_results = [{"doc_id": "d1", "text": "t1", "score": 0.9}]
    with _client_with_mocks(search_results=search_results) as client:
        r = client.post(
            "/search",
            json={
                "query": "hello",
                "n_results": 2,
                "use_hybrid": True,
                "use_reranking": False,
            },
        )
        assert r.status_code == 200
        data = r.json()

        # keep this robust: count should match returned results length
        assert data["query"] == "hello"
        assert data["count"] == len(data["results"])
        assert data["results"][0]["doc_id"] == "d1"

        mock_retriever = client._mock_retriever  # type: ignore[attr-defined]
        mock_retriever.search.assert_called_once()

        args, kwargs = mock_retriever.search.call_args
        assert args[0] == "hello"
        assert args[1] == 2
        assert kwargs["use_hybrid"] is True
        assert kwargs["use_reranking"] is False


def test_rag_success():
    rag_result = {
        "question": "What is RAG?",
        "answer": "RAG combines retrieval + generation.",
        "sources": [{"doc_id": "doc1", "text": "RAG info"}],
    }

    with _client_with_mocks(rag_result=rag_result) as client:
        r = client.post(
            "/rag",
            json={"question": "What is RAG?", "n_context_docs": 3, "temperature": 0.3},
        )
        assert r.status_code == 200
        data = r.json()

        assert data["question"] == "What is RAG?"
        assert data["answer"] == "RAG combines retrieval + generation."
        assert data["context_count"] == 1
        assert data["context"][0]["doc_id"] == "doc1"

        mock_rag = client._mock_rag  # type: ignore[attr-defined]
        mock_rag.query.assert_called_once()

        _, kwargs = mock_rag.query.call_args
        assert kwargs["question"] == "What is RAG?"
        assert kwargs["n_results"] == 3
        assert kwargs["temperature"] == 0.3
        assert kwargs["system_prompt"] is None
        assert kwargs["use_hybrid"] is True
        assert kwargs["use_reranking"] is True
        assert kwargs["pre_rerank_docs"] == 20
        assert kwargs["conversation"] == []


def test_rag_handles_llm_error_502():
    with _client_with_mocks(rag_raises=True) as client:
        r = client.post("/rag", json={"question": "hi"})
        assert r.status_code == 502
        assert "llm failed" in r.text


def test_rag_system_not_initialized_503():
    with _client_with_mocks() as client:
        main.rag_system = None
        r = client.post("/rag", json={"question": "hi"})
        assert r.status_code == 503


def test_rag_value_error_returns_400():
    with _client_with_mocks() as client:
        client._mock_rag.query.side_effect = ValueError("bad question")  # type: ignore[attr-defined]
        r = client.post("/rag", json={"question": "hi"})
        assert r.status_code == 400


def test_search_invalid_n_results_low_400():
    with _client_with_mocks() as client:
        r = client.post("/search", json={"query": "hello", "n_results": 0})
        assert r.status_code == 400


def test_search_invalid_n_results_high_400():
    with _client_with_mocks() as client:
        r = client.post("/search", json={"query": "hello", "n_results": 999})
        assert r.status_code == 400


def test_rag_empty_question_400():
    with _client_with_mocks() as client:
        r = client.post("/rag", json={"question": "   "})
        assert r.status_code == 400


def test_rag_invalid_n_context_docs_422():
    # Pydantic validation should reject this
    with _client_with_mocks() as client:
        r = client.post("/rag", json={"question": "hi", "n_context_docs": 0})
        assert r.status_code in (400, 422)


def test_rag_invalid_temperature_422():
    # Pydantic validation should reject if you used Field bounds
    with _client_with_mocks() as client:
        r = client.post("/rag", json={"question": "hi", "temperature": -1})
        assert r.status_code in (400, 422)


def test_search_handles_retriever_exception_returns_500():
    # Force retriever.search to throw so we cover the error handling branch
    search_results = [{"doc_id": "d1", "text": "ok"}]
    with _client_with_mocks(search_results=search_results) as client:
        client._mock_retriever.search.side_effect = RuntimeError("search failed")  # type: ignore[attr-defined]

        r = client.post("/search", json={"query": "hello", "n_results": 3})
        assert r.status_code in (500, 502)


def test_startup_failure_sets_unhealthy_and_search_503():
    main.retriever = None
    main.rag_system = None  # important

    with patch("retrieval.main.DocumentRetriever", side_effect=RuntimeError("boom")):
        with TestClient(main.app, raise_server_exceptions=False) as client:
            r = client.get("/health")
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "unhealthy"
            assert "files_indexed" in data

            r2 = client.post("/search", json={"query": "hi", "n_results": 3})
            assert r2.status_code == 503


def test_test_error_endpoint_uses_general_exception_handler():
    with TestClient(main.app, raise_server_exceptions=False) as client:
        r = client.get("/test/error")
        assert r.status_code == 500
        assert r.json() == {"detail": "Internal server error"}


def test_main_prints_run_instructions(capsys):
    # Covers the __main__ prints (lines 222-224)
    runpy.run_module("retrieval.main", run_name="__main__")
    out = capsys.readouterr().out
    assert "To run this application" in out
    assert "uv run uvicorn" in out
    assert "http://127.0.0.1:8081" in out
