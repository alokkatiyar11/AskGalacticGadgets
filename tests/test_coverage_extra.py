from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import retrieval.main as main
from retrieval.llm import LLMClient, LLMClientError
from retrieval.loader import DocumentLoader
from retrieval.rag import RAGSystem


def test_llm_native_invalid_json_raises(monkeypatch):
    class _BadResp:
        status_code = 200
        text = "not-json"

        def json(self):
            raise ValueError("nope")

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            return _BadResp()

    monkeypatch.setattr("retrieval.llm.httpx.Client", _Client)

    client = LLMClient()
    with pytest.raises(LLMClientError, match="Invalid JSON"):
        client.generate("hello")


def test_llm_openai_bad_shape_raises(monkeypatch):
    class _NativeFailResp:
        status_code = 404
        text = "no"

        def json(self):
            return {}

    class _OpenAIResp:
        status_code = 200
        text = '{"wrong":true}'

        def json(self):
            return {"wrong": True}

    class _Client:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, *args, **kwargs):
            self.calls += 1
            if "/api/generate" in url:
                return _NativeFailResp()
            return _OpenAIResp()

    monkeypatch.setattr("retrieval.llm.httpx.Client", _Client)

    client = LLMClient()
    with pytest.raises(LLMClientError, match="Unexpected OpenAI-compatible"):
        client.generate("hello")


def test_llm_openai_empty_content_raises(monkeypatch):
    class _NativeFailResp:
        status_code = 404
        text = "no"

        def json(self):
            return {}

    class _OpenAIResp:
        status_code = 200
        text = "ok"

        def json(self):
            return {"choices": [{"message": {"content": "   "}}]}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, *args, **kwargs):
            if "/api/generate" in url:
                return _NativeFailResp()
            return _OpenAIResp()

    monkeypatch.setattr("retrieval.llm.httpx.Client", _Client)

    client = LLMClient()
    with pytest.raises(LLMClientError, match="Empty response"):
        client.generate("hello")


def test_loader_pdf_no_chunker_returns_single_doc(monkeypatch, tmp_path: Path):
    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    class _Page:
        def extract_text(self):
            return "Hello PDF"

    class _Reader:
        pages = [_Page()]

        def __init__(self, _filepath):
            pass

    monkeypatch.setattr("retrieval.loader.pypdf.PdfReader", _Reader)

    loader = DocumentLoader(chunker=None)
    out = loader._load_pdf_file(fake_pdf)

    assert len(out) == 1
    assert out[0]["text"] == "Hello PDF"
    assert out[0]["metadata"]["type"] == "pdf"


def test_rag_domain_flow_with_rewrite_and_llm_answer(capsys):
    retriever = Mock()
    retriever.search.return_value = [{"doc_id": "d1", "text": "ctx", "score": 0.9}]

    llm = Mock()

    def _generate(prompt: str, **kwargs):
        if "Classify the user's question" in prompt:
            return "domain_qa"
        if "Rewrite the following user question" in prompt:
            return "galactic gadgets battery"
        return "final answer"

    llm.generate.side_effect = _generate

    rag = RAGSystem(retriever=retriever, llm_client=llm)
    result = rag.query("Battery?", conversation=[{"role": "user", "content": "hi"}])

    assert result["answer"] == "final answer"
    assert result["sources"][0]["doc_id"] == "d1"
    assert result["error"] is None

    out = capsys.readouterr().out
    assert "Original query:" in out
    assert "Rewritten query:" in out


def test_rag_rewrite_exception_falls_back_to_original(monkeypatch):
    retriever = Mock()
    retriever.search.return_value = [{"doc_id": "d1", "text": "ctx", "score": 0.9}]

    llm = Mock()

    def _generate(prompt: str, **kwargs):
        if "Classify the user's question" in prompt:
            return "domain_qa"
        if "Rewrite the following user question" in prompt:
            raise LLMClientError("boom")
        return "answer"

    llm.generate.side_effect = _generate

    rag = RAGSystem(retriever=retriever, llm_client=llm)
    result = rag.query("galactic gadgets setup")

    assert result["answer"] == "answer"
    retriever.search.assert_called_once()
    args, _kwargs = retriever.search.call_args
    assert args[0] == "galactic gadgets setup"


def test_rag_fallback_answer_no_results():
    rag = RAGSystem(retriever=Mock(), llm_client=None)
    msg = rag._fallback_answer([], "err")
    assert "no relevant documents" in msg


def test_rag_smalltalk_without_llm_returns_error():
    rag = RAGSystem(retriever=Mock(), llm_client=None)
    result = rag.query("hi")
    assert result["sources"] == []
    assert result["error"] == "LLM not configured"


def test_main_chat_endpoints_and_chat_id_history_used():
    mock_retriever = Mock()
    mock_retriever.index_documents.return_value = 1
    mock_retriever.search.return_value = [{"doc_id": "d1", "text": "ctx", "score": 1.0}]
    mock_retriever.document_count = "bad"
    mock_retriever.file_count = None

    mock_llm = Mock()
    mock_llm.is_available.return_value = True

    mock_rag = Mock()
    mock_rag.llm_client = mock_llm
    mock_rag.query.return_value = {
        "question": "q",
        "answer": "a",
        "sources": [{"doc_id": "d1", "text": "ctx"}],
        "error": None,
    }

    with (
        pytest.MonkeyPatch().context() as mp,
    ):
        mp.setattr(main, "retriever", mock_retriever)
        mp.setattr(main, "rag_system", mock_rag)
        mp.setattr(main, "chat_sessions", {})

        client = TestClient(main.app)

        r = client.post("/chats/new")
        assert r.status_code == 200
        chat_id = r.json()["chat_id"]

        r = client.get("/chats")
        assert r.status_code == 200
        assert r.json()[0]["chat_id"] == chat_id

        r = client.get(f"/chats/{chat_id}")
        assert r.status_code == 200

        r = client.post(f"/chats/{chat_id}/clear")
        assert r.status_code == 200

        r = client.post(
            "/rag",
            json={"question": "galactic gadgets", "chat_id": chat_id, "conversation": []},
        )
        assert r.status_code == 200
        assert (
            mock_rag.query.call_args.kwargs["conversation"]
            is main.chat_sessions[chat_id]["messages"]
        )

        r = client.post(
            "/rag",
            json={
                "question": "galactic gadgets",
                "chat_id": chat_id,
                "conversation": [{"role": "user", "content": "x"}],
            },
        )
        assert r.status_code == 200
        assert (
            mock_rag.query.call_args.kwargs["conversation"]
            == main.chat_sessions[chat_id]["messages"]
        )

        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["documents_indexed"] == 0
        assert data["files_indexed"] == 0

        r = client.delete(f"/chats/{chat_id}")
        assert r.status_code == 200

        r = client.get(f"/chats/{chat_id}")
        assert r.status_code == 404


def test_main_search_retriever_missing_and_search_error():
    client = TestClient(main.app)

    mp = pytest.MonkeyPatch()
    mp.setattr(main, "retriever", None)

    r = client.post("/search", json={"query": "hi", "n_results": 3})
    assert r.status_code == 503

    mock_retriever = Mock()
    mock_retriever.search.side_effect = RuntimeError("boom")
    mp.setattr(main, "retriever", mock_retriever)

    r = client.post("/search", json={"query": "hi", "n_results": 3})
    assert r.status_code == 500

    mp.undo()


def test_main_about_and_missing_chat_paths():
    client = TestClient(main.app)

    r = client.get("/about")
    assert r.status_code == 200
    data = r.json()
    assert data["app"]
    assert data["author"]

    missing_id = "missing"
    r = client.get(f"/chats/{missing_id}")
    assert r.status_code == 404
    r = client.post(f"/chats/{missing_id}/clear")
    assert r.status_code == 404
    r = client.delete(f"/chats/{missing_id}")
    assert r.status_code == 404


def test_main_rag_creates_chat_session_when_missing():
    mock_retriever = Mock()
    mock_retriever.search.return_value = []
    mock_retriever.index_documents.return_value = 0
    mock_retriever.document_count = 0
    mock_retriever.file_count = 0

    mock_llm = Mock()
    mock_llm.is_available.return_value = True

    mock_rag = Mock()
    mock_rag.llm_client = mock_llm
    mock_rag.query.return_value = {"question": "q", "answer": "a", "sources": [], "error": None}

    with (
        pytest.MonkeyPatch().context() as mp,
    ):
        mp.setattr(main, "retriever", mock_retriever)
        mp.setattr(main, "rag_system", mock_rag)
        mp.setattr(main, "chat_sessions", {})

        client = TestClient(main.app)

        new_chat_id = "new-chat"
        r = client.post("/rag", json={"question": "galactic gadgets", "chat_id": new_chat_id})
        assert r.status_code == 200
        assert new_chat_id in main.chat_sessions


def test_rag_fallback_answer_with_snippets_and_no_snippets():
    rag = RAGSystem(retriever=Mock(), llm_client=None)

    msg = rag._fallback_answer(
        [{"doc_id": "d1", "text": "hello"}, {"doc_id": "d2", "text": ""}],
        "err",
    )
    assert "most relevant retrieved information" in msg
    assert "Technical note: err" in msg
    assert "hello" in msg

    msg2 = rag._fallback_answer([{"doc_id": "d1", "text": "   "}], "err")
    assert "Relevant context was found" in msg2


def test_rag_smalltalk_llm_success_path():
    retriever = Mock()
    llm = Mock()
    llm.generate.return_value = "Hi there!"
    rag = RAGSystem(retriever=retriever, llm_client=llm)

    out = rag.query("hi")
    assert out["answer"] == "Hi there!"
    assert out["sources"] == []
    assert out["error"] is None


def test_rag_domain_llm_error_uses_fallback_answer():
    retriever = Mock()
    retriever.search.return_value = [{"doc_id": "d1", "text": "ctx"}]

    llm = Mock()

    def _generate(prompt: str, **kwargs):
        if "Classify the user's question" in prompt:
            return "domain_qa"
        raise LLMClientError("timeout")

    llm.generate.side_effect = _generate
    rag = RAGSystem(retriever=retriever, llm_client=llm)

    out = rag.query("galactic gadgets setup")
    assert out["error"] == "timeout"
    assert "Technical note: timeout" in out["answer"]


def test_rag_classify_question_llm_label_parsing_domain_qa():
    retriever = Mock()
    llm = Mock()
    llm.generate.return_value = "domain_qa"
    rag = RAGSystem(retriever=retriever, llm_client=llm)

    assert rag._classify_question("What is this?") == "domain_qa"


def test_rag_classify_question_llm_label_parsing_smalltalk():
    retriever = Mock()
    llm = Mock()
    llm.generate.return_value = "smalltalk"
    rag = RAGSystem(retriever=retriever, llm_client=llm)

    assert rag._classify_question("What is this?") == "smalltalk"


def test_rag_classify_question_llm_label_parsing_out_of_scope():
    retriever = Mock()
    llm = Mock()
    llm.generate.return_value = "out_of_scope"
    rag = RAGSystem(retriever=retriever, llm_client=llm)

    assert rag._classify_question("What is this?") == "out_of_scope"


def test_rag_rewrite_query_no_llm_returns_original():
    rag = RAGSystem(retriever=Mock(), llm_client=None)
    assert rag._rewrite_query("hello") == "hello"


def test_rag_rewrite_query_empty_result_returns_original():
    llm = Mock()

    def _generate(prompt: str, **kwargs):
        if "Rewrite the following user question" in prompt:
            return "   "
        return "out_of_scope"

    llm.generate.side_effect = _generate
    rag = RAGSystem(retriever=Mock(), llm_client=llm)
    assert rag._rewrite_query("galactic gadgets setup") == "galactic gadgets setup"


def test_rag_domain_route_with_no_llm_hits_not_configured_fallback():
    class _Retriever:
        def search(self, *args, **kwargs):
            return [{"doc_id": "d1", "text": "ctx"}]

    rag = RAGSystem(retriever=_Retriever(), llm_client=None)
    # Force domain route so we reach the fallback branch (instead of out_of_scope).
    rag._classify_question = lambda _q: "domain_qa"  # type: ignore[assignment]

    out = rag.query("anything")
    assert out["error"] == "LLM not configured"
    assert "Technical note: LLM not configured" in out["answer"]


def test_rag_classify_question_llm_exception_falls_back_to_out_of_scope():
    retriever = Mock()
    llm = Mock()
    llm.generate.side_effect = RuntimeError("boom")
    rag = RAGSystem(retriever=retriever, llm_client=llm)

    # No domain keywords in this question, so it will attempt LLM classification.
    # If that fails, _classify_question should fall back to out_of_scope.
    assert rag._classify_question("What does it say?") == "out_of_scope"
