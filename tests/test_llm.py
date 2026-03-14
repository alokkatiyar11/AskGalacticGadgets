# tests/test_llm.py

from unittest.mock import Mock, patch

import pytest

from retrieval.llm import LLMClient, LLMClientError


def _setup_client(mock_client_class):
    """
    Makes httpx.Client() act like:
        with httpx.Client(...) as client:
            ...
    """
    mock_client = Mock()
    mock_client_class.return_value.__enter__.return_value = mock_client
    mock_client_class.return_value.__exit__.return_value = False
    return mock_client


@patch("retrieval.llm.httpx.Client")
def test_generate_ollama(mock_client_class):
    mock_client = _setup_client(mock_client_class)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "Test answer"}}]}
    mock_client.post.return_value = mock_response

    client = LLMClient()
    out = client.generate("hello", system_prompt="be helpful")

    assert out == "Test answer"

    args, kwargs = mock_client.post.call_args
    assert args[0].endswith("/v1/chat/completions")
    assert kwargs["headers"]["Authorization"] == "Bearer ollama"


@patch("retrieval.llm.httpx.Client")
def test_generate_openai(mock_client_class):
    mock_client = _setup_client(mock_client_class)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "OpenAI answer"}}]}
    mock_client.post.return_value = mock_response

    client = LLMClient(
        base_url="https://api.openai.com",
        model="gpt-4o-mini",
        api_key="sk-test",
    )
    out = client.generate("what is rag?")

    assert out == "OpenAI answer"

    args, kwargs = mock_client.post.call_args
    assert args[0] == "https://api.openai.com/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"


@patch("retrieval.llm.httpx.Client")
def test_generate_http_error(mock_client_class):
    mock_client = _setup_client(mock_client_class)

    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "server error"
    mock_client.post.return_value = mock_response

    client = LLMClient()
    with pytest.raises(LLMClientError):
        client.generate("hello")


@patch("retrieval.llm.httpx.Client")
def test_generate_bad_json_shape(mock_client_class):
    mock_client = _setup_client(mock_client_class)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = '{"response":"oops"}'
    mock_response.json.return_value = {"response": "oops"}
    mock_client.post.return_value = mock_response

    client = LLMClient()
    with pytest.raises(LLMClientError):
        client.generate("hello")


@patch("retrieval.llm.httpx.Client")
def test_is_available_true(mock_client_class):
    mock_client = _setup_client(mock_client_class)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_client.get.return_value = mock_response

    client = LLMClient()
    assert client.is_available() is True


@patch("retrieval.llm.httpx.Client")
def test_is_available_false(mock_client_class):
    mock_client = _setup_client(mock_client_class)

    mock_response = Mock()
    mock_response.status_code = 404
    mock_client.get.return_value = mock_response

    client = LLMClient()
    assert client.is_available() is False


@patch("retrieval.llm.httpx.Client")
def test_generate_timeout(mock_client_class):
    import httpx

    mock_client = mock_client_class.return_value.__enter__.return_value
    mock_client.post.side_effect = httpx.TimeoutException("timeout")

    client = LLMClient()

    with pytest.raises(LLMClientError):
        client.generate("hello")


@patch("retrieval.llm.httpx.Client")
def test_generate_request_error(mock_client_class):
    import httpx

    mock_client = mock_client_class.return_value.__enter__.return_value
    mock_client.post.side_effect = httpx.RequestError("boom", request=Mock())

    client = LLMClient()

    with pytest.raises(LLMClientError):
        client.generate("hello")
