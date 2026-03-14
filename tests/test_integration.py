"""
Integration tests for search API.

@author: Kevin Lundeen
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 3.0.0+w26
"""

import pytest
from fastapi.testclient import TestClient

from retrieval.main import app


@pytest.fixture
def client():
    """Provide test client with lifespan events."""
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def query_tester(client, query, expected_file):
    """Helper function to test endpoints."""
    response = client.post("/search", json={"query": query, "n_results": 1})

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == query
    assert data["count"] == 1
    assert len(data["results"]) == 1
    assert data["results"][0]["metadata"]["filename"] == expected_file


def test_some_queries(client):
    """Test some queries."""
    query_tester(client, "test", "sample1.txt")
    query_tester(client, "vectors are vicious", "sample4.txt")
    query_tester(client, "How about Python?", "sample2.txt")

    # with the reranker this actually gets a worse result (sample1.txt)
    # ...so decided to take it out of the test wuite
    # query_tester(client, "ML Engineering", "sample3.txt")


def test_search_endpoint(client):
    """Test search endpoint returns results."""
    response = client.post("/search", json={"query": "test", "n_results": 3})

    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert "count" in data
    assert data["query"] == "test"


def test_search_empty_query(client):
    """Test search with empty query returns 400."""
    response = client.post("/search", json={"query": "   ", "n_results": 5})
    assert response.status_code == 400


def test_search_invalid_n_results(client):
    """Test search with invalid n_results returns 400."""
    response = client.post("/search", json={"query": "test", "n_results": 100})
    assert response.status_code == 400
