"""
Lab 3 FastAPI API.
@author: Alok katiyar, Aarti Dashore
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 0.1.0+w26
"""

import pytest
from fastapi.testclient import TestClient

from src.retrieval.main import app


@pytest.fixture
def client():
    """Fixture provides a fresh test client for each test"""
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_healthcheck(client):  # Client injected as the parameter
    """Smoke test uses the /health endpoint"""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert len(data["message"]) > 0


def test_not_found(client):
    """Try to get something nonsensical"""
    res = client.get("/nonsensical?x=1&y=2&z=3")
    assert res.status_code == 404


def test_generic_exception_handler(client):
    """Use the purpose-build /test/error to raise"""
    res = client.get("/test/error")
    assert res.status_code == 500
    data = res.json()
    assert "Internal server error" in data["detail"]
