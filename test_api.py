from fastapi.testclient import TestClient
from api import app, documents
client = TestClient(app)
headers = {
    "api-key": "trainee123"
}
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
def test_search_empty():
    documents.clear()
    response = client.post(
        "/search",
        headers=headers,
        json={
            "query": "employee rules",
            "top_k": 5
        }
    )
    assert response.status_code == 200
    assert response.json()["results"] == []
def test_invalid_api_key():
    response = client.post(
        "/search",
        headers={"api-key": "wrong"},
        json={
            "query": "employee rules",
            "top_k": 5
        }
    )
    assert response.status_code == 401
def test_invalid_top_k():
    response = client.post(
        "/search",
        headers=headers,
        json={
            "query": "employee rules",
            "top_k": 25
        }
    )
    assert response.status_code == 422
def test_search_response():
    response = client.post(
        "/search",
        headers=headers,
        json={
            "query": "company policy",
            "top_k": 3
        }
    )
    assert response.status_code == 200
    assert isinstance(response.json()["results"], list)
def test_document_upload():
    response = client.post(
        "/documents",
        headers=headers,
        files={
            "file": (
                "test.txt",
                b"Employees must follow company rules and policies.",
                "text/plain"
            )
        }
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "test.txt"
    assert response.json()["total_chunks"] >= 1
