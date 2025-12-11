"""
Tests de l'API FastAPI
Vérifie les endpoints principaux de l'application.
"""
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)


def test_root_redirect():
    """Vérifie que la racine redirige vers /docs"""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health_router():
    """Vérifie que le endpoint de santé fonctionne"""
    response = client.get("/health_router")
    assert response.status_code == 200
    assert response.json() == {"status": "Router ok"}


def test_index_info():
    """Vérifie que le endpoint d'informations de l'index fonctionne"""
    response = client.get("/index_info")
    assert response.status_code == 200
    data = response.json()
    assert "exists" in data


def test_ask_endpoint():
    """Vérifie que le endpoint ask fonctionne avec une requête valide"""
    response = client.post(
        "/ask",
        json={"query": "Quels sont les concerts rock à Rennes ?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data


def test_search_raw_endpoint():
    """Vérifie que le endpoint search_raw fonctionne"""
    response = client.post(
        "/search_raw",
        json={"query": "concert jazz"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
