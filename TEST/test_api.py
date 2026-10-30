import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "classes" in data


def test_predict_valid():
    response = client.post("/predict", json={"text": "Stock markets surge on positive economic data"})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "category" in data
    assert data["category"] in ["World", "Sports", "Business", "Sci/Tech"]


def test_predict_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422
