import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from src.app import app
import src.app as appmod

appmod.load_model()
client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict():
    response = client.post("/predict", json={"text": "Football team wins championship"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] in ["World", "Sports", "Business", "Sci/Tech"]
