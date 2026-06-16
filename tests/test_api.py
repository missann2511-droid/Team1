"""Smoke tests for the prediction API (used by CI and local `pytest`)."""
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

SAMPLE = {
    "year": 2014, "make": "Ford", "model": "Fusion", "trim": "SE", "body": "Sedan",
    "transmission": "automatic", "state": "ca", "condition": 35.0, "odometer": 42000.0,
    "color": "white", "interior": "black",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_single():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code in (200, 503)  # 503 if model artifact absent in CI cache
    if r.status_code == 200:
        assert r.json()["predicted_price"] > 0


def test_predict_batch():
    r = client.post("/predict_batch", json={"items": [SAMPLE, SAMPLE]})
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert r.json()["count"] == 2
