"""Tests for the inference server.

The classifier is stubbed so these run without downloading the model.
TestClient is used without a context manager on purpose: that skips the
lifespan handler, so the real model is never loaded during tests.
"""

import pytest
from fastapi.testclient import TestClient

import app as service


def stub_classifier(texts):
    return [{"label": "POSITIVE", "score": 0.99} for _ in texts]


@pytest.fixture
def client():
    service.state["classifier"] = stub_classifier
    service.state["ready"] = True
    return TestClient(service.app)


def test_predict_returns_one_prediction_per_input(client):
    response = client.post("/predict", json={"texts": ["alpha", "beta"]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_empty_string_is_rejected(client):
    assert client.post("/predict", json={"texts": ["   "]}).status_code == 422


def test_oversized_input_is_rejected(client):
    oversized = "x" * (service.MAX_CHARS + 1)
    assert client.post("/predict", json={"texts": [oversized]}).status_code == 422


def test_batch_over_limit_is_rejected(client):
    batch = ["ok"] * (service.MAX_BATCH + 1)
    assert client.post("/predict", json={"texts": batch}).status_code == 422


def test_predict_is_503_when_not_ready(client):
    service.state["ready"] = False
    assert client.post("/predict", json={"texts": ["alpha"]}).status_code == 503


def test_readyz_is_503_before_model_loads(client):
    service.state["ready"] = False
    assert client.get("/readyz").status_code == 503


def test_inference_failure_returns_generic_500(client):
    def failing(_texts):
        raise RuntimeError("internal detail that must not leak")

    service.state["classifier"] = failing
    response = client.post("/predict", json={"texts": ["alpha"]})
    assert response.status_code == 500
    assert "internal detail" not in response.text
