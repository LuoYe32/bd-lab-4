from pathlib import Path
import io

import joblib
import numpy as np
from PIL import Image
from typing import cast
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import src.api as api


class DummyQdrant:
    def __init__(self):
        self.saved = []
        self.search_calls = []

    def save_prediction(self, vector, prediction: dict):
        self.saved.append(
            {
                "vector": vector.tolist() if hasattr(vector, "tolist") else vector,
                "prediction": prediction,
            }
        )

    def search_similar(self, vector, limit: int = 5):
        self.search_calls.append(
            {
                "vector": vector.tolist() if hasattr(vector, "tolist") else vector,
                "limit": limit,
            }
        )
        return [
            {
                "id": "point-1",
                "score": 0.99,
                "payload": {
                    "class_id": 1,
                    "class_name": "Trouser",
                    "proba": [0.0] * 10,
                },
            },
            {
                "id": "point-2",
                "score": 0.95,
                "payload": {
                    "class_id": 6,
                    "class_name": "Shirt",
                    "proba": [0.0] * 10,
                },
            },
        ][:limit]


def get_dummy_qdrant():
    return cast(DummyQdrant, api.qdrant)


def create_dummy_model():
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    X = np.random.random((50, 784)).astype("float32")
    y = np.array([i % 10 for i in range(50)], dtype="int64")

    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(X, y)

    joblib.dump(model, "artifacts/model.joblib")


def setup_test_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_dummy_model()
    api._model = None
    api.qdrant = DummyQdrant()
    return TestClient(api.app)


def test_health_endpoint(tmp_path, monkeypatch):
    client = setup_test_env(tmp_path, monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_present"] is True


def test_predict_fill(tmp_path, monkeypatch):
    client = setup_test_env(tmp_path, monkeypatch)

    response = client.post("/predict", json={"fill": 0})

    assert response.status_code == 200
    data = response.json()

    assert "class_id" in data
    assert "class_name" in data
    assert "proba" in data
    assert len(data["proba"]) == 10

    assert api.qdrant is not None
    assert len(get_dummy_qdrant().saved) == 1


def test_predict_random_endpoint(tmp_path, monkeypatch):
    client = setup_test_env(tmp_path, monkeypatch)

    response = client.get("/predict/random?seed=42")

    assert response.status_code == 200
    data = response.json()

    assert "class_id" in data
    assert "class_name" in data
    assert len(data["proba"]) == 10

    assert len(get_dummy_qdrant().saved) == 1


def test_predict_image(tmp_path, monkeypatch):
    client = setup_test_env(tmp_path, monkeypatch)

    img = Image.fromarray((np.random.rand(28, 28) * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict/image",
        files={"file": ("test.png", buf, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()

    assert "class_id" in data
    assert "class_name" in data
    assert len(data["proba"]) == 10

    assert len(get_dummy_qdrant().saved) == 1


def test_similar_endpoint(tmp_path, monkeypatch):
    client = setup_test_env(tmp_path, monkeypatch)

    response = client.post("/similar?limit=2", json={"fill": 0})

    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["id"] == "point-1"
    assert "score" in data["results"][0]
    assert "payload" in data["results"][0]

    assert len(get_dummy_qdrant().search_calls) == 1
    assert get_dummy_qdrant().search_calls[0]["limit"] == 2


def test_predict_rejects_multiple_inputs(tmp_path, monkeypatch):
    client = setup_test_env(tmp_path, monkeypatch)

    response = client.post(
        "/predict",
        json={"fill": 0, "random_seed": 42},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only one of pixels, fill or random_seed can be provided"


def test_predict_rejects_missing_input(tmp_path, monkeypatch):
    client = setup_test_env(tmp_path, monkeypatch)

    response = client.post("/predict", json={})

    assert response.status_code == 400
    assert response.json()["detail"] == "One of pixels, fill or random_seed must be provided"


def test_predict_rejects_wrong_pixels_length(tmp_path, monkeypatch):
    client = setup_test_env(tmp_path, monkeypatch)

    response = client.post("/predict", json={"pixels": [0.1, 0.2]})

    assert response.status_code == 400
    assert response.json()["detail"] == "pixels must contain 784 values"


def test_predict_rejects_negative_fill(tmp_path, monkeypatch):
    client = setup_test_env(tmp_path, monkeypatch)

    response = client.post("/predict", json={"fill": -1})

    assert response.status_code == 400
    assert response.json()["detail"] == "fill must be non-negative"


def test_predict_random_rejects_negative_seed(tmp_path, monkeypatch):
    client = setup_test_env(tmp_path, monkeypatch)

    response = client.get("/predict/random?seed=-1")

    assert response.status_code == 400
    assert response.json()["detail"] == "seed must be non-negative"


def test_predict_image_rejects_wrong_content_type(tmp_path, monkeypatch):
    client = setup_test_env(tmp_path, monkeypatch)

    response = client.post(
        "/predict/image",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type. Allowed: png, jpg, jpeg, bmp"


def test_predict_image_rejects_empty_file(tmp_path, monkeypatch):
    client = setup_test_env(tmp_path, monkeypatch)

    response = client.post(
        "/predict/image",
        files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Empty file"


def test_similar_returns_500_when_qdrant_unavailable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_dummy_model()
    api._model = None
    api.qdrant = None

    client = TestClient(api.app)

    response = client.post("/similar", json={"fill": 0})

    assert response.status_code == 500
    assert response.json()["detail"] == "Qdrant not available"