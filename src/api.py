from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image
import io

from src.services.prediction import PredictionService
from src.services.qdrant import QdrantService
from src.schemas import SimilarResponse, PredictRequest, PredictResponse

MODEL_PATH = Path("artifacts/model.joblib")

app = FastAPI(title="Fashion-MNIST Classic ML API")

_model = None
_prediction_service = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found")
        _model = joblib.load(MODEL_PATH)
    return _model


def get_prediction_service():
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService(get_model())
    return _prediction_service


def build_vector_from_request(req: PredictRequest) -> np.ndarray:

    provided = [
        req.pixels is not None,
        req.fill is not None,
        req.random_seed is not None,
    ]

    if sum(provided) != 1:
        raise HTTPException(status_code=400, detail="Provide exactly one input")

    if req.pixels is not None:
        x = np.array(req.pixels, dtype=np.float32)

    elif req.fill is not None:
        x = np.full((784,), float(req.fill), dtype=np.float32)

    else:
        assert req.random_seed is not None
        rng = np.random.default_rng(req.random_seed)
        x = rng.random(784)

    if x.max() > 1.5:
        x = x / 255.0

    return x


@app.get("/health")
def health():
    return {"status": "ok", "model_present": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = build_vector_from_request(req)
    return get_prediction_service().predict(x)


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")
    image = image.resize((28, 28))
    arr = np.array(image, dtype=np.float32).flatten()

    return get_prediction_service().predict(arr)


@app.get("/predict/random", response_model=PredictResponse)
def predict_random(seed: Optional[int] = None):
    rng = np.random.default_rng(seed) if seed else np.random
    x = rng.random(784)
    return get_prediction_service().predict(x)


@app.post("/similar", response_model=SimilarResponse)
def find_similar(req: PredictRequest, limit: int = 5):

    qdrant = QdrantService()

    x = build_vector_from_request(req)
    results = qdrant.search_similar(x, limit=limit)

    return {"results": results}