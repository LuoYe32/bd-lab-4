import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile

from src.services.qdrant import QdrantService
from src.schemas import PredictRequest, PredictResponse, SimilarResponse
from src.services.prediction import PredictionService

logger = logging.getLogger(__name__)

app = FastAPI(title="Fashion-MNIST Classic ML API")

try:
    qdrant_service = QdrantService()
except RuntimeError as exc:
    logger.warning("Qdrant init error: %s", exc)
    qdrant_service = None

prediction_service = PredictionService(qdrant_service=qdrant_service)


@app.get("/health")
def health():
    return prediction_service.health()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = prediction_service.build_vector_from_request(req)
    return prediction_service.predict_array(x)


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
    x = await prediction_service.build_vector_from_image(file)
    return prediction_service.predict_array(x)


@app.get("/predict/random", response_model=PredictResponse)
def predict_random(seed: Optional[int] = None):
    x = prediction_service.build_random_vector(seed)
    return prediction_service.predict_array(x)


@app.post("/similar", response_model=SimilarResponse)
def find_similar(req: PredictRequest, limit: int = 5):
    x = prediction_service.build_vector_from_request(req)
    results = prediction_service.search_similar(x, limit=limit)
    return {"results": results}