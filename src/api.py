from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image
import io

from src.database.qdrant_client import QdrantService
from src.schemas import SimilarResponse, PredictRequest, PredictResponse

MODEL_PATH = Path("artifacts/model.joblib")

CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


app = FastAPI(title="Fashion-MNIST Classic ML API")
_model = None

try:
    qdrant = QdrantService()
except Exception as e:
    print(f"Qdrant init error: {e}")
    qdrant = None


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run training or dvc pull artifacts."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def build_vector_from_request(req: PredictRequest) -> np.ndarray:

    provided = [
        req.pixels is not None,
        req.fill is not None,
        req.random_seed is not None,
    ]

    if sum(provided) == 0:
        raise HTTPException(
            status_code=400,
            detail="One of pixels, fill or random_seed must be provided"
        )

    if sum(provided) > 1:
        raise HTTPException(
            status_code=400,
            detail="Only one of pixels, fill or random_seed can be provided"
        )

    if req.pixels is not None:

        if len(req.pixels) != 784:
            raise HTTPException(status_code=400, detail="pixels must contain 784 values")

        try:
            x = np.array(req.pixels, dtype=np.float32)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid pixel values")

        if not np.isfinite(x).all():
            raise HTTPException(status_code=400, detail="pixels contain NaN or inf")

        if x.min() < 0:
            raise HTTPException(status_code=400, detail="pixels must be non-negative")

    elif req.fill is not None:

        try:
            fill_value = float(req.fill)
        except Exception:
            raise HTTPException(status_code=400, detail="fill must be numeric")

        if not np.isfinite(fill_value):
            raise HTTPException(status_code=400, detail="fill must be finite")

        if fill_value < 0:
            raise HTTPException(status_code=400, detail="fill must be non-negative")

        x = np.full((784,), fill_value, dtype=np.float32)

    elif req.random_seed is not None:

        try:
            seed = int(req.random_seed)
        except Exception:
            raise HTTPException(status_code=400, detail="random_seed must be int")

        if seed < 0:
            raise HTTPException(status_code=400, detail="random_seed must be non-negative")

        rng = np.random.default_rng(seed)
        x = rng.random(784)

    else:
        raise HTTPException(status_code=500, detail="Unexpected error")

    try:
        if x.max() > 1.5:
            x = x / 255.0
    except Exception:
        raise HTTPException(status_code=400, detail="Normalization error")

    return x


def _predict_array(x: np.ndarray):

    model = _load_model()

    if x.max() > 1.5:
        x = x / 255.0

    X = x.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        class_id = int(np.argmax(proba))
    else:
        class_id = int(model.predict(X)[0])
        proba = np.zeros(10)
        proba[class_id] = 1.0

    result = {
        "class_id": class_id,
        "class_name": CLASS_NAMES.get(class_id, str(class_id)),
        "proba": [float(p) for p in proba],
    }

    if qdrant is not None:
        try:
            qdrant.save_prediction(x, result)
        except Exception as e:
            print(f"Qdrant save error: {e}")

    return result


@app.get("/health")
def health():
    ok = MODEL_PATH.exists()
    return {"status": "ok", "model_present": ok}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    x = build_vector_from_request(req)

    return _predict_array(x)


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):

    if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/bmp"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: png, jpg, jpeg, bmp"
        )

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        image = Image.open(io.BytesIO(contents))
        image = image.convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if image.width > 4096 or image.height > 4096:
        raise HTTPException(status_code=400, detail="Image resolution too large")

    image = image.resize((28, 28))

    arr = np.array(image, dtype=np.float32).flatten()

    if not np.isfinite(arr).all():
        raise HTTPException(status_code=400, detail="Invalid pixel values")

    return _predict_array(arr)


@app.get("/predict/random", response_model=PredictResponse)
def predict_random(seed: Optional[int] = None):

    if seed is not None:
        if seed < 0:
            raise HTTPException(status_code=400, detail="seed must be non-negative")
        rng = np.random.default_rng(seed)
        x = rng.random(784)
    else:
        x = np.random.random(784)

    return _predict_array(x)


@app.post("/similar", response_model=SimilarResponse)
def find_similar(req: PredictRequest, limit: int = 5):

    if qdrant is None:
        raise HTTPException(status_code=500, detail="Qdrant not available")

    try:
        x = build_vector_from_request(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    try:
        results = qdrant.search_similar(x, limit=limit)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Qdrant search error: {str(e)}"
        )

    return {
        "results": results
    }