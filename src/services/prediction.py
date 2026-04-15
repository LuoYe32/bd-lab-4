import io
import logging
import uuid
import datetime
from pathlib import Path
from typing import Optional, Any

import joblib
import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from kafka.errors import KafkaError
from pydantic import ValidationError

from src.services.qdrant import QdrantService
from src.messaging.kafka_producer import KafkaProducerService
from src.schemas import PredictRequest, KafkaPredictionEvent

logger = logging.getLogger(__name__)


class PredictionService:
    MODEL_PATH = Path("artifacts/model.joblib")
    VECTOR_SIZE = 784
    MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
    MAX_IMAGE_DIMENSION = 4096
    ALLOWED_CONTENT_TYPES = {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/bmp",
    }

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

    def __init__(self, qdrant_service: Optional[QdrantService] = None) -> None:
        self._model: Any = None
        self.qdrant = qdrant_service
        self.kafka: Optional[KafkaProducerService] = None

        try:
            self.kafka = KafkaProducerService()
            logger.info("Kafka producer initialized")
        except KafkaError:
            logger.exception("Kafka init failed")
        except ValueError:
            logger.exception("Kafka init failed due to invalid configuration")
        except OSError:
            logger.exception("Kafka init failed due to OS/network error")

    def _load_model(self):
        if self._model is None:
            if not self.MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.MODEL_PATH}. "
                    f"Run training or dvc pull artifacts."
                )
            self._model = joblib.load(self.MODEL_PATH)
        return self._model

    @staticmethod
    def _normalize_vector(x: np.ndarray) -> np.ndarray:
        if x.max() > 1.5:
            x = x / 255.0
        return x

    def build_vector_from_request(self, req: PredictRequest) -> np.ndarray:
        provided = [
            req.pixels is not None,
            req.fill is not None,
            req.random_seed is not None,
        ]

        if sum(provided) == 0:
            raise HTTPException(
                status_code=400,
                detail="One of pixels, fill or random_seed must be provided",
            )

        if sum(provided) > 1:
            raise HTTPException(
                status_code=400,
                detail="Only one of pixels, fill or random_seed can be provided",
            )

        if req.pixels is not None:
            return self._build_from_pixels(req.pixels)

        if req.fill is not None:
            return self._build_from_fill(req.fill)

        if req.random_seed is not None:
            return self._build_from_seed(req.random_seed)

        raise HTTPException(status_code=500, detail="Unexpected request state")

    def _build_from_pixels(self, pixels: list[float]) -> np.ndarray:
        if len(pixels) != self.VECTOR_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"pixels must contain {self.VECTOR_SIZE} values",
            )

        try:
            x = np.array(pixels, dtype=np.float32)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid pixel values",
            ) from exc

        if not np.isfinite(x).all():
            raise HTTPException(
                status_code=400,
                detail="pixels contain NaN or inf",
            )

        if x.min() < 0:
            raise HTTPException(
                status_code=400,
                detail="pixels must be non-negative",
            )

        return self._normalize_vector(x)

    def _build_from_fill(self, fill: float) -> np.ndarray:
        try:
            fill_value = float(fill)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="fill must be numeric",
            ) from exc

        if not np.isfinite(fill_value):
            raise HTTPException(
                status_code=400,
                detail="fill must be finite",
            )

        if fill_value < 0:
            raise HTTPException(
                status_code=400,
                detail="fill must be non-negative",
            )

        x = np.full((self.VECTOR_SIZE,), fill_value, dtype=np.float32)
        return self._normalize_vector(x)

    def _build_from_seed(self, random_seed: int) -> np.ndarray:
        try:
            seed = int(random_seed)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="random_seed must be int",
            ) from exc

        if seed < 0:
            raise HTTPException(
                status_code=400,
                detail="random_seed must be non-negative",
            )

        rng = np.random.default_rng(seed)
        x = rng.random(self.VECTOR_SIZE, dtype=np.float32)
        return self._normalize_vector(x)

    async def build_vector_from_image(self, file: UploadFile) -> np.ndarray:
        if file.content_type not in self.ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Allowed: png, jpg, jpeg, bmp",
            )

        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        if len(contents) > self.MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(status_code=400, detail="File too large")

        try:
            image = Image.open(io.BytesIO(contents))
            image = image.convert("L")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file",
            ) from exc

        if image.width > self.MAX_IMAGE_DIMENSION or image.height > self.MAX_IMAGE_DIMENSION:
            raise HTTPException(
                status_code=400,
                detail="Image resolution too large",
            )

        image = image.resize((28, 28))
        arr = np.array(image, dtype=np.float32).flatten()

        if not np.isfinite(arr).all():
            raise HTTPException(
                status_code=400,
                detail="Invalid pixel values",
            )

        return self._normalize_vector(arr)

    def build_random_vector(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            if seed < 0:
                raise HTTPException(
                    status_code=400,
                    detail="seed must be non-negative",
                )
            rng = np.random.default_rng(seed)
            x = rng.random(self.VECTOR_SIZE, dtype=np.float32)
        else:
            x = np.random.random(self.VECTOR_SIZE).astype(np.float32)

        return self._normalize_vector(x)

    def predict_array(self, x: np.ndarray) -> dict:
        model = self._load_model()
        x = self._normalize_vector(x)

        X = x.reshape(1, -1)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            class_id = int(np.argmax(proba))
        else:
            class_id = int(model.predict(X)[0])
            proba = np.zeros(10, dtype=np.float32)
            proba[class_id] = 1.0

        result = {
            "class_id": class_id,
            "class_name": self.CLASS_NAMES.get(class_id, str(class_id)),
            "proba": [float(p) for p in proba],
        }

        self._send_event(x, result)

        return result

    def _send_event(self, x: np.ndarray, result: dict) -> None:
        if self.kafka is None:
            logger.warning("Kafka not available")
            return

        try:
            event = KafkaPredictionEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                source="fashion-api",
                vector=x.tolist(),
                prediction=result,
            )
            payload = event.model_dump()
        except ValidationError:
            logger.exception("Kafka event validation failed")
            return
        except ValueError:
            logger.exception("Kafka event serialization failed")
            return

        try:
            self.kafka.send_prediction(payload)
            logger.info("Event sent to Kafka: %s", event.event_id)
        except KafkaError:
            logger.exception("Kafka send failed")
        except OSError:
            logger.exception("Kafka send failed due to OS/network error")

    def search_similar(self, x: np.ndarray, limit: int = 5):
        if self.qdrant is None:
            raise HTTPException(status_code=500, detail="Qdrant not available")

        try:
            return self.qdrant.search_similar(x, limit=limit)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Qdrant search error: {exc}",
            ) from exc

    def health(self) -> dict:
        return {
            "status": "ok",
            "model_present": self.MODEL_PATH.exists(),
        }