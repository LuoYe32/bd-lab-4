import logging
import uuid
import datetime
import numpy as np

from src.services.qdrant import QdrantService
from src.messaging.kafka_producer import KafkaProducerService
from src.schemas import KafkaPredictionEvent

logger = logging.getLogger(__name__)

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


class PredictionService:

    def __init__(self, model):
        self.model = model

        try:
            self.qdrant = QdrantService()
            logger.info("Qdrant initialized")
        except Exception as e:
            logger.error(f"Qdrant init error: {e}")
            self.qdrant = None

        try:
            self.kafka = KafkaProducerService()
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.error(f"Kafka init error: {e}")
            self.kafka = None

    def predict(self, x: np.ndarray):

        X = x.reshape(1, -1)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            class_id = int(np.argmax(proba))
        else:
            class_id = int(self.model.predict(X)[0])
            proba = np.zeros(10)
            proba[class_id] = 1.0

        result = {
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, str(class_id)),
            "proba": [float(p) for p in proba],
        }

        self._save_to_qdrant(x, result)
        self._send_event(result)

        return result

    def _save_to_qdrant(self, x, result):
        if self.qdrant is None:
            logger.warning("Qdrant not available")
            return

        try:
            self.qdrant.save_prediction(x, result)
        except Exception as e:
            logger.exception("Qdrant save failed")

    def _send_event(self, result):
        if self.kafka is None:
            logger.warning("Kafka not available")
            return

        try:
            event = KafkaPredictionEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                source="fashion-api",
                prediction=result,
            )

            self.kafka.send_prediction(event.model_dump())

        except Exception as e:
            logger.exception("Kafka send failed")