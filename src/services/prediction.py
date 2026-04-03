import logging
import uuid
import datetime
import numpy as np

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
            self.kafka = KafkaProducerService()
            logger.info("Kafka producer initialized")
        except Exception:
            logger.exception("Kafka init error")
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

        self._send_event(x, result)

        return result

    def _send_event(self, x: np.ndarray, result: dict):
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

            self.kafka.send_prediction(event.model_dump())

            logger.info(f"Event sent to Kafka: {event.event_id}")

        except Exception:
            logger.exception("Kafka send failed")