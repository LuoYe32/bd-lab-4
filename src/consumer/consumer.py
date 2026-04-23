import json
import time
import logging
import numpy as np

from kafka import KafkaConsumer
from kafka.errors import KafkaError, NoBrokersAvailable
from pydantic import ValidationError

from src.settings.settings import settings
from src.schemas import KafkaPredictionEvent
from src.services.qdrant import QdrantService
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class KafkaConsumerService:

    def __init__(self):
        self.consumer = None
        self.qdrant = None

    def _create_consumer(self) -> KafkaConsumer:
        return KafkaConsumer(
            settings.kafka_topic_predictions,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),

            security_protocol=settings.kafka_security_protocol,
            sasl_mechanism=settings.kafka_sasl_mechanism,
            sasl_plain_username=settings.kafka_username,
            sasl_plain_password=settings.kafka_password,
        )

    def _init_qdrant(self) -> None:
        try:
            self.qdrant = QdrantService()
            logger.info("Qdrant initialized in consumer")
        except RuntimeError:
            logger.exception("Failed to init Qdrant")
            self.qdrant = None

    def _connect_kafka(self) -> None:
        while True:
            try:
                self.consumer = self._create_consumer()
                logger.info("Kafka connected")
                return

            except NoBrokersAvailable:
                logger.error("Kafka not available, retrying...")
                time.sleep(3)

            except KafkaError:
                logger.exception("Kafka connection error, retrying...")
                time.sleep(3)

    def _process_message(self, message) -> None:
        try:
            event = KafkaPredictionEvent(**message.value)
        except ValidationError:
            logger.error("Invalid message format")
            return

        logger.info(
            "Processing event %s class=%s",
            event.event_id,
            event.prediction["class_name"],
        )

        if self.qdrant is None:
            logger.warning("Qdrant not available, skipping save")
            return

        try:
            vector = np.array(event.vector, dtype=np.float32)

            self.qdrant.save_prediction(
                vector=vector,
                prediction=event.prediction,
            )

        except RuntimeError:
            logger.exception("Qdrant save failed")

        except ValueError:
            logger.exception("Invalid vector data")

    def run(self) -> None:
        logger.info("Starting Kafka consumer...")

        self._init_qdrant()
        self._connect_kafka()

        assert self.consumer is not None

        for message in self.consumer:
            self._process_message(message)


if __name__ == "__main__":
    service = KafkaConsumerService()
    service.run()