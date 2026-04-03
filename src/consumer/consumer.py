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


def create_consumer():
    return KafkaConsumer(
        settings.kafka_topic_predictions,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),

        security_protocol=settings.kafka_security_protocol,
        sasl_mechanism=settings.kafka_sasl_mechanism,
        sasl_plain_username=settings.kafka_username,
        sasl_plain_password=settings.kafka_password,
    )


def run_consumer():

    logger.info("Starting Kafka consumer...")

    try:
        qdrant = QdrantService()
        logger.info("Qdrant initialized in consumer")
    except Exception:
        logger.exception("Failed to init Qdrant")
        qdrant = None

    while True:
        try:
            consumer = create_consumer()
            logger.info("Kafka connected")
            break

        except NoBrokersAvailable:
            logger.error("Kafka not available")
            time.sleep(3)

        except KafkaError as e:
            logger.error(f"Kafka error: {e}")
            time.sleep(3)

    for message in consumer:
        try:
            event = KafkaPredictionEvent(**message.value)

            logger.info(
                f"Processing event {event.event_id} class={event.prediction['class_name']}"
            )

            if qdrant is not None:
                vector = np.array(event.vector, dtype=np.float32)

                qdrant.save_prediction(
                    vector=vector,
                    prediction=event.prediction,
                )

        except ValidationError as e:
            logger.error(f"Invalid message: {e}")

        except Exception:
            logger.exception("Failed to process message")


if __name__ == "__main__":
    run_consumer()