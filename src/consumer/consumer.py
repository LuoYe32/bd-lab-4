import json
import time
import logging

from kafka import KafkaConsumer
from kafka.errors import KafkaError, NoBrokersAvailable
from pydantic import ValidationError

from src.settings.settings import settings
from src.schemas import KafkaPredictionEvent
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def create_consumer():
    return KafkaConsumer(
        settings.kafka_topic_predictions,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="fashion-consumer-group",
        enable_auto_commit=True,
    )


def process_message(event: KafkaPredictionEvent):
    logger.info("New prediction received")
    logger.info(f"ID={event.event_id} class={event.prediction['class_name']}")


def run_consumer():

    logger.info("Starting Kafka consumer...")

    while True:
        try:
            consumer = create_consumer()
            logger.info("Kafka connected")
            break

        except NoBrokersAvailable:
            logger.error("Kafka not available (No brokers)")
            time.sleep(3)

        except KafkaError as e:
            logger.error(f"Kafka error during connection: {e}")
            time.sleep(3)

        except Exception:
            logger.exception("Unexpected error during Kafka connection")
            time.sleep(3)

    for message in consumer:
        try:
            event = KafkaPredictionEvent(**message.value)
            process_message(event)

        except ValidationError as e:
            logger.error(f"Invalid message schema: {e}")

        except KafkaError as e:
            logger.error(f"Kafka error while consuming: {e}")

        except Exception:
            logger.exception("Unexpected error while processing message")


if __name__ == "__main__":
    run_consumer()