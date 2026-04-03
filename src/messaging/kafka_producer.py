import json
import logging

from kafka import KafkaProducer
from kafka.errors import KafkaError

from src.settings.settings import settings

logger = logging.getLogger(__name__)


class KafkaProducerService:

    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,

            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),

            security_protocol=settings.kafka_security_protocol,
            sasl_mechanism=settings.kafka_sasl_mechanism,
            sasl_plain_username=settings.kafka_username,
            sasl_plain_password=settings.kafka_password,

            acks="all",
            retries=5,
        )

        self.topic = settings.kafka_topic_predictions

    def send_prediction(self, event: dict):
        try:
            event_id = event.get("event_id")

            future = self.producer.send(
                self.topic,
                key=event_id,
                value=event,
            )

            metadata = future.get(timeout=10)

            logger.info(
                f"Kafka message sent: topic={metadata.topic}, partition={metadata.partition}"
            )

        except KafkaError:
            logger.exception("Kafka send failed")

        except Exception:
            logger.exception("Unexpected error during Kafka send")