import json
from kafka import KafkaProducer
from src.settings.settings import settings


class KafkaProducerService:

    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        self.topic = settings.kafka_topic_predictions

    def send_prediction(self, data: dict):
        self.producer.send(self.topic, value=data)
        self.producer.flush()