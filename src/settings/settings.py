from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # API
    app_name: str = "Fashion MNIST API"

    # Qdrant
    qdrant_host: Optional[str] = None
    qdrant_port: Optional[int] = None
    qdrant_api_key: Optional[str] = None

    # DagsHub
    dagshub_access_key: Optional[str] = None
    dagshub_secret_key: Optional[str] = None

    # Kafka
    kafka_bootstrap_servers: Optional[str] = None
    kafka_topic_predictions: str = "predictions"

    kafka_security_protocol: Optional[str] = None
    kafka_sasl_mechanism: Optional[str] = None
    kafka_username: Optional[str] = None
    kafka_password: Optional[str] = None
    kafka_cluster_id: Optional[str] = None

    # Docker
    docker_image: str = "bd-lab-1-6:latest"
    dockerhub_username: Optional[str] = None
    dockerhub_token: Optional[str] = None

    model_config = {
        "env_file": ".env",
    }

    @model_validator(mode="after")
    def verify_config(self) -> "Settings":

        missing = []

        if self.qdrant_host is None:
            missing.append("qdrant_host")
        if self.qdrant_port is None:
            missing.append("qdrant_port")
        if self.qdrant_api_key is None:
            missing.append("qdrant_api_key")

        if self.kafka_bootstrap_servers is None:
            missing.append("kafka_bootstrap_servers")
        if self.kafka_cluster_id is None:
            missing.append("kafka_cluster_id")

        if self.kafka_security_protocol == "SASL_PLAINTEXT":
            if self.kafka_username is None:
                missing.append("kafka_username")
            if self.kafka_password is None:
                missing.append("kafka_password")

        if self.dagshub_access_key is None:
            missing.append("dagshub_access_key")
        if self.dagshub_secret_key is None:
            missing.append("dagshub_secret_key")

        if missing:
            raise ValueError(
                f"The following settings are required: "
                f"{', '.join(missing)}"
            )

        return self


settings = Settings()