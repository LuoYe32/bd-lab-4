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