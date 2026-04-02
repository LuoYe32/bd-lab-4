from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid

from src.schemas import SimilarItem
from src.settings.settings import settings


class QdrantService:

    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=False,
        )
        self.collection_name = "predictions"

        self._init_collection()

    def _init_collection(self):

        exists = self.client.collection_exists(self.collection_name)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=784,
                    distance=Distance.COSINE,
                ),
            )

    def save_prediction(self, vector, prediction: dict):

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload=prediction,
                )
            ],
        )

    def search_similar(self, vector, limit: int = 5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector.tolist(),
            limit=limit,
        )

        return [
            SimilarItem(
                id=str(point.id),
                score=point.score,
                payload=point.payload,
            )
            for point in results
        ]