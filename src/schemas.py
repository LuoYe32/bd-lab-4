from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    pixels: Optional[List[float]] = Field(
        default=None,
        description="Length 784 array"
    )
    fill: Optional[float] = Field(
        default=None,
        description="Fill all pixels with one value"
    )
    random_seed: Optional[int] = Field(
        default=None,
        description="Generate deterministic random pixels"
    )


class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    proba: List[float]


class SimilarItem(BaseModel):
    id: str
    score: float
    payload: Optional[Dict[str, Any]]


class SimilarResponse(BaseModel):
    results: List[SimilarItem]


class KafkaPredictionEvent(BaseModel):
    event_id: str
    timestamp: str
    source: str

    vector: List[float]
    prediction: Dict[str, Any]