from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid


class FaceState(str, Enum):
    NO_FACE = "NO_FACE"
    LOW_QUALITY = "LOW_QUALITY"
    PROCESSING = "PROCESSING"
    UNKNOWN = "UNKNOWN"
    UNCERTAIN = "UNCERTAIN"
    MATCHED = "MATCHED"
    ERROR = "ERROR"


class FaceRecognitionResult(BaseModel):
    track_id: str
    face_id: Optional[str] = None
    person_id: Optional[str] = None
    identity: Optional[str] = None
    match_status: FaceState
    similarity: Optional[float] = None
    quality_score: Optional[float] = None
    timestamp: float
    model_version: Optional[str] = None


class Person(BaseModel):
    person_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    display_name: str
    external_identifier: Optional[str] = None
    status: str = "ACTIVE"
    created_at: float


class FaceTemplate(BaseModel):
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    person_id: str

    embedding: List[float]
    model_version: str
    quality_score: float
    created_at: float
