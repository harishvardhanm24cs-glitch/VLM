from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ExpressionState(str, Enum):
    NO_FACE = "NO_FACE"
    LOW_QUALITY = "LOW_QUALITY"
    UNCERTAIN = "UNCERTAIN"
    CLASSIFIED = "CLASSIFIED"


@dataclass
class ExpressionResult:
    track_id: str
    expression: Optional[str]
    confidence: float
    timestamp: float
    face_detected: bool
    face_quality: float
    observation_count: int
    status: ExpressionState
