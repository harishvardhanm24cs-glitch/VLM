from dataclasses import dataclass
from typing import List

@dataclass
class Event:
    event_id: str
    timestamp: float
    track_id: int
    object_type: str
    event_type: str
    duration: float
    severity: str
    bbox: List[float]
    snapshot: str = ""
    description: str = ""
    plate_text: str = ""
    vehicle_info: dict = None
    vehicle_category: str = None
    classification_confidence: float = 0.0
