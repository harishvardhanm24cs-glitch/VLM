import os
from typing import List


class L5Config:
    MIN_FACE_SIZE: int = 48
    MIN_FACE_QUALITY: float = 0.5
    MIN_CLASSIFICATION_CONFIDENCE: float = 0.6
    MIN_OBSERVATIONS: int = 3
    STABILITY_RATIO: float = 0.6
    MIN_CONFIDENCE: float = 0.5
    HISTORY_LENGTH: int = 10
    INFERENCE_INTERVAL: float = 0.1

    MIN_BRIGHTNESS: float = 40.0
    MAX_BRIGHTNESS: float = 240.0
    MIN_BLUR_VARIANCE: float = 50.0
    MIN_FACE_AREA: int = 48 * 48
    MIN_FACE_VISIBILITY: float = 15.0

    EXPRESSION_LABELS: List[str] = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]

    def __init__(self):

        for key in dir(self):
            if not key.startswith("_"):
                env_val = os.getenv(f"L5_{key}")
                if env_val is not None:
                    attr_type = type(getattr(self, key))
                    if attr_type == list:
                        setattr(self, key, env_val.split(","))
                    else:
                        setattr(self, key, attr_type(env_val))


config = L5Config()
