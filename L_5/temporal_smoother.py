from collections import deque
from enum import Enum
from typing import Dict, Tuple, Optional
from .utils import get_logger
from .config import config

logger = get_logger(__name__)


class TemporalStatus(str, Enum):
    STABLE = "STABLE"
    UNCERTAIN = "UNCERTAIN"
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"


class TemporalSmoother:
    def __init__(self):
        self.min_observations = config.MIN_OBSERVATIONS
        self.stability_ratio = config.STABILITY_RATIO
        self.min_confidence = config.MIN_CONFIDENCE
        self.history_length = config.HISTORY_LENGTH

        self.history: Dict[str, deque] = {}

    def add_observation(
        self, track_id: str, expression: str, confidence: float, timestamp: float
    ):
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.history_length)
        self.history[track_id].append(
            {"expression": expression, "confidence": confidence, "timestamp": timestamp}
        )

    def get_smoothed_expression(
        self, track_id: str
    ) -> Tuple[Optional[str], float, TemporalStatus]:
        if track_id not in self.history:
            return None, 0.0, TemporalStatus.INSUFFICIENT_OBSERVATIONS

        history = self.history[track_id]
        if len(history) < self.min_observations:
            return None, 0.0, TemporalStatus.INSUFFICIENT_OBSERVATIONS

        counts = {}
        total_conf = {}

        for obs in history:
            exp = obs["expression"]
            conf = obs["confidence"]
            counts[exp] = counts.get(exp, 0) + 1
            total_conf[exp] = total_conf.get(exp, 0.0) + conf

        most_frequent = max(counts, key=counts.get)
        freq_ratio = counts[most_frequent] / len(history)
        avg_conf = total_conf[most_frequent] / counts[most_frequent]

        if freq_ratio >= self.stability_ratio and avg_conf >= self.min_confidence:
            return most_frequent, avg_conf, TemporalStatus.STABLE

        return most_frequent, avg_conf, TemporalStatus.UNCERTAIN

    def clean_stale_tracks(self, current_time: float, timeout: float = 5.0):
        """Remove tracks that haven't been seen recently to prevent memory leaks."""
        stale_tracks = []
        for track_id, history in self.history.items():
            if not history:
                stale_tracks.append(track_id)
                continue
            last_seen = history[-1]["timestamp"]
            if current_time - last_seen > timeout:
                stale_tracks.append(track_id)

        for track_id in stale_tracks:
            del self.history[track_id]
            logger.debug(f"Cleaned up stale track: {track_id}")
