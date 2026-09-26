from .schemas import ExpressionResult, ExpressionState
from .face_quality import FaceQualityAssessor
from .temporal_smoother import TemporalSmoother, TemporalStatus
from .expression_model import ExpressionModel
from .utils import get_logger
import time
import numpy as np
from typing import Tuple, Optional

logger = get_logger(__name__)


class ExpressionAnalyzer:
    def __init__(self, model_path: str = None):
        self.quality_assessor = FaceQualityAssessor()
        self.expression_model = ExpressionModel(model_path=model_path)
        self.temporal_smoother = TemporalSmoother()

    def process_frame(
        self,
        track_id: str,
        face_bbox: Optional[Tuple[int, int, int, int]],
        face_crop: Optional[np.ndarray],
    ) -> ExpressionResult:
        """
        Process a single frame for a tracked individual.
        Pipeline: Face Crop -> Quality -> Expression Model -> Temporal Smoothing -> ExpressionResult
        """
        timestamp = time.time()

        self.temporal_smoother.clean_stale_tracks(timestamp)

        if face_bbox is None or face_crop is None:
            return ExpressionResult(
                track_id=track_id,
                expression=None,
                confidence=0.0,
                timestamp=timestamp,
                face_detected=False,
                face_quality=0.0,
                observation_count=0,
                status=ExpressionState.NO_FACE,
            )

        quality_res = self.quality_assessor.assess_face_quality(face_crop)
        final_quality = quality_res.score

        if quality_res.status in ["LOW_QUALITY", "REJECTED"]:
            return ExpressionResult(
                track_id=track_id,
                expression=None,
                confidence=0.0,
                timestamp=timestamp,
                face_detected=True,
                face_quality=final_quality,
                observation_count=0,
                status=ExpressionState.LOW_QUALITY,
            )

        expr, conf, probs, status = self.expression_model.analyze(face_crop)

        if status == ExpressionState.UNCERTAIN:
            return ExpressionResult(
                track_id=track_id,
                expression=None,
                confidence=conf,
                timestamp=timestamp,
                face_detected=True,
                face_quality=final_quality,
                observation_count=0,
                status=ExpressionState.UNCERTAIN,
            )

        if expr:
            self.temporal_smoother.add_observation(track_id, expr, conf, timestamp)

        smoothed_expr, smoothed_conf, temp_status = (
            self.temporal_smoother.get_smoothed_expression(track_id)
        )
        obs_count = len(self.temporal_smoother.history.get(track_id, []))

        if temp_status == TemporalStatus.INSUFFICIENT_OBSERVATIONS:
            final_status = ExpressionState.UNCERTAIN
        elif temp_status == TemporalStatus.UNCERTAIN:
            final_status = ExpressionState.UNCERTAIN
        else:
            final_status = ExpressionState.CLASSIFIED

        return ExpressionResult(
            track_id=track_id,
            expression=smoothed_expr,
            confidence=smoothed_conf,
            timestamp=timestamp,
            face_detected=True,
            face_quality=final_quality,
            observation_count=obs_count,
            status=final_status,
        )
