import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List
from .utils import get_logger
from .config import config

logger = get_logger(__name__)


class QualityStatus(str, Enum):
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    LOW_QUALITY = "LOW_QUALITY"
    REJECTED = "REJECTED"


@dataclass
class QualityResult:
    score: float
    status: QualityStatus
    reasons: List[str]


class FaceQualityAssessor:
    def __init__(self):
        self.min_blur = config.MIN_BLUR_VARIANCE
        self.min_brightness = config.MIN_BRIGHTNESS
        self.max_brightness = config.MAX_BRIGHTNESS
        self.min_area = config.MIN_FACE_AREA
        self.min_visibility = config.MIN_FACE_VISIBILITY

    def assess_face_quality(self, face_crop: np.ndarray) -> QualityResult:
        if face_crop is None or face_crop.size == 0:
            return QualityResult(0.0, QualityStatus.REJECTED, ["Empty crop"])

        h, w = face_crop.shape[:2]
        area = h * w
        reasons = []
        score_deductions = 0.0

        if area < self.min_area:
            reasons.append(f"Area too small ({area} < {self.min_area})")
            score_deductions += 0.6

        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop

        mean_brightness = np.mean(gray)
        if mean_brightness < self.min_brightness:
            reasons.append(f"Too dark (brightness: {mean_brightness:.1f})")
            score_deductions += 0.4
        elif mean_brightness > self.max_brightness:
            reasons.append(f"Too bright (brightness: {mean_brightness:.1f})")
            score_deductions += 0.4

        blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_variance < self.min_blur:
            reasons.append(
                f"Too blurry (variance: {blur_variance:.1f} < {self.min_blur})"
            )
            score_deductions += 0.5

        std_dev = np.std(gray)
        if std_dev < self.min_visibility:
            reasons.append(f"Low contrast/visibility (std dev: {std_dev:.1f})")
            score_deductions += 0.3

        final_score = max(0.0, 1.0 - score_deductions)

        if final_score >= 0.8:
            status = QualityStatus.GOOD
        elif final_score >= 0.5:
            status = QualityStatus.ACCEPTABLE
        elif final_score > 0.0:
            status = QualityStatus.LOW_QUALITY
        else:
            status = QualityStatus.REJECTED

        if len(reasons) >= 2 or final_score == 0.0:
            status = QualityStatus.REJECTED

        return QualityResult(final_score, status, reasons)
