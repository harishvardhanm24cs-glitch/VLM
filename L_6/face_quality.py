import numpy as np
from typing import Dict, Any
from L_5.face_quality import (
    FaceQualityAssessor as L5FaceQualityAssessor,
    QualityStatus,
    QualityResult,
)


class FaceQualityChecker:
    def __init__(self):

        self.assessor = L5FaceQualityAssessor()

    def assess(self, face_crop: np.ndarray) -> Dict[str, Any]:
        """
        Assess the quality of a face crop for L_6 Face Recognition.

        Returns:
            Dict containing:
                - 'status': GOOD, ACCEPTABLE, LOW_QUALITY, or REJECTED
                - 'score': float (0.0 to 1.0)
                - 'reasons': List of reasons for deductions
                - 'is_matchable': bool (True if GOOD or ACCEPTABLE)
        """
        if face_crop is None or face_crop.size == 0:
            return {
                "status": QualityStatus.REJECTED.value,
                "score": 0.0,
                "reasons": ["Empty crop or invalid image"],
                "is_matchable": False,
            }

        result: QualityResult = self.assessor.assess_face_quality(face_crop)

        is_matchable = result.status in (QualityStatus.GOOD, QualityStatus.ACCEPTABLE)

        return {
            "status": result.status.value,
            "score": result.score,
            "reasons": result.reasons,
            "is_matchable": is_matchable,
        }
