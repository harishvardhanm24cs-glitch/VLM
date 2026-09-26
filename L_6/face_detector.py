import numpy as np
from typing import Dict, Any, Tuple, Optional
from L_5.face_detector import FaceDetector as L5FaceDetector


class FaceDetectionError(Exception):
    """Exception raised for face detection errors (e.g., no face, multiple faces)."""


class FaceDetector:
    def __init__(self, min_face_size: int = 48):

        self.detector = L5FaceDetector(min_face_size=min_face_size)

    def extract_face(
        self, frame: np.ndarray, bounding_box: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extract a face crop using the given bounding box.
        Delegates to L_5 implementation for dimension safety.
        """
        return self.detector.extract_face(frame, bounding_box)

    def detect_face(
        self, frame: np.ndarray, enforce_single_face: bool = True
    ) -> Dict[str, Any]:
        """
        Detect a face in the provided frame/person crop.

        Args:
            frame: Image numpy array.
            enforce_single_face: If True (e.g., for enrollment), rejects images with > 1 face.

        Returns:
            Dict containing 'bbox', 'confidence', 'crop', 'timestamp'.

        Raises:
            FaceDetectionError: If no face, invalid box, small face, or multiple faces (when enforced).
        """
        results = self.detector.detect_faces(frame)

        if not results:
            raise FaceDetectionError(
                "No valid face detected (could be no face, too small, or out of bounds)."
            )

        if enforce_single_face and len(results) > 1:
            raise FaceDetectionError(
                f"Multiple faces detected ({len(results)}). Expected exactly one."
            )

        best_face = max(results, key=lambda x: x["confidence"])
        return best_face
