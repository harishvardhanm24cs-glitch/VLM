import cv2
import time
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from .utils import get_logger

logger = get_logger(__name__)


class FaceDetector:
    def __init__(self, min_face_size: int = 48):
        self.min_face_size = min_face_size
        self.model_path = os.path.join(
            os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx"
        )

        if not os.path.exists(self.model_path):
            logger.error(f"Face detector model not found at {self.model_path}")
            raise RuntimeError("Face detector model failed to load")

        self.face_detector = cv2.FaceDetectorYN_create(
            self.model_path,
            "",
            (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )

    def _is_valid_bbox(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> bool:
        """Validate bounding box against frame dimensions and min size."""
        x, y, w, h = bbox
        img_h, img_w = frame.shape[:2]

        if w <= 0 or h <= 0:
            logger.debug(f"Rejected zero-area face: {bbox}")
            return False

        if w < self.min_face_size or h < self.min_face_size:
            logger.debug(f"Rejected small face: {bbox} (min: {self.min_face_size})")
            return False

        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            logger.debug(f"Rejected out-of-bounds face: {bbox}")
            return False

        return True

    def extract_face(
        self, frame: np.ndarray, bounding_box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Extract a face crop using the given bounding box."""
        if not self._is_valid_bbox(frame, bounding_box):
            return None
        x, y, w, h = bounding_box
        return frame[y : y + h, x : x + w].copy()

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in a frame and return their bounding boxes, crops, and confidence."""
        if frame is None or frame.size == 0:
            return []

        img_h, img_w = frame.shape[:2]
        self.face_detector.setInputSize((img_w, img_h))

        retval, faces = self.face_detector.detect(frame)

        timestamp = time.time()
        results = []

        if faces is None:
            return results

        for face in faces:

            box = face[0:4].astype(int)
            score = float(face[-1])

            bbox = tuple(box)
            if not self._is_valid_bbox(frame, bbox):
                continue

            crop = self.extract_face(frame, bbox)
            if crop is not None:
                results.append(
                    {
                        "bbox": bbox,
                        "confidence": score,
                        "crop": crop,
                        "timestamp": timestamp,
                    }
                )

        return results

    def draw_debug_visualization(
        self, frame: np.ndarray, results: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on the frame for debugging."""
        vis_frame = frame.copy()
        for i, res in enumerate(results):
            x, y, w, h = res["bbox"]
            conf = res["confidence"]
            label = f"FACE #{i+1} ({conf:.2f})"
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                vis_frame,
                label,
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return vis_frame
