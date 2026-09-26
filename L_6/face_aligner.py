import cv2
import numpy as np


class FaceAligner:
    def __init__(self, target_size=(112, 112)):
        self.target_size = target_size

    def align_face(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Align the face crop to the target dimensions expected by the embedder.
        In a full landmarks-based pipeline, this would rotate and center eyes/nose.
        For now, we enforce a strict high-quality resize to prevent embedding distortion.
        """
        if face_crop is None or face_crop.size == 0:
            raise ValueError("Invalid face crop provided for alignment.")

        if face_crop.shape[:2] == self.target_size:
            return face_crop.copy()

        h, w = face_crop.shape[:2]
        interpolation = (
            cv2.INTER_AREA
            if (h > self.target_size[0] or w > self.target_size[1])
            else cv2.INTER_CUBIC
        )

        aligned = cv2.resize(face_crop, self.target_size, interpolation=interpolation)
        return aligned
