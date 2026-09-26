import cv2
import os
import numpy as np
from typing import Tuple, Optional, Dict, List
from .config import config
from .schemas import ExpressionState
from .utils import get_logger

logger = get_logger(__name__)


class ExpressionModel:
    def __init__(self, model_path: str = None, class_mapping: List[str] = None):
        self.class_mapping = (
            class_mapping if class_mapping is not None else config.EXPRESSION_LABELS
        )

        if not self.class_mapping:
            logger.error("No class mapping provided to ExpressionModel")

        self.model_path = model_path
        self.net = None

        if self.model_path and os.path.exists(self.model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
            except Exception as e:
                logger.error(
                    f"Failed to load expression model from {self.model_path}: {e}"
                )
                self.net = None
        else:
            if self.model_path:
                logger.error(f"Model path {self.model_path} does not exist.")

    def analyze(
        self, face_crop: np.ndarray
    ) -> Tuple[Optional[str], float, Dict[str, float], ExpressionState]:
        """
        Analyze the facial expression of a face crop.
        Returns: expression_label, confidence, probabilities_dict, status
        """
        if face_crop is None or face_crop.size == 0:
            return None, 0.0, {}, ExpressionState.NO_FACE

        if not self.class_mapping:
            return None, 0.0, {}, ExpressionState.UNCERTAIN

        if self.net is None:

            probs = {
                label: 1.0 / len(self.class_mapping) for label in self.class_mapping
            }
            return None, 0.0, probs, ExpressionState.UNCERTAIN

        try:

            blob = cv2.dnn.blobFromImage(
                face_crop, 1.0 / 255.0, (64, 64), (0, 0, 0), swapRB=True, crop=False
            )
            self.net.setInput(blob)
            preds = self.net.forward()

            preds = preds[0]
            exp_preds = np.exp(preds - np.max(preds))
            probs_array = exp_preds / np.sum(exp_preds)

            probs = {
                self.class_mapping[i]: float(probs_array[i])
                for i in range(min(len(self.class_mapping), len(probs_array)))
            }
            best_idx = np.argmax(probs_array)
            best_class = self.class_mapping[best_idx]
            confidence = float(probs_array[best_idx])

            if confidence < config.MIN_CLASSIFICATION_CONFIDENCE:
                status = ExpressionState.UNCERTAIN
            else:
                status = ExpressionState.CLASSIFIED

            return best_class, confidence, probs, status

        except Exception as e:
            logger.error(f"Expression model inference failed: {e}")
            return None, 0.0, {}, ExpressionState.UNCERTAIN
