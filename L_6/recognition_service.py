import time
import numpy as np

from L_6.schemas import FaceRecognitionResult, FaceState
from L_6.face_detector import FaceDetector, FaceDetectionError
from L_6.face_quality import FaceQualityChecker
from L_6.face_aligner import FaceAligner
from L_6.face_embedder import FaceEmbedder, FaceEmbedderError
from L_6.face_matcher import FaceMatcher
from L_6.database import FaceDatabase


class FaceRecognitionService:
    def __init__(self, database: FaceDatabase):
        self.db = database
        self.detector = FaceDetector()
        self.quality_checker = FaceQualityChecker()
        self.aligner = FaceAligner()
        self.embedder = FaceEmbedder()
        self.matcher = FaceMatcher(database)

    def process_frame(self, track_id: str, frame: np.ndarray) -> FaceRecognitionResult:
        """
        Execute the full L_6 Face Recognition pipeline.

        Pipeline:
        Detected Face -> Quality Check -> Embedding -> Database/vector search -> Similarity calc -> Decision
        """
        timestamp = time.time()

        result = FaceRecognitionResult(
            track_id=track_id, match_status=FaceState.PROCESSING, timestamp=timestamp
        )

        try:

            detect_res = self.detector.detect_face(frame, enforce_single_face=False)
            face_crop = detect_res["crop"]
        except FaceDetectionError:
            result.match_status = FaceState.NO_FACE
            return result

        quality = self.quality_checker.assess(face_crop)
        result.quality_score = quality["score"]

        if not quality["is_matchable"]:
            result.match_status = FaceState.LOW_QUALITY
            return result

        try:

            aligned = self.aligner.align_face(face_crop)
            embed_res = self.embedder.generate_embedding(aligned)
            embedding_vector = embed_res["embedding"]
            result.model_version = embed_res["model_version"]
        except (ValueError, FaceEmbedderError):
            result.match_status = FaceState.ERROR
            return result

        match_result = self.matcher.find_best_match(embedding_vector)

        result.match_status = FaceState(match_result["match_status"])
        result.person_id = match_result["person_id"]
        result.identity = match_result["identity"]
        result.similarity = match_result["similarity"]

        return result
