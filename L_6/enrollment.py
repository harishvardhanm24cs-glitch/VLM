import time
import numpy as np
from typing import Tuple

from L_6.schemas import Person, FaceTemplate
from L_6.face_detector import FaceDetector, FaceDetectionError
from L_6.face_quality import FaceQualityChecker
from L_6.face_aligner import FaceAligner
from L_6.face_embedder import FaceEmbedder, FaceEmbedderError
from L_6.database import FaceDatabase, FaceDatabaseError


class EnrollmentError(Exception):
    pass


class FaceEnrollmentService:
    def __init__(self, database: FaceDatabase):
        self.db = database
        self.detector = FaceDetector()
        self.quality_checker = FaceQualityChecker()
        self.aligner = FaceAligner()
        self.embedder = FaceEmbedder()

    def enroll_person(
        self, display_name: str, frame: np.ndarray, external_id: str = None
    ) -> Tuple[Person, FaceTemplate]:
        """
        Executes the strict face enrollment workflow.

        Args:
            display_name: The name of the person being enrolled.
            frame: A high-quality numpy image array containing exactly one face.
            external_id: Optional external identifier.

        Returns:
            Tuple of the created (Person, FaceTemplate) objects.

        Raises:
            EnrollmentError: If any step of the strict pipeline fails.
        """

        if not display_name or not display_name.strip():
            raise EnrollmentError("A valid display_name is required for enrollment.")

        if frame is None or frame.size == 0:
            raise EnrollmentError("Invalid image frame provided.")

        try:

            detect_result = self.detector.detect_face(frame, enforce_single_face=True)
            face_crop = detect_result["crop"]
        except FaceDetectionError as e:
            raise EnrollmentError(f"Detection failed: {str(e)}")

        quality = self.quality_checker.assess(face_crop)
        if not quality["is_matchable"]:
            raise EnrollmentError(
                f"Quality check failed ({quality['status']}). Reasons: {', '.join(quality['reasons'])}"
            )

        try:

            aligned_face = self.aligner.align_face(face_crop)

            embed_result = self.embedder.generate_embedding(aligned_face)
            embedding_vector = embed_result["embedding"]

            embedding_list = [float(x) for x in embedding_vector]
        except (ValueError, FaceEmbedderError) as e:
            raise EnrollmentError(f"Embedding failed: {str(e)}")

        timestamp = time.time()

        person = Person(
            display_name=display_name,
            external_identifier=external_id,
            created_at=timestamp,
        )

        template = FaceTemplate(
            person_id=person.person_id,
            embedding=embedding_list,
            model_version=embed_result["model_version"],
            quality_score=quality["score"],
            created_at=timestamp,
        )

        try:
            self.db.save_person(person)
            self.db.save_template(template)
        except FaceDatabaseError as e:
            raise EnrollmentError(f"Database error: {str(e)}")

        return person, template
