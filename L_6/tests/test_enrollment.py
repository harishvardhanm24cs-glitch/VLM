import unittest
from unittest.mock import patch
import numpy as np

from L_6.enrollment import FaceEnrollmentService, EnrollmentError
from L_6.database import FaceDatabase
from L_6.face_detector import FaceDetectionError


class TestL6Enrollment(unittest.TestCase):
    def setUp(self):
        self.db = FaceDatabase()
        self.service = FaceEnrollmentService(self.db)
        self.dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)

    @patch("L_6.face_detector.FaceDetector.detect_face")
    @patch("L_6.face_quality.FaceQualityChecker.assess")
    @patch("L_6.face_embedder.FaceEmbedder.generate_embedding")
    def test_successful_enrollment(self, mock_embed, mock_quality, mock_detect):

        mock_detect.return_value = {"crop": np.zeros((100, 100, 3), dtype=np.uint8)}
        mock_quality.return_value = {
            "is_matchable": True,
            "score": 0.95,
            "status": "GOOD",
            "reasons": [],
        }
        mock_embed.return_value = {
            "embedding": np.ones(128, dtype=np.float32),
            "model_version": "v1.0",
        }

        person, template = self.service.enroll_person("John Doe", self.dummy_frame)

        self.assertEqual(person.display_name, "John Doe")
        self.assertEqual(template.person_id, person.person_id)
        self.assertEqual(len(template.embedding), 128)
        self.assertAlmostEqual(template.quality_score, 0.95)

        self.assertIsNotNone(self.db.get_person(person.person_id))
        self.assertEqual(len(self.db.get_templates_for_person(person.person_id)), 1)

    @patch("L_6.face_detector.FaceDetector.detect_face")
    def test_rejection_no_face(self, mock_detect):
        mock_detect.side_effect = FaceDetectionError("No valid face detected")

        with self.assertRaisesRegex(EnrollmentError, "Detection failed: No valid face"):
            self.service.enroll_person("Jane Doe", self.dummy_frame)

    @patch("L_6.face_detector.FaceDetector.detect_face")
    def test_rejection_multiple_faces(self, mock_detect):
        mock_detect.side_effect = FaceDetectionError("Multiple faces detected")

        with self.assertRaisesRegex(
            EnrollmentError, "Detection failed: Multiple faces"
        ):
            self.service.enroll_person("Jane Doe", self.dummy_frame)

    @patch("L_6.face_detector.FaceDetector.detect_face")
    @patch("L_6.face_quality.FaceQualityChecker.assess")
    def test_rejection_low_quality(self, mock_quality, mock_detect):
        mock_detect.return_value = {"crop": np.zeros((100, 100, 3), dtype=np.uint8)}
        mock_quality.return_value = {
            "is_matchable": False,
            "status": "LOW_QUALITY",
            "reasons": ["Too blurry"],
        }

        with self.assertRaisesRegex(EnrollmentError, "Quality check failed"):
            self.service.enroll_person("Jane Doe", self.dummy_frame)

    def test_rejection_invalid_image(self):
        with self.assertRaisesRegex(EnrollmentError, "Invalid image frame"):
            self.service.enroll_person("Jane Doe", np.array([]))


if __name__ == "__main__":
    unittest.main()
