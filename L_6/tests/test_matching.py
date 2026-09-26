import unittest
from unittest.mock import patch
import numpy as np

from L_6.schemas import FaceState, Person, FaceTemplate
from L_6.database import FaceDatabase
from L_6.config import config
from L_6.recognition_service import FaceRecognitionService
from L_6.face_detector import FaceDetectionError


class TestL6Matching(unittest.TestCase):
    def setUp(self):
        self.db = FaceDatabase()
        self.service = FaceRecognitionService(self.db)
        self.dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)

        self.person = Person(display_name="Target", created_at=0.0)
        self.db.save_person(self.person)
        self.template = FaceTemplate(
            person_id=self.person.person_id,
            embedding=list(np.ones(128, dtype=np.float32)),
            model_version="v1.0",
            quality_score=0.99,
            created_at=0.0,
        )
        self.db.save_template(self.template)

    @patch("L_6.face_detector.FaceDetector.detect_face")
    @patch("L_6.face_quality.FaceQualityChecker.assess")
    @patch("L_6.face_embedder.FaceEmbedder.generate_embedding")
    @patch("L_6.face_embedder.FaceEmbedder.compute_similarity")
    def test_matched(self, mock_sim, mock_embed, mock_quality, mock_detect):
        mock_detect.return_value = {"crop": np.zeros((100, 100, 3), dtype=np.uint8)}
        mock_quality.return_value = {"is_matchable": True, "score": 0.95}
        mock_embed.return_value = {
            "embedding": np.ones(128, dtype=np.float32),
            "model_version": "v1.0",
        }

        mock_sim.return_value = config.FACE_MATCH_THRESHOLD + 0.1

        result = self.service.process_frame("track_1", self.dummy_frame)
        self.assertEqual(result.match_status, FaceState.MATCHED)
        self.assertEqual(result.identity, "Target")
        self.assertEqual(result.person_id, self.person.person_id)

    @patch("L_6.face_detector.FaceDetector.detect_face")
    @patch("L_6.face_quality.FaceQualityChecker.assess")
    @patch("L_6.face_embedder.FaceEmbedder.generate_embedding")
    @patch("L_6.face_embedder.FaceEmbedder.compute_similarity")
    def test_uncertain(self, mock_sim, mock_embed, mock_quality, mock_detect):
        mock_detect.return_value = {"crop": np.zeros((100, 100, 3), dtype=np.uint8)}
        mock_quality.return_value = {"is_matchable": True, "score": 0.95}
        mock_embed.return_value = {
            "embedding": np.ones(128, dtype=np.float32),
            "model_version": "v1.0",
        }

        mock_sim.return_value = config.FACE_MATCH_THRESHOLD - (
            config.FACE_UNCERTAIN_MARGIN / 2
        )

        result = self.service.process_frame("track_1", self.dummy_frame)
        self.assertEqual(result.match_status, FaceState.UNCERTAIN)
        self.assertIsNone(result.identity)

    @patch("L_6.face_detector.FaceDetector.detect_face")
    @patch("L_6.face_quality.FaceQualityChecker.assess")
    @patch("L_6.face_embedder.FaceEmbedder.generate_embedding")
    @patch("L_6.face_embedder.FaceEmbedder.compute_similarity")
    def test_unknown(self, mock_sim, mock_embed, mock_quality, mock_detect):
        mock_detect.return_value = {"crop": np.zeros((100, 100, 3), dtype=np.uint8)}
        mock_quality.return_value = {"is_matchable": True, "score": 0.95}
        mock_embed.return_value = {
            "embedding": np.ones(128, dtype=np.float32),
            "model_version": "v1.0",
        }

        mock_sim.return_value = config.FACE_MATCH_THRESHOLD - 0.2

        result = self.service.process_frame("track_1", self.dummy_frame)
        self.assertEqual(result.match_status, FaceState.UNKNOWN)
        self.assertIsNone(result.identity)

    @patch("L_6.face_detector.FaceDetector.detect_face")
    @patch("L_6.face_quality.FaceQualityChecker.assess")
    def test_low_quality_abortion(self, mock_quality, mock_detect):
        mock_detect.return_value = {"crop": np.zeros((100, 100, 3), dtype=np.uint8)}
        mock_quality.return_value = {"is_matchable": False, "score": 0.2}

        result = self.service.process_frame("track_1", self.dummy_frame)
        self.assertEqual(result.match_status, FaceState.LOW_QUALITY)

    @patch("L_6.face_detector.FaceDetector.detect_face")
    def test_no_face_abortion(self, mock_detect):
        mock_detect.side_effect = FaceDetectionError()

        result = self.service.process_frame("track_1", self.dummy_frame)
        self.assertEqual(result.match_status, FaceState.NO_FACE)


if __name__ == "__main__":
    unittest.main()
