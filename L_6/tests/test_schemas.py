import unittest
import time
from L_6.schemas import FaceRecognitionResult, FaceState


class TestSchemas(unittest.TestCase):
    def test_face_recognition_result_creation(self):
        result = FaceRecognitionResult(
            track_id="track_123",
            face_id="face_abc",
            person_id="person_456",
            identity="John Doe",
            match_status=FaceState.MATCHED,
            similarity=0.95,
            quality_score=0.99,
            timestamp=time.time(),
            model_version="v1.0",
        )
        self.assertEqual(result.track_id, "track_123")
        self.assertEqual(result.match_status, FaceState.MATCHED)
        self.assertEqual(result.similarity, 0.95)

    def test_face_recognition_result_defaults(self):
        result = FaceRecognitionResult(
            track_id="track_123", match_status=FaceState.NO_FACE, timestamp=time.time()
        )
        self.assertEqual(result.track_id, "track_123")
        self.assertEqual(result.match_status, FaceState.NO_FACE)
        self.assertIsNone(result.face_id)
        self.assertIsNone(result.similarity)


if __name__ == "__main__":
    unittest.main()
