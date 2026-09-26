import unittest
from unittest.mock import patch
import numpy as np
from L_6.face_detector import FaceDetector, FaceDetectionError


class TestL6FaceDetector(unittest.TestCase):
    def setUp(self):
        self.detector = FaceDetector(min_face_size=48)
        self.dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)

    @patch("L_5.face_detector.FaceDetector.detect_faces")
    def test_clear_face(self, mock_detect):

        mock_detect.return_value = [
            {
                "bbox": (50, 50, 100, 100),
                "confidence": 0.99,
                "crop": np.zeros((100, 100, 3), dtype=np.uint8),
                "timestamp": 123456789.0,
            }
        ]

        result = self.detector.detect_face(self.dummy_frame)
        self.assertEqual(result["confidence"], 0.99)
        self.assertEqual(result["bbox"], (50, 50, 100, 100))

    @patch("L_5.face_detector.FaceDetector.detect_faces")
    def test_no_face(self, mock_detect):

        mock_detect.return_value = []

        with self.assertRaisesRegex(FaceDetectionError, "No valid face detected"):
            self.detector.detect_face(self.dummy_frame)

    @patch("L_5.face_detector.FaceDetector.detect_faces")
    def test_multiple_faces_enforced(self, mock_detect):

        mock_detect.return_value = [
            {"bbox": (10, 10, 50, 50), "confidence": 0.9},
            {"bbox": (100, 100, 50, 50), "confidence": 0.85},
        ]

        with self.assertRaisesRegex(FaceDetectionError, "Multiple faces detected"):
            self.detector.detect_face(self.dummy_frame, enforce_single_face=True)

    @patch("L_5.face_detector.FaceDetector.detect_faces")
    def test_multiple_faces_not_enforced(self, mock_detect):

        mock_detect.return_value = [
            {"bbox": (10, 10, 50, 50), "confidence": 0.85},
            {"bbox": (100, 100, 50, 50), "confidence": 0.95},
        ]

        result = self.detector.detect_face(self.dummy_frame, enforce_single_face=False)
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["bbox"], (100, 100, 50, 50))

    @patch("L_5.face_detector.FaceDetector.detect_faces")
    def test_small_face(self, mock_detect):

        mock_detect.return_value = []
        with self.assertRaisesRegex(FaceDetectionError, "No valid face detected"):
            self.detector.detect_face(self.dummy_frame)

    @patch("L_5.face_detector.FaceDetector.extract_face")
    def test_extract_face_partial(self, mock_extract):

        expected_crop = np.zeros((30, 30, 3), dtype=np.uint8)
        mock_extract.return_value = expected_crop

        crop = self.detector.extract_face(self.dummy_frame, (-10, 0, 50, 50))

        np.testing.assert_array_equal(crop, expected_crop)


if __name__ == "__main__":
    unittest.main()
