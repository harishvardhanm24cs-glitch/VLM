import unittest
from unittest.mock import patch
import numpy as np
from L_6.face_quality import FaceQualityChecker
from L_5.face_quality import QualityStatus, QualityResult


class TestL6FaceQuality(unittest.TestCase):
    def setUp(self):
        self.checker = FaceQualityChecker()
        self.dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)

    @patch("L_5.face_quality.FaceQualityAssessor.assess_face_quality")
    def test_clear_face(self, mock_assess):
        mock_assess.return_value = QualityResult(
            score=0.95, status=QualityStatus.GOOD, reasons=[]
        )

        result = self.checker.assess(self.dummy_crop)
        self.assertEqual(result["status"], "GOOD")
        self.assertTrue(result["is_matchable"])

    @patch("L_5.face_quality.FaceQualityAssessor.assess_face_quality")
    def test_blurred_face(self, mock_assess):
        mock_assess.return_value = QualityResult(
            score=0.2, status=QualityStatus.LOW_QUALITY, reasons=["Too blurry"]
        )

        result = self.checker.assess(self.dummy_crop)
        self.assertEqual(result["status"], "LOW_QUALITY")
        self.assertFalse(result["is_matchable"])

    @patch("L_5.face_quality.FaceQualityAssessor.assess_face_quality")
    def test_dark_face(self, mock_assess):
        mock_assess.return_value = QualityResult(
            score=0.4, status=QualityStatus.LOW_QUALITY, reasons=["Too dark"]
        )

        result = self.checker.assess(self.dummy_crop)
        self.assertEqual(result["status"], "LOW_QUALITY")
        self.assertFalse(result["is_matchable"])

    @patch("L_5.face_quality.FaceQualityAssessor.assess_face_quality")
    def test_tiny_face(self, mock_assess):
        mock_assess.return_value = QualityResult(
            score=0.0, status=QualityStatus.REJECTED, reasons=["Area too small"]
        )

        result = self.checker.assess(self.dummy_crop)
        self.assertEqual(result["status"], "REJECTED")
        self.assertFalse(result["is_matchable"])

    @patch("L_5.face_quality.FaceQualityAssessor.assess_face_quality")
    def test_partially_visible_face(self, mock_assess):

        mock_assess.return_value = QualityResult(
            score=0.6, status=QualityStatus.ACCEPTABLE, reasons=["Low contrast"]
        )

        result = self.checker.assess(self.dummy_crop)
        self.assertEqual(result["status"], "ACCEPTABLE")
        self.assertTrue(result["is_matchable"])

    def test_invalid_image(self):

        empty = np.array([])
        result = self.checker.assess(empty)
        self.assertEqual(result["status"], "REJECTED")
        self.assertFalse(result["is_matchable"])


if __name__ == "__main__":
    unittest.main()
