import unittest
import numpy as np
import cv2
from L_5.face_quality import FaceQualityAssessor, QualityStatus


class TestFaceQuality(unittest.TestCase):
    def setUp(self):
        self.assessor = FaceQualityAssessor()

    def test_clear_face(self):

        crop = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)

        for i in range(0, 100, 2):
            crop[i, :] = 255
            crop[:, i] = 0

        result = self.assessor.assess_face_quality(crop)
        self.assertIn(result.status, [QualityStatus.GOOD, QualityStatus.ACCEPTABLE])

    def test_tiny_face(self):

        crop = np.random.randint(50, 200, (20, 20, 3), dtype=np.uint8)

        for i in range(0, 20, 2):
            crop[i, :] = 255
            crop[:, i] = 0

        result = self.assessor.assess_face_quality(crop)
        self.assertIn(
            result.status, [QualityStatus.LOW_QUALITY, QualityStatus.REJECTED]
        )
        self.assertTrue(any("Area too small" in r for r in result.reasons))

    def test_blurred_face(self):

        crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(crop, (25, 25), 15)
        result = self.assessor.assess_face_quality(blurred)
        self.assertTrue(any("blurry" in r.lower() for r in result.reasons))

    def test_dark_face(self):

        crop = np.ones((100, 100, 3), dtype=np.uint8) * 10

        crop[50:, :] = 15
        result = self.assessor.assess_face_quality(crop)
        self.assertTrue(any("Too dark" in r for r in result.reasons))

    def test_partially_visible_face(self):

        crop = np.ones((100, 100, 3), dtype=np.uint8) * 128
        crop[50:, :] = 130
        result = self.assessor.assess_face_quality(crop)
        self.assertTrue(
            any(
                "visibility" in r.lower() or "contrast" in r.lower()
                for r in result.reasons
            )
        )


if __name__ == "__main__":
    unittest.main()
