import unittest
import numpy as np
from L_5.face_detector import FaceDetector


class TestFaceDetector(unittest.TestCase):
    def setUp(self):
        self.detector = FaceDetector(min_face_size=20)

        self.dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)

    def test_invalid_coordinates(self):

        bbox = (-10, 20, 50, 50)
        self.assertFalse(self.detector._is_valid_bbox(self.dummy_frame, bbox))

        bbox = (180, 20, 50, 50)
        self.assertFalse(self.detector._is_valid_bbox(self.dummy_frame, bbox))

    def test_zero_area(self):
        bbox = (50, 50, 0, 50)
        self.assertFalse(self.detector._is_valid_bbox(self.dummy_frame, bbox))

    def test_extremely_small_faces(self):

        bbox = (50, 50, 10, 10)
        self.assertFalse(self.detector._is_valid_bbox(self.dummy_frame, bbox))

    def test_valid_face(self):
        bbox = (50, 50, 40, 40)
        self.assertTrue(self.detector._is_valid_bbox(self.dummy_frame, bbox))

    def test_extract_face(self):
        bbox = (50, 50, 40, 40)

        self.dummy_frame[50:90, 50:90] = 255
        crop = self.detector.extract_face(self.dummy_frame, bbox)

        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape, (40, 40, 3))

        self.assertTrue(np.all(crop == 255))

    def test_detect_faces_empty_frame(self):
        res = self.detector.detect_faces(np.array([]))
        self.assertEqual(len(res), 0)

    def test_draw_visualization(self):
        results = [
            {
                "bbox": (50, 50, 40, 40),
                "confidence": 0.85,
                "crop": None,
                "timestamp": 12345.0,
            }
        ]
        vis_frame = self.detector.draw_debug_visualization(self.dummy_frame, results)
        self.assertEqual(vis_frame.shape, self.dummy_frame.shape)


if __name__ == "__main__":
    unittest.main()
