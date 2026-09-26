import unittest
import numpy as np
from L_5.schemas import ExpressionResult, ExpressionState
from L_5.config import config
from L_5.expression_analyzer import ExpressionAnalyzer


class TestFoundation(unittest.TestCase):
    def setUp(self):

        self.dummy_crop = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)

        for i in range(0, 100, 2):
            self.dummy_crop[i, :] = 255
            self.dummy_crop[:, i] = 0

    def test_result_schema(self):
        result = ExpressionResult(
            track_id="test_1",
            expression="happy",
            confidence=0.9,
            timestamp=123456789.0,
            face_detected=True,
            face_quality=0.8,
            observation_count=1,
            status=ExpressionState.CLASSIFIED,
        )
        self.assertEqual(result.track_id, "test_1")
        self.assertEqual(result.status, ExpressionState.CLASSIFIED)

    def test_configuration_loading(self):
        self.assertGreater(config.MIN_FACE_SIZE, 0)
        self.assertGreater(config.MIN_FACE_QUALITY, 0)
        self.assertGreater(config.MIN_CLASSIFICATION_CONFIDENCE, 0)
        self.assertGreater(len(config.EXPRESSION_LABELS), 0)

    def test_missing_face(self):
        analyzer = ExpressionAnalyzer()
        result = analyzer.process_frame(
            track_id="test_2", face_bbox=None, face_crop=None
        )
        self.assertEqual(result.status, ExpressionState.NO_FACE)
        self.assertFalse(result.face_detected)

    def test_low_quality_face(self):
        analyzer = ExpressionAnalyzer()
        bad_crop = np.ones((20, 20, 3), dtype=np.uint8) * 128
        result = analyzer.process_frame(
            track_id="test_3", face_bbox=(0, 0, 20, 20), face_crop=bad_crop
        )
        self.assertEqual(result.status, ExpressionState.LOW_QUALITY)

    def test_uncertain_prediction(self):
        analyzer = ExpressionAnalyzer()

        analyzer.expression_model.analyze = lambda crop: (
            None,
            0.1,
            {},
            ExpressionState.UNCERTAIN,
        )
        result = analyzer.process_frame(
            track_id="test_4", face_bbox=(0, 0, 100, 100), face_crop=self.dummy_crop
        )
        self.assertEqual(result.status, ExpressionState.UNCERTAIN)

    def test_valid_classification(self):
        analyzer = ExpressionAnalyzer()

        analyzer.expression_model.analyze = lambda crop: (
            "happy",
            0.9,
            {"happy": 0.9},
            ExpressionState.CLASSIFIED,
        )
        for _ in range(config.MIN_OBSERVATIONS):
            result = analyzer.process_frame(
                track_id="test_5", face_bbox=(0, 0, 100, 100), face_crop=self.dummy_crop
            )

        self.assertEqual(result.status, ExpressionState.CLASSIFIED)
        self.assertEqual(result.expression, "happy")


if __name__ == "__main__":
    unittest.main()
