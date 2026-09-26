import unittest
import numpy as np
from L_5.expression_model import ExpressionModel
from L_5.schemas import ExpressionState


class TestExpressionModel(unittest.TestCase):
    def setUp(self):
        self.dummy_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_model_loading_failure(self):

        model = ExpressionModel(model_path="non_existent_path.onnx")
        self.assertIsNone(model.net)

        expr, conf, probs, status = model.analyze(self.dummy_crop)
        self.assertEqual(status, ExpressionState.UNCERTAIN)

    def test_unknown_class_mapping(self):
        model = ExpressionModel(class_mapping=[])
        expr, conf, probs, status = model.analyze(self.dummy_crop)
        self.assertEqual(status, ExpressionState.UNCERTAIN)
        self.assertIsNone(expr)

    def test_invalid_image(self):
        model = ExpressionModel()
        expr, conf, probs, status = model.analyze(np.array([]))
        self.assertEqual(status, ExpressionState.NO_FACE)

    def test_mock_low_confidence(self):

        model = ExpressionModel()
        expr, conf, probs, status = model.analyze(self.dummy_crop)
        self.assertEqual(status, ExpressionState.UNCERTAIN)

        self.assertLess(conf, 0.5)

    def test_valid_face_mock_override(self):

        model = ExpressionModel()

        class MockNet:
            def setInput(self, blob):
                pass

            def forward(self):
                return [np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]

        model.net = MockNet()
        expr, conf, probs, status = model.analyze(self.dummy_crop)
        self.assertEqual(status, ExpressionState.CLASSIFIED)
        self.assertGreater(conf, 0.9)
        self.assertEqual(expr, model.class_mapping[0])


if __name__ == "__main__":
    unittest.main()
