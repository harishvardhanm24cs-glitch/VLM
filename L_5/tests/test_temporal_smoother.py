import unittest
import time
from L_5.temporal_smoother import TemporalSmoother, TemporalStatus


class TestTemporalSmoother(unittest.TestCase):
    def setUp(self):
        self.smoother = TemporalSmoother()
        self.smoother.min_observations = 3
        self.smoother.stability_ratio = 0.6
        self.smoother.min_confidence = 0.5
        self.smoother.history_length = 5

    def test_insufficient_observations(self):
        self.smoother.add_observation("t1", "happy", 0.9, time.time())
        expr, conf, status = self.smoother.get_smoothed_expression("t1")
        self.assertEqual(status, TemporalStatus.INSUFFICIENT_OBSERVATIONS)

    def test_stable_expression(self):
        for _ in range(4):
            self.smoother.add_observation("t2", "neutral", 0.8, time.time())
        expr, conf, status = self.smoother.get_smoothed_expression("t2")
        self.assertEqual(status, TemporalStatus.STABLE)
        self.assertEqual(expr, "neutral")

    def test_changing_expression(self):
        self.smoother.add_observation("t3", "neutral", 0.8, time.time())
        self.smoother.add_observation("t3", "happy", 0.8, time.time())
        self.smoother.add_observation("t3", "sad", 0.8, time.time())
        expr, conf, status = self.smoother.get_smoothed_expression("t3")
        self.assertEqual(status, TemporalStatus.UNCERTAIN)

    def test_disappeared_track(self):
        old_time = time.time() - 10.0
        self.smoother.add_observation("t4", "happy", 0.9, old_time)
        self.smoother.clean_stale_tracks(time.time(), timeout=5.0)
        expr, conf, status = self.smoother.get_smoothed_expression("t4")
        self.assertEqual(status, TemporalStatus.INSUFFICIENT_OBSERVATIONS)
        self.assertNotIn("t4", self.smoother.history)

    def test_history_overflow(self):
        for i in range(10):
            self.smoother.add_observation("t5", "happy", 0.9, time.time())
        self.assertEqual(len(self.smoother.history["t5"]), self.smoother.history_length)

    def test_multiple_simultaneous_tracks(self):
        for _ in range(3):
            self.smoother.add_observation("t6", "happy", 0.9, time.time())
            self.smoother.add_observation("t7", "sad", 0.9, time.time())

        expr1, conf1, status1 = self.smoother.get_smoothed_expression("t6")
        expr2, conf2, status2 = self.smoother.get_smoothed_expression("t7")
        self.assertEqual(expr1, "happy")
        self.assertEqual(expr2, "sad")


if __name__ == "__main__":
    unittest.main()
