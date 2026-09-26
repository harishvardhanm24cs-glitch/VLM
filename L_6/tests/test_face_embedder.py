import unittest
import numpy as np
import cv2
from L_6.face_embedder import FaceEmbedder, FaceEmbedderError


class TestL6FaceEmbedder(unittest.TestCase):
    def setUp(self):
        self.embedder = FaceEmbedder()

        np.random.seed(42)
        self.face1 = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)

        noise = np.random.normal(0, 5, (112, 112, 3)).astype(np.uint8)
        self.face1_variant = cv2.add(self.face1, noise)

        np.random.seed(99)
        self.face2 = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)

    def test_embedding_generation(self):
        result = self.embedder.generate_embedding(self.face1)

        self.assertIn("embedding", result)
        self.assertIn("model_name", result)
        self.assertIn("model_version", result)
        self.assertIn("dimension", result)

        self.assertEqual(result["dimension"], 128)
        self.assertEqual(len(result["embedding"]), 128)

    def test_meaningful_comparison(self):
        res1 = self.embedder.generate_embedding(self.face1)
        res1_var = self.embedder.generate_embedding(self.face1_variant)
        res2 = self.embedder.generate_embedding(self.face2)

        emb1 = res1["embedding"]
        emb1_var = res1_var["embedding"]
        emb2 = res2["embedding"]

        sim_identical = self.embedder.compute_similarity(emb1, emb1)
        self.assertAlmostEqual(sim_identical, 1.0, places=4)

        sim_same = self.embedder.compute_similarity(emb1, emb1_var)
        self.assertGreater(sim_same, 0.8)

        sim_diff = self.embedder.compute_similarity(emb1, emb2)

        self.assertLess(sim_diff, sim_same)

    def test_invalid_image(self):
        empty = np.array([])
        with self.assertRaises(FaceEmbedderError):
            self.embedder.generate_embedding(empty)


if __name__ == "__main__":
    unittest.main()
