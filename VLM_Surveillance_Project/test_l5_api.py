import unittest
from fastapi.testclient import TestClient
from backend.unified_server import app, state

client = TestClient(app)


class TestL5API(unittest.TestCase):
    def setUp(self):
        state.expressions.clear()

    def test_api_expressions_empty(self):
        response = client.get("/api/expressions")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_api_edge_sync_populates_expressions(self):
        mock_payload = {
            "frame_number": 1,
            "tracks": {
                "21": {
                    "track_id": 21,
                    "class_name": "Person",
                    "object_type": "person",
                    "l5_expression": "happy",
                    "l5_expression_confidence": 0.95,
                    "l5_status": "CLASSIFIED",
                }
            },
        }

        response = client.post("/api/internal/edge_sync", json=mock_payload)
        self.assertEqual(response.status_code, 200)

        self.assertIn("21", state.expressions)

        resp = client.get("/api/expressions")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["expression"], "happy")
        self.assertEqual(data[0]["track_id"], "21")

        resp_tid = client.get("/api/expressions/21")
        self.assertEqual(resp_tid.status_code, 200)
        self.assertEqual(resp_tid.json()["expression"], "happy")

    def test_api_expressions_not_found(self):
        resp = client.get("/api/expressions/999")
        self.assertEqual(resp.status_code, 404)

    def test_cleanup_stale_tracks(self):

        mock_payload = {
            "frame_number": 1,
            "tracks": {
                "21": {
                    "track_id": 21,
                    "class_name": "Person",
                    "object_type": "person",
                    "l5_expression": "happy",
                    "l5_expression_confidence": 0.95,
                    "l5_status": "CLASSIFIED",
                }
            },
        }
        client.post("/api/internal/edge_sync", json=mock_payload)
        self.assertIn("21", state.expressions)

        mock_payload_2 = {"frame_number": 2, "tracks": {}}
        client.post("/api/internal/edge_sync", json=mock_payload_2)

        resp = client.get("/api/expressions")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])


if __name__ == "__main__":
    unittest.main()
