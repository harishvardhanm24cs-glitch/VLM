import unittest
import numpy as np
import cv2
from fastapi.testclient import TestClient


import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "VLM_Surveillance_Project")
)

from unittest.mock import patch
from L_6.schemas import Person, FaceTemplate
from backend.unified_server import app, state, l6_database


class TestL6API(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        state.face_recognitions.clear()
        l6_database.persons.clear()
        l6_database.templates.clear()

    def test_get_persons_empty(self):
        response = self.client.get("/api/persons")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    @patch("backend.unified_server.l6_enrollment.enroll_person")
    def test_enroll_and_get_person(self, mock_enroll):

        mock_person = Person(
            person_id="P123", display_name="Test Subject", created_at=0.0
        )
        mock_template = FaceTemplate(
            person_id="P123",
            embedding=[0.1] * 128,
            model_version="v1",
            quality_score=0.99,
            created_at=0.0,
        )
        mock_enroll.return_value = (mock_person, mock_template)

        l6_database.save_person(mock_person)
        l6_database.save_template(mock_template)

        img = np.ones((112, 112, 3), dtype=np.uint8) * 100
        ret, buf = cv2.imencode(".jpg", img)

        files = {"file": ("test.jpg", buf.tobytes(), "image/jpeg")}
        data = {"display_name": "Test Subject"}

        response = self.client.post("/api/persons/enroll", data=data, files=files)
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data["status"], "success")
        self.assertEqual(res_data["display_name"], "Test Subject")

        person_id = res_data["person_id"]

        response_get = self.client.get("/api/persons")
        self.assertEqual(response_get.status_code, 200)

        persons = response_get.json()
        self.assertEqual(len(persons), 1)

        self.assertNotIn("embedding", persons[0])
        self.assertEqual(persons[0]["person_id"], person_id)

    def test_face_recognition_state(self):

        state.face_recognitions["1"] = {
            "event_type": "FACE_RECOGNITION",
            "track_id": "1",
            "person_id": "P001",
            "identity": "Target",
            "match_status": "MATCHED",
            "similarity": 0.95,
            "timestamp": 12345.0,
        }

        response = self.client.get("/api/face-recognition")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["identity"], "Target")


if __name__ == "__main__":
    unittest.main()
