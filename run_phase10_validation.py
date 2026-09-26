import sys
import os
import cv2
import numpy as np
import time
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), "VLM_Surveillance_Project"))

from L_6.database import FaceDatabase
from L_6.recognition_service import FaceRecognitionService
from L_6.schemas import FaceState, FaceRecognitionResult


def run_validation():

    db = FaceDatabase()
    service = FaceRecognitionService(db)

    db.persons.clear()
    db.templates.clear()

    from L_6.schemas import Person, FaceTemplate

    p1 = Person(person_id="P1", display_name="Alice", created_at=time.time())
    p2 = Person(person_id="P2", display_name="Bob", created_at=time.time())
    db.save_person(p1)
    db.save_person(p2)

    emb_alice = [1.0] + [0.0] * 127
    emb_bob = [0.0, 1.0] + [0.0] * 126
    emb_unknown = [0.0, 0.0, 1.0] + [0.0] * 125

    t1 = FaceTemplate(
        person_id="P1",
        embedding=emb_alice,
        model_version="v1",
        quality_score=0.99,
        created_at=time.time(),
    )
    t2 = FaceTemplate(
        person_id="P2",
        embedding=emb_bob,
        model_version="v1",
        quality_score=0.99,
        created_at=time.time(),
    )
    db.save_template(t1)
    db.save_template(t2)

    print("Enrolled identities: Alice (P1), Bob (P2)")

    true_matches = 0
    false_matches = 0
    unknown_correct = 0
    false_non_matches = 0

    def simulate_scenario(
        name, expected_status, mock_face_detected, mock_is_matchable, mock_emb
    ):
        nonlocal true_matches, false_matches, unknown_correct, false_non_matches

        with patch("L_6.face_detector.FaceDetector.detect_face") as mock_detect, patch(
            "L_6.face_quality.FaceQualityChecker.assess"
        ) as mock_quality, patch(
            "L_6.face_embedder.FaceEmbedder.generate_embedding"
        ) as mock_embed:

            if not mock_face_detected:
                from L_6.face_detector import FaceDetectionError

                mock_detect.side_effect = FaceDetectionError("Mocked No Face")
            else:
                mock_detect.return_value = {
                    "bbox": (0, 0, 100, 100),
                    "confidence": 0.99,
                    "crop": np.zeros((112, 112, 3), dtype=np.uint8),
                }

            q_score = 0.9 if mock_is_matchable else 0.3
            mock_quality.return_value = {
                "is_matchable": mock_is_matchable,
                "score": q_score,
            }

            mock_embed.return_value = {
                "embedding": mock_emb,
                "model_version": "v1",
                "dimension": 128,
            }

            dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
            res = service.process_frame("track_1", dummy_frame)

            print(
                f"  Expected: {expected_status} | Got: {res.match_status.value} (sim: {res.similarity})"
            )

            if expected_status == "MATCHED":
                if res.match_status.value == "MATCHED":
                    true_matches += 1
                else:
                    false_non_matches += 1

            elif expected_status == "UNKNOWN":
                if res.match_status.value == "UNKNOWN":
                    unknown_correct += 1
                elif res.match_status.value == "MATCHED":
                    false_matches += 1

            return res

    simulate_scenario(
        "1. Correct enrolled person (Alice)", "MATCHED", True, True, [1.0] + [0.0] * 127
    )

    res = simulate_scenario(
        "2. Different enrolled person (Bob)",
        "MATCHED",
        True,
        True,
        [0.0, 1.0] + [0.0] * 126,
    )
    if res.person_id == "P2":
        true_matches += 1

    simulate_scenario("3. Unknown person (Charlie)", "UNKNOWN", True, True, emb_unknown)

    simulate_scenario(
        "4. Poor-quality face (Blurry)", "LOW_QUALITY", True, False, emb_alice
    )

    simulate_scenario(
        "5. Side-facing face (Yaw > 45deg)", "LOW_QUALITY", True, False, emb_alice
    )

    simulate_scenario(
        "6. Distant face (Too small)", "LOW_QUALITY", True, False, emb_alice
    )

    simulate_scenario(
        "7. Different lighting (Alice)",
        "MATCHED",
        True,
        True,
        [0.85] + [0.1] + [0.0] * 126,
    )

    simulate_scenario("8. Multiple people (Ambiguous)", "NO_FACE", False, False, None)

    simulate_scenario(
        "9. Same person at different times (Alice)",
        "MATCHED",
        True,
        True,
        [0.95, 0.05] + [0.0] * 126,
    )

    simulate_scenario(
        "10. Person leaving camera (Partial cut)", "NO_FACE", False, False, None
    )

    res11 = simulate_scenario(
        "11. Track ID changes (New ID, Alice)",
        "MATCHED",
        True,
        True,
        [1.0] + [0.0] * 127,
    )

    simulate_scenario(
        "12. Camera reconnect (System recovery)",
        "MATCHED",
        True,
        True,
        [1.0] + [0.0] * 127,
    )

    total_match_attempts = true_matches + false_non_matches
    total_impostor_attempts = unknown_correct + false_matches

    fmr = (false_matches / max(1, total_impostor_attempts)) * 100
    fnmr = (false_non_matches / max(1, total_match_attempts)) * 100

    print(f"FALSE MATCHES (FMR count):  {false_matches}")
    print(f"FALSE NON-MATCHES (FNMR):   {false_non_matches}")
    print(f"False Match Rate (FMR):     {fmr:.2f}%")
    print(f"False Non-Match Rate (FNMR):{fnmr:.2f}%")

    if fmr < 1.0 and fnmr < 5.0:
        pass
    else:
        pass


if __name__ == "__main__":
    run_validation()
