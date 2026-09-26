import numpy as np
from typing import Dict, Any
from L_6.database import FaceDatabase
from L_6.face_embedder import FaceEmbedder
from L_6.config import config


class FaceMatcher:
    def __init__(self, database: FaceDatabase):
        self.db = database
        self.embedder = FaceEmbedder()

    def find_best_match(self, query_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Search the database for the closest match to the query embedding.

        Returns:
            Dict containing:
                - person_id: Optional[str]
                - identity: Optional[str]
                - similarity: Optional[float]
                - match_status: "MATCHED", "UNCERTAIN", or "UNKNOWN"
        """
        all_templates = self.db.get_all_templates()

        if not all_templates:
            return {
                "person_id": None,
                "identity": None,
                "similarity": None,
                "match_status": "UNKNOWN",
            }

        best_score = -1.0
        best_template = None

        for template in all_templates:

            db_emb = np.array(template.embedding, dtype=np.float32)
            sim = self.embedder.compute_similarity(query_embedding, db_emb)

            if sim > best_score:
                best_score = sim
                best_template = template

        threshold = config.FACE_MATCH_THRESHOLD
        margin = config.FACE_UNCERTAIN_MARGIN

        if best_score >= threshold:
            person = self.db.get_person(best_template.person_id)
            return {
                "person_id": person.person_id if person else None,
                "identity": person.display_name if person else None,
                "similarity": best_score,
                "match_status": "MATCHED",
            }
        elif best_score >= (threshold - margin):

            return {
                "person_id": None,
                "identity": None,
                "similarity": best_score,
                "match_status": "UNCERTAIN",
            }
        else:
            return {
                "person_id": None,
                "identity": None,
                "similarity": best_score if best_score > 0 else None,
                "match_status": "UNKNOWN",
            }
