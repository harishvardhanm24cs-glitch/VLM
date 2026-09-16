import os
from L3.movement import MovementAnalyzer

class BehaviourAnalyzer:
    def __init__(self, config):
        threshold = float(os.getenv("LOITERING_MOVEMENT_THRESHOLD", config.get("movement_threshold_pixels", 20)))
        self.movement = MovementAnalyzer(threshold)
        self.object_hand_analysis = os.getenv("OBJECT_HAND_ANALYSIS_ENABLED", "true").lower() == "true"
        
    def analyze(self, track_dict, all_tracks=None):
        """
        Enhances track dictionary with movement state, duration, and object interaction.
        """
        state, displacement, stationary_duration = self.movement.analyze_movement(track_dict)
        track_dict["movement_state"] = state
        track_dict["displacement"] = displacement
        track_dict["stationary_duration"] = stationary_duration
        
        track_dict["recently_entered"] = len(track_dict.get("movement_history", [])) == 1
        track_dict["recently_exited"] = not track_dict.get("active", True)
        
        # Object-in-hand logic
        track_dict["potential_object_in_hand"] = False
        if self.object_hand_analysis and all_tracks and track_dict.get("class") == "person":
            person_bbox = track_dict.get("current_bbox")
            if person_bbox:
                px1, py1, px2, py2 = person_bbox
                # Check for nearby objects like mobile phone, bag, tool
                candidate_classes = ["mobile phone", "handbag", "suitcase", "bottle", "umbrella", "Object"]
                
                for oid, otrack in all_tracks.items():
                    if oid != track_dict.get("track_id") and otrack.get("class") in candidate_classes:
                        obbox = otrack.get("current_bbox")
                        if obbox:
                            ox1, oy1, ox2, oy2 = obbox
                            # Simple bounding box distance/overlap check
                            # Check if the object is generally within the person's bounding box area or very close
                            if (ox1 >= px1 - 20 and ox2 <= px2 + 20 and oy1 >= py1 and oy2 <= py2 + 50):
                                track_dict["potential_object_in_hand"] = True
                                break
        
        return track_dict
