import os

class SuspiciousActivityRules:
    def __init__(self, config):
        self.person_time = float(os.getenv("LOITERING_THRESHOLD_SECONDS", config.get("person_stationary_time", 15)))
        self.vehicle_time = config.get("vehicle_stationary_time", 120)
        self.restricted_zones = config.get("restricted_zones", [])
        
    def _is_in_zone(self, bbox, zone):
        # zone format [x1, y1, x2, y2]
        # Just check if center of bbox is in zone
        bx1, by1, bx2, by2 = bbox
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        zx1, zy1, zx2, zy2 = zone
        return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

    def evaluate(self, enhanced_track):
        events = []
        obj_type = enhanced_track.get("class", "").lower()
        duration = enhanced_track.get("stationary_duration", 0.0)
        state = enhanced_track.get("movement_state", "")
        bbox = enhanced_track.get("current_bbox", [])
        
        # RULE 1: Person stationary (LOITERING)
        if obj_type == "person" and state == "stationary" and duration >= self.person_time:
            events.append({
                "event_type": "LOITERING",
                "severity": "WARNING",
                "duration": duration,
                "description": f"Person has been stationary for more than {self.person_time} seconds."
            })
            
        # RULE 2: Vehicle stationary
        if obj_type in ["car", "truck", "bus", "motorcycle"] and state == "stationary" and duration >= self.vehicle_time:
            events.append({
                "event_type": "stationary_vehicle",
                "severity": "WARNING",
                "duration": duration
            })
            
        # RULE 3: Restricted Zone Entry
        if obj_type == "person" and self.restricted_zones:
            for zone in self.restricted_zones:
                if self._is_in_zone(bbox, zone):
                    events.append({
                        "event_type": "restricted_zone_entry",
                        "severity": "CRITICAL",
                        "duration": 0.0
                    })
                    break
                    
        # RULE 4: Potential Object In Hand
        if enhanced_track.get("potential_object_in_hand"):
            events.append({
                "event_type": "OBJECT_IN_HAND",
                "severity": "HIGH",
                "duration": 0.0,
                "description": "Potential object detected near person's hand."
            })
            
        # RULE 5: Normal Vehicle Classification
        v_class = enhanced_track.get("vehicle_category")
        v_conf = enhanced_track.get("classification_confidence", 0.0)
        
        if v_class == "normal_vehicle":
            events.append({
                "event_type": "NORMAL_VEHICLE_ALERT",
                "severity": "HIGH",
                "duration": 0.0,
                "description": f"Normal vehicle detected with {v_conf*100:.1f}% confidence."
            })
            
        return events
