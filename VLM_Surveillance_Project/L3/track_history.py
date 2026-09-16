import logging

logger = logging.getLogger(__name__)

class TrackHistory:
    def __init__(self, max_lost_frames=30):
        self.tracks = {}
        self.max_lost_frames = max_lost_frames

    def update(self, frame_number, timestamp, detections):
        """
        Updates tracking history with new detections from the current frame.
        detections should be a list of dicts with at least:
        - track_id, class_name, x1, y1, x2, y2
        """
        current_frame_ids = set()
        
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue
                
            current_frame_ids.add(track_id)
            
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            if track_id not in self.tracks:
                # New track
                self.tracks[track_id] = {
                    "track_id": track_id,
                    "class": det.get("class_name"),
                    "object_type": det.get("object_type"),
                    "vehicle_category": det.get("vehicle_category"),
                    "subtype": det.get("subtype"),
                    "alert": det.get("alert"),
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "current_bbox": [x1, y1, x2, y2],
                    "center_x": center_x,
                    "center_y": center_y,
                    "movement_history": [{"x": center_x, "y": center_y, "time": timestamp}],
                    "lost_frames": 0,
                    "active": True
                }
            else:
                # Update existing track
                track = self.tracks[track_id]
                track["class"] = det.get("class_name")
                track["object_type"] = det.get("object_type", track.get("object_type"))
                track["vehicle_category"] = det.get("vehicle_category", track.get("vehicle_category"))
                track["subtype"] = det.get("subtype", track.get("subtype"))
                track["alert"] = det.get("alert", track.get("alert"))
                track["last_seen"] = timestamp
                track["current_bbox"] = [x1, y1, x2, y2]
                track["center_x"] = center_x
                track["center_y"] = center_y
                track["movement_history"].append({"x": center_x, "y": center_y, "time": timestamp})
                track["lost_frames"] = 0
                track["active"] = True

        # Handle lost tracks
        for tid, track in self.tracks.items():
            if tid not in current_frame_ids and track["active"]:
                track["lost_frames"] += 1
                if track["lost_frames"] > self.max_lost_frames:
                    track["active"] = False

    def get_active_tracks(self):
        return {tid: track for tid, track in self.tracks.items() if track["active"]}
        
    def get_all_tracks(self):
        return self.tracks
