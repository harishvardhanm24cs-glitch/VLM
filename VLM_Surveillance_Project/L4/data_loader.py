import json
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.tracks_path = self.project_root / "outputs" / "detections" / "tracks.json"
        self.events_path = self.project_root / "outputs" / "alerts" / "events.json"
        self.vlm_path = self.project_root / "outputs" / "alerts" / "vlm_analysis.json"

    def _read_json(self, path):
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return []

    def get_tracks(self):
        """Returns a dict of active tracks from tracks.json."""
        tracks_dict = self._read_json(self.tracks_path)
        # Handle case where tracks.json might be a list or dict
        if isinstance(tracks_dict, list):
            return {t.get("track_id", i): t for i, t in enumerate(tracks_dict) if t.get("active", False)}
        elif isinstance(tracks_dict, dict):
            return {tid: t for tid, t in tracks_dict.items() if t.get("active", False)}
        return {}

    def get_stats(self):
        """Returns counts for persons, vehicles, active tracks."""
        tracks = self.get_tracks()
        active_tracks = len(tracks)
        persons = 0
        vehicles = 0
        
        vehicle_classes = ["car", "truck", "bus", "motorcycle"]
        
        for t in tracks.values():
            obj_class = t.get("class", "").lower()
            if obj_class == "person":
                persons += 1
            elif obj_class in vehicle_classes:
                vehicles += 1
                
        return {
            "active_tracks": active_tracks,
            "persons": persons,
            "vehicles": vehicles
        }

    def get_events(self):
        """Returns a merged DataFrame of events and VLM explanations."""
        events = self._read_json(self.events_path)
        vlm_analyses = self._read_json(self.vlm_path)
        
        if not events:
            return pd.DataFrame(columns=[
                "event_id", "timestamp", "track_id", "object_type", 
                "event_type", "duration", "severity", "vlm_analysis"
            ])
            
        events_df = pd.DataFrame(events)
        
        if vlm_analyses:
            vlm_df = pd.DataFrame(vlm_analyses)
            # Merge VLM explanation if available
            if "event_id" in events_df.columns and "event_id" in vlm_df.columns:
                events_df = pd.merge(events_df, vlm_df[['event_id', 'vlm_analysis']], on="event_id", how="left")
            else:
                events_df["vlm_analysis"] = "N/A"
        else:
            events_df["vlm_analysis"] = "No VLM analysis available."
            
        # Ensure all required columns exist
        required_cols = ["event_id", "timestamp", "track_id", "object_type", "event_type", "duration", "severity", "vlm_analysis"]
        for col in required_cols:
            if col not in events_df.columns:
                events_df[col] = "N/A"
                
        # Fill NaN values
        events_df.fillna("N/A", inplace=True)
        return events_df
