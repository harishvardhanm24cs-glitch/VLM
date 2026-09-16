import uuid
import json
import csv
from pathlib import Path
from L3.models import Event
import logging

logger = logging.getLogger(__name__)

class EventEngine:
    def __init__(self, output_dir):
        self.active_events = {} # format: track_id_event_type -> event_id
        self.all_events = []
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.output_dir / "events.json"
        self.csv_path = self.output_dir / "events.csv"
        
    def process_rule_matches(self, track, rule_matches, current_timestamp, snapshot_path=""):
        track_id = track.get("track_id")
        
        for match in rule_matches:
            event_type = match["event_type"]
            state_key = f"{track_id}_{event_type}"
            
            # Debounce: only create event if not already active
            if state_key not in self.active_events:
                event_id = str(uuid.uuid4())
                event = Event(
                    event_id=event_id,
                    timestamp=current_timestamp,
                    track_id=track_id,
                    object_type=track.get("class", "unknown"),
                    event_type=event_type,
                    duration=round(match.get("duration", 0.0), 2),
                    severity=match.get("severity", "unknown"),
                    bbox=track.get("current_bbox", []),
                    description=match.get("description", ""),
                    snapshot=snapshot_path,
                    plate_text=match.get("plate_text", ""),
                    vehicle_info=match.get("vehicle_info", None),
                    vehicle_category=track.get("vehicle_category", None),
                    classification_confidence=track.get("classification_confidence", 0.0)
                )
                self.active_events[state_key] = event_id
                self.all_events.append(event)
                logger.warning(f"SUSPICIOUS EVENT DETECTED: {event_type} for Track ID {track_id}")

    def save_events(self):
        events_dicts = [vars(e) for e in self.all_events]
        
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(events_dicts, f, indent=4)
            
        if events_dicts:
            keys = events_dicts[0].keys()
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(events_dicts)
                
        logger.info(f"Saved {len(events_dicts)} events to JSON and CSV.")
