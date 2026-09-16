import os
import sys
import logging
import json
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from L1.video_reader import VideoReader
from L3.track_history import TrackHistory
from L3.behaviour import BehaviourAnalyzer
from L3.suspicious_activity import SuspiciousActivityRules
from L3.event_engine import EventEngine
from models.vlm.qwen_model import QwenVLWrapper
from models.vlm.vlm_analyzer import VLMAnalyzer

logger = logging.getLogger(__name__)

class ObjectTracker:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        
        self.project_root = Path(config_path).parent.parent
        video_source = self.config.get("video", {}).get("source", "videos/test.mp4")
        self.video_path = str(self.project_root / video_source)
        
        l2_cfg = self.config.get("L2", {})
        trk_cfg = self.config.get("tracking", {})
        
        self.model_path = l2_cfg.get("model_path", "yolov8n.pt")
        self.conf_thresh = trk_cfg.get("tracking_confidence", 0.5)
        self.tracker_type = trk_cfg.get("tracker_type", "bytetrack.yaml")
        self.max_lost_frames = trk_cfg.get("maximum_lost_frames", 30)
        self.frame_sampling = trk_cfg.get("frame_sampling", 1)
        self.classes = l2_cfg.get("classes_to_detect", None)
        
        self.output_dir = self.project_root / "outputs" / "detections"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracks_json = self.output_dir / "tracks.json"
        
        self.history = TrackHistory(max_lost_frames=self.max_lost_frames)
        
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except ImportError:
            logger.error("Ultralytics YOLO is not installed. Run `pip install ultralytics`")
        except Exception as e:
            logger.error(f"Failed to load YOLO model for tracking: {e}")
            
        # Initialize L3 modules
        susp_cfg = self.config.get("suspicious_activity", {})
        self.behaviour = BehaviourAnalyzer(susp_cfg)
        self.rules = SuspiciousActivityRules(susp_cfg)
        self.event_engine = EventEngine(self.project_root / "outputs" / "alerts")
        
        # Initialize VLM modules
        vlm_cfg = self.config.get("VLM", {})
        if vlm_cfg.get("enabled", True):
            self.vlm_wrapper = QwenVLWrapper(
                model_id=vlm_cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct"),
                device=vlm_cfg.get("device", "auto")
            )
            self.vlm_analyzer = VLMAnalyzer(vlm_cfg, self.vlm_wrapper)
        else:
            self.vlm_wrapper = None
            self.vlm_analyzer = None
            
        self.frame_cache = {}

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def save_tracks(self):
        with open(self.tracks_json, "w", encoding="utf-8") as f:
            json.dump(self.history.get_all_tracks(), f, indent=4)
        logger.info(f"Saved tracks to {self.tracks_json}")

    def run(self):
        logger.info(f"Starting L3 Object Tracking on video: {self.video_path}")
        
        if self.model is None:
            logger.error("YOLO model not loaded. Aborting tracking.")
            return
            
        reader = VideoReader(self.video_path)
        if not reader.is_opened():
            logger.error("Aborting tracking due to video read failure.")
            return

        fps = reader.fps if reader.fps > 0 else 30
        
        try:
            import cv2
            video_out_path = self.output_dir / "tracking_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (reader.width, reader.height))
            logger.info(f"Initialized annotated video writer at {video_out_path}")
        except ImportError:
            video_writer = None
            logger.warning("cv2 not available, skipping annotated video generation.")

        frame_number = 0
        
        while True:
            ret, frame = reader.read_frame()
            if not ret:
                break
                
            frame_number += 1
            timestamp = round(frame_number / fps, 3)
            
            # Cache 1 frame per second for VLM analysis
            if frame_number % int(fps) == 0:
                try:
                    import cv2
                    from PIL import Image
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frame_cache[timestamp] = Image.fromarray(img_rgb)
                    # Cleanup old frames (> 5 mins)
                    old_keys = [k for k in self.frame_cache.keys() if k < timestamp - 300]
                    for k in old_keys:
                        del self.frame_cache[k]
                except Exception as e:
                    logger.warning(f"Failed to cache frame for VLM: {e}")
            
            if self.frame_sampling > 1 and frame_number % self.frame_sampling != 0:
                continue
                
            try:
                results = self.model.track(
                    source=frame,
                    conf=self.conf_thresh,
                    tracker=self.tracker_type,
                    classes=self.classes,
                    persist=True,
                    verbose=False
                )
            except Exception as e:
                logger.warning(f"Tracking failed on frame {frame_number}: {e}")
                continue
                
            frame_detections = []
            
            annotated_frame = frame.copy() if video_writer else None
            
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                if boxes is not None and boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                    
                    for i, box in enumerate(boxes):
                        if i >= len(track_ids):
                            break
                        
                        track_id = track_ids[i]
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = result.names[cls_id] if result.names else str(cls_id)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        det = {
                            "track_id": track_id,
                            "class_id": cls_id,
                            "class_name": class_name,
                            "confidence": round(conf, 4),
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2)
                        }
                        frame_detections.append(det)
                        
                        if video_writer:
                            import cv2
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                            label = f"ID:{track_id} {class_name}"
                            cv2.putText(annotated_frame, label, (int(x1), max(int(y1)-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            self.history.update(frame_number, timestamp, frame_detections)
            
            # Run L3 analysis
            active_tracks = self.history.get_active_tracks()
            for tid, track in active_tracks.items():
                # Enhance track with movement logic
                enhanced_track = self.behaviour.analyze(track)
                
                # Check suspicious rules
                events = self.rules.evaluate(enhanced_track)
                if events:
                    # Capture the number of pre-existing events to detect if a new one was added
                    num_events_before = len(self.event_engine.all_events)
                    self.event_engine.process_rule_matches(enhanced_track, events, timestamp)
                    
                    # If new event was added and VLM is enabled
                    if len(self.event_engine.all_events) > num_events_before and self.vlm_analyzer:
                        new_event = self.event_engine.all_events[-1]
                        
                        # Collect selected frames
                        duration = new_event.duration
                        start_time = max(0, timestamp - duration)
                        mid_time = timestamp - (duration / 2)
                        end_time = timestamp
                        
                        target_times = [start_time, mid_time, end_time]
                        selected_frames = []
                        
                        if self.frame_cache:
                            for tt in target_times:
                                # Find closest timestamp in cache
                                closest_t = min(self.frame_cache.keys(), key=lambda k: abs(k - tt))
                                selected_frames.append(self.frame_cache[closest_t])
                                
                        if selected_frames:
                            self.vlm_analyzer.analyze_event(vars(new_event), selected_frames)
            
            if video_writer:
                video_writer.write(annotated_frame)
            
            if frame_number % 50 == 0:
                active = len(self.history.get_active_tracks())
                logger.info(f"Tracking Frame {frame_number}. Active tracks: {active}")
                
        reader.release()
        if video_writer:
            video_writer.release()
            
        self.save_tracks()
        self.event_engine.save_events()
        logger.info(f"L3 Tracking complete. Total unique tracks identified: {len(self.history.get_all_tracks())}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(module)s: %(message)s")
    config_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    tracker = ObjectTracker(config_path)
    tracker.run()
