import os
import sys
import logging
import json
import csv
import time
import threading
from pathlib import Path
import yaml
import cv2
from PIL import Image

# Setup logging before any heavy imports
base_dir = Path(__file__).resolve().parent
log_dir = base_dir / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s: %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Subsystem Imports
# ---------------------------------------------------------
try:
    from L1.video_reader import VideoReader
    from L1.change_detector import ChangeDetector
    
    from L3.track_history import TrackHistory
    from L3.behaviour import BehaviourAnalyzer
    from L3.suspicious_activity import SuspiciousActivityRules
    from L3.event_engine import EventEngine
    
    from L2.anpr import ANPRPipeline
    from L2.vehicle_db import VehicleDB
    
    from models.vlm.opus_model import OpusVLMWrapper
    from models.vlm.vlm_analyzer import VLMAnalyzer
    
    # We use Ultralytics YOLO directly for L2 + Tracking
    from ultralytics import YOLO
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------
class SurveillancePipeline:
    def __init__(self, config_path: Path):
        self.project_root = config_path.parent.parent
        self.config = self._load_config(config_path)
        self._setup_directories()
        
        # Pipeline execution variables
        self.l1_results = []
        self.frame_cache = {}
        
        self._init_subsystems()

    def _load_config(self, config_path: Path) -> dict:
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _setup_directories(self):
        self.out_dir = self.project_root / "outputs"
        self.l1_json_path = self.out_dir / "l1_results.json"
        self.l1_csv_path = self.out_dir / "l1_results.csv"
        
        self.detections_dir = self.out_dir / "detections"
        self.tracks_json_path = self.detections_dir / "tracks.json"
        
        self.alerts_dir = self.out_dir / "alerts"
        
        self.frames_dir = self.out_dir / "frames"
        
        # Create directories
        for d in [self.out_dir, self.detections_dir, self.alerts_dir, self.frames_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _init_subsystems(self):
        logger.info("Initializing Subsystems...")
        
        # --- L1 Initialization ---
        l1_cfg = self.config.get("L1", {})
        self.l1_enabled = l1_cfg.get("enabled", True)
        self.pixel_thresh = l1_cfg.get("pixel_difference_threshold", 25)
        self.min_change_pct = l1_cfg.get("minimum_change_percentage", 0.1)
        self.change_detector = ChangeDetector(self.pixel_thresh, self.min_change_pct)
        
        # --- L2 & Tracking Initialization ---
        l2_cfg = self.config.get("L2", {})
        trk_cfg = self.config.get("tracking", {})
        
        self.l2_enabled = l2_cfg.get("enabled", True)
        self.tracking_enabled = trk_cfg.get("enabled", True)
        
        model_path = l2_cfg.get("model_path", "yolov8n.pt")
        self.classes = l2_cfg.get("classes_to_detect", None)
        self.conf_thresh = trk_cfg.get("tracking_confidence", 0.5)
        self.tracker_type = trk_cfg.get("tracker_type", "bytetrack.yaml")
        max_lost_frames = trk_cfg.get("maximum_lost_frames", 30)
        self.frame_sampling = trk_cfg.get("frame_sampling", 1)
        
        try:
            self.yolo_model = YOLO(model_path)
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.yolo_model = None
            
        self.track_history = TrackHistory(max_lost_frames=max_lost_frames)
        self.anpr = ANPRPipeline(self.config)
        self.vehicle_db = VehicleDB()
        
        # --- L3 Initialization ---
        susp_cfg = self.config.get("suspicious_activity", {})
        self.l3_enabled = susp_cfg.get("enabled", True)
        self.behaviour_analyzer = BehaviourAnalyzer(susp_cfg)
        self.activity_rules = SuspiciousActivityRules(susp_cfg)
        self.event_engine = EventEngine(self.alerts_dir)
        
        # --- VLM Initialization ---
        vlm_cfg = self.config.get("VLM", {})
        self.vlm_enabled = vlm_cfg.get("enabled", True)
        
        if self.vlm_enabled:
            model_id = vlm_cfg.get("model_name", "claude-3-opus-20240229")
            api_key = vlm_cfg.get("api_key")
            max_new = vlm_cfg.get("max_new_tokens", 256)
            temp = vlm_cfg.get("temperature", 0.1)
            self.vlm_wrapper = OpusVLMWrapper(model_id=model_id, max_new_tokens=max_new, temperature=temp, api_key=api_key)
            self.vlm_analyzer = VLMAnalyzer(vlm_cfg, self.vlm_wrapper)
        else:
            self.vlm_wrapper = None
            self.vlm_analyzer = None
            
        # --- Vehicle Classifier Initialization ---
        v_class_cfg = self.config.get("vehicle_classifier", {})
        if v_class_cfg.get("enabled", True):
            try:
                from L2.vehicle_classifier import init_classifier
                model_dir = v_class_cfg.get("model_dir", "d:/VLM/models/vehicle_classifier")
                subtype_dir = v_class_cfg.get("subtype_model_dir", None)
                min_conf = v_class_cfg.get("min_confidence", 0.80)
                interval = v_class_cfg.get("interval_frames", 5)
                min_obs = v_class_cfg.get("min_observations", 3)
                stability = v_class_cfg.get("stability_ratio", 0.67)
                init_classifier(model_dir=model_dir, subtype_dir=subtype_dir, min_confidence=min_conf, interval_frames=interval, min_obs=min_obs, stability_ratio=stability)
                logger.info("Vehicle Classifier initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Vehicle Classifier: {e}")
            
        logger.info("Subsystem initialization complete.")

    def save_l1_results(self):
        with open(self.l1_json_path, "w", encoding="utf-8") as f:
            json.dump(self.l1_results, f, indent=4)
        if self.l1_results:
            keys = self.l1_results[0].keys()
            with open(self.l1_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.l1_results)

    def save_tracks(self):
        with open(self.tracks_json_path, "w", encoding="utf-8") as f:
            json.dump(self.track_history.get_all_tracks(), f, indent=4)

    def shutdown(self, video_writer=None, reader=None):
        logger.info("Shutting down pipeline and saving all results...")
        if video_writer:
            video_writer.release()
        if reader:
            reader.release()
            reader.release()
            
        self.save_l1_results()
        self.save_tracks()
        self.event_engine.save_events()
        
        logger.info("Shutdown complete. All data saved.")

    def run(self, on_frame=None, on_stats=None, on_event=None, on_vlm=None,
             video_path_override: str = None, stop_event: threading.Event = None):
        if video_path_override:
            video_path = video_path_override
        else:
            video_source = self.config.get("video", {}).get("source", "videos/input.mp4")
            video_path = str(self.project_root / video_source)
        
        logger.info(f"Opening video source: {video_path}")
        reader = VideoReader(video_path)
        if not reader.is_opened():
            logger.error("Failed to open video source.")
            return
            
        fps = reader.fps if reader.fps > 0 else 30
        
        # Annotated video writer
        video_out_path = self.detections_dir / "tracking_annotated.webm"
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (reader.width, reader.height))
        
        frame_number = 0
        prev_frame_gray = None
        
        try:
            while True:
                # Allow external stop signal
                if stop_event and stop_event.is_set():
                    logger.info("Stop event received. Terminating pipeline.")
                    break

                ret, frame = reader.read_frame()
                if not ret:
                    break
                    
                frame_number += 1
                timestamp = round(frame_number / fps, 3)
                
                # Enforce 1 FPS processing for performance and detail
                if frame_number % int(fps) != 0:
                    continue
                
                # --- VLM Frame Caching ---
                if frame_number % int(fps) == 0:
                    try:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frame_cache[timestamp] = Image.fromarray(img_rgb)
                        # Cleanup cache (> 5 mins)
                        old_keys = [k for k in self.frame_cache.keys() if k < timestamp - 300]
                        for k in old_keys:
                            del self.frame_cache[k]
                    except Exception as e:
                        logger.warning(f"Frame caching failed: {e}")
                
                # --- L1 Frame Analysis ---
                l1_change_detected = True # Default to True so L2 runs if L1 is disabled
                if self.l1_enabled:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_frame_gray is not None:
                        changed_pixels, total_pixels, change_pct = self.change_detector.process_frame_pair(prev_frame_gray, gray)
                        
                        self.l1_results.append({
                            "frame_number": frame_number,
                            "timestamp_sec": timestamp,
                            "changed_pixels": int(changed_pixels),
                            "total_pixels": int(total_pixels),
                            "change_percentage": round(change_pct, 4)
                        })
                        
                        if change_pct < self.min_change_pct:
                            l1_change_detected = False
                    prev_frame_gray = gray
                
                # If L1 detects no significant change, we can optionally skip L2
                # However, for smooth tracking, we often still need to update tracks
                # So we run L2/Tracking regardless in this strict unified pipeline, 
                # or skip it if we want aggressive optimization. Let's run it.
                
                # --- L2 & Tracking ---
                annotated_frame = frame.copy()
                frame_detections = []
                
                if self.l2_enabled and self.tracking_enabled and self.yolo_model:
                    try:
                        results = self.yolo_model.track(
                            source=frame,
                            conf=self.conf_thresh,
                            tracker=self.tracker_type,
                            classes=self.classes,
                            persist=True,
                            verbose=False,
                            imgsz=1280
                        )
                        
                        if results and len(results) > 0:
                            result = results[0]
                            boxes = result.boxes
                            
                            if boxes is not None and boxes.id is not None:
                                track_ids = boxes.id.int().cpu().tolist()
                                
                                for i, box in enumerate(boxes):
                                    if i >= len(track_ids): break
                                    
                                    track_id = track_ids[i]
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    class_name = result.names[cls_id] if result.names else str(cls_id)
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    
                                    # --- Object / Vehicle Classification Logic ---
                                    display_name = class_name
                                    v_class = None
                                    v_conf = 0.0
                                    v_sub = None
                                    v_sub_conf = 0.0
                                    
                                    if class_name in ["bicycle", "motorcycle"]:
                                        display_name = "Two-Wheeler"
                                    elif class_name in ["car", "truck", "bus"]:
                                        display_name = "Four-Wheeler"
                                        
                                        # Run Vehicle Classifier
                                        try:
                                            from L2.vehicle_classifier import get_classifier
                                            classifier = get_classifier()
                                            if classifier and classifier.top_model:
                                                cx1, cy1, cx2, cy2 = int(x1), int(y1), int(x2), int(y2)
                                                cy1, cy2 = max(0, cy1), min(frame.shape[0], cy2)
                                                cx1, cx2 = max(0, cx1), min(frame.shape[1], cx2)
                                                if cx2 > cx1 and cy2 > cy1:
                                                    crop = frame[cy1:cy2, cx1:cx2]
                                                    res = classifier.process_track(track_id, crop)
                                                    
                                                    v_class = res.get("class", "uncertain")
                                                    v_conf = res.get("confidence", 0.0)
                                                    v_sub = res.get("subtype", None)
                                                    v_sub_conf = res.get("subtype_confidence", 0.0)
                                                    
                                                    alert_status = "true" if v_class == "Normal" else "false"
                                                    
                                                    # Debug Output
                                                    print(f"\n[YOLO]\nclass={class_name}\nconfidence={conf:.2f}")
                                                    print(f"\n[TRACK]\ntrack_id={track_id}")
                                                    print(f"\n[VEHICLE CROP]\nsize={cx2-cx1}x{cy2-cy1}")
                                                    print(f"\n[TOP LEVEL CLASSIFIER]\n{v_class}\nconfidence={v_conf:.2f}")
                                                    if v_class == "Military" and v_sub:
                                                        print(f"\n[MILITARY SUBTYPE]\n{v_sub}\nconfidence={v_sub_conf:.2f}")
                                                    elif v_class == "Normal" and v_sub:
                                                        print(f"\n[CIVILIAN SUBTYPE]\n{v_sub}\nconfidence={v_sub_conf:.2f}")
                                                    print(f"\n[FINAL VEHICLE]\nYOLO={class_name}\nCATEGORY={v_class}\nTYPE={v_sub if v_sub else 'N/A'}\nALERT={alert_status}\n")

                                        except Exception as e:
                                            logger.warning(f"Vehicle classification failed: {e}")
                                            
                                    elif class_name != "person":
                                        display_name = "Object"
                                    
                                    frame_detections.append({
                                        "track_id": track_id,
                                        "class_id": cls_id,
                                        "class_name": display_name,
                                        "object_type": class_name,
                                        "confidence": round(conf, 4),
                                        "x1": round(x1, 2),
                                        "y1": round(y1, 2),
                                        "x2": round(x2, 2),
                                        "y2": round(y2, 2),
                                        "vehicle_category": v_class,
                                        "classification_confidence": round(v_conf, 4),
                                        "subtype": v_sub,
                                        "subtype_confidence": round(v_sub_conf, 4),
                                        "alert": v_class == "Normal"
                                    })
                                    
                                    # Annotate
                                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                                    label = f"ID:{track_id} {display_name}"
                                    cv2.putText(annotated_frame, label, (int(x1), max(int(y1)-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                    
                    except Exception as e:
                        logger.warning(f"Tracking failed on frame {frame_number}: {e}")

                self.track_history.update(frame_number, timestamp, frame_detections)
                if on_stats:
                    active_tracks = self.track_history.get_active_tracks()
                    person_count = sum(1 for t in active_tracks.values() if t.get("class") == "person")
                    vehicle_count = sum(1 for t in active_tracks.values() if t.get("class") in ["car", "truck", "bus", "motorcycle", "Two-Wheeler", "Four-Wheeler"])
                    on_stats({
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "active_tracks_count": len(active_tracks),
                        "person_count": person_count,
                        "vehicle_count": vehicle_count,
                        "tracks": active_tracks
                    })
                
                # --- L3 Behaviour & Events ---
                if self.l3_enabled:
                    active_tracks = self.track_history.get_active_tracks()
                    
                    # --- ANPR Processing ---
                    for tid, track in active_tracks.items():
                        obj_class = track.get("class", "").lower()
                        if obj_class in ["car", "truck", "bus", "motorcycle"]:
                            bx1, by1, bx2, by2 = track.get("current_bbox", [0,0,10,10])
                            if int(bx2) > int(bx1) and int(by2) > int(by1):
                                crop = frame[int(by1):int(by2), int(bx1):int(bx2)]
                                if crop.size > 0:
                                    best_plate, conf = self.anpr.process_vehicle_crop(tid, crop, timestamp)
                                    if best_plate:
                                        state_key = f"{tid}_ANPR"
                                        if state_key not in self.event_engine.active_events:
                                            vehicle_info = self.vehicle_db.lookup_plate(best_plate)
                                            # Save snapshot
                                            snap_filename = f"snapshot_{tid}_{timestamp}_anpr.jpg"
                                            snap_filepath = self.out_dir / "alerts" / snap_filename
                                            cv2.imwrite(str(snap_filepath), annotated_frame)
                                            snapshot_path = f"/api/evidence/{snap_filename}"
                                            
                                            self.event_engine.process_rule_matches(track, [{
                                                "event_type": "ANPR",
                                                "severity": "INFO",
                                                "duration": 0.0,
                                                "plate_text": best_plate,
                                                "description": f"Plate recognized: {best_plate} (Conf: {conf*100:.1f}%)",
                                                "vehicle_info": vehicle_info
                                            }], timestamp, snapshot_path=snapshot_path)
                    
                    # --- Behaviour Processing ---
                    for tid, track in active_tracks.items():
                        enhanced_track = self.behaviour_analyzer.analyze(track, all_tracks=active_tracks)
                        events = self.activity_rules.evaluate(enhanced_track)
                        
                        if events:
                            num_events_before = len(self.event_engine.all_events)
                            
                            snapshot_path = ""
                            snap_filename = f"snapshot_{tid}_{timestamp}.jpg"
                            snap_filepath = self.out_dir / "alerts" / snap_filename
                            cv2.imwrite(str(snap_filepath), annotated_frame)
                            snapshot_path = f"/api/evidence/{snap_filename}"
                            
                            self.event_engine.process_rule_matches(enhanced_track, events, timestamp, snapshot_path=snapshot_path)
                            
                            if len(self.event_engine.all_events) > num_events_before:
                                new_event = self.event_engine.all_events[-1]
                                if on_event:
                                    on_event(vars(new_event))
                                    
                                # --- VLM Analysis Trigger ---
                                if self.vlm_enabled and self.vlm_analyzer:
                                    duration = new_event.duration
                                    
                                    start_t = max(0, timestamp - duration)
                                    mid_t = timestamp - (duration / 2)
                                    end_t = timestamp
                                    
                                    target_times = [start_t, mid_t, end_t]
                                    selected_frames = []
                                    
                                    if self.frame_cache:
                                        for tt in target_times:
                                            closest_t = min(self.frame_cache.keys(), key=lambda k: abs(k - tt))
                                            selected_frames.append(self.frame_cache[closest_t])
                                            
                                    if selected_frames:
                                        vlm_res = self.vlm_analyzer.analyze_event(vars(new_event), selected_frames)
                                        if vlm_res and on_vlm:
                                            on_vlm(vlm_res)
                
                # Live Dashboard Stream
                live_frame_path = self.out_dir / "live_frame.jpg"
                cv2.imwrite(str(live_frame_path), annotated_frame)
                        
                video_writer.write(annotated_frame)
                if on_frame:
                    on_frame(annotated_frame)
                
                if frame_number % 100 == 0:
                    logger.info(f"Processed Frame {frame_number} ({timestamp:.2f}s). Active tracks: {len(self.track_history.get_active_tracks())}")
                    
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")
        except Exception as e:
            logger.error(f"Pipeline crashed: {e}")
        finally:
            self.shutdown(video_writer, reader)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VLM Surveillance Pipeline")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    args = parser.parse_args()

    config_path = base_dir / "config" / "settings.yaml"
    pipeline = SurveillancePipeline(config_path)
    pipeline.run(video_path_override=args.video)
