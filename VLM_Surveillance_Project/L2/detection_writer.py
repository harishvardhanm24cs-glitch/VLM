import json
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DetectionWriter:
    def __init__(self, output_dir, video_path=None, fps=30, width=640, height=480, create_video=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.json_path = self.output_dir / "detections.json"
        self.csv_path = self.output_dir / "detections.csv"
        
        self.detections = []
        
        self.create_video = create_video
        self.video_writer = None
        
        if self.create_video:
            try:
                import cv2
                video_out_path = self.output_dir / "annotated.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))
                logger.info(f"Initialized annotated video writer at {video_out_path}")
            except ImportError:
                logger.warning("cv2 not available, skipping annotated video generation.")
                self.video_writer = None

    def add_detections(self, frame_number, timestamp, detections):
        for det in detections:
            # We copy to avoid mutating the original dict if it's reused
            det_copy = det.copy()
            det_copy["frame_number"] = frame_number
            det_copy["timestamp_sec"] = timestamp
            self.detections.append(det_copy)

    def write_annotated_frame(self, frame, detections):
        if self.video_writer is not None:
            try:
                import cv2
                annotated = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = map(int, [det["x1"], det["y1"], det["x2"], det["y2"]])
                    label = f"{det['class_name']} {det['confidence']:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                self.video_writer.write(annotated)
            except Exception as e:
                logger.error(f"Failed to write annotated frame: {e}")

    def close(self):
        if self.video_writer:
            self.video_writer.release()
            
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.detections, f, indent=4)
            
        if self.detections:
            keys = ["frame_number", "timestamp_sec", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.detections)
                
        logger.info(f"Saved {len(self.detections)} total detections to JSON and CSV.")
