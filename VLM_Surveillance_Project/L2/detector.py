import os
import sys
import logging
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from L1.video_reader import VideoReader
from L2.detection_models import YOLOModelWrapper
from L2.detection_writer import DetectionWriter

logger = logging.getLogger(__name__)

class L2Detector:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        
        self.project_root = Path(config_path).parent.parent
        video_source = self.config.get("video", {}).get("source", "videos/test.mp4")
        self.video_path = str(self.project_root / video_source)
        
        l2_cfg = self.config.get("L2", {})
        self.model_path = l2_cfg.get("model_path", "yolov8n.pt")
        self.conf_thresh = l2_cfg.get("confidence_threshold", 0.5)
        self.iou_thresh = l2_cfg.get("iou_threshold", 0.45)
        self.imgsz = l2_cfg.get("image_size", 640)
        self.sampling_rate = l2_cfg.get("frame_sampling_rate", 1)
        self.classes = l2_cfg.get("classes_to_detect", None)
        
        self.output_dir = self.project_root / "outputs" / "detections"
        
        self.model = YOLOModelWrapper(
            model_path=self.model_path,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh,
            imgsz=self.imgsz,
            classes=self.classes
        )

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run(self):
        logger.info(f"Starting L2 Object Detection on video: {self.video_path}")
        
        reader = VideoReader(self.video_path)
        if not reader.is_opened():
            logger.error("Aborting L2 analysis due to video read failure.")
            return

        fps = reader.fps if reader.fps > 0 else 30
        
        writer = DetectionWriter(
            output_dir=self.output_dir,
            fps=fps,
            width=reader.width,
            height=reader.height,
            create_video=True
        )
        
        frame_number = 0
        total_detections = 0
        class_counts = {}
        confidence_sum = 0.0
        
        while True:
            ret, frame = reader.read_frame()
            if not ret:
                break
                
            frame_number += 1
            
            if self.sampling_rate > 1 and frame_number % self.sampling_rate != 0:
                continue
                
            results = self.model.predict(frame)
            
            frame_detections = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = result.names[cls_id] if result.names else str(cls_id)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        det = {
                            "class_id": cls_id,
                            "class_name": class_name,
                            "confidence": round(conf, 4),
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2)
                        }
                        frame_detections.append(det)
                        
                        total_detections += 1
                        confidence_sum += conf
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            timestamp = round(frame_number / fps, 3)
            writer.add_detections(frame_number, timestamp, frame_detections)
            writer.write_annotated_frame(frame, frame_detections)
            
            if frame_number % 50 == 0:
                logger.info(f"L2 Processed {frame_number} frames. Found {len(frame_detections)} objects in current frame.")
                
        reader.release()
        writer.close()
        
        avg_conf = (confidence_sum / total_detections) if total_detections > 0 else 0
        logger.info("=== L2 Detection Statistics ===")
        logger.info(f"Total frames processed: {frame_number}")
        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Average confidence: {avg_conf:.4f}")
        logger.info(f"Detections by class: {class_counts}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(module)s: %(message)s")
    config_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    detector = L2Detector(config_path)
    detector.run()
