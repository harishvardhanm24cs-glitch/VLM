import logging

logger = logging.getLogger(__name__)

class YOLOModelWrapper:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.5, iou_thresh=0.45, imgsz=640, classes=None):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.classes = classes
        
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
        except ImportError:
            logger.error("Ultralytics YOLO is not installed. Run `pip install ultralytics`")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO model {model_path}: {e}")
            self.model = None

    def predict(self, frame):
        if self.model is None:
            return []
            
        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                imgsz=self.imgsz,
                classes=self.classes,
                verbose=False
            )
            return results
        except Exception as e:
            logger.warning(f"Detection failed on frame: {e}")
            return []
