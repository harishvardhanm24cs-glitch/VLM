import cv2
import sys

sys.path.append("d:/VLM/VLM_Surveillance_Project")

from L2.detection_models import YOLOModelWrapper
from L2.vehicle_classifier import HierarchicalVehicleClassifier

yolo = YOLOModelWrapper("d:/VLM/yolov8s.pt", conf_thresh=0.5, iou_thresh=0.45)
clf = HierarchicalVehicleClassifier(
    "d:/VLM/models/vehicle_classifier",
    "d:/VLM/models/military_subtype/v_FULL_20260914_210557",
)

frame = cv2.imread("d:/VLM/debug_vehicle_frame.jpg")
results = yolo.predict(frame)

if results and len(results) > 0:
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = results[0].names[cls_id] if results[0].names else str(cls_id)
        if name in ["car", "truck", "bus"]:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1

            tx1, ty1, tx2, ty2 = map(
                int,
                [
                    max(0, x1),
                    max(0, y1),
                    min(frame.shape[1], x2),
                    min(frame.shape[0], y2),
                ],
            )
            tight_crop = frame[ty1:ty2, tx1:tx2]
            print(f"Tight {name}:", clf.classify_crop(tight_crop))

            ex1 = max(0, x1 - w * 0.3)
            ey1 = max(0, y1 - h * 0.3)
            ex2 = min(frame.shape[1], x2 + w * 0.3)
            ey2 = min(frame.shape[0], y2 + h * 0.3)

            ex1, ey1, ex2, ey2 = map(int, [ex1, ey1, ex2, ey2])
            exp_crop = frame[ey1:ey2, ex1:ex2]
            print(f"Expanded 30% {name}:", clf.classify_crop(exp_crop))
