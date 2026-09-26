import cv2
import json
from ultralytics import YOLO


def main():
    model = YOLO("yolov8s.pt")
    print(f"YOLO CLASSES: {json.dumps(model.names)}")

    for cls_name in ["person", "car", "motorcycle", "bus", "truck"]:
        cls_id = None
        for k, v in model.names.items():
            if v == cls_name:
                cls_id = k
                break

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        return

    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.get(cv2.CAP_PROP_FPS)

    for _ in range(15):
        ret, frame = cap.read()

    ret, frame = cap.read()
    if not ret:
        return

    cv2.imwrite("debug_vehicle_frame.jpg", frame)
    cap.release()

    print("STEP 3 & 4: TEST CONFIDENCE THRESHOLDS (imgsz=640)")
    for conf in [0.10, 0.20, 0.25, 0.35, 0.50]:
        results = model.predict(frame, conf=conf, verbose=False, imgsz=640)
        boxes = results[0].boxes
        if boxes is None:
            count = 0
        else:
            count = len(boxes)
        if conf == 0.10 or conf == 0.50:
            if count > 0:
                for cls_t, conf_t, xyxy_t in zip(boxes.cls, boxes.conf, boxes.xyxy):
                    cls_id = int(cls_t.item())
                    cls_name = model.names.get(cls_id, "unknown")
                    float(conf_t.item())
                    x1, y1, x2, y2 = map(int, xyxy_t.tolist())

    print("STEP 5: TEST IMAGE SIZES (conf=0.25)")
    for imgsz in [640, 960, 1280]:
        results = model.predict(frame, conf=0.25, verbose=False, imgsz=imgsz)
        boxes = results[0].boxes
        count = len(boxes) if boxes is not None else 0
        if count > 0:
            for cls_t, conf_t, xyxy_t in zip(boxes.cls, boxes.conf, boxes.xyxy):
                cls_id = int(cls_t.item())
                if cls_id in (0, 2, 3, 5, 7):
                    cls_name = model.names.get(cls_id, "unknown")
                    print(f"    - {cls_name} ({conf_t.item():.2f})")

    results = model.predict(frame, conf=0.50, verbose=False)
    boxes = results[0].boxes
    len(boxes) if boxes is not None else 0

    results_track = model.track(
        frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.50
    )
    t_boxes = results_track[0].boxes
    if t_boxes is not None and t_boxes.id is not None:
        len(t_boxes.id)


if __name__ == "__main__":
    main()
