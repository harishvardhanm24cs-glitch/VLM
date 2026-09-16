import cv2
import json
from ultralytics import YOLO

def main():
    print("==================================================")
    print("STEP 1 & 2 & 9: MODEL INFO")
    model = YOLO("yolov8s.pt")
    print(f"YOLO MODEL PATH: yolov8s.pt")
    print(f"YOLO CLASSES: {json.dumps(model.names)}")
    
    # Check if standard COCO classes exist
    for cls_name in ["person", "car", "motorcycle", "bus", "truck"]:
        cls_id = None
        for k, v in model.names.items():
            if v == cls_name:
                cls_id = k
                break
        print(f"Class '{cls_name}' is ID {cls_id}")
    
    print("==================================================")
    print("STEP 6: CAPTURE FRAME")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("FAILED TO OPEN CAMERA")
        return
        
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("STEP 10: CAMERA FRAME")
    print(f"Frame width: {w}, height: {h}, FPS: {fps}")
    
    # Warm up camera
    for _ in range(15):
        ret, frame = cap.read()
    
    ret, frame = cap.read()
    if not ret:
        print("FAILED TO READ FRAME")
        return
        
    cv2.imwrite("debug_vehicle_frame.jpg", frame)
    print("Saved debug_vehicle_frame.jpg")
    cap.release()
    
    print("==================================================")
    print("STEP 3 & 4: TEST CONFIDENCE THRESHOLDS (imgsz=640)")
    for conf in [0.10, 0.20, 0.25, 0.35, 0.50]:
        results = model.predict(frame, conf=conf, verbose=False, imgsz=640)
        boxes = results[0].boxes
        if boxes is None:
            count = 0
        else:
            count = len(boxes)
        print(f"CONF={conf:.2f} -> total detections={count}")
        if conf == 0.10 or conf == 0.50:
            if count > 0:
                for cls_t, conf_t, xyxy_t in zip(boxes.cls, boxes.conf, boxes.xyxy):
                    cls_id = int(cls_t.item())
                    cls_name = model.names.get(cls_id, "unknown")
                    conf_val = float(conf_t.item())
                    x1, y1, x2, y2 = map(int, xyxy_t.tolist())
                    print(f"  [YOLO DETECTION] class={cls_name} class_id={cls_id} confidence={conf_val:.3f} bbox={x1},{y1},{x2},{y2}")
    
    print("==================================================")
    print("STEP 5: TEST IMAGE SIZES (conf=0.25)")
    for imgsz in [640, 960, 1280]:
        results = model.predict(frame, conf=0.25, verbose=False, imgsz=imgsz)
        boxes = results[0].boxes
        count = len(boxes) if boxes is not None else 0
        print(f"IMGSZ={imgsz} -> detections={count}")
        if count > 0:
            for cls_t, conf_t, xyxy_t in zip(boxes.cls, boxes.conf, boxes.xyxy):
                cls_id = int(cls_t.item())
                if cls_id in (0, 2, 3, 5, 7):
                    cls_name = model.names.get(cls_id, "unknown")
                    print(f"    - {cls_name} ({conf_t.item():.2f})")

    print("==================================================")
    print("STEP 7: TEST WITHOUT BYTETRACK")
    results = model.predict(frame, conf=0.50, verbose=False)
    boxes = results[0].boxes
    det_count = len(boxes) if boxes is not None else 0
    print(f"YOLO detections = {det_count}")
    
    results_track = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.50)
    t_boxes = results_track[0].boxes
    trk_count = 0
    if t_boxes is not None and t_boxes.id is not None:
        trk_count = len(t_boxes.id)
    print(f"YOLO + ByteTrack tracks = {trk_count}")

if __name__ == "__main__":
    main()
