import sys
import time
import json
import cv2
from pathlib import Path

sys.path.append("d:/VLM/VLM_Surveillance_Project")

from L2.vehicle_classifier import HierarchicalVehicleClassifier
from L2.detection_models import YOLOModelWrapper

clf = HierarchicalVehicleClassifier(
    top_level_dir="d:/VLM/models/vehicle_classifier",
    subtype_dir="d:/VLM/models/military_subtype/v_FULL_20260914_210557",
)

print("Preprocessing mode: Aspect-ratio preserving padding (letterbox_image)")

yolo = YOLOModelWrapper("d:/VLM/yolov8s.pt", conf_thresh=0.4, iou_thresh=0.45)

hard_neg_dir = Path("d:/VLM/datasets/hard_negatives/normal_misclassified")
hard_neg_dir.mkdir(parents=True, exist_ok=True)

stats = {
    "normal": {"tested": 0, "correct": 0, "false_military": 0, "uncertain": 0},
    "military": {"tested": 0, "correct": 0, "false_normal": 0, "uncertain": 0},
}


def simulate_pipeline(img_paths, ground_truth, max_tests=50):
    global stats
    tested_count = 0
    track_id_counter = 1

    for img_path in img_paths:
        if tested_count >= max_tests:
            break

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        results = yolo.predict(frame)
        if not results or len(results) == 0:
            continue

        for box in results[0].boxes:
            c_id = int(box.cls[0])
            name = results[0].names[c_id] if results[0].names else ""
            if name in ["car", "truck", "bus"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                if x2 - x1 > 20 and y2 - y1 > 20:
                    crop = frame[y1:y2, x1:x2]
                    track_id = f"T_{ground_truth}_{track_id_counter}"
                    track_id_counter += 1

                    final_res = None
                    for i in range(25):
                        final_res = clf.process_track(
                            track_id,
                            crop,
                            yolo_class=name,
                            yolo_conf=box.conf[0].item(),
                        )

                    predicted = final_res["class"]
                    stats[ground_truth]["tested"] += 1

                    if predicted == "Military":
                        if ground_truth == "military":
                            stats["military"]["correct"] += 1
                        else:
                            stats["normal"]["false_military"] += 1

                            ts = int(time.time())
                            out_img_name = f"normal_fp_{ts}_{track_id}_{name}.jpg"
                            out_json_name = f"normal_fp_{ts}_{track_id}_{name}.json"
                            cv2.imwrite(str(hard_neg_dir / out_img_name), crop)

                            history = clf.track_history.get(track_id, {})
                            last_pred = history.get("predictions", [{}])[-1]

                            metadata = {
                                "timestamp": ts,
                                "track_id": track_id,
                                "yolo_class": name,
                                "yolo_confidence": box.conf[0].item(),
                                "crop_dimensions": [x2 - x1, y2 - y1],
                                "mil_prob": last_pred.get("mil_prob", 0.0),
                                "nor_prob": last_pred.get("nor_prob", 0.0),
                                "margin": last_pred.get("margin", 0.0),
                                "temporal_decision": predicted,
                                "final_decision": predicted,
                            }
                            with open(hard_neg_dir / out_json_name, "w") as f:
                                json.dump(metadata, f, indent=4)
                    elif predicted == "Normal":
                        if ground_truth == "normal":
                            stats["normal"]["correct"] += 1
                        else:
                            stats["military"]["false_normal"] += 1
                    else:
                        stats[ground_truth]["uncertain"] += 1

                    tested_count += 1
                    break


normal_paths = [
    p
    for p in Path("d:/VLM/gadiya/Normal Indian Vehicle").rglob("*.*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]
simulate_pipeline(normal_paths, "normal", max_tests=200)

military_paths = [
    p
    for p in Path("d:/VLM/gadiya/Military").rglob("*.*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]
simulate_pipeline(military_paths, "military", max_tests=200)

print(json.dumps(stats, indent=4))
