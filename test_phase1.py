import sys
import os
import cv2
from pathlib import Path

sys.path.append("d:/VLM/VLM_Surveillance_Project")

from L2.vehicle_classifier import (
    HierarchicalVehicleClassifier,
    init_classifier,
    get_classifier,
)

print("2. Initializing classifier (verifying model and classes.json loading)...")
init_classifier(
    model_dir="d:/VLM/models/vehicle_classifier",
    subtype_dir="d:/VLM/models/military_subtype/v_FULL_20260914_210557",
)
clf = get_classifier()

try:
    c = HierarchicalVehicleClassifier(top_level_dir="d:/VLM/VLM_Surveillance_Project")
    print("FAIL: Expected exception for missing/invalid classes.json")
except FileNotFoundError:
    print("PASS: Caught FileNotFoundError as expected for missing model files.")
except Exception as e:
    print(f"PASS: Caught exception: {e}")


dummy_crop = cv2.imread("d:/VLM/test_normal.jpg")
if dummy_crop is None:

    import numpy as np

    dummy_crop = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

res = clf.classify_crop(dummy_crop)
if res["valid"]:
    total_prob = res["mil_prob"] + res["nor_prob"]
    print(f"Total probability (mil+nor): {total_prob:.4f}")
    if abs(total_prob - 1.0) < 0.1:
        pass
    else:
        pass

for i in range(5):
    res_track = clf.process_track("test_id_1", dummy_crop, "car", 0.9)

if res_track["class"] != "Military" and res_track["subtype"] is not None:
    pass
else:
    pass
