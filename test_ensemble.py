import os
import sys
import cv2

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "VLM_Surveillance_Project")
)
from L2.vehicle_classifier import init_classifier, get_classifier

init_classifier(
    old_model_dir="d:/VLM/models/vehicle_classifier/v_20260916_235302",
    new_model_dir="d:/VLM/models/vehicle_classifier",
    interval_frames=1,
)
clf = get_classifier()


crop_path = "d:/VLM/debug_vehicle_crop_105.jpg"
if not os.path.exists(crop_path):
    sys.exit(1)

img = cv2.imread(crop_path)

for i in range(5):
    res = clf.process_track(999, img, yolo_class="car", yolo_conf=0.90)
