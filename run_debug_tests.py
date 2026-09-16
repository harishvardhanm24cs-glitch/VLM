import sys
import os
import cv2
from PIL import Image

sys.path.append(r"d:\VLM\VLM_Surveillance_Project")
from L2.vehicle_classifier import init_classifier, get_classifier

def run_tests():
    print("Initializing classifier...")
    init_classifier(
        model_dir="d:/VLM/models/vehicle_classifier", 
        subtype_dir="d:/VLM/models/military_subtype/v_FULL_20260914_210557",
        min_confidence=0.50, 
        interval_frames=1, 
        min_obs=1, 
        stability_ratio=0.0
    )
    classifier = get_classifier()
    
    img_path = r"d:\VLM\gadiya\Military\test\tanks\tanks_0_9102.jpeg"
    img = cv2.imread(img_path)
    
    print("\n====================================================")
    print("TEST A: Original Army vehicle image")
    print("====================================================")
    res_a = classifier.classify_crop(img)
    print(f"Military confidence: {res_a.get('mil_prob', 0.0):.4f}")
    print(f"Normal confidence: {res_a.get('nor_prob', 0.0):.4f}")
    print(f"Final category: {res_a.get('top_class')}")
    
    print("\n====================================================")
    print("TEST B: Crop generated from image")
    print("====================================================")
    h, w = img.shape[:2]
    # create a mock YOLO crop
    crop = img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    res_b = classifier.classify_crop(crop)
    print(f"Military confidence: {res_b.get('mil_prob', 0.0):.4f}")
    print(f"Normal confidence: {res_b.get('nor_prob', 0.0):.4f}")
    print(f"Final category: {res_b.get('top_class')}")
    
    print("\n====================================================")
    print("TEST C: COMPLETE HIERARCHY")
    print("====================================================")
    # Using process_track to simulate temporal logic
    res_c = classifier.process_track(track_id=999, cv2_image=crop)
    # the debug prints in process_track will output the exact information requested!
    print("\n[TEST C COMPLETE]")

if __name__ == "__main__":
    run_tests()
