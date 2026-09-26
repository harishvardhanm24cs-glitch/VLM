import sys
import cv2

sys.path.append(r"d:\VLM\VLM_Surveillance_Project")
from L2.vehicle_classifier import init_classifier, get_classifier


def run_tests():
    init_classifier(
        new_model_dir="d:/VLM/models/vehicle_classifier",
        subtype_dir="d:/VLM/models/military_subtype/v_FULL_20260914_210557",
        interval_frames=1,
    )
    classifier = get_classifier()

    img_path = r"d:\VLM\gadiya\Military\test\tanks\tanks_0_9102.jpeg"
    img = cv2.imread(img_path)

    res_a = classifier.classify_crop(img)
    print(f"Military confidence: {res_a.get('mil_prob', 0.0):.4f}")
    print(f"Normal confidence: {res_a.get('nor_prob', 0.0):.4f}")
    print(f"Final category: {res_a.get('top_class')}")

    h, w = img.shape[:2]

    crop = img[int(h * 0.1) : int(h * 0.9), int(w * 0.1) : int(w * 0.9)]
    res_b = classifier.classify_crop(crop)
    print(f"Military confidence: {res_b.get('mil_prob', 0.0):.4f}")
    print(f"Normal confidence: {res_b.get('nor_prob', 0.0):.4f}")
    print(f"Final category: {res_b.get('top_class')}")

    res_c = classifier.process_track(track_id=999, cv2_image=crop)


if __name__ == "__main__":
    run_tests()
