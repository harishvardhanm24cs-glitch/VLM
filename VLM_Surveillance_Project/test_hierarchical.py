import os
import cv2
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add L2 to sys.path to easily import the classifier
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from L2.vehicle_classifier import init_classifier, get_classifier

def generate_dummy_crop(color=(0, 255, 0)):
    # Generate a random colored crop to test pipeline logic without relying on specific images
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = color
    return img

def test_pipeline():
    logger.info("Initializing hierarchical classifier...")
    init_classifier(
        model_dir="d:/VLM/models/vehicle_classifier", 
        subtype_dir="d:/VLM/models/military_subtype/v_FULL_20260914_210557",
        min_confidence=0.50, 
        interval_frames=1, 
        min_obs=1, 
        stability_ratio=0.0
    )
    
    classifier = get_classifier()
    
    import cv2
    import os
    import random
    
    test_dir = r"d:\VLM\gadiya\Military\test"
    # Find a random image
    classes = os.listdir(test_dir)
    random_class = random.choice(classes)
    class_dir = os.path.join(test_dir, random_class)
    images = os.listdir(class_dir)
    random_image = random.choice(images)
    image_path = os.path.join(class_dir, random_image)
    
    logger.info(f"Running test on real image: {image_path} (True class: {random_class})")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Failed to load image.")
        return
        
    res = classifier.process_track("test_id_1", img)
    
    logger.info("--- Test Results ---")
    logger.info(f"Class: {res.get('class')}")
    logger.info(f"Confidence: {res.get('confidence')}")
    logger.info(f"Subtype: {res.get('subtype')}")
    logger.info(f"Subtype Conf: {res.get('subtype_confidence')}")
    
    if res.get('class') in ["Military", "Normal"]:
        logger.info("✅ Top-level classification pipeline works.")
    else:
        logger.error("❌ Pipeline failed to produce a valid top-level class.")
        
if __name__ == '__main__':
    test_pipeline()
