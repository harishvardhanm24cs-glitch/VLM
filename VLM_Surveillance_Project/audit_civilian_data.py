import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def audit_civilian_dataset(data_dir):
    logger.info(f"Auditing civilian dataset at: {data_dir}")
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    class_counts = {}
    
    for root, dirs, files in os.walk(data_dir):
        rel_path = os.path.relpath(root, data_dir)
        if rel_path == '.':
            continue
            
        imgs = [f for f in files if Path(f).suffix.lower() in valid_exts]
        if len(imgs) > 0:
            class_name = os.path.basename(root)
            if class_name in class_counts:
                class_counts[class_name] += len(imgs)
            else:
                class_counts[class_name] = len(imgs)
                
    logger.info("--- Data Audit Results ---")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{cls}: {count} images")
        
    logger.info("--- Imbalance Analysis ---")
    total_imgs = sum(class_counts.values())
    if total_imgs > 0:
        max_class = max(class_counts, key=class_counts.get)
        min_class = min(class_counts, key=class_counts.get)
        logger.info(f"Most represented: {max_class} ({class_counts[max_class]} images, {class_counts[max_class]/total_imgs*100:.1f}%)")
        logger.info(f"Least represented: {min_class} ({class_counts[min_class]} images, {class_counts[min_class]/total_imgs*100:.1f}%)")
        logger.info(f"Ratio Max:Min = {class_counts[max_class] / class_counts[min_class]:.1f}:1")
        
    logger.info("\nRECOMMENDATION:")
    logger.info("B) collect more images for underrepresented classes")
    logger.info("Explanation: The dataset exhibits severe class imbalance (e.g., ~6480 for bikes vs ~10 for cars/trucks). "
                "Training on 10 images per class for most civilian classes will result in severe overfitting and a model that predicts "
                "'bike' for almost any civilian vehicle. Class weighting cannot mathematically invent the missing features for 10-image classes. "
                "Data collection is mandatory before training the civilian subtype classifier.")

if __name__ == '__main__':
    data_dir = r'd:\VLM\gadiya\Normal Indian Vehicle'
    audit_civilian_dataset(data_dir)
