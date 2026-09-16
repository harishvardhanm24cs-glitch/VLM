import os
import json
import collections
from pathlib import Path

def analyze_dataset(root_dir):
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    results = {}
    
    for root, dirs, files in os.walk(root_dir):
        # We want to store info for each directory
        rel_path = os.path.relpath(root, root_dir)
        if rel_path == '.':
            rel_path = root_dir

        img_files = [f for f in files if Path(f).suffix.lower() in valid_exts]
        other_files = [f for f in files if Path(f).suffix.lower() not in valid_exts]
        exts = list(set(Path(f).suffix.lower() for f in img_files))
        
        is_class = False
        # It represents a class if it contains images. Let's mark it as a class if it has images and no subdirectories (or we can just list everything).
        
        results[rel_path] = {
            'num_images': len(img_files),
            'num_other': len(other_files),
            'extensions': exts,
            'subdirs': dirs
        }

    with open('dataset_audit.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    analyze_dataset(r'd:\VLM\gadiya')
