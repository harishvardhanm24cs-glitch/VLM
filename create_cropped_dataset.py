import cv2
import os
import random
import sys
from pathlib import Path
sys.path.append('d:/VLM/VLM_Surveillance_Project')
from L2.detection_models import YOLOModelWrapper

def crop_dataset(dest_dir):
    dest_dir = Path(dest_dir)
    yolo = YOLOModelWrapper('d:/VLM/yolov8s.pt', conf_thresh=0.4, iou_thresh=0.45)
    
    classes = {
        "Military": [Path("d:/VLM/gadiya/Military")],
        "Normal Indian Vehicle": [
            Path("d:/VLM/gadiya/Normal Indian Vehicle"),
            Path("d:/VLM/Indian Car Recommendation System/All car images")
        ]
    }
    
    for cls_name, cls_paths in classes.items():
        print(f"Processing {cls_name}...")
        all_files = []
        for p_dir in cls_paths:
            all_files.extend([p for p in p_dir.rglob("*.*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        random.shuffle(all_files)
        
        saved = 0
        dest_cls_dir = dest_dir / cls_name
        dest_cls_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in all_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
                
            results = yolo.predict(frame)
            if results and len(results) > 0:
                for box in results[0].boxes:
                    c_id = int(box.cls[0])
                    name = results[0].names[c_id] if results[0].names else ""
                    if name in ['car', 'truck', 'bus']:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                        
                        if x2 - x1 > 20 and y2 - y1 > 20:
                            crop = frame[y1:y2, x1:x2]
                            out_name = f"{img_path.stem}_{saved}.jpg"
                            cv2.imwrite(str(dest_cls_dir / out_name), crop)
                            saved += 1
                            break # Only take one crop per image to maintain diversity

        print(f"Saved {saved} crops for {cls_name}")

if __name__ == "__main__":
    crop_dataset('d:/VLM/gadiya_cropped')
