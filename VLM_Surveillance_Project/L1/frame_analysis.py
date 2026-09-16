import os
import sys
import logging
import json
import csv
from pathlib import Path
import cv2
import yaml

# Ensure imports work whether run from main.py or directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from L1.video_reader import VideoReader
from L1.change_detector import ChangeDetector

logger = logging.getLogger(__name__)

class FrameAnalyzer:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        
        # Parse L1 configuration
        l1_config = self.config.get("L1", {})
        self.threshold = l1_config.get("pixel_difference_threshold", 25)
        self.frame_skip = l1_config.get("frame_skip", 1)
        self.min_change_pct = l1_config.get("minimum_change_percentage", 0.1)
        
        # Ensure paths are absolute relative to the project root
        self.project_root = Path(config_path).parent.parent
        video_source = self.config.get("video", {}).get("source", "videos/test.mp4")
        self.video_path = str(self.project_root / video_source)
        
        self.output_dir = self.project_root / "outputs"
        self.output_dir.mkdir(exist_ok=True)
        self.json_output = self.output_dir / "l1_results.json"
        self.csv_output = self.output_dir / "l1_results.csv"
        
        self.results = []
        
    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
            
    def save_results(self):
        # Save JSON
        with open(self.json_output, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
            
        # Save CSV
        if self.results:
            keys = self.results[0].keys()
            with open(self.csv_output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)
                
        logger.info(f"Results saved to {self.json_output} and {self.csv_output}")
        
    def run(self):
        logger.info(f"Starting L1 Frame Analysis on video: {self.video_path}")
        
        reader = VideoReader(self.video_path)
        if not reader.is_opened():
            logger.error("Aborting analysis due to video read failure. Is the path correct?")
            return
            
        logger.info(f"Video specs - FPS: {reader.fps:.2f}, "
                    f"Resolution: {reader.width}x{reader.height}, "
                    f"Total Frames: {reader.total_frames}")
                    
        detector = ChangeDetector(self.threshold, self.min_change_pct)
        
        prev_frame_gray = None
        frame_number = 0
        
        while True:
            ret, frame = reader.read_frame()
            if not ret:
                logger.info("End of video or cannot read further frames.")
                break
                
            frame_number += 1
            
            if self.frame_skip > 1 and frame_number % self.frame_skip != 0:
                continue
                
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.warning(f"Failed to convert frame {frame_number} to grayscale: {e}")
                continue
                
            # Compare N with N-1
            if prev_frame_gray is not None:
                changed_pixels, total_pixels, change_pct = detector.process_frame_pair(prev_frame_gray, gray)
                
                timestamp = frame_number / reader.fps if reader.fps > 0 else 0
                
                result_entry = {
                    "frame_number": frame_number,
                    "timestamp_sec": round(timestamp, 3),
                    "changed_pixels": int(changed_pixels),
                    "total_pixels": int(total_pixels),
                    "change_percentage": round(change_pct, 4)
                }
                
                if change_pct >= self.min_change_pct:
                    self.results.append(result_entry)
                    if frame_number % 100 == 0:
                        logger.info(f"Frame {frame_number} ({timestamp:.2f}s) - Change: {change_pct:.2f}%")
                    
            prev_frame_gray = gray
            
        reader.release()
        self.save_results()
        logger.info(f"L1 Analysis complete. Processed {frame_number} frames. Found {len(self.results)} events.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(module)s: %(message)s")
    config_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    analyzer = FrameAnalyzer(config_path)
    analyzer.run()
