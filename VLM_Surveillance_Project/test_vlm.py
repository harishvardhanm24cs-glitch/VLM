import os
import sys
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.vlm.opus_model import OpusVLMWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def create_synthetic_frame(x, y, text=""):
    img_array = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img_array, (x, y), (x+100, y+100), (0, 0, 255), -1) 
    cv2.putText(img_array, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

def main():
    logger.info("Starting VLM standalone multi-frame test...")
    
    import yaml
    config_path = Path(__file__).resolve().parent / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    api_key = cfg.get("VLM", {}).get("api_key")
    
    wrapper = OpusVLMWrapper(model_id="claude-3-opus-20240229", max_new_tokens=256, temperature=0.1, api_key=api_key)
    
    if not wrapper.is_ready():
        logger.error("Wrapper failed to initialize. Ensure ANTHROPIC_API_KEY is set in environment or config/settings.yaml.")
        return
        
    logger.info("Creating 3 synthetic frames indicating motion...")
    frames = [
        create_synthetic_frame(100, 150, "Frame 1 (Start)"),
        create_synthetic_frame(250, 150, "Frame 2 (Middle)"),
        create_synthetic_frame(400, 150, "Frame 3 (End)")
    ]
    
    prompt = (
        "Analyze these CCTV frames for the detected event.\n"
        "Describe what is visibly happening.\n"
        "Compare the person/object's position and visible activity across the frames.\n"
        "Mention relevant visible contextual information."
    )
    logger.info(f"Sending prompt to VLM: '{prompt}'")
    
    result = wrapper.analyze(frames, prompt)
    
    logger.info("\n--- VLM RESPONSE ---")
    logger.info(result)
    logger.info("--------------------\n")

if __name__ == "__main__":
    main()
