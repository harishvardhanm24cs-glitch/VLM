import os
import cv2
import time
import logging
from collections import defaultdict
import numpy as np

# Lazy load easyocr so it doesn't break if not enabled/installed immediately
try:
    import easyocr
except ImportError:
    easyocr = None

logger = logging.getLogger(__name__)

class ANPRPipeline:
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        self.enabled = config.get("enabled", os.getenv("ANPR_ENABLED", "false").lower() == "true")
        self.min_confidence = float(config.get("min_confidence", os.getenv("ANPR_MIN_CONFIDENCE", "0.8")))
        
        # Temporal Validation Storage
        # track_id -> {"frames": list_of_plates, "last_seen": timestamp, "best_confidence": float, "best_plate": str}
        self.history = defaultdict(lambda: {"frames": [], "last_seen": 0, "best_confidence": 0, "best_plate": ""})
        self.history_max_age = 10.0 # Clear history older than 10 seconds
        
        self.reader = None
        if self.enabled and easyocr is not None:
            logger.info("Initializing EasyOCR for ANPR...")
            # Use english, disable gpu by default unless configured
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        elif self.enabled and easyocr is None:
            logger.error("EasyOCR is not installed. ANPR will be disabled.")
            self.enabled = False

    def clean_history(self, current_time):
        expired = [tid for tid, data in self.history.items() if current_time - data["last_seen"] > self.history_max_age]
        for tid in expired:
            del self.history[tid]

    def _normalize_plate(self, text):
        """Basic normalization to remove spaces and non-alphanumeric chars."""
        return ''.join(e for e in text if e.isalnum()).upper()

    def process_vehicle_crop(self, track_id, vehicle_crop, current_time):
        """
        Process a vehicle crop to detect its license plate and perform OCR.
        Returns the temporally validated plate text and confidence if valid, else None, None.
        """
        if not self.enabled or self.reader is None:
            return None, None
            
        self.clean_history(current_time)
        self.history[track_id]["last_seen"] = current_time
        
        # Run EasyOCR on the vehicle crop
        # In a full production system, we would first use a License Plate Detector model (like a specific YOLO model)
        # to crop the plate specifically. For this unified pipeline, we run OCR on the vehicle crop.
        try:
            results = self.reader.readtext(vehicle_crop, detail=1)
            
            # Filter results for plate-like text
            valid_plates = []
            for (bbox, text, prob) in results:
                normalized = self.normalize_plate(text)
                # Simple heuristic: plate should be between 5 and 10 characters long
                if 5 <= len(normalized) <= 12 and prob > 0.4:
                    valid_plates.append((normalized, prob, bbox))
            
            if valid_plates:
                # Sort by confidence
                valid_plates.sort(key=lambda x: x[1], reverse=True)
                best_read, best_prob, _ = valid_plates[0]
                
                self.history[track_id]["frames"].append((best_read, best_prob))
                
                # Update best overall record for this track
                if best_prob > self.history[track_id]["best_confidence"]:
                    self.history[track_id]["best_confidence"] = best_prob
                    self.history[track_id]["best_plate"] = best_read

            # Temporal Voting
            frames = self.history[track_id]["frames"]
            if len(frames) >= 2: # Require at least 2 observations
                # Count frequencies
                plate_counts = defaultdict(float)
                for plate, prob in frames:
                    plate_counts[plate] += prob # Weighted by confidence
                
                # Get the plate with the highest weighted score
                best_plate = max(plate_counts, key=plate_counts.get)
                # Calculate average confidence for this plate
                matching_probs = [p for pl, p in frames if pl == best_plate]
                avg_prob = sum(matching_probs) / len(matching_probs) if matching_probs else 0
                
                if avg_prob >= self.min_confidence:
                    return best_plate, avg_prob
                    
        except Exception as e:
            logger.warning(f"OCR failed for track {track_id}: {e}")
            
        return None, None
        
    def normalize_plate(self, text):
        return self._normalize_plate(text)
