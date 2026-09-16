import cv2
import numpy as np

class ChangeDetector:
    def __init__(self, threshold, min_change_percentage):
        self.threshold = threshold
        self.min_change_percentage = min_change_percentage
        
    def process_frame_pair(self, prev_frame, current_frame):
        """
        Calculates pixel difference between prev_frame and current_frame.
        Both frames should be grayscale.
        Returns: changed_pixels, total_pixels, change_percentage
        """
        # Calculate absolute pixel difference
        difference = cv2.absdiff(prev_frame, current_frame)
        
        # Apply configurable thresholding
        _, thresh_image = cv2.threshold(difference, self.threshold, 255, cv2.THRESH_BINARY)
        
        changed_pixels = np.count_nonzero(thresh_image)
        total_pixels = current_frame.shape[0] * current_frame.shape[1]
        
        if total_pixels == 0:
            return 0, 0, 0.0
            
        change_percentage = (changed_pixels / total_pixels) * 100
        
        return changed_pixels, total_pixels, change_percentage
