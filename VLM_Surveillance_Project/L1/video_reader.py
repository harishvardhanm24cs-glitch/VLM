import cv2
import logging

logger = logging.getLogger(__name__)

class VideoReader:
    def __init__(self, video_path):
        self.video_path = video_path
        
        # Support for Live WebCam
        source = 0 if str(video_path) == "0" else self.video_path
        if source == 0:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {self.video_path}")
            self.fps = 0
            self.width = 0
            self.height = 0
            self.total_frames = 0
            return
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.fps <= 0:
            logger.warning(f"Invalid FPS ({self.fps}) detected. Defaulting to 30.")
            self.fps = 30
            
    def is_opened(self):
        return self.cap.isOpened()
        
    def read_frame(self):
        if not self.is_opened():
            return False, None
            
        ret, frame = self.cap.read()
        return ret, frame
        
    def release(self):
        if self.cap:
            self.cap.release()
