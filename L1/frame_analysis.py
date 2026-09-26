import cv2
import numpy as np
import yaml

with open("../config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

VIDEO_PATH = config["video"]["source"]


def run_frame_analysis():
    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    previous_frame = None
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if previous_frame is not None:
            difference = cv2.absdiff(previous_frame, gray)
            _, threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

            changed_pixels = np.count_nonzero(threshold)
            total_pixels = gray.shape[0] * gray.shape[1]
            (changed_pixels / total_pixels) * 100

            frame_number / fps

        previous_frame = gray

    cap.release()


if __name__ == "__main__":
    run_frame_analysis()
