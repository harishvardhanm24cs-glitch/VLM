import cv2
import numpy as np


def create_test_video(
    path="videos/test.mp4", width=640, height=480, fps=30, frames=150
):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    x, y = 100, 100
    dx, dy = 5, 3

    for i in range(frames):

        frame = np.zeros((height, width, 3), dtype=np.uint8)

        x += dx
        y += dy

        if x <= 0 or x + 100 >= width:
            dx = -dx
        if y <= 0 or y + 100 >= height:
            dy = -dy

        cv2.rectangle(frame, (x, y), (x + 100, y + 100), (255, 255, 255), -1)

        out.write(frame)

    out.release()


if __name__ == "__main__":
    create_test_video()
