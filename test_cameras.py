import cv2
import sys


def test_cameras(max_tested=5):
    print(
        f"Testing camera indexes 0 to {max_tested - 1} using DirectShow (cv2.CAP_DSHOW)..."
    )

    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            pass
        else:
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                available.append(i)
            else:
                print(
                    f"Index {i}: Opened, but failed to read a frame (could be used by another app)."
                )
            cap.release()

    if available:
        pass
    else:
        pass


if __name__ == "__main__":
    test_cameras()
