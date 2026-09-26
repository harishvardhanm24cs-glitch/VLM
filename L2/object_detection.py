from ultralytics import YOLO


def run_object_detection(video_path):
    model = YOLO("yolo11n.pt")

    results = model.track(source=video_path, tracker="bytetrack.yaml", stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            int(box.cls[0])
            float(box.conf[0])


if __name__ == "__main__":
    run_object_detection("../videos/input.mp4")
