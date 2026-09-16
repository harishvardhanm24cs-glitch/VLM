from ultralytics import YOLO

def run_object_detection(video_path):
    model = YOLO("yolo11n.pt")
    
    # Run tracking instead of just detection to give IDs to objects
    results = model.track(source=video_path, tracker="bytetrack.yaml", stream=True)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            print("Class:", cls, "Confidence:", confidence)

if __name__ == "__main__":
    run_object_detection("../videos/input.mp4")
