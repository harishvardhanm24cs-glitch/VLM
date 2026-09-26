import base64
import requests
import time

snapshot_path = (
    r"D:\VLM\CCTV\prototype\snapshots\00580afe-fe11-4d31-a8ce-7c66b30e00c2.jpg"
)
with open(snapshot_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "camera_id": "TEST-CAM-01",
    "event_type": "Person Detected",
    "confidence": 0.95,
    "snapshot_b64": b64,
}

resp = requests.post("http://localhost:8000/cctv/api/events", json=payload)

time.sleep(5)
