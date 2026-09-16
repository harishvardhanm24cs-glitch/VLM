import base64
import requests
import os
import time

snapshot_path = r"D:\VLM\CCTV\prototype\snapshots\00580afe-fe11-4d31-a8ce-7c66b30e00c2.jpg"
with open(snapshot_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "camera_id": "TEST-CAM-01",
    "event_type": "Person Detected",
    "confidence": 0.95,
    "snapshot_b64": b64
}

print("Sending simulated event to CCTV Edge API...")
resp = requests.post("http://localhost:8000/cctv/api/events", json=payload)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text}")

print("Waiting 5 seconds to let bridge and VLM process the event...")
time.sleep(5)
print("Done.")
