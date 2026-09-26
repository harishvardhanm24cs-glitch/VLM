import requests

url = "http://127.0.0.1:8000/api/internal/edge_sync"

payload = {
    "frame_number": 1,
    "active_tracks_count": 1,
    "person_count": 1,
    "vehicle_count": 0,
    "object_count": 1,
    "tracks": {
        "99": {
            "track_id": 99,
            "object_type": "person",
            "class": "person",
            "confidence": 0.95,
        }
    },
}

try:
    resp = requests.post(url, json=payload, timeout=2.0)
except Exception as e:
    print("Error:", e)
