import requests

try:
    resp = requests.post(
        "http://127.0.0.1:8000/api/internal/edge_sync",
        json={
            "frame_number": 999999,
            "active_tracks_count": 0,
            "person_count": 0,
            "vehicle_count": 0,
            "object_count": 0,
            "tracks": {},
        },
    )
except Exception as e:
    print("Error:", e)
