import time
import requests
import subprocess
import os

PORT = 8123


proc = subprocess.Popen(
    [
        "d:\\VLM\\VLM_Surveillance_Project\\.venv\\Scripts\\python.exe",
        "-m",
        "uvicorn",
        "VLM_Surveillance_Project.backend.unified_server:app",
        "--port",
        str(PORT),
    ],
    cwd="d:\\VLM",
    env=dict(os.environ, BACKEND_PORT=str(PORT)),
)

time.sleep(15)


def send_tracks(tracks_dict):
    url = f"http://127.0.0.1:{PORT}/api/internal/edge_sync"
    payload = {"frame_number": 1, "tracks": tracks_dict}
    requests.post(url, json=payload)
    time.sleep(1)


try:
    send_tracks({"1": {"object_type": "person", "class": "person"}})

    send_tracks({})
    send_tracks({"2": {"object_type": "car", "vehicle_class": "NORMAL"}})

    send_tracks({})
    send_tracks({"3": {"object_type": "truck", "vehicle_class": "UNCERTAIN"}})

    send_tracks({})
    send_tracks({"4": {"object_type": "car", "vehicle_class": "MILITARY"}})

    send_tracks({})

    send_tracks(
        {
            "1": {"object_type": "person"},
            "4": {"object_type": "car", "vehicle_class": "MILITARY"},
        }
    )

    send_tracks({})
    send_tracks(
        {
            "2": {"object_type": "car", "vehicle_class": "NORMAL"},
            "4": {"object_type": "car", "vehicle_class": "MILITARY"},
        }
    )

finally:
    proc.terminate()
    proc.wait()
