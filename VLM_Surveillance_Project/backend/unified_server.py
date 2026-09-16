import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(root_path / "VLM_Surveillance_Project"))
load_dotenv(root_path / ".env")

import json
import logging
import threading
import asyncio
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import VLM Pipeline
from VLM_Surveillance_Project.main import SurveillancePipeline

# Import CCTV App and DB functions
from CCTV.prototype.main import app as cctv_app, startup as cctv_startup, shutdown as cctv_shutdown

logger = logging.getLogger(__name__)

app = FastAPI(title="Unified Surveillance SOC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

# Mount the legacy CCTV prototype app under /cctv
app.mount("/cctv", cctv_app)

# Mount evidence snapshots
evidence_dir = root_path / "VLM_Surveillance_Project" / "outputs" / "alerts"
evidence_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/evidence", StaticFiles(directory=str(evidence_dir)), name="evidence")

# Global State for VLM
class SOCState:
    def __init__(self):
        self.tracks = {}
        self.events = []
        self.vlm_results = []
        self.current_frame = None
        self.pipeline_running = False

state = SOCState()

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# Pipeline Hooks
def on_frame(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    if ret:
        state.current_frame = buffer.tobytes()

def on_stats(stats):
    state.tracks = stats["tracks"]
    msg = {"type": "STATS", "data": stats}
    asyncio.run(manager.broadcast(msg))

def on_event(event):
    state.events.append(event)
    msg = {"type": "EVENT", "data": event}
    asyncio.run(manager.broadcast(msg))

def on_vlm(vlm_result):
    state.vlm_results.append(vlm_result)
    msg = {"type": "VLM", "data": vlm_result}
    asyncio.run(manager.broadcast(msg))

def run_pipeline():
    try:
        config_path = root_path / "VLM_Surveillance_Project" / "config" / "settings.yaml"
        pipeline = SurveillancePipeline(config_path)
        state.pipeline_running = True
        logger.info("Starting background VLM surveillance pipeline...")
        pipeline.run(
            on_frame=on_frame,
            on_stats=on_stats,
            on_event=on_event,
            on_vlm=on_vlm
        )
    except Exception as e:
        logger.error(f"Pipeline thread crashed: {e}")
    finally:
        state.pipeline_running = False

# API Endpoints
@app.on_event("startup")
async def startup_event():
    # Initialize the mounted CCTV PostgreSQL Database
    await cctv_startup()
    # Start the VLM pipeline in a background thread
    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()

@app.on_event("shutdown")
async def shutdown_event():
    await cctv_shutdown()

@app.get("/api/health")
async def get_health():
    # Base status
    health = {
        "FastAPI": "ONLINE",
        "pipeline_running": state.pipeline_running,
        "Video": "ONLINE" if state.current_frame else "OFFLINE"
    }
    
    # Read orchestrator status if available
    status_file = root_path / "system_status.json"
    if status_file.exists():
        try:
            with open(status_file, "r") as f:
                orchestrator_status = json.load(f)
                health.update(orchestrator_status)
        except:
            pass
            
    return health

@app.get("/api/events")
async def get_events():
    return state.events

@app.get("/api/tracks")
async def get_tracks():
    return state.tracks

@app.get("/api/vlm")
async def get_vlm():
    return state.vlm_results

@app.get("/api/cameras")
async def get_cameras():
    return [{"id": "cam_01", "name": "Main Entrance", "status": "ONLINE"}]

# --- INTERNAL BRIDGE ENDPOINT ---
# Receives enriched events from bridge.py and pushes them to the React Dashboard
@app.post("/api/internal/bridge_event")
async def receive_bridge_event(payload: dict):
    state.events.append(payload)
    await manager.broadcast({"type": "EVENT", "data": payload})
    
    if "vlm_analysis" in payload:
        vlm_data = {
            "event_id": payload.get("event_id"),
            "vlm_analysis": payload["vlm_analysis"],
            "model": payload.get("model", "VLM Bridge"),
            "frames_used": 1
        }
        state.vlm_results.append(vlm_data)
        await manager.broadcast({"type": "VLM", "data": vlm_data})
        
    return {"status": "ok"}

# --- EDGE SYNC ENDPOINT ---
# Receives real-time camera frames and tracking stats from the legacy edge_pipeline.py
@app.post("/api/internal/edge_sync")
async def receive_edge_sync(payload: dict):
    import base64
    if "frame_b64" in payload:
        try:
            state.current_frame = base64.b64decode(payload["frame_b64"])
        except:
            pass
            
    stats = {
        "frame_number": payload.get("frame_number", 0),
        "active_tracks_count": payload.get("active_tracks_count", 0),
        "person_count": payload.get("person_count", 0),
        "vehicle_count": payload.get("vehicle_count", 0),
        "object_count": payload.get("object_count", 0),
        "tracks": payload.get("tracks", {})
    }
    
    # Update state
    state.tracks = stats["tracks"]
    
    # Broadcast to dashboard
    await manager.broadcast({"type": "STATS", "data": stats})
    
    return {"status": "ok"}

def generate_video_stream():
    import time
    while True:
        if state.current_frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + state.current_frame + b'\r\n')
        time.sleep(0.03)

@app.get("/api/stream")
async def video_stream():
    return StreamingResponse(generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("BACKEND_PORT", 8000)))
