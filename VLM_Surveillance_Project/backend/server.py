import sys
from pathlib import Path
import json
import logging
import threading
import asyncio
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from main import SurveillancePipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="Surveillance SOC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
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
    # Encode frame to JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    if ret:
        state.current_frame = buffer.tobytes()

def on_stats(stats):
    state.tracks = stats["tracks"]
    # Broadcast stats
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
        config_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
        pipeline = SurveillancePipeline(config_path)
        state.pipeline_running = True
        logger.info("Starting background surveillance pipeline...")
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
    # Start the pipeline in a background thread
    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()

@app.get("/api/health")
async def get_health():
    return {
        "status": "ONLINE",
        "pipeline_running": state.pipeline_running,
        "L1": "ONLINE",
        "L2": "ONLINE",
        "L3": "ONLINE",
        "VLM": "ONLINE",
        "Video": "ONLINE" if state.current_frame else "OFFLINE"
    }

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

def generate_video_stream():
    while True:
        if state.current_frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + state.current_frame + b'\r\n')
        # Tiny sleep to prevent aggressive looping if no frame
        import time
        time.sleep(0.03)

@app.get("/api/stream")
async def video_stream():
    return StreamingResponse(generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
