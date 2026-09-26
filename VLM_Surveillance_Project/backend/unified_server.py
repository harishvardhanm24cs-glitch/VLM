import sys
import os
from pathlib import Path
from dotenv import load_dotenv

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


from VLM_Surveillance_Project.main import SurveillancePipeline


from CCTV.prototype.main import (
    app as cctv_app,
    startup as cctv_startup,
    shutdown as cctv_shutdown,
)

try:
    from iot.esp32_client import ESP32Client

    esp32_client = ESP32Client()
except Exception as e:
    logger.error(f"Failed to load ESP32 module: {e}")
    esp32_client = None


from L_6.database import FaceDatabase
from L_6.enrollment import FaceEnrollmentService
from fastapi import UploadFile, File, Form
import numpy as np

l6_database = FaceDatabase()
l6_enrollment = FaceEnrollmentService(l6_database)

logger = logging.getLogger(__name__)


fh = logging.FileHandler("alarm_debug.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)


bg_tasks = set()

app = FastAPI(title="Unified Surveillance SOC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

app.mount("/cctv", cctv_app)


evidence_dir = root_path / "VLM_Surveillance_Project" / "outputs" / "alerts"
evidence_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/evidence", StaticFiles(directory=str(evidence_dir)), name="evidence")


class AlarmManager:
    def __init__(self):
        self.camera_alarm_end_time = 0.0
        self.sensor_alarm_end_time = 0.0

        self.camera_alerted_track_ids = set()
        self.evidence_dir = r"d:\VLM\outputs\evidence"

        self.prev_sensor_state = {
            "pir": False,
            "ir": False,
            "hc_sr04_cm": 0.0,
            "ul53ldk_cm": 0.0,
            "mpu6050": {},
        }

        self.physical_buzzer_active = True
        self.CAMERA_DURATION = 2.0
        self.SENSOR_DURATION = 5.0

    def _save_evidence(self) -> str:
        import uuid
        import time

        if not state.current_frame:
            return ""

        try:
            filename = f"{uuid.uuid4().hex}_{int(time.time())}.jpg"
            filepath = os.path.join(self.evidence_dir, filename)
            with open(filepath, "wb") as f:
                f.write(state.current_frame)
            return f"/api/evidence/{filename}"
        except Exception as e:
            logger.error(f"Failed to save evidence: {e}")
            return ""

    async def create_and_broadcast_event(
        self, event_type: str, severity: str, description: str, track_id: str = None
    ):
        import uuid
        import time

        snapshot_url = self._save_evidence()

        payload = {
            "event_id": uuid.uuid4().hex,
            "event_type": event_type,
            "severity": severity,
            "timestamp": time.time(),
            "description": description,
            "track_id": track_id,
        }

        if snapshot_url:
            payload["snapshot"] = snapshot_url

        state.events.append(payload)
        await manager.broadcast({"type": "EVENT", "data": payload})

    def process_camera_tracks(self, tracks: dict, timestamp: float):
        new_alarm_triggered = False
        current_active_tids = set()

        for track_id, track in tracks.items():
            current_active_tids.add(track_id)
            obj_type = track.get("object_type", track.get("class", "")).lower()
            veh_class = track.get("vehicle_class", "UNCERTAIN").upper()

            is_alarming = False
            trigger_reason = ""

            if obj_type == "person":
                is_alarming = True
                trigger_reason = "PERSON detected"
            elif obj_type in ["car", "truck", "bus", "motorcycle", "vehicle"]:
                if veh_class == "MILITARY":
                    pass
                elif veh_class == "NORMAL":
                    is_alarming = True
                    trigger_reason = "NORMAL VEHICLE detected"
                else:
                    is_alarming = True
                    trigger_reason = "UNKNOWN VEHICLE detected"

            if is_alarming:
                if track_id not in self.camera_alerted_track_ids:
                    self.camera_alerted_track_ids.add(track_id)
                    new_alarm_triggered = True
                    logger.info(
                        f"[ALARM][CAMERA] {trigger_reason} -> BUZZER ON for 2 seconds"
                    )

                    import asyncio

                    asyncio.create_task(
                        self.create_and_broadcast_event(
                            event_type=trigger_reason.replace(" detected", "")
                            .replace(" ", "_")
                            .upper()
                            + "_ALERT",
                            severity="CRITICAL",
                            description=f"{trigger_reason}. Triggered Buzzer for 2s.",
                            track_id=track_id,
                        )
                    )

        for track_id, track in tracks.items():
            obj_type = track.get("object_type", track.get("class", "")).lower()
            veh_class = track.get("vehicle_class", "UNCERTAIN").upper()
            if (
                obj_type in ["car", "truck", "bus", "motorcycle", "vehicle"]
                and veh_class == "MILITARY"
            ):
                if track_id not in self.camera_alerted_track_ids:
                    self.camera_alerted_track_ids.add(track_id)
                    logger.info("[ALARM][CAMERA] MILITARY VEHICLE -> BUZZER OFF")

        stale_ids = [
            tid
            for tid in self.camera_alerted_track_ids
            if tid not in current_active_tids
        ]
        for tid in stale_ids:
            self.camera_alerted_track_ids.remove(tid)

        if new_alarm_triggered:
            self.camera_alarm_end_time = timestamp + self.CAMERA_DURATION

    def process_sensor_data(self, data: dict, timestamp: float):
        current_pir = data.get("pir", False)
        current_ir = data.get("ir", False)

        new_alarm_triggered = False

        hc_sr04 = self.prev_sensor_state.get("hc_sr04_cm", 0.0)
        vl53l0x = self.prev_sensor_state.get("ul53ldk_cm", 0.0)
        distance_str = f"HC-SR04: {hc_sr04}cm | VL53L0X: {vl53l0x}cm"

        if current_pir and not self.prev_sensor_state.get("pir", False):
            logger.info("[ALARM][SENSOR][PIR] Detection -> BUZZER ON for 5 seconds")
            new_alarm_triggered = True
            import asyncio

            asyncio.create_task(
                self.create_and_broadcast_event(
                    event_type="PIR_SENSOR_ALERT",
                    severity="HIGH",
                    description=f"PIR Motion Detected! {distance_str}. Buzzer ON for 5s.",
                )
            )

        if current_ir and not self.prev_sensor_state.get("ir", False):
            logger.info("[ALARM][SENSOR][IR] Detection -> BUZZER ON for 5 seconds")
            new_alarm_triggered = True
            import asyncio

            asyncio.create_task(
                self.create_and_broadcast_event(
                    event_type="IR_SENSOR_ALERT",
                    severity="HIGH",
                    description=f"IR Obstacle Detected! {distance_str}. Buzzer ON for 5s.",
                )
            )

        self.prev_sensor_state = data.copy()

        if new_alarm_triggered:
            self.sensor_alarm_end_time = timestamp + self.SENSOR_DURATION

    def check_expirations(self, timestamp: float):
        if self.camera_alarm_end_time > 0 and timestamp > self.camera_alarm_end_time:
            logger.info("[ALARM][CAMERA] 2-second alarm expired -> BUZZER OFF")
            self.camera_alarm_end_time = 0.0

        if self.sensor_alarm_end_time > 0 and timestamp > self.sensor_alarm_end_time:
            logger.info("[ALARM][SENSOR] 5-second alarm expired -> BUZZER OFF")
            self.sensor_alarm_end_time = 0.0

    def get_desired_physical_state(self, timestamp: float) -> bool:
        self.check_expirations(timestamp)
        camera_active = self.camera_alarm_end_time > 0
        sensor_active = self.sensor_alarm_end_time > 0
        return camera_active or sensor_active


class SOCState:

    def __init__(self):
        self.tracks = {}
        self.events = []
        self.vlm_results = []
        self.expressions = {}
        self.face_recognitions = {}
        self.current_frame = None
        self.pipeline_running = False

        self.esp32_sensors = {}
        self.esp32_last_alert_time = 0.0
        self.esp32_alarm_active = False
        self.alarm_manager = AlarmManager()


state = SOCState()


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


async def process_l5_tracks(tracks: dict, timestamp: float):
    for tid, track in tracks.items():
        if "l5_status" in track:
            expr = track.get("l5_expression")
            conf = track.get("l5_expression_confidence", 0.0)
            status = track.get("l5_status")
            obs = track.get("l5_observation_count", 0)
            duration = track.get("time_in_frame", 0.0)

            prev = state.expressions.get(tid)
            changed = False
            if not prev:
                changed = True
            elif prev["expression"] != expr or prev["status"] != status:
                changed = True

            if changed:
                payload = {
                    "event_type": "FACE_EXPRESSION",
                    "track_id": tid,
                    "expression": expr,
                    "confidence": conf,
                    "status": status,
                    "observation_count": obs,
                    "duration": duration,
                    "timestamp": timestamp,
                }
                state.expressions[tid] = payload
                await manager.broadcast(payload)

    active_tids = set(tracks.keys())
    stale_tids = [t for t in state.expressions.keys() if t not in active_tids]
    for t in stale_tids:
        del state.expressions[t]


async def process_l6_tracks(tracks: dict, timestamp: float):
    for tid, track in tracks.items():
        if "match_status" in track:
            status = track.get("match_status", "PROCESSING")
            person_id = track.get("person_id")
            identity = track.get("identity")
            similarity = track.get("similarity", 0.0)

            prev = state.face_recognitions.get(tid)
            changed = False

            if status not in ("PROCESSING", "NO_FACE"):
                if not prev:
                    changed = True
                elif prev["match_status"] != status or prev["identity"] != identity:
                    changed = True

            if changed:
                payload = {
                    "event_type": "FACE_RECOGNITION",
                    "track_id": tid,
                    "person_id": person_id,
                    "identity": identity,
                    "match_status": status,
                    "similarity": similarity,
                    "timestamp": timestamp,
                }
                state.face_recognitions[tid] = payload
                await manager.broadcast(payload)

    active_tids = set(tracks.keys())
    stale_tids = [
        t
        for t in state.face_recognitions.keys()
        if str(t) not in active_tids and int(t) not in active_tids
    ]
    for t in stale_tids:
        del state.face_recognitions[t]


def on_frame(frame):
    ret, buffer = cv2.imencode(".jpg", frame)
    if ret:
        state.current_frame = buffer.tobytes()


def on_stats(stats):
    state.tracks = stats["tracks"]
    asyncio.run(process_l5_tracks(stats["tracks"], stats.get("timestamp", 0.0)))
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
        config_path = (
            root_path / "VLM_Surveillance_Project" / "config" / "settings.yaml"
        )
        pipeline = SurveillancePipeline(config_path)
        state.pipeline_running = True
        logger.info("Starting background VLM surveillance pipeline...")
        pipeline.run(
            on_frame=on_frame, on_stats=on_stats, on_event=on_event, on_vlm=on_vlm
        )
    except Exception as e:
        logger.error(f"Pipeline thread crashed: {e}")
    finally:
        state.pipeline_running = False


async def poll_esp32_sensors():
    import time

    while True:
        if esp32_client:
            try:
                data = await asyncio.to_thread(esp32_client.get_sensor_data)
                state.esp32_sensors = data
                state.alarm_manager.process_sensor_data(data, time.time())

            except Exception as e:
                logger.error(f"ESP32 polling error: {e}")
        await asyncio.sleep(3)


async def alarm_controller_loop():
    import time

    while True:
        if esp32_client:
            now = time.time()
            desired = state.alarm_manager.get_desired_physical_state(now)

            if desired and not state.alarm_manager.physical_buzzer_active:
                state.alarm_manager.physical_buzzer_active = True
                state.esp32_alarm_active = True
                asyncio.create_task(asyncio.to_thread(esp32_client.alarm_on))

            elif not desired and state.alarm_manager.physical_buzzer_active:
                state.alarm_manager.physical_buzzer_active = False
                state.esp32_alarm_active = False
                asyncio.create_task(asyncio.to_thread(esp32_client.alarm_off))

        await asyncio.sleep(0.5)


from fastapi.responses import FileResponse
import os


@app.get("/api/evidence/{filename}")
async def get_evidence(filename: str):
    filepath = os.path.join(r"d:\VLM\outputs\evidence", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="Evidence not found")


@app.on_event("startup")
async def startup_event():

    await cctv_startup()

    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()

    asyncio.create_task(poll_esp32_sensors())
    asyncio.create_task(alarm_controller_loop())


@app.on_event("shutdown")
async def shutdown_event():
    await cctv_shutdown()


@app.get("/api/health")
async def get_health():

    health = {
        "FastAPI": "ONLINE",
        "pipeline_running": state.pipeline_running,
        "Video": "ONLINE" if state.current_frame else "OFFLINE",
    }

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


@app.get("/api/expressions")
async def get_expressions():
    return list(state.expressions.values())


from fastapi import HTTPException


@app.get("/api/expressions/{track_id}")
async def get_expression_by_track(track_id: str):

    if track_id in state.expressions:
        return state.expressions[track_id]
    try:
        tid_int = int(track_id)
        if tid_int in state.expressions:
            return state.expressions[tid_int]
    except ValueError:
        pass
    raise HTTPException(status_code=404, detail="Track not found or no expression data")


@app.get("/api/vlm")
async def get_vlm():
    return state.vlm_results


@app.get("/api/cameras")
async def get_cameras():
    return [{"id": "cam_01", "name": "Main Entrance", "status": "ONLINE"}]


@app.get("/api/face-recognition")
async def get_face_recognition():
    """Returns the live face recognition state of active tracks."""
    return list(state.face_recognitions.values())


@app.get("/api/persons")
async def get_persons():
    """Returns enrolled persons while strictly stripping biometric embeddings."""
    enrolled = []

    for template in l6_database.get_all_templates():
        person = l6_database.get_person(template.person_id)
        if person:

            safe_person = person.model_dump()

            enrolled.append(
                {
                    "person_id": safe_person["person_id"],
                    "display_name": safe_person["display_name"],
                    "external_identifier": safe_person["external_identifier"],
                    "status": safe_person["status"],
                    "created_at": safe_person["created_at"],
                    "quality_score": template.quality_score,
                }
            )
    return enrolled


@app.post("/api/persons/enroll")
async def enroll_person(display_name: str = Form(...), file: UploadFile = File(...)):
    """Enrolls a new identity via image upload."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        person, template = l6_enrollment.enroll_person(display_name, frame)

        return {
            "status": "success",
            "person_id": person.person_id,
            "display_name": person.display_name,
            "quality_score": template.quality_score,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/iot/esp32/status")
async def get_esp32_status():
    connected = state.esp32_sensors.get("status") != "OFFLINE"
    return {"connected": connected, "last_alert_time": state.esp32_last_alert_time}


@app.get("/api/iot/esp32/sensors")
async def get_esp32_sensors():
    return state.esp32_sensors


@app.post("/api/iot/esp32/alarm/on")
async def manual_alarm_on():
    import time

    state.esp32_last_alert_time = time.time()
    if esp32_client:
        success = await asyncio.to_thread(esp32_client.alarm_on)
        return {"success": success}
    return {"success": False, "error": "Module not loaded"}


@app.post("/api/iot/esp32/alarm/off")
async def manual_alarm_off():
    state.esp32_last_alert_time = 0.0
    if esp32_client:
        success = await asyncio.to_thread(esp32_client.alarm_off)
        return {"success": success}
    return {"success": False, "error": "Module not loaded"}


@app.post("/api/internal/bridge_event")
async def receive_bridge_event(payload: dict):
    state.events.append(payload)
    await manager.broadcast({"type": "EVENT", "data": payload})

    import time

    state.alarm_manager.camera_alarm_end_time = max(
        state.alarm_manager.camera_alarm_end_time,
        time.time() + state.alarm_manager.CAMERA_DURATION,
    )

    if "vlm_analysis" in payload:
        vlm_data = {
            "event_id": payload.get("event_id"),
            "vlm_analysis": payload["vlm_analysis"],
            "model": payload.get("model", "VLM Bridge"),
            "frames_used": 1,
        }
        state.vlm_results.append(vlm_data)
        await manager.broadcast({"type": "VLM", "data": vlm_data})

    return {"status": "ok"}


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
        "tracks": payload.get("tracks", {}),
    }

    state.tracks = stats["tracks"]

    import time

    state.alarm_manager.process_camera_tracks(state.tracks, time.time())

    await process_l5_tracks(stats["tracks"], time.time())
    await process_l6_tracks(stats["tracks"], time.time())

    await manager.broadcast({"type": "STATS", "data": stats})

    return {"status": "ok"}


def generate_video_stream():
    import time

    while True:
        if state.current_frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + state.current_frame + b"\r\n"
            )
        time.sleep(0.03)


@app.get("/api/stream")
async def video_stream():
    return StreamingResponse(
        generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


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
