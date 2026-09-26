import os

evidence_dir = r"d:\VLM\outputs\evidence"
os.makedirs(evidence_dir, exist_ok=True)

filepath = r"d:\VLM\VLM_Surveillance_Project\backend\unified_server.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()


alarm_manager_old_start = """class AlarmManager:
    def __init__(self):
        self.camera_alarm_end_time = 0.0
        self.sensor_alarm_end_time = 0.0
        
        self.camera_alerted_track_ids = set()"""

alarm_manager_new_start = """class AlarmManager:
    def __init__(self):
        self.camera_alarm_end_time = 0.0
        self.sensor_alarm_end_time = 0.0
        
        self.camera_alerted_track_ids = set()
        self.evidence_dir = r"d:\\VLM\\outputs\\evidence"
        
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
            
    async def create_and_broadcast_event(self, event_type: str, severity: str, description: str, track_id: str = None):
        import uuid
        import time
        
        snapshot_url = self._save_evidence()
        
        payload = {
            "event_id": uuid.uuid4().hex,
            "event_type": event_type,
            "severity": severity,
            "timestamp": time.time(),
            "description": description,
            "track_id": track_id
        }
        
        if snapshot_url:
            payload["snapshot"] = snapshot_url
            
        state.events.append(payload)
        await manager.broadcast({"type": "EVENT", "data": payload})
"""
content = content.replace(alarm_manager_old_start, alarm_manager_new_start)


camera_old = """            if is_alarming:
                if track_id not in self.camera_alerted_track_ids:
                    self.camera_alerted_track_ids.add(track_id)
                    new_alarm_triggered = True
                    logger.info(f"[ALARM][CAMERA] {trigger_reason} -> BUZZER ON for 10 seconds")"""

camera_new = """            if is_alarming:
                if track_id not in self.camera_alerted_track_ids:
                    self.camera_alerted_track_ids.add(track_id)
                    new_alarm_triggered = True
                    logger.info(f"[ALARM][CAMERA] {trigger_reason} -> BUZZER ON for 10 seconds")
                    # Fire async event
                    import asyncio
                    asyncio.create_task(self.create_and_broadcast_event(
                        event_type=trigger_reason.replace(' detected', '').replace(' ', '_').upper() + "_ALERT",
                        severity="CRITICAL",
                        description=f"{trigger_reason}. Triggered Buzzer for 10s.",
                        track_id=track_id
                    ))"""
content = content.replace(camera_old, camera_new)


sensor_old = """        if current_pir and not self.prev_sensor_state.get("pir", False):
            logger.info("[ALARM][SENSOR][PIR] Detection -> BUZZER ON for 5 seconds")
            new_alarm_triggered = True
            
        if current_ir and not self.prev_sensor_state.get("ir", False):
            logger.info("[ALARM][SENSOR][IR] Detection -> BUZZER ON for 5 seconds")
            new_alarm_triggered = True"""

sensor_new = """        hc_sr04 = self.prev_sensor_state.get("hc_sr04_cm", 0.0)
        vl53l0x = self.prev_sensor_state.get("ul53ldk_cm", 0.0)
        distance_str = f"HC-SR04: {hc_sr04}cm | VL53L0X: {vl53l0x}cm"
        
        if current_pir and not self.prev_sensor_state.get("pir", False):
            logger.info("[ALARM][SENSOR][PIR] Detection -> BUZZER ON for 5 seconds")
            new_alarm_triggered = True
            import asyncio
            asyncio.create_task(self.create_and_broadcast_event(
                event_type="PIR_SENSOR_ALERT",
                severity="HIGH",
                description=f"PIR Motion Detected! {distance_str}. Buzzer ON for 5s."
            ))
            
        if current_ir and not self.prev_sensor_state.get("ir", False):
            logger.info("[ALARM][SENSOR][IR] Detection -> BUZZER ON for 5 seconds")
            new_alarm_triggered = True
            import asyncio
            asyncio.create_task(self.create_and_broadcast_event(
                event_type="IR_SENSOR_ALERT",
                severity="HIGH",
                description=f"IR Obstacle Detected! {distance_str}. Buzzer ON for 5s."
            ))"""
content = content.replace(sensor_old, sensor_new)


api_endpoints_start = """# API Endpoints
@app.on_event("startup")"""

api_endpoints_new = """# API Endpoints
from fastapi.responses import FileResponse
import os

@app.get("/api/evidence/{filename}")
async def get_evidence(filename: str):
    filepath = os.path.join(r"d:\\VLM\\outputs\\evidence", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="Evidence not found")

@app.on_event("startup")"""

content = content.replace(api_endpoints_start, api_endpoints_new)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)
