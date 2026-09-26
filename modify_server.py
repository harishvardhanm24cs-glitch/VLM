import re

filepath = r"d:\VLM\VLM_Surveillance_Project\backend\unified_server.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()


alarm_manager_code = """
class AlarmManager:
    def __init__(self):
        self.camera_alarm_end_time = 0.0
        self.sensor_alarm_end_time = 0.0
        
        self.camera_alerted_track_ids = set()
        self.prev_sensor_state = {
            "pir": False,
            "ir": False,
            "hc_sr04_cm": 0.0,
            "ul53ldk_cm": 0.0,
            "mpu6050": {}
        }
        
        self.physical_buzzer_active = False
        self.CAMERA_DURATION = 10.0
        self.SENSOR_DURATION = 5.0
        
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
                    logger.info(f"[ALARM][CAMERA] {trigger_reason} -> BUZZER ON for 10 seconds")
                    
        for track_id, track in tracks.items():
            obj_type = track.get("object_type", track.get("class", "")).lower()
            veh_class = track.get("vehicle_class", "UNCERTAIN").upper()
            if obj_type in ["car", "truck", "bus", "motorcycle", "vehicle"] and veh_class == "MILITARY":
                if track_id not in self.camera_alerted_track_ids:
                    self.camera_alerted_track_ids.add(track_id)
                    logger.info("[ALARM][CAMERA] MILITARY VEHICLE -> BUZZER OFF")
        
        stale_ids = [tid for tid in self.camera_alerted_track_ids if tid not in current_active_tids]
        for tid in stale_ids:
            self.camera_alerted_track_ids.remove(tid)
            
        if new_alarm_triggered:
            self.camera_alarm_end_time = timestamp + self.CAMERA_DURATION

    def process_sensor_data(self, data: dict, timestamp: float):
        current_pir = data.get("pir", False)
        current_ir = data.get("ir", False)
        
        new_alarm_triggered = False
        
        if current_pir and not self.prev_sensor_state.get("pir", False):
            logger.info("[ALARM][SENSOR][PIR] Detection -> BUZZER ON for 5 seconds")
            new_alarm_triggered = True
            
        if current_ir and not self.prev_sensor_state.get("ir", False):
            logger.info("[ALARM][SENSOR][IR] Detection -> BUZZER ON for 5 seconds")
            new_alarm_triggered = True
            
        self.prev_sensor_state = data.copy()
        
        if new_alarm_triggered:
            self.sensor_alarm_end_time = timestamp + self.SENSOR_DURATION

    def check_expirations(self, timestamp: float):
        if self.camera_alarm_end_time > 0 and timestamp > self.camera_alarm_end_time:
            logger.info("[ALARM][CAMERA] 10-second alarm expired -> BUZZER OFF")
            self.camera_alarm_end_time = 0.0
            
        if self.sensor_alarm_end_time > 0 and timestamp > self.sensor_alarm_end_time:
            logger.info("[ALARM][SENSOR] 5-second alarm expired -> BUZZER OFF")
            self.sensor_alarm_end_time = 0.0

    def get_desired_physical_state(self, timestamp: float) -> bool:
        self.check_expirations(timestamp)
        camera_active = self.camera_alarm_end_time > 0
        sensor_active = self.sensor_alarm_end_time > 0
        return camera_active or sensor_active

# Global State for VLM
class SOCState:
"""

content = content.replace("# Global State for VLM\nclass SOCState:", alarm_manager_code)


socstate_update = """        # Phase 3: ESP32 State
        self.esp32_sensors = {}
        self.esp32_last_alert_time = 0.0
        self.esp32_alarm_active = False
        self.alarm_manager = AlarmManager()"""
content = content.replace(
    """        # Phase 3: ESP32 State
        self.esp32_sensors = {}
        self.esp32_last_alert_time = 0.0
        self.esp32_alarm_active = False""",
    socstate_update,
)


poll_sensor_old = """                data = await asyncio.to_thread(esp32_client.get_sensor_data)
                state.esp32_sensors = data
                        
            except"""
poll_sensor_new = """                data = await asyncio.to_thread(esp32_client.get_sensor_data)
                state.esp32_sensors = data
                state.alarm_manager.process_sensor_data(data, time.time())
                        
            except"""
content = content.replace(poll_sensor_old, poll_sensor_new)


controller_code = """
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

# API Endpoints
"""
content = content.replace("# API Endpoints\n", controller_code)


startup_old = """    # Start ESP32 Polling Background Task
    asyncio.create_task(poll_esp32_sensors())"""
startup_new = """    # Start ESP32 Polling Background Task
    asyncio.create_task(poll_esp32_sensors())
    asyncio.create_task(alarm_controller_loop())"""
content = content.replace(startup_old, startup_new)


import re

content = re.sub(
    r"def update_alarm_state\(\).*?elif not alarm_required and state\.esp32_alarm_active:.*?task\.add_done_callback\(bg_tasks\.discard\)\n",
    "",
    content,
    flags=re.DOTALL,
)


receive_old = """    # Update state
    state.tracks = stats["tracks"]
    update_alarm_state()
    
    import time"""
receive_new = """    # Update state
    state.tracks = stats["tracks"]
    
    import time
    state.alarm_manager.process_camera_tracks(state.tracks, time.time())
    """
content = content.replace(receive_old, receive_new)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)
