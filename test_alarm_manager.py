import sys

sys.path.append(r"d:\VLM")

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


try:
    from VLM_Surveillance_Project.backend.unified_server import AlarmManager
except Exception as e:
    sys.exit(1)


def run_tests():
    manager = AlarmManager()

    current_time = 100.0

    def step_time(dt=1.0):
        nonlocal current_time
        current_time += dt
        return manager.get_desired_physical_state(current_time)

    tracks = {"t1": {"class": "person"}}
    manager.process_camera_tracks(tracks, current_time)
    assert step_time(0.1) == True, "Buzzer should be ON"

    tracks = {"t1": {"class": "person"}}
    manager.process_camera_tracks(tracks, current_time)
    assert manager.camera_alarm_end_time == 100.0 + 10.0, "Timer should NOT restart"

    tracks = {}
    manager.process_camera_tracks(tracks, current_time)
    state = step_time(11.0)
    assert state == False, "Buzzer should turn OFF after 10 seconds"

    tracks = {"t2": {"class": "car", "vehicle_class": "NORMAL"}}
    manager.process_camera_tracks(tracks, current_time)
    assert step_time(0.1) == True, "Buzzer should be ON"
    step_time(10.0)

    tracks = {"t3": {"class": "truck", "vehicle_class": "UNCERTAIN"}}
    manager.process_camera_tracks(tracks, current_time)
    assert step_time(0.1) == True, "Buzzer should be ON"
    step_time(10.0)

    tracks = {"t4": {"class": "car", "vehicle_class": "MILITARY"}}
    manager.process_camera_tracks(tracks, current_time)
    assert step_time(0.1) == False, "Buzzer should be OFF for military"

    sensors = {"pir": True, "ir": False}
    manager.process_sensor_data(sensors, current_time)
    assert step_time(0.1) == True, "Buzzer should be ON"

    manager.process_sensor_data(sensors, current_time)

    assert (
        manager.sensor_alarm_end_time < current_time + 5.0
    ), "Timer should NOT restart"

    sensors = {"pir": True, "ir": True}
    manager.process_sensor_data(sensors, current_time)
    assert step_time(0.1) == True, "Buzzer should be ON"

    step_time(10.0)
    assert step_time(0) == False, "Should be completely off"

    tracks = {"t5": {"class": "person"}}
    manager.process_camera_tracks(tracks, current_time)

    step_time(3.0)
    sensors = {"pir": False, "ir": False}
    manager.process_sensor_data(sensors, current_time)
    sensors = {"pir": True, "ir": False}
    manager.process_sensor_data(sensors, current_time)

    assert step_time(5.0) == True

    assert step_time(3.0) == False, "Buzzer should be OFF now"


if __name__ == "__main__":
    run_tests()
