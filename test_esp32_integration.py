import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from iot.esp32_client import ESP32Client


def run_tests():

    client = ESP32Client(timeout=2.0)

    is_connected = client.get_connection_status()
    print(f"Result: {'PASS' if is_connected else 'FAIL (Is the ESP32 powered on?)'}")

    sensors = client.get_sensor_data()
    if "status" in sensors and sensors["status"] == "OFFLINE":
        pass
    else:
        pass
    print("\n[Test 3] Triggering alarm_on()...")
    on_success = client.alarm_on()
    time.sleep(1)

    print("\n[Test 4] Triggering alarm_off()...")
    off_success = client.alarm_off()

    print("\n[Test 5] Simulating Timeout Handling (using a non-existent IP)...")
    bad_client = ESP32Client(ip="10.255.255.255", timeout=1.0)
    start = time.time()
    bad_data = bad_client.get_sensor_data()
    elapsed = time.time() - start
    print(
        f"Time elapsed: {elapsed:.2f}s (should be ~1.0s). Safe fallback data returned: {bad_data}"
    )
    if bad_data.get("status") == "OFFLINE" and elapsed < 1.5:
        pass
    else:
        pass


if __name__ == "__main__":
    run_tests()
