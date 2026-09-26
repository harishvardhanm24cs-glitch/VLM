import serial
import time
import json


def main():

    port = "COM7"
    baudrate = 115200

    try:
        ser = serial.Serial(port, baudrate, timeout=1)
    except Exception as e:
        print(f"Error opening serial port {port}: {e}")
        return

    while True:
        try:
            line = ser.readline().decode("utf-8").strip()
            if line:
                if (
                    "PIR: HUMAN MOVEMENT DETECTED" in line
                    or "IR: OBJECT / VEHICLE DETECTED" in line
                ):
                    payload = {"trigger_time": time.time(), "sensor_event": line}
                    with open("hardware_state.json", "w") as f:
                        json.dump(payload, f)
        except Exception as e:
            print(f"Serial read error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
