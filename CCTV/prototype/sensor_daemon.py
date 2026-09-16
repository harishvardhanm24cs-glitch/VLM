import serial
import time
import json

def main():
    # Instructing user to change 'COM3' to the ESP32's actual port
    port = 'COM7'
    baudrate = 115200
    
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"Listening on {port} at {baudrate} baud...")
    except Exception as e:
        print(f"Error opening serial port {port}: {e}")
        return

    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                if "PIR: HUMAN MOVEMENT DETECTED" in line or "IR: OBJECT / VEHICLE DETECTED" in line:
                    payload = {
                        "trigger_time": time.time(),
                        "sensor_event": line
                    }
                    with open("hardware_state.json", "w") as f:
                        json.dump(payload, f)
                    print(f"Hardware state updated: {payload}")
        except Exception as e:
            print(f"Serial read error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
