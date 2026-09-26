import urllib.request
import urllib.error
import json
import os
import logging

logger = logging.getLogger(__name__)


class ESP32Client:
    def __init__(self, ip=None, timeout=1.5):
        self.ip = ip or os.getenv("ESP32_IP", "10.0.36.27")
        self.timeout = timeout
        self.base_url = f"http://{self.ip}"

    def get_connection_status(self):
        """Checks if the ESP32 root endpoint is reachable."""
        try:
            req = urllib.request.Request(self.base_url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        return False

    def get_sensor_data(self):
        """Fetches the latest JSON sensor data from the ESP32."""
        url = f"{self.base_url}/sensors"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    data = response.read().decode("utf-8")
                    return json.loads(data)
        except urllib.error.URLError as e:
            logger.debug(f"[ESP32] Connection error fetching sensors: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"[ESP32] Invalid JSON received from sensors: {e}")
        except Exception as e:
            logger.error(f"[ESP32] Unexpected error fetching sensors: {e}")

        return {
            "pir": False,
            "ir": False,
            "hc_sr04_cm": 0.0,
            "ul53ldk_cm": 0.0,
            "mpu6050": {
                "acc_x": 0,
                "acc_y": 0,
                "acc_z": 0,
                "gyro_x": 0,
                "gyro_y": 0,
                "gyro_z": 0,
            },
            "alarm": False,
            "status": "OFFLINE",
        }

    def alarm_on(self):
        """Turns the physical FC-07 buzzer ON (active LOW)."""
        url = f"{self.base_url}/alarm/on"
        logger.info(f"[ESP32] GET {url}")
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                logger.info(f"[ESP32] Response: {response.status}")
                if response.status == 200:
                    logger.info("[ESP32] Alarm ON")
                    return True
        except urllib.error.URLError as e:
            logger.error(f"[ESP32] Connection error triggering alarm_on: {e}")
        except Exception as e:
            logger.error(f"[ESP32] Unexpected error triggering alarm_on: {e}")
        return False

    def alarm_off(self):
        """Turns the physical FC-07 buzzer OFF (active HIGH)."""
        url = f"{self.base_url}/alarm/off"
        logger.info(f"[ESP32] GET {url}")
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                logger.info(f"[ESP32] Response: {response.status}")
                if response.status == 200:
                    logger.info("[ESP32] Alarm OFF")
                    return True
        except urllib.error.URLError as e:
            logger.error(f"[ESP32] Connection error triggering alarm_off: {e}")
        except Exception as e:
            logger.error(f"[ESP32] Unexpected error triggering alarm_off: {e}")
        return False
