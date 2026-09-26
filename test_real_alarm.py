import asyncio
from VLM_Surveillance_Project.backend.unified_server import state, update_alarm_state
import VLM_Surveillance_Project.backend.unified_server as backend
from iot.esp32_client import ESP32Client

backend.esp32_client = ESP32Client()
state.esp32_alarm_active = False

state.tracks = {"1": {"object_type": "person"}}


async def test():

    update_alarm_state()

    await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(test())
