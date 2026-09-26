import asyncio
import time
from VLM_Surveillance_Project.backend.unified_server import state, AlarmManager


async def main():
    manager = AlarmManager()

    tracks = {"t999": {"class": "person"}}

    manager.process_camera_tracks(tracks, time.time())

    await asyncio.sleep(1.0)

    print("Events length:", len(state.events))
    if state.events:
        pass
    else:
        pass


if __name__ == "__main__":
    asyncio.run(main())
