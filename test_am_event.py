import sys

sys.path.append(r"d:\VLM")
import asyncio
from VLM_Surveillance_Project.backend.unified_server import AlarmManager


async def test():
    am = AlarmManager()
    am.camera_alarm_end_time = 0.0
    am.sensor_alarm_end_time = 0.0
    am.camera_alerted_track_ids = set()
    am.evidence_dir = r"d:\VLM\outputs\evidence"

    try:
        await am.create_and_broadcast_event(
            event_type="TEST_ALERT",
            severity="CRITICAL",
            description="Test alert description",
            track_id="123",
        )
    except Exception as e:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test())
