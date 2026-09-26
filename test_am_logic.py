import asyncio
import traceback


class MockState:
    def __init__(self):
        self.events = []
        self.current_frame = b"fakeimage"


state = MockState()


class MockManager:
    async def broadcast(self, msg):
        pass


manager = MockManager()

import os
import uuid
import time
import logging

logger = logging.getLogger("test")


class AlarmManager:
    def __init__(self):
        self.evidence_dir = r"d:\VLM\outputs\evidence"

    def _save_evidence(self) -> str:
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

    async def create_and_broadcast_event(
        self, event_type: str, severity: str, description: str, track_id: str = None
    ):
        snapshot_url = self._save_evidence()

        payload = {
            "event_id": uuid.uuid4().hex,
            "event_type": event_type,
            "severity": severity,
            "timestamp": time.time(),
            "description": description,
            "track_id": track_id,
        }

        if snapshot_url:
            payload["snapshot"] = snapshot_url

        state.events.append(payload)
        await manager.broadcast({"type": "EVENT", "data": payload})


async def main():
    am = AlarmManager()
    await am.create_and_broadcast_event("TEST", "HIGH", "TEST DESC", "123")


if __name__ == "__main__":
    asyncio.run(main())
