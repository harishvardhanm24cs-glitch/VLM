"""
IBVAP - Intelligent Border Video Analytics Platform
Phase 2 (v2): Advanced Edge AI Pipeline with ByteTrack & Spatial Analytics
WITH IOT SENSOR INTEGRATION

Upgrades over v1:
  - YOLOv8s (Small) model for higher accuracy vs. YOLOv8n
  - ByteTrack object tracking (persist=True) — stable IDs across frames
  - Virtual Fence Intrusion: cv2.pointPolygonTest on each object's foot-point
  - Suspicious Loitering: per-track dwell-time > LOITERING_THRESHOLD_S seconds
  - Low-Light CLAHE Enhancement on LAB L-channel when avg brightness < 60
  - Per-track alert cooldown (replaces the old single global cooldown)
  - Stale-track pruning with configurable grace period

Hard Constraints (preserved from v1):
  - PyTorch 2.6+ safe_globals fix applied BEFORE YOLO is imported
  - Windows DirectShow camera init (prevents the 1000 FPS bug)
  - API payload is unchanged: {camera_id, event_type, confidence, bounding_box, snapshot_b64}
"""

import torch
import ultralytics.nn.tasks  # noqa: F401

torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])


import base64
import logging
import math
import time
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ibvap.edge")


WAKE_DURATION_S: float = 5.0


LOITER_THRESHOLD_SECONDS: int = 15
MOVEMENT_TOLERANCE_PIXELS: int = 50

loiter_tracker: dict = {}


API_ENDPOINT: str = "http://127.0.0.1:8000/api/events"


CAMERA_INDEX: int = 0
CAMERA_ID: str = "CAM-EDGE-01"


MODEL_WEIGHTS: str = "yolov8s.pt"


CONFIDENCE_THRESHOLD: float = 0.50


LOITERING_THRESHOLD_S: float = 8.0


PER_TRACK_COOLDOWN_S: float = 10.0


GLOBAL_ALERT_GAP_S: float = 2.0


TRACK_GRACE_PERIOD_S: float = 5.0


LOW_LIGHT_THRESHOLD: int = 60


SNAPSHOT_JPEG_QUALITY: int = 85


WATCHED_CLASSES: dict[int, str] = {
    0: "PERSON_DETECTED",
    2: "VEHICLE_DETECTED",
    3: "VEHICLE_DETECTED",
    5: "VEHICLE_DETECTED",
    7: "VEHICLE_DETECTED",
}


EVENT_INTRUSION: str = "VIRTUAL_FENCE_INTRUSION"
EVENT_LOITERING: str = "SUSPICIOUS_LOITERING"


COLOR_INTRUSION: tuple[int, int, int] = (0, 0, 255)
COLOR_LOITERING: tuple[int, int, int] = (0, 140, 255)
COLOR_PERSON: tuple[int, int, int] = (0, 255, 80)
COLOR_VEHICLE: tuple[int, int, int] = (0, 160, 255)
COLOR_FENCE: tuple[int, int, int] = (255, 80, 0)


_ALERT_PRIORITY: dict[str, int] = {
    EVENT_INTRUSION: 0,
    EVENT_LOITERING: 1,
}


@dataclass
class TrackedObject:
    """Represents a single ByteTrack-assigned detection in one frame."""

    track_id: int
    class_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    event_type: str = field(default="")
    time_in_frame: float = field(default=0.0)

    @property
    def foot_point(self) -> tuple[int, int]:
        """Bottom-center of bbox — represents feet (person) or tires (vehicle)."""
        return ((self.x1 + self.x2) // 2, self.y2)

    @property
    def bounding_box(self) -> dict:
        return {
            "x": self.x1,
            "y": self.y1,
            "width": self.x2 - self.x1,
            "height": self.y2 - self.y1,
        }

    @property
    def color(self) -> tuple[int, int, int]:
        if self.event_type == EVENT_INTRUSION:
            return COLOR_INTRUSION
        if self.event_type == EVENT_LOITERING:
            return COLOR_LOITERING
        return COLOR_PERSON if self.class_id == 0 else COLOR_VEHICLE


def build_fence_polygon(frame_w: int, frame_h: int) -> np.ndarray:
    """
    Define the virtual restricted zone as the bottom half of the frame.
    Adjust this polygon to match the real-world boundary you want to monitor.
    """
    mid_y = frame_h // 2
    return np.array(
        [[0, mid_y], [frame_w, mid_y], [frame_w, frame_h], [0, frame_h]],
        dtype=np.int32,
    )


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE on the L-channel of the LAB color space when the scene is dark.
    Returns the enhanced frame if brightness < LOW_LIGHT_THRESHOLD, else unchanged.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() >= LOW_LIGHT_THRESHOLD:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_lab = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def load_model(weights: str) -> YOLO:
    """Load (or download) YOLOv8 weights. safe_globals fix has already been applied."""
    log.info("Loading model: %s", weights)
    model = YOLO(weights)
    log.info("Model loaded — %d classes available", len(model.names))
    return model


def open_camera(index: int) -> cv2.VideoCapture:
    """
    Open webcam using DirectShow backend on Windows.
    This pins the camera to 640x480@30fps and prevents the runaway 1000 FPS bug
    caused by Windows' default camera backend buffering.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {index}. "
            "Ensure the webcam is connected and not used by another process."
        )

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Camera %d opened via DirectShow: %dx%d", index, w, h)
    return cap


def run_tracking(model: YOLO, frame: np.ndarray) -> list[TrackedObject]:
    """
    Run YOLOv8 + ByteTrack on a single frame.

    Returns a list of TrackedObject for every watched-class detection that:
      - Has a valid (non-None) track_id assigned by ByteTrack
      - Meets the CONFIDENCE_THRESHOLD
    """
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    boxes = results[0].boxes

    if boxes is None or boxes.id is None:
        return []

    detected: list[TrackedObject] = []
    for cls_t, conf_t, id_t, xyxy_t in zip(boxes.cls, boxes.conf, boxes.id, boxes.xyxy):
        cls_id = int(cls_t.item())
        if cls_id not in WATCHED_CLASSES:
            continue

        conf = float(conf_t.item())
        if conf < CONFIDENCE_THRESHOLD:
            continue

        track_id = int(id_t.item())
        x1, y1, x2, y2 = map(int, xyxy_t.tolist())

        detected.append(
            TrackedObject(
                track_id=track_id,
                class_id=cls_id,
                confidence=conf,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )

    return detected


def classify_objects(
    objects: list[TrackedObject],
    fence_polygon: np.ndarray,
    track_first_seen: dict[int, float],
    now: float,
) -> None:
    """
    Determine event_type for each tracked object (mutates in-place).

    Priority:
      1. VIRTUAL_FENCE_INTRUSION  (foot-point inside restricted zone polygon)
      2. SUSPICIOUS_LOITERING     (dwell time > LOITERING_THRESHOLD_S, outside zone)
      3. Base class label          (PERSON_DETECTED / VEHICLE_DETECTED)
    """
    for obj in objects:

        if obj.track_id not in track_first_seen:
            track_first_seen[obj.track_id] = now

        obj.time_in_frame = now - track_first_seen[obj.track_id]

        dist = cv2.pointPolygonTest(fence_polygon, obj.foot_point, measureDist=False)
        inside_fence = dist >= 0

        if inside_fence:
            obj.event_type = EVENT_INTRUSION
        elif obj.time_in_frame > LOITERING_THRESHOLD_S:
            obj.event_type = EVENT_LOITERING
        else:
            obj.event_type = WATCHED_CLASSES[obj.class_id]


def annotate_frame(
    frame: np.ndarray,
    objects: list[TrackedObject],
    fence_polygon: np.ndarray,
) -> None:
    """Draw the restricted zone, bounding boxes, labels, and foot-points (in-place)."""

    overlay = frame.copy()
    cv2.fillPoly(overlay, [fence_polygon], color=(200, 40, 0))
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    cv2.polylines(frame, [fence_polygon], isClosed=True, color=COLOR_FENCE, thickness=2)

    lx, ly = int(fence_polygon[0][0]) + 10, int(fence_polygon[0][1]) - 8
    cv2.putText(
        frame,
        "[ RESTRICTED ZONE ]",
        (lx, ly),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        COLOR_FENCE,
        1,
        cv2.LINE_AA,
    )

    for obj in objects:
        colour = obj.color

        cv2.rectangle(frame, (obj.x1, obj.y1), (obj.x2, obj.y2), colour, 2)

        cv2.circle(frame, obj.foot_point, 5, colour, -1)

        base = obj.event_type.replace("_DETECTED", "").replace("_", " ")
        if obj.event_type == EVENT_LOITERING:
            label = f"ID:{obj.track_id}  LOITERING  {obj.time_in_frame:.0f}s"
        elif obj.event_type == EVENT_INTRUSION:
            label = f"ID:{obj.track_id}  !! INTRUSION !!"
        else:
            label = f"ID:{obj.track_id}  {base}  {obj.confidence:.0%}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)

        cv2.rectangle(
            frame,
            (obj.x1, obj.y1 - th - baseline - 6),
            (obj.x1 + tw + 8, obj.y1),
            colour,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (obj.x1 + 4, obj.y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def encode_snapshot(frame: np.ndarray) -> str:
    """JPEG-encode and Base64-encode the annotated frame for the API payload."""
    ok, buffer = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed — cannot create snapshot")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def post_alert(
    session: requests.Session,
    obj: TrackedObject,
    snapshot_b64: str,
) -> bool:
    """
    POST an alert to the IBVAP backend.

    Payload schema (unchanged from v1):
      { camera_id, event_type, confidence, bounding_box, snapshot_b64 }
    Returns True on HTTP 201, False on any error (non-blocking).
    """
    payload = {
        "camera_id": CAMERA_ID,
        "event_type": obj.event_type,
        "confidence": round(obj.confidence, 6),
        "bounding_box": obj.bounding_box,
        "snapshot_b64": snapshot_b64,
    }
    try:
        resp = session.post(API_ENDPOINT, json=payload, timeout=5)
        if resp.status_code == 201:
            data = resp.json()
            log.info(
                "Alert sent → id=%s  type=%-26s  track=%d  conf=%.1f%%",
                data.get("id", "?"),
                obj.event_type,
                obj.track_id,
                obj.confidence * 100,
            )
            return True
        log.warning("API returned %d: %s", resp.status_code, resp.text[:200])
    except requests.exceptions.ConnectionError:
        log.error("Cannot reach API at %s — is the backend running?", API_ENDPOINT)
    except requests.exceptions.Timeout:
        log.error("API request timed out after 5 s")
    except Exception as exc:
        log.error("Unexpected error posting alert: %s", exc)
    return False


def draw_hud(
    frame: np.ndarray,
    frame_count: int,
    fps: float,
    objects: list[TrackedObject],
    alert_counts: dict[str, int],
    low_light: bool,
) -> None:
    """Overlay a translucent HUD bar with live telemetry at the top of the frame."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (10, 10, 10), cv2.FILLED)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    intrusions = alert_counts.get(EVENT_INTRUSION, 0)
    loitering = alert_counts.get(EVENT_LOITERING, 0)
    hud = (
        f"{CAMERA_ID} | FPS:{fps:.1f} | Frm:{frame_count} | "
        f"Trk:{len(objects)} | "
        f"INTR:{intrusions} | LOIT:{loitering}"
    )
    cv2.putText(
        frame,
        hud,
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    if low_light:
        cv2.putText(
            frame,
            "[CLAHE ENHANCED]",
            (w - 185, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )


def run_pipeline() -> None:
    """Entry point — runs until the user presses 'q' in the OpenCV window."""

    model = load_model(MODEL_WEIGHTS)
    cap = open_camera(CAMERA_INDEX)
    session = requests.Session()

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fence_polygon = build_fence_polygon(frame_w, frame_h)

    frame_count: int = 0
    fps_frame_count: int = 0
    fps_timer: float = time.monotonic()
    fps: float = 0.0

    track_first_seen: dict[int, float] = {}
    track_last_seen: dict[int, float] = {}
    track_last_alert: dict[int, float] = {}

    last_global_alert: float = 0.0
    alert_counts: dict[str, int] = {}

    log.info(
        "Pipeline v2 running | ByteTrack | Model: %s | "
        "Fence: bottom-half %dx%d | Loiter: %.0fs | "
        "CLAHE threshold: brightness < %d",
        MODEL_WEIGHTS,
        frame_w,
        frame_h,
        LOITERING_THRESHOLD_S,
        LOW_LIGHT_THRESHOLD,
    )
    log.info("Press 'q' in the OpenCV window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            log.error("Failed to read frame from camera — exiting.")
            break

        is_awake = True
        try:
            if os.path.exists("hardware_state.json"):
                with open("hardware_state.json", "r") as f:
                    state = json.load(f)
                trigger_time = state.get("trigger_time", 0)
                is_awake = (time.time() - trigger_time) < WAKE_DURATION_S
        except Exception:
            is_awake = True

        if is_awake:
            frame_count += 1
            fps_frame_count += 1
            now = time.monotonic()

            gray_mean = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
            low_light = bool(gray_mean < LOW_LIGHT_THRESHOLD)
            if low_light:
                frame = enhance_low_light(frame)

            objects = run_tracking(model, frame)

            for obj in objects:
                track_last_seen[obj.track_id] = now

            classify_objects(objects, fence_polygon, track_first_seen, now)

            annotate_frame(frame, objects, fence_polygon)

            for obj in objects:
                if obj.class_id != 0:
                    continue
                cx = (obj.x1 + obj.x2) / 2
                cy = (obj.y1 + obj.y2) / 2
                tid = obj.track_id
                if tid not in loiter_tracker:
                    loiter_tracker[tid] = {
                        "start_time": time.time(),
                        "centroid": (cx, cy),
                        "alerted": False,
                    }
                else:
                    entry = loiter_tracker[tid]
                    dist = math.hypot(
                        cx - entry["centroid"][0], cy - entry["centroid"][1]
                    )
                    if dist > MOVEMENT_TOLERANCE_PIXELS:

                        entry["start_time"] = time.time()
                        entry["centroid"] = (cx, cy)
                        entry["alerted"] = False
                    else:
                        elapsed = time.time() - entry["start_time"]
                        if elapsed >= LOITER_THRESHOLD_SECONDS and not entry["alerted"]:
                            entry["alerted"] = True
                            log.warning(
                                "CRITICAL LOITERING — track_id=%d stationary %.0fs",
                                tid,
                                elapsed,
                            )
                            try:
                                session.post(
                                    API_ENDPOINT,
                                    json={
                                        "camera_id": CAMERA_ID,
                                        "event_type": "CRITICAL: LOITERING DETECTED",
                                        "confidence": round(obj.confidence, 6),
                                        "bounding_box": obj.bounding_box,
                                        "snapshot_b64": encode_snapshot(frame),
                                    },
                                    timeout=5,
                                )
                            except Exception as exc:
                                log.error("Loiter alert POST failed: %s", exc)

            actionable = [
                o for o in objects if o.event_type in (EVENT_INTRUSION, EVENT_LOITERING)
            ]
            actionable.sort(key=lambda o: _ALERT_PRIORITY.get(o.event_type, 99))

            for obj in actionable:

                if now - track_last_alert.get(obj.track_id, 0.0) < PER_TRACK_COOLDOWN_S:
                    continue

                if now - last_global_alert < GLOBAL_ALERT_GAP_S:
                    break

                snapshot_b64 = encode_snapshot(frame)
                if post_alert(session, obj, snapshot_b64):
                    track_last_alert[obj.track_id] = now
                    last_global_alert = now
                    alert_counts[obj.event_type] = (
                        alert_counts.get(obj.event_type, 0) + 1
                    )
                break

            stale_ids = [
                tid
                for tid, ts in track_last_seen.items()
                if now - ts > TRACK_GRACE_PERIOD_S
            ]
            for tid in stale_ids:
                track_first_seen.pop(tid, None)
                track_last_seen.pop(tid, None)
                track_last_alert.pop(tid, None)
                loiter_tracker.pop(tid, None)

            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                fps_timer = now
                fps_frame_count = 0

            draw_hud(frame, frame_count, fps, objects, alert_counts, low_light)
        else:
            frame = cv2.addWeighted(
                frame, 0.3, np.zeros(frame.shape, frame.dtype), 0.7, 0
            )
            text = "[ STANDBY MODE - CPU IDLE - AWAITING SENSOR ]"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(
                frame,
                text,
                ((frame.shape[1] - tw) // 2, (frame.shape[0] + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("IBVAP v2 — Edge Pipeline  [q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("Quit key pressed — shutting down.")
            break

    cap.release()
    session.close()
    cv2.destroyAllWindows()
    log.info("Pipeline stopped cleanly.")


if __name__ == "__main__":
    run_pipeline()
