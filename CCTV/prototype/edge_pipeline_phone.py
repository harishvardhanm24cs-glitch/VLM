"""
IBVAP - Intelligent Border Video Analytics Platform
Secondary Edge Node: Mobile Phone IP Camera (Wi-Fi Stream)

Differences from edge_pipeline.py (primary webcam node):
  - Connects to a phone IP camera via Wi-Fi HTTP stream (e.g., DroidCam / IP Webcam app)
  - Auto-reconnect loop: if the stream drops, waits 2 s and re-initialises VideoCapture
  - CAP_PROP_BUFFERSIZE = 2 to prevent multi-second network-buffer latency
  - Instant "SUSPICIOUS_VEHICLE" alert on any vehicle class (bypasses time-based checks)
  - Virtual fence & loitering logic identical to primary node (humans only)
  - CAMERA_ID = "BOP-Mobile-Patrol"

Hard Constraints (preserved from primary node):
  - PyTorch 2.6+ safe_globals fix applied BEFORE YOLO is imported
  - API payload unchanged: {camera_id, event_type, confidence, bounding_box, snapshot_b64}
"""

# ── CRITICAL: PyTorch 2.6+ fix — MUST come before `from ultralytics import YOLO` ──
import torch
import ultralytics.nn.tasks  # noqa: F401
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
# ────────────────────────────────────────────────────────────────────────────────

import base64
import logging
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests
from ultralytics import YOLO
import easyocr
import re

reader = easyocr.Reader(['en'], gpu=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ibvap.phone_edge")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ── Phone stream ─────────────────────────────────────────────────────────────
# Replace YOUR_PHONE_IP with your phone's IP (shown in the IP Webcam / DroidCam app).
# Common port is 8080 (IP Webcam) or 4747 (DroidCam).
PHONE_URL: str = "http://10.249.88.65:8080/video"

# ── Identity ──────────────────────────────────────────────────────────────────
CAMERA_ID: str = "BOP-Mobile-Patrol"

# ── Backend API ───────────────────────────────────────────────────────────────
API_ENDPOINT: str = "http://127.0.0.1:8000/api/events"

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_WEIGHTS: str = "yolov8s.pt"

# ── Detection & tracking ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.50

# ── Loitering (humans only) ───────────────────────────────────────────────────
LOITERING_THRESHOLD_S: float = 8.0

# ── Alert rate-limiting ───────────────────────────────────────────────────────
# Vehicles: shorter cooldown because every sighting is suspicious
VEHICLE_COOLDOWN_S:   float = 8.0
# Humans: loitering / intrusion cooldown per track
HUMAN_COOLDOWN_S:     float = 10.0
# Minimum wall-clock gap between any two API POSTs (flood guard)
GLOBAL_ALERT_GAP_S:   float = 2.0

# ── Track state lifetime ──────────────────────────────────────────────────────
TRACK_GRACE_PERIOD_S: float = 5.0

# ── CLAHE low-light enhancement ───────────────────────────────────────────────
LOW_LIGHT_THRESHOLD: int = 60

# ── Snapshot quality ─────────────────────────────────────────────────────────
SNAPSHOT_JPEG_QUALITY: int = 85

# ── Reconnect delay on stream drop ───────────────────────────────────────────
RECONNECT_DELAY_S: float = 2.0

# Centroid-based loitering detection (position-aware, independent of dwell-time check)
LOITER_THRESHOLD_SECONDS: int  = 10   # seconds stationary before critical alert
MOVEMENT_TOLERANCE_PIXELS: int = 150   # pixel radius considered "not moved"
# { track_id: {"start_time": float, "centroid": (cx, cy), "alerted": bool} }
loiter_tracker: dict = {}

loiter_count: int = 0
intrusion_count: int = 0

# COCO class IDs
VEHICLE_CLASSES: set[int] = {2, 3, 5, 7}   # car, motorcycle, bus, truck
PERSON_CLASS: int = 0
WATCHED_CLASSES: set[int] = VEHICLE_CLASSES | {PERSON_CLASS}

# Event types
EVENT_VEHICLE:    str = "SUSPICIOUS_VEHICLE"
EVENT_INTRUSION:  str = "VIRTUAL_FENCE_INTRUSION"
EVENT_LOITERING:  str = "SUSPICIOUS_LOITERING"
EVENT_PERSON:     str = "PERSON_DETECTED"

# Annotation colours (BGR)
COLOR_VEHICLE:   tuple[int, int, int] = (180,  0, 180)   # Purple  — suspicious vehicle
COLOR_INTRUSION: tuple[int, int, int] = (0,    0, 255)   # Red     — fence breach
COLOR_LOITERING: tuple[int, int, int] = (0,  140, 255)   # Orange  — loitering
COLOR_PERSON:    tuple[int, int, int] = (0,  255,  80)   # Neon green — normal person
COLOR_FENCE:     tuple[int, int, int] = (255,  80,   0)  # Blue    — fence overlay

# Alert priority (lower value = dispatched first when multiple objects compete)
_ALERT_PRIORITY: dict[str, int] = {
    EVENT_VEHICLE:   0,
    EVENT_INTRUSION: 1,
    EVENT_LOITERING: 2,
}


# ---------------------------------------------------------------------------
# Tracked-Object dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """Represents a single ByteTrack-assigned detection in one frame."""

    track_id:      int
    class_id:      int
    confidence:    float
    x1: int; y1: int; x2: int; y2: int
    event_type:    str   = field(default="")
    time_in_frame: float = field(default=0.0)

    @property
    def foot_point(self) -> tuple[int, int]:
        """Bottom-centre of bbox — represents feet (person) or tires (vehicle)."""
        return ((self.x1 + self.x2) // 2, self.y2)

    @property
    def is_vehicle(self) -> bool:
        return self.class_id in VEHICLE_CLASSES

    @property
    def bounding_box(self) -> dict:
        return {
            "x":      self.x1,
            "y":      self.y1,
            "width":  self.x2 - self.x1,
            "height": self.y2 - self.y1,
        }

    @property
    def color(self) -> tuple[int, int, int]:
        if self.event_type == EVENT_VEHICLE:    return COLOR_VEHICLE
        if self.event_type == EVENT_INTRUSION:  return COLOR_INTRUSION
        if self.event_type == EVENT_LOITERING:  return COLOR_LOITERING
        return COLOR_PERSON

    @property
    def cooldown(self) -> float:
        """Return the appropriate per-track cooldown based on class."""
        return VEHICLE_COOLDOWN_S if self.is_vehicle else HUMAN_COOLDOWN_S


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE on the L-channel of LAB color space when the scene is dark.
    Returns an enhanced frame if grayscale mean < LOW_LIGHT_THRESHOLD, else unchanged.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() >= LOW_LIGHT_THRESHOLD:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_lab = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
    log.debug("CLAHE applied — avg brightness was %.1f", gray.mean())
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def load_model(weights: str) -> YOLO:
    """Load (or download) YOLOv8 weights. safe_globals fix already applied."""
    log.info("Loading model: %s", weights)
    model = YOLO(weights)
    model.to('cuda')
    log.info("Model loaded — %d classes available", len(model.names))
    return model


def open_stream(url: str) -> cv2.VideoCapture:
    """
    Open the phone IP camera stream.

    CAP_PROP_BUFFERSIZE = 2 is critical for network streams:
    without it, OpenCV accumulates a large frame buffer causing multi-second lag
    between real-world events and what YOLO actually processes.
    """
    log.info("Connecting to stream: %s", url)
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)   # CRITICAL — keep buffer small
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Stream connected: %dx%d", w, h)
    else:
        log.warning("Stream not immediately available — will retry in loop.")
    return cap


def build_fence_polygon(width: int, height: int) -> np.ndarray:
    """Virtual restricted zone: the bottom half of the frame."""
    pts = np.array(
        [[0, height // 2], [width, height // 2], [width, height], [0, height]],
        np.int32,
    )
    return pts


def run_tracking(model: YOLO, frame: np.ndarray) -> list[TrackedObject]:
    """
    Run YOLOv8 + ByteTrack on a frame.
    Returns only detections for watched classes with valid track IDs and
    confidence above CONFIDENCE_THRESHOLD.
    """
    results = model.track(
        frame,
        device=0,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    boxes = results[0].boxes
    # Guard: tracker hasn't assigned IDs yet (common on first few frames)
    if boxes is None or boxes.id is None:
        return []

    detected: list[TrackedObject] = []
    for cls_t, conf_t, id_t, xyxy_t in zip(
        boxes.cls, boxes.conf, boxes.id, boxes.xyxy
    ):
        cls_id = int(cls_t.item())
        if cls_id not in WATCHED_CLASSES:
            continue

        conf = float(conf_t.item())
        if conf < CONFIDENCE_THRESHOLD:
            continue

        track_id         = int(id_t.item())
        x1, y1, x2, y2  = map(int, xyxy_t.tolist())
        detected.append(TrackedObject(
            track_id=track_id, class_id=cls_id, confidence=conf,
            x1=x1, y1=y1, x2=x2, y2=y2,
        ))

    return detected


def classify_objects(
    objects:          list[TrackedObject],
    fence_polygon:    np.ndarray,
    track_first_seen: dict[int, float],
    now:              float,
) -> None:
    """
    Assign event_type to each tracked object (mutates in-place).

    Rules:
      • Vehicle (class 2/3/5/7)  → SUSPICIOUS_VEHICLE  [bypasses all time checks]
      • Human inside fence       → VIRTUAL_FENCE_INTRUSION
      • Human, dwell > 8 s       → SUSPICIOUS_LOITERING
      • Human, dwell <= 8 s      → PERSON_DETECTED (informational only)
    """
    for obj in objects:
        # ── Vehicles: instant alert, bypass all timers ────────────────────────
        if obj.is_vehicle:
            obj.event_type = EVENT_VEHICLE
            continue

        # ── Human: register first-seen time ──────────────────────────────────
        if obj.track_id not in track_first_seen:
            track_first_seen[obj.track_id] = now
        obj.time_in_frame = now - track_first_seen[obj.track_id]

        # ── Virtual fence check ───────────────────────────────────────────────
        dist = cv2.pointPolygonTest(fence_polygon, obj.foot_point, measureDist=False)
        inside_fence = dist >= 0

        if inside_fence:
            obj.event_type = EVENT_INTRUSION
        elif obj.time_in_frame > LOITERING_THRESHOLD_S:
            obj.event_type = EVENT_LOITERING
        else:
            obj.event_type = EVENT_PERSON


def annotate_frame(
    frame:         np.ndarray,
    objects:       list[TrackedObject],
    fence_polygon: np.ndarray,
    low_light:     bool,
) -> None:
    """Draw restricted zone overlay, bounding boxes, and labels (in-place)."""
    h, w = frame.shape[:2]

    # ── Virtual fence ─────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.fillPoly(overlay, [fence_polygon], color=(200, 40, 0))
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
    cv2.polylines(frame, [fence_polygon], isClosed=True, color=COLOR_FENCE, thickness=2)
    lx = int(fence_polygon[0][0]) + 10
    ly = int(fence_polygon[0][1]) - 8
    cv2.putText(frame, "[ RESTRICTED ZONE ]", (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_FENCE, 1, cv2.LINE_AA)

    # ── Per-object boxes ──────────────────────────────────────────────────────
    for obj in objects:
        colour = obj.color
        cv2.rectangle(frame, (obj.x1, obj.y1), (obj.x2, obj.y2), colour, 2)
        cv2.circle(frame, obj.foot_point, 5, colour, -1)

        # Label
        if obj.event_type == EVENT_VEHICLE:
            label = f"ID:{obj.track_id}  !! VEHICLE !!"
        elif obj.event_type == EVENT_INTRUSION:
            label = f"ID:{obj.track_id}  !! INTRUSION !!"
        elif obj.event_type == EVENT_LOITERING:
            label = f"ID:{obj.track_id}  LOITERING  {obj.time_in_frame:.0f}s"
        else:
            label = f"ID:{obj.track_id}  PERSON  {obj.confidence:.0%}"

        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(frame,
                      (obj.x1, obj.y1 - th - bl - 6),
                      (obj.x1 + tw + 8, obj.y1),
                      colour, cv2.FILLED)
        cv2.putText(frame, label, (obj.x1 + 4, obj.y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(
    frame:        np.ndarray,
    frame_count:  int,
    fps:          float,
    objects:      list[TrackedObject],
    alert_counts: dict[str, int],
    low_light:    bool,
    connected:    bool,
) -> None:
    """Translucent HUD bar with stream info and running alert totals."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (10, 10, 10), cv2.FILLED)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    conn_str  = "LIVE" if connected else "RECONNECTING..."
    track_count = len(objects)

    line1 = f"IBVAP v2 | CAM-EDGE | FPS:{fps:.1f} | Frame:{frame_count}"
    line2 = f"Tracks:{track_count} | INTRUSIONS:{intrusion_count} | LOITERING:{loiter_count}"

    cv2.putText(frame, line1, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 1, cv2.LINE_AA)

    if low_light:
        cv2.putText(frame, "[CLAHE ENHANCED]", (w - 185, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)


def encode_snapshot(frame: np.ndarray) -> str:
    """JPEG-encode and Base64-encode the annotated frame."""
    ok, buf = cv2.imencode(".jpg", frame,
                           [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def post_alert(
    session:      requests.Session,
    obj:          TrackedObject,
    snapshot_b64: str,
) -> bool:
    """
    POST an alert to the IBVAP backend.

    Payload (unchanged from primary node spec):
      { camera_id, event_type, confidence, bounding_box, snapshot_b64 }
    """
    payload = {
        "camera_id":    CAMERA_ID,
        "event_type":   obj.event_type,
        "confidence":   round(obj.confidence, 6),
        "bounding_box": obj.bounding_box,
        "snapshot_b64": snapshot_b64,
    }
    try:
        resp = session.post(API_ENDPOINT, json=payload, timeout=5)
        if resp.status_code == 201:
            data = resp.json()
            log.info(
                "Alert sent → id=%s  type=%-22s  track=%d  conf=%.1f%%",
                data.get("id", "?"), obj.event_type, obj.track_id, obj.confidence * 100,
            )
            return True
        log.warning("API %d: %s", resp.status_code, resp.text[:200])
    except requests.exceptions.ConnectionError:
        log.error("Cannot reach API at %s — is the backend running?", API_ENDPOINT)
    except requests.exceptions.Timeout:
        log.error("API request timed out after 5 s")
    except Exception as exc:
        log.error("Unexpected POST error: %s", exc)
    return False


# ---------------------------------------------------------------------------
# Main pipeline loop (with auto-reconnect)
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """
    Entry point. Outer loop handles stream reconnection;
    inner loop is the per-frame inference cycle.
    """
    if "YOUR_PHONE_IP" in PHONE_URL:
        log.warning(
            "PHONE_URL still contains placeholder 'YOUR_PHONE_IP'. "
            "Edit the PHONE_URL constant before running for real."
        )

    model   = load_model(MODEL_WEIGHTS)
    session = requests.Session()

    # Persistent tracking state (survives reconnects)
    track_first_seen: dict[int, float] = {}
    track_last_seen:  dict[int, float] = {}
    track_last_alert: dict[int, float] = {}
    last_global_alert: float           = 0.0
    alert_counts:      dict[str, int]  = {}

    frame_count:     int   = 0
    fps_frame_count: int   = 0
    fps_timer:       float = time.monotonic()
    fps:             float = 0.0
    fence_polygon:   np.ndarray | None = None

    log.info(
        "Phone edge node starting | CAMERA_ID=%s | Stream=%s",
        CAMERA_ID, PHONE_URL,
    )
    log.info("Press 'q' in the OpenCV window to quit.")

    WIN_NAME = f"IBVAP — Phone Node ({CAMERA_ID})  [q to quit]"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    # ── Outer reconnect loop ──────────────────────────────────────────────────
    while True:
        cap       = open_stream(PHONE_URL)
        connected = cap.isOpened()

        # ── Inner per-frame loop ──────────────────────────────────────────────
        while True:
            ret, frame = cap.read()

            # ── Network drop / stream unavailable ────────────────────────────
            if not ret or frame is None:
                log.warning(
                    "Stream read failed (network drop or unavailable). "
                    "Reconnecting in %.0f s...", RECONNECT_DELAY_S
                )
                cap.release()
                connected = False
                time.sleep(RECONNECT_DELAY_S)
                cap = cv2.VideoCapture(PHONE_URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)   # reapply after re-init
                connected = cap.isOpened()
                if connected:
                    log.info("Reconnected successfully.")
                continue   # retry cap.read()

            frame_count     += 1
            fps_frame_count += 1
            now              = time.monotonic()

            # ── Build fence polygon once we know frame dimensions ─────────────
            if fence_polygon is None:
                h, w    = frame.shape[:2]
                fence_polygon = build_fence_polygon(w, h)
                cv2.resizeWindow(WIN_NAME, w, h)   # snap window to actual stream size
                log.info("Fence polygon built for %dx%d frame.", w, h)

            # ── 1. Low-light CLAHE enhancement ───────────────────────────────
            gray_mean = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
            low_light = bool(gray_mean < LOW_LIGHT_THRESHOLD)
            if low_light:
                frame = enhance_low_light(frame)

            # ── 2. ByteTrack inference ────────────────────────────────────────
            objects = run_tracking(model, frame)

            # Update last-seen timestamps
            for obj in objects:
                track_last_seen[obj.track_id] = now

            # ── 3. Classify ───────────────────────────────────────────────────
            classify_objects(objects, fence_polygon, track_first_seen, now)

            # ── 4. Annotate ───────────────────────────────────────────────────
            annotate_frame(frame, objects, fence_polygon, low_light)

            # ── 4b. Centroid-based loitering detection ─────────────────────────
            for obj in objects:
                cx = (obj.x1 + obj.x2) / 2
                cy = (obj.y1 + obj.y2) / 2
                tid = obj.track_id
                if tid not in loiter_tracker:
                    loiter_tracker[tid] = {
                        "start_time": time.time(),
                        "centroid":   (cx, cy),
                        "alerted":    False,
                        "ocr_scanned": False,
                    }
                else:
                    entry = loiter_tracker[tid]
                    dist  = math.hypot(cx - entry["centroid"][0], cy - entry["centroid"][1])
                    if dist > MOVEMENT_TOLERANCE_PIXELS:
                        entry["start_time"] = time.time()
                        entry["centroid"]   = (cx, cy)
                        entry["alerted"]    = False
                    else:
                        if obj.class_id == 0:    # persons only for loitering alert
                            loiter_elapsed = time.time() - entry["start_time"]
                            if loiter_elapsed >= LOITER_THRESHOLD_SECONDS and not entry["alerted"]:
                                global loiter_count
                                loiter_count += 1
                                entry["alerted"] = True
                                log.warning(
                                    "CRITICAL LOITERING — track_id=%d stationary %.0fs",
                                    tid, loiter_elapsed,
                                )
                                try:
                                    session.post(
                                        API_ENDPOINT,
                                        json={
                                            "camera_id":    CAMERA_ID,
                                            "event_type":   "CRITICAL: LOITERING DETECTED",
                                            "confidence":   round(obj.confidence, 6),
                                            "bounding_box": obj.bounding_box,
                                            "snapshot_b64": encode_snapshot(frame),
                                        },
                                        timeout=5,
                                    )
                                except Exception as exc:
                                    log.error("Loiter alert POST failed: %s", exc)

            # ── 5. Alert dispatch ─────────────────────────────────────────────
            # Sort by priority — vehicles first, then intrusions, then loitering
            actionable = [
                o for o in objects
                if o.event_type in (EVENT_VEHICLE, EVENT_INTRUSION, EVENT_LOITERING)
            ]
            actionable.sort(key=lambda o: _ALERT_PRIORITY.get(o.event_type, 99))

            for obj in actionable:
                # Per-track cooldown (uses obj.cooldown which differs by class)
                if now - track_last_alert.get(obj.track_id, 0.0) < obj.cooldown:
                    continue

                # Global minimum gap
                if now - last_global_alert < GLOBAL_ALERT_GAP_S:
                    break

                if obj.event_type in (EVENT_INTRUSION, EVENT_VEHICLE):
                    tid = obj.track_id
                    if tid in loiter_tracker and not loiter_tracker[tid].get("ocr_scanned", False):
                        h_frame, w_frame = frame.shape[:2]
                        w_obj = obj.x2 - obj.x1
                        h_obj = obj.y2 - obj.y1
                        
                        x1_crop = int(obj.x1 + w_obj * 0.2)
                        x2_crop = int(obj.x2 - w_obj * 0.2)
                        y1_crop = int(obj.y1 + h_obj * 0.6)
                        y2_crop = int(obj.y2)

                        x1_crop, y1_crop = max(0, x1_crop), max(0, y1_crop)
                        x2_crop, y2_crop = min(w_frame, x2_crop), min(h_frame, y2_crop)
                        
                        plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                        if plate_crop.size > 0:
                            ocr_results = reader.readtext(plate_crop)
                            loiter_tracker[tid]["ocr_scanned"] = True
                            for bbox, text, prob in ocr_results:
                                cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                                if len(cleaned) >= 4:  # Alphanumeric pattern match
                                    obj.event_type = f"ANPR: PLATE DETECTED -> {cleaned}"
                                    break

                snapshot_b64 = encode_snapshot(frame)
                if post_alert(session, obj, snapshot_b64):
                    track_last_alert[obj.track_id] = now
                    last_global_alert              = now
                    alert_counts[obj.event_type]   = \
                        alert_counts.get(obj.event_type, 0) + 1
                    if obj.event_type == EVENT_INTRUSION:
                        global intrusion_count
                        intrusion_count += 1
                break   # one alert per frame cycle

            # ── 6. Stale-track pruning ────────────────────────────────────────
            stale = [
                tid for tid, ts in track_last_seen.items()
                if now - ts > TRACK_GRACE_PERIOD_S
            ]
            for tid in stale:
                track_first_seen.pop(tid, None)
                track_last_seen.pop(tid, None)
                track_last_alert.pop(tid, None)
                loiter_tracker.pop(tid, None)    # prune centroid-loiter state too

            # ── 7. FPS (sliding 1-second window) ─────────────────────────────
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps             = fps_frame_count / elapsed
                fps_timer       = now
                fps_frame_count = 0

            # ── 8. HUD overlay ────────────────────────────────────────────────
            draw_hud(frame, frame_count, fps, objects,
                     alert_counts, low_light, connected)

            # ── 9. Display ────────────────────────────────────────────────────
            cv2.imshow(WIN_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Quit key pressed — shutting down.")
                cap.release()
                session.close()
                cv2.destroyAllWindows()
                log.info("Phone edge pipeline stopped cleanly.")
                return   # exit both loops


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
