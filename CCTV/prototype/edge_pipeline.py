"""
IBVAP - Intelligent Border Video Analytics Platform
Phase 2 (v2): Advanced Edge AI Pipeline with ByteTrack & Spatial Analytics

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

reader = easyocr.Reader(['en'], gpu=False)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ibvap.edge")

# ---------------------------------------------------------------------------
# Configuration  (edit these constants to tune the pipeline)
# ---------------------------------------------------------------------------

# Backend API — updated to point to unified backend CCTV routes
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

API_ENDPOINT: str = "http://127.0.0.1:8000/cctv/api/events"

# Camera
CAMERA_INDEX: int = 0
CAMERA_ID: str = "CAM-EDGE-01"

# Model — upgraded to yolov8s for higher accuracy
MODEL_WEIGHTS: str = "yolov8s.pt"

# Detection confidence minimum (tracker smooths noise so 0.50 is safe)
CONFIDENCE_THRESHOLD: float = 0.25

# Loitering: seconds a tracked object must dwell before triggering loitering alert
LOITERING_THRESHOLD_S: float = 8.0

# How long (seconds) to suppress additional alerts for the same track_id
PER_TRACK_COOLDOWN_S: float = 10.0

# Minimum wall-clock gap between *any* two POST requests (flood prevention)
GLOBAL_ALERT_GAP_S: float = 2.0

# Seconds after a track disappears before its state is purged
TRACK_GRACE_PERIOD_S: float = 5.0

# Centroid-based loitering detection (position-aware, independent of dwell-time check)
LOITER_THRESHOLD_SECONDS: int  = 10   # seconds stationary before critical alert
MOVEMENT_TOLERANCE_PIXELS: int = 150   # pixel radius considered "not moved"
# { track_id: {"start_time": float, "centroid": (cx, cy), "alerted": bool} }
loiter_tracker: dict = {}

loiter_count: int = 0
intrusion_count: int = 0

# Low-light threshold — average grayscale brightness below this triggers CLAHE
LOW_LIGHT_THRESHOLD: int = 60

# JPEG quality for snapshots sent to API
SNAPSHOT_JPEG_QUALITY: int = 85

# COCO class IDs and their base event labels
WATCHED_CLASSES: dict[int, str] = {
    0: "PERSON_DETECTED",
    2: "VEHICLE_DETECTED",
    3: "VEHICLE_DETECTED",
    5: "VEHICLE_DETECTED",
    7: "VEHICLE_DETECTED",
}

# Event type constants
EVENT_INTRUSION: str = "VIRTUAL_FENCE_INTRUSION"
EVENT_LOITERING: str = "SUSPICIOUS_LOITERING"

# Annotation colours (BGR)
COLOR_INTRUSION: tuple[int, int, int] = (0,   0,   255)   # Red
COLOR_LOITERING: tuple[int, int, int] = (0,   140, 255)   # Orange
COLOR_PERSON:    tuple[int, int, int] = (0,   255, 80)    # Neon green
COLOR_VEHICLE:   tuple[int, int, int] = (0,   160, 255)   # Amber
COLOR_FENCE:     tuple[int, int, int] = (255, 80,  0)     # Blue

# Alert priority order (lower = higher priority)
_ALERT_PRIORITY: dict[str, int] = {
    EVENT_INTRUSION: 0,
    EVENT_LOITERING: 1,
    "VEHICLE_DETECTED": 2,
    "SUSPICIOUS_VEHICLE": 2,
}


# ---------------------------------------------------------------------------
# Tracked Object dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """Represents a single ByteTrack-assigned detection in one frame."""

    track_id:     int
    class_id:     int
    confidence:   float
    x1: int; y1: int; x2: int; y2: int
    event_type:   str = field(default="")
    time_in_frame: float = field(default=0.0)  # seconds since first seen
    vehicle_category: str = None
    classification_confidence: float = 0.0

    @property
    def foot_point(self) -> tuple[int, int]:
        """Bottom-center of bbox — represents feet (person) or tires (vehicle)."""
        return ((self.x1 + self.x2) // 2, self.y2)

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
        if self.event_type == EVENT_INTRUSION:
            return COLOR_INTRUSION
        if self.event_type == EVENT_LOITERING:
            return COLOR_LOITERING
        return COLOR_PERSON if self.class_id == 0 else COLOR_VEHICLE


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

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

    # Guard: tracker hasn't assigned IDs yet (common on the first few frames)
    if boxes is None or boxes.id is None:
        return []

    detected: list[TrackedObject] = []
    
    # PART 3: RAW YOLO DETECTIONS
    if results[0].boxes is not None:
        for cls_t, conf_t, xyxy_t in zip(results[0].boxes.cls, results[0].boxes.conf, results[0].boxes.xyxy):
            cls_id = int(cls_t.item())
            cls_name = model.names.get(cls_id, f"unknown_{cls_id}")
            conf_val = float(conf_t.item())
            x1, y1, x2, y2 = map(int, xyxy_t.tolist())
            print(f"[YOLO]\nclass={cls_name}\nconfidence={conf_val:.2f}\nbbox=({x1},{y1},{x2},{y2})\n")

    for cls_t, conf_t, id_t, xyxy_t in zip(
        boxes.cls, boxes.conf, boxes.id, boxes.xyxy
    ):
        cls_id  = int(cls_t.item())
        if cls_id not in WATCHED_CLASSES:
            continue

        conf = float(conf_t.item())
        if conf < CONFIDENCE_THRESHOLD:
            continue

        track_id          = int(id_t.item())
        x1, y1, x2, y2   = map(int, xyxy_t.tolist())
        
        cls_name = model.names.get(cls_id, "unknown")
        print(f"[TRACK]\ntrack_id={track_id}\nyolo_class={cls_name}\nconfidence={conf:.2f}\nbbox=({x1},{y1},{x2},{y2})\n")

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
    Determine event_type for each tracked object (mutates in-place).

    Priority:
      1. VIRTUAL_FENCE_INTRUSION  (foot-point inside restricted zone polygon)
      2. SUSPICIOUS_LOITERING     (dwell time > LOITERING_THRESHOLD_S, outside zone)
      3. Base class label          (PERSON_DETECTED / VEHICLE_DETECTED)
    """
    for obj in objects:
        # Register first-seen timestamp for new track IDs
        if obj.track_id not in track_first_seen:
            track_first_seen[obj.track_id] = now

        obj.time_in_frame = now - track_first_seen[obj.track_id]

        # 1. Virtual fence check — foot-point inside polygon?
        dist = cv2.pointPolygonTest(fence_polygon, obj.foot_point, measureDist=False)
        inside_fence = dist >= 0

        if inside_fence:
            obj.event_type = EVENT_INTRUSION
        elif obj.time_in_frame > LOITERING_THRESHOLD_S:
            obj.event_type = EVENT_LOITERING
        else:
            obj.event_type = WATCHED_CLASSES[obj.class_id]


def annotate_frame(
    frame:         np.ndarray,
    objects:       list[TrackedObject],
    fence_polygon: np.ndarray,
) -> None:
    """Draw the restricted zone, bounding boxes, labels, and foot-points (in-place)."""

    # ── Virtual fence polygon ────────────────────────────────────────────────
    # Semi-transparent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [fence_polygon], color=(200, 40, 0))
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    # Solid border
    cv2.polylines(frame, [fence_polygon], isClosed=True, color=COLOR_FENCE, thickness=2)

    # Zone label
    lx, ly = int(fence_polygon[0][0]) + 10, int(fence_polygon[0][1]) - 8
    cv2.putText(frame, "[ RESTRICTED ZONE ]", (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_FENCE, 1, cv2.LINE_AA)

    # ── Per-object annotations ───────────────────────────────────────────────
    for obj in objects:
        colour = obj.color

        # Bounding box
        cv2.rectangle(frame, (obj.x1, obj.y1), (obj.x2, obj.y2), colour, 2)

        # Foot-point circle
        cv2.circle(frame, obj.foot_point, 5, colour, -1)

        # Build label text
        base = obj.event_type.replace("_DETECTED", "").replace("_", " ")
        if obj.event_type == EVENT_LOITERING:
            label = f"ID:{obj.track_id}  LOITERING  {obj.time_in_frame:.0f}s"
        elif obj.event_type == EVENT_INTRUSION:
            label = f"ID:{obj.track_id}  !! INTRUSION !!"
        else:
            label = f"ID:{obj.track_id}  {base}  {obj.confidence:.0%}"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
        )
        # Label background pill
        cv2.rectangle(
            frame,
            (obj.x1, obj.y1 - th - baseline - 6),
            (obj.x1 + tw + 8, obj.y1),
            colour, cv2.FILLED,
        )
        cv2.putText(
            frame, label,
            (obj.x1 + 4, obj.y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA,
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
    session:      requests.Session,
    obj:          TrackedObject,
    snapshot_b64: str,
) -> bool:
    """
    POST an alert to the IBVAP backend.

    Payload schema (unchanged from v1):
      { camera_id, event_type, confidence, bounding_box, snapshot_b64 }
    Returns True on HTTP 201, False on any error (non-blocking).
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
                "Alert sent → id=%s  type=%-26s  track=%d  conf=%.1f%%",
                data.get("id", "?"), obj.event_type, obj.track_id, obj.confidence * 100,
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
    frame:        np.ndarray,
    frame_count:  int,
    fps:          float,
    objects:      list[TrackedObject],
    alert_counts: dict[str, int],
    low_light:    bool,
) -> None:
    """Overlay a translucent HUD bar with live telemetry at the top of the frame."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (10, 10, 10), cv2.FILLED)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

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


# ---------------------------------------------------------------------------
# Main pipeline loop
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """Entry point — runs until the user presses 'q' in the OpenCV window."""

    model         = load_model(MODEL_WEIGHTS)
    cap           = open_camera(CAMERA_INDEX)
    session       = requests.Session()  # reuse TCP connection for all POSTs

    frame_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fence_polygon = build_fence_polygon(frame_w, frame_h)

    # ── Loop state ───────────────────────────────────────────────────────────
    frame_count:  int   = 0
    fps_frame_count: int = 0
    fps_timer:    float = time.monotonic()
    fps:          float = 0.0

    # Tracking temporal state
    track_first_seen: dict[int, float] = {}   # track_id → first seen (monotonic)
    track_last_seen:  dict[int, float] = {}   # track_id → most recent frame time
    track_last_alert: dict[int, float] = {}   # track_id → last alert sent time

    last_global_alert: float     = 0.0
    alert_counts:      dict[str, int] = {}

    log.info(
        "Pipeline v2 running | ByteTrack | Model: %s | "
        "Fence: bottom-half %dx%d | Loiter: %.0fs | "
        "CLAHE threshold: brightness < %d",
        MODEL_WEIGHTS, frame_w, frame_h,
        LOITERING_THRESHOLD_S, LOW_LIGHT_THRESHOLD,
    )
    print("==================================================")
    print(f"YOLO MODEL:\n{MODEL_WEIGHTS}")
    print(f"YOLO TASK:\ndetect")
    print(f"YOLO CLASS NAMES:\n{model.names}")
    print(f"YOLO CONFIDENCE:\n{CONFIDENCE_THRESHOLD}")
    print(f"YOLO IMAGE SIZE:\n640") # Ultralytics default for .track() unless imgsz passed
    print(f"YOLO CLASS FILTER:\n{WATCHED_CLASSES}")
    print("==================================================")
    
    log.info("Press 'q' in the OpenCV window to quit.")
    
    # Initialize Vehicle Classifier
    vehicle_classifier = None
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "VLM_Surveillance_Project"))
        
        from L2.vehicle_classifier import init_classifier, get_classifier
        model_dir = os.getenv("VEHICLE_CLASSIFIER_MODEL", "d:/VLM/models/vehicle_classifier")
        interval = int(os.getenv("VEHICLE_CLASSIFICATION_INTERVAL", "5"))
        min_obs = int(os.getenv("VEHICLE_CLASSIFICATION_MIN_OBSERVATIONS", "3"))
        stability = float(os.getenv("VEHICLE_CLASSIFICATION_STABILITY_RATIO", "0.67"))
        min_conf = float(os.getenv("VEHICLE_CLASSIFICATION_MIN_CONFIDENCE", "0.80"))
        
        subtype_dir = "d:/VLM/models/military_subtype/v_FULL_20260914_210557"
        init_classifier(model_dir=model_dir, subtype_dir=subtype_dir, min_confidence=min_conf, interval_frames=interval, min_obs=min_obs, stability_ratio=stability)
        vehicle_classifier = get_classifier()
        log.info(f"Vehicle Classifier loaded successfully on edge (Interval:{interval}, Obs:{min_obs}, Stable:{stability}).")
    except Exception as e:
        log.warning(f"Could not load vehicle classifier: {e}")

    WIN_NAME = "IBVAP v2 — Edge Pipeline  [q to quit]"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, frame_w, frame_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            log.error("Failed to read frame from camera — exiting.")
            break

        frame_count      += 1
        fps_frame_count  += 1
        now               = time.monotonic()

        # ── 1. Low-light enhancement ─────────────────────────────────────────
        gray_mean = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        low_light = bool(gray_mean < LOW_LIGHT_THRESHOLD)
        if low_light:
            frame = enhance_low_light(frame)

        # ── 2. ByteTrack inference ────────────────────────────────────────────
        objects = run_tracking(model, frame)

        # Update last-seen timestamps (for stale-track pruning)
        for obj in objects:
            track_last_seen[obj.track_id] = now

        # ── 3. Classify each tracked object ──────────────────────────────────
        classify_objects(objects, fence_polygon, track_first_seen, now)
        
        if vehicle_classifier:
            for obj in objects:
                if obj.class_id in (2, 3, 5, 7): # car, motorcycle, bus, truck
                    try:
                        x1, y1, x2, y2 = obj.x1, obj.y1, obj.x2, obj.y2
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]
                            
                            print(f"[VEHICLE CLASSIFIER INPUT]\ntrack_id={obj.track_id}\nyolo_class={model.names.get(obj.class_id, 'unknown').lower()}\ncrop_size={crop.shape[1]}x{crop.shape[0]}\n")
                            cv2.imwrite(f"d:/VLM/debug_vehicle_crop_{obj.track_id}.jpg", crop)
                            
                            res = vehicle_classifier.process_track(obj.track_id, crop)
                            obj.vehicle_category = res["class"]
                            obj.classification_confidence = res["confidence"]
                            obj.subtype = res.get("subtype", None)
                            obj.subtype_confidence = res.get("subtype_confidence", 0.0)
                            
                            print(f"[VEHICLE CLASSIFIER OUTPUT]\ntrack_id={obj.track_id}\nprediction={obj.vehicle_category.upper()}\nconfidence={obj.classification_confidence:.2f}\nsubtype={obj.subtype}\n")
                            
                            # If it's a normal vehicle, escalate its priority!
                            if obj.vehicle_category == "Normal":
                                obj.event_type = "SUSPICIOUS_VEHICLE"
                            elif obj.vehicle_category == "Military":
                                obj.event_type = "VEHICLE_DETECTED"
                            elif obj.vehicle_category == "uncertain":
                                obj.event_type = "UNCERTAIN_VEHICLE"
                    except Exception as e:
                        log.warning(f"Vehicle classifier error on edge: {e}")

        # ── 4. Annotate frame ─────────────────────────────────────────────────
        annotate_frame(frame, objects, fence_polygon)

        # ── 4b. Centroid-based loitering detection ───────────────────────────
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

        # ── 5. Alert dispatch ─────────────────────────────────────────────────
        # Sort by event priority (INTRUSION first, then LOITERING, skip normals)
        actionable = [
            o for o in objects
            if o.event_type in (EVENT_INTRUSION, EVENT_LOITERING, "VEHICLE_DETECTED", "SUSPICIOUS_VEHICLE")
        ]
        actionable.sort(key=lambda o: _ALERT_PRIORITY.get(o.event_type, 99))

        for obj in actionable:
            # Per-track cooldown
            if now - track_last_alert.get(obj.track_id, 0.0) < PER_TRACK_COOLDOWN_S:
                continue

            # Global minimum gap between any two API calls
            if now - last_global_alert < GLOBAL_ALERT_GAP_S:
                break  # No point checking more — gap not elapsed yet

            if obj.event_type in (EVENT_INTRUSION, "VEHICLE_DETECTED", "SUSPICIOUS_VEHICLE"):
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
                alert_counts[obj.event_type]   = alert_counts.get(obj.event_type, 0) + 1
                if obj.event_type == EVENT_INTRUSION:
                    global intrusion_count
                    intrusion_count += 1
            break  # One alert per frame cycle max

        # ── 6. Stale-track pruning ────────────────────────────────────────────
        stale_ids = [
            tid for tid, ts in track_last_seen.items()
            if now - ts > TRACK_GRACE_PERIOD_S
        ]
        for tid in stale_ids:
            track_first_seen.pop(tid, None)
            track_last_seen.pop(tid, None)
            track_last_alert.pop(tid, None)
            loiter_tracker.pop(tid, None)    # prune centroid-loiter state too

        # ── 7. FPS (per-window, resets each second) ───────────────────────────
        elapsed = now - fps_timer
        if elapsed >= 1.0:
            fps             = fps_frame_count / elapsed
            fps_timer       = now
            fps_frame_count = 0

        # ── 8. HUD overlay ────────────────────────────────────────────────────
        draw_hud(frame, frame_count, fps, objects, alert_counts, low_light)
        
        # ── 8b. Dashboard Live Sync Adapter ───────────────────────────────────
        # Send a minimal payload to the unified server so the dashboard sees the live camera and true counts
        if frame_count % 3 == 0:  # Sync ~10 fps to save bandwidth
            try:
                person_count = sum(1 for o in objects if o.class_id == 0)
                vehicle_count = sum(1 for o in objects if o.class_id in (2, 3, 5, 7))
                object_count = len(objects) - person_count - vehicle_count
                
                # Create a small summary of tracks for the dashboard
                track_summary = {}
                for o in objects:
                    yolo_class = model.names.get(o.class_id, "object").lower()
                    
                    veh_class = "UNCERTAIN"
                    if getattr(o, "vehicle_category", None) == "Military":
                        veh_class = "Military"
                    elif getattr(o, "vehicle_category", None) == "Normal":
                        veh_class = "Normal"
                        
                    track_summary[str(o.track_id)] = {
                        "track_id": o.track_id,
                        "object_type": yolo_class,
                        "class": yolo_class, # Kept for backward compatibility
                        "confidence": o.confidence,
                        "event_type": o.event_type,
                        "time_in_frame": o.time_in_frame,
                        "vehicle_class": veh_class,
                        "vehicle_category": getattr(o, "vehicle_category", None), # Kept for backward compatibility
                        "vehicle_class_confidence": getattr(o, "classification_confidence", 0.0),
                        "subtype": getattr(o, "subtype", None),
                        "subtype_confidence": getattr(o, "subtype_confidence", 0.0)
                    }
                    
                sync_payload = {
                    "frame_number": frame_count,
                    "frame_b64": encode_snapshot(frame),
                    "active_tracks_count": len(objects),
                    "person_count": person_count,
                    "vehicle_count": vehicle_count,
                    "object_count": object_count,
                    "tracks": track_summary
                }
                session.post("http://127.0.0.1:8000/api/internal/edge_sync", json=sync_payload, timeout=0.1)
            except Exception:
                pass  # Ignore if unified backend is not running

        # ── 9. Display ────────────────────────────────────────────────────────
        cv2.imshow(WIN_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("Quit key pressed — shutting down.")
            break

    cap.release()
    session.close()
    cv2.destroyAllWindows()
    log.info("Pipeline stopped cleanly.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
