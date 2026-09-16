"""
bridge.py - VLM & CCTV Database Bridge
======================================
This standalone microservice polls the CCTV prototype PostgreSQL database for
new events. When a new anomaly is found, it loads the corresponding snapshot,
feeds it into the VLM (Claude), and appends the results to the VLM Dashboard files.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import yaml
import requests
from dotenv import load_dotenv

# Setup paths and load env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Import existing VLM models
from models.vlm.opus_model import OpusVLMWrapper
from models.vlm.qwen_model import QwenVLWrapper
from models.vlm.vlm_analyzer import VLMAnalyzer

try:
    import psycopg2
except ImportError:
    print("FATAL: psycopg2 is required. Run: pip install psycopg2-binary")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] bridge: %(message)s")
logger = logging.getLogger(__name__)

# --- Paths & DB Config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CCTV_ROOT = PROJECT_ROOT / "CCTV" / "prototype"
VLM_PROJ_ROOT = PROJECT_ROOT / "VLM_Surveillance_Project"

CONFIG_PATH = VLM_PROJ_ROOT / "config" / "settings.yaml"
OUTPUT_DIR = Path(os.getenv("EVENTS_DIR", str(VLM_PROJ_ROOT / "outputs" / "alerts")))
EVENTS_JSON_PATH = OUTPUT_DIR / "events.json"

DB_PARAMS = {
    "dbname":   os.getenv("DATABASE_NAME", "ibvap_db"),
    "user":     os.getenv("DATABASE_USER", "ibvap_user"),
    "password": os.getenv("DATABASE_PASSWORD", "ibvap_secret"),
    "host":     os.getenv("DATABASE_HOST", "127.0.0.1"),
    "port":     int(os.getenv("DATABASE_PORT", 5433)),
}

BACKEND_PORT = os.getenv("BACKEND_PORT", "8000")
INTERNAL_API = f"http://127.0.0.1:{BACKEND_PORT}/api/internal/bridge_event"

POLL_INTERVAL_SEC = 3

def load_vlm_config():
    if not CONFIG_PATH.exists():
        logger.error(f"Config file not found: {CONFIG_PATH}")
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("VLM", {})

def load_vlm_analyzer():
    vlm_cfg = load_vlm_config()
    model_id = vlm_cfg.get("model_name", "claude-3-opus-20240229")
    api_key = vlm_cfg.get("api_key")
    max_new = vlm_cfg.get("max_new_tokens", 256)
    temp = vlm_cfg.get("temperature", 0.1)
    
    if not api_key or api_key == "YOUR_ANTHROPIC_API_KEY_HERE":
        logger.error("API Key not set in config/settings.yaml!")
        return None
        
    wrapper = OpusVLMWrapper(model_id=model_id, max_new_tokens=max_new, temperature=temp, api_key=api_key)
    return VLMAnalyzer(vlm_cfg, wrapper)

def get_db_connection():
    try:
        return psycopg2.connect(**DB_PARAMS)
    except psycopg2.OperationalError as exc:
        logger.error(f"DB connection failed: {exc}")
        return None

def append_to_vlm_events(event_dict):
    """Appends a CCTV event to the VLM events.json file so the Dashboard can read it."""
    events = []
    if EVENTS_JSON_PATH.exists():
        try:
            with open(EVENTS_JSON_PATH, "r", encoding="utf-8") as f:
                events = json.load(f)
        except json.JSONDecodeError:
            pass
            
    events.append(event_dict)
    
    with open(EVENTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=4)

def main():
    logger.info("Starting CCTV -> VLM Database Bridge...")
    
    vlm_analyzer = load_vlm_analyzer()
    if not vlm_analyzer:
        logger.error("Could not initialize VLM Analyzer. Waiting for valid API key.")
        # We will continue polling anyway so we don't crash, but VLM analysis won't run.

    # Start polling from the current time to ignore old events
    last_processed_time = datetime.now(timezone.utc)

    while True:
        conn = get_db_connection()
        if conn is not None:
            cur = None
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT id, camera_id, event_type, confidence, snapshot_path, timestamp
                    FROM events
                    WHERE timestamp > %s
                    ORDER BY timestamp ASC
                    """,
                    (last_processed_time,)
                )
                rows = cur.fetchall()

                for row in rows:
                    event_id, camera_id, event_type, confidence, snapshot_path, timestamp = row
                    
                    # 1. Translate to VLM format
                    vlm_event = {
                        "event_id": str(event_id),
                        "timestamp": timestamp.timestamp(),
                        "track_id": hash(camera_id) % 10000, # Fake track ID from camera ID
                        "object_type": "cctv_anomaly",
                        "event_type": event_type,
                        "duration": 0.0,
                        "severity": "high" if confidence > 0.8 else "medium",
                        "bbox": []
                    }
                    
                    logger.info(f"New CCTV Event Detected: {event_type} on {camera_id}")
                    
                    # 2. Append to VLM events.json
                    append_to_vlm_events(vlm_event)
                    
                    # 3. Perform VLM Analysis on the snapshot
                    if snapshot_path and vlm_analyzer and vlm_analyzer.vlm.is_ready():
                        local_snapshot_path = CCTV_ROOT / "snapshots" / Path(snapshot_path).name
                        
                        if local_snapshot_path.exists():
                            try:
                                pil_img = Image.open(local_snapshot_path).convert("RGB")
                                vlm_res = vlm_analyzer.analyze_event(vlm_event, [pil_img])
                                if vlm_res:
                                    vlm_event["vlm_analysis"] = vlm_res.get("vlm_analysis", "No analysis")
                                    vlm_event["model"] = vlm_res.get("model", "VLM")
                            except Exception as e:
                                logger.error(f"Failed to run VLM on snapshot {local_snapshot_path}: {e}")
                        else:
                            logger.warning(f"Snapshot file not found locally: {local_snapshot_path}")

                    # 4. Push to Unified Server Dashboard via internal API
                    try:
                        requests.post(INTERNAL_API, json=vlm_event, timeout=2)
                    except Exception as e:
                        logger.error(f"Failed to push event to unified backend: {e}")
                            
                    last_processed_time = timestamp

            except Exception as exc:
                logger.error(f"Error processing events: {exc}")
            finally:
                if cur is not None:
                    cur.close()
                conn.close()
        
        time.sleep(POLL_INTERVAL_SEC)

if __name__ == "__main__":
    main()
