"""
IBVAP - Intelligent Border Video Analytics Platform
Phase 1: FastAPI Backend with PostgreSQL

Architecture:
  - Async FastAPI with the `databases` library for non-blocking DB access
  - SQLAlchemy Core for schema definition (not ORM)
  - Payload separation: images saved to /snapshots, only filepath stored in DB
  - Static file mounting so frontend can reference images via URL
"""

import base64
import os
import uuid
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import databases
import sqlalchemy
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://ibvap_user:ibvap_secret@localhost:5433/ibvap_db",
)

SNAPSHOTS_DIR = Path(os.getenv("SNAPSHOTS_DIR", "./snapshots"))
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database setup  (SQLAlchemy Core + databases async driver)
# ---------------------------------------------------------------------------

database = databases.Database(DATABASE_URL)

metadata = sqlalchemy.MetaData()

events_table = sqlalchemy.Table(
    "events",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String, primary_key=True),
    sqlalchemy.Column("camera_id", sqlalchemy.String, nullable=False, index=True),
    sqlalchemy.Column("event_type", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("confidence", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("bounding_box", sqlalchemy.JSON, nullable=True),
    sqlalchemy.Column("snapshot_path", sqlalchemy.String, nullable=True),
    sqlalchemy.Column(
        "timestamp",
        sqlalchemy.DateTime(timezone=True),
        nullable=False,
        index=True,
    ),
)

# Sync engine used only at startup to run CREATE TABLE IF NOT EXISTS
engine = sqlalchemy.create_engine(
    DATABASE_URL.replace("+asyncpg", ""),
    pool_pre_ping=True,
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EventIngest(BaseModel):
    """Payload accepted by POST /api/events from the edge pipeline."""

    camera_id: str = Field(..., min_length=1, max_length=128)
    event_type: str = Field(..., min_length=1, max_length=128)
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: dict | None = Field(default=None)
    snapshot_b64: str | None = Field(
        default=None,
        description="Base64-encoded JPEG snapshot. Written to filesystem; never stored in DB.",
    )

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 6)


class EventResponse(BaseModel):
    """Serialised event returned by the read API."""

    id: str
    camera_id: str
    event_type: str
    confidence: float
    bounding_box: dict | None
    snapshot_url: str | None   # Absolute URL; resolvable via the /snapshots static mount
    timestamp: datetime


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IBVAP Backend API",
    description="Intelligent Border Video Analytics Platform — Ingestion & Query API",
    version="1.0.0",
)

# Allow all origins for the local prototype; restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount /snapshots as a static route so the frontend can display images directly.
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")

# Mount /static to serve offline CSS, JS, and other local assets.
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup() -> None:
    """Create tables (idempotent) and open the async DB connection pool."""
    metadata.create_all(engine)  # sync DDL — runs once at boot
    await database.connect()


@app.on_event("shutdown")
async def shutdown() -> None:
    await database.disconnect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SENDER_EMAIL = " "
APP_PASSWORD = " "
RECEIVER_EMAIL = " "

def send_escalation_email(camera_id: str, event_type: str, timestamp: datetime) -> None:
    """Synchronously sends an escalation email for critical events."""
    subject = f"CRITICAL ALERT: {event_type} on {camera_id}"
    body = (
        f"Camera ID: {camera_id}\n"
        f"Event: {event_type}\n"
        f"Timestamp: {timestamp}\n\n"
        "Immediate action is required."
    )
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
            print(f"Escalation email sent for {event_type} on {camera_id}.")
    except Exception as exc:
        print(f"Failed to send email: {exc}")


def _save_snapshot(b64_data: str, event_id: str) -> str:
    """
    Decode a Base64 JPEG string and persist it to SNAPSHOTS_DIR.

    Returns the URL-accessible path: /snapshots/<event_id>.jpg
    Raises HTTP 400 on invalid Base64 input.
    """
    try:
        # Strip optional data-URI prefix (data:image/jpeg;base64,...)
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_data, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Base64 snapshot payload: {exc}",
        ) from exc

    filename = f"{event_id}.jpg"
    (SNAPSHOTS_DIR / filename).write_bytes(image_bytes)
    return f"/snapshots/{filename}"


def _row_to_response(row: Any, base_url: str) -> EventResponse:
    """Map a database row mapping to an EventResponse."""
    snap = row["snapshot_path"]
    return EventResponse(
        id=row["id"],
        camera_id=row["camera_id"],
        event_type=row["event_type"],
        confidence=row["confidence"],
        bounding_box=row["bounding_box"],
        snapshot_url=f"{base_url}{snap}" if snap else None,
        timestamp=row["timestamp"],
    )


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@app.post(
    "/api/events",
    response_model=EventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a new anomaly event from the edge pipeline",
    tags=["Events"],
)
async def ingest_event(payload: EventIngest, background_tasks: BackgroundTasks) -> EventResponse:
    """
    Accept a JSON alert from an edge node:
      1. Decodes `snapshot_b64` and writes JPEG to disk.
      2. Inserts event metadata + filesystem path into the `events` table.
      3. Returns the persisted event with a resolvable `snapshot_url`.

    The DB never receives raw image bytes — only the string file path.
    """
    event_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    snapshot_path: str | None = None

    if payload.snapshot_b64:
        snapshot_path = _save_snapshot(payload.snapshot_b64, event_id)

    await database.execute(
        events_table.insert().values(
            id=event_id,
            camera_id=payload.camera_id,
            event_type=payload.event_type,
            confidence=payload.confidence,
            bounding_box=payload.bounding_box,
            snapshot_path=snapshot_path,
            timestamp=now,
        )
    )

    if "LOITERING" in payload.event_type.upper():
        background_tasks.add_task(send_escalation_email, payload.camera_id, payload.event_type, now)

    base_url = "http://localhost:8000"
    return EventResponse(
        id=event_id,
        camera_id=payload.camera_id,
        event_type=payload.event_type,
        confidence=payload.confidence,
        bounding_box=payload.bounding_box,
        snapshot_url=f"{base_url}{snapshot_path}" if snapshot_path else None,
        timestamp=now,
    )


@app.get(
    "/api/events",
    response_model=list[EventResponse],
    summary="Retrieve the latest 50 anomaly events",
    tags=["Events"],
)
async def get_events() -> list[EventResponse]:
    """Return up to 50 most-recent events, ordered by timestamp descending."""
    rows = await database.fetch_all(
        events_table.select()
        .order_by(events_table.c.timestamp.desc())
        .limit(50)
    )
    base_url = "http://localhost:8000"
    return [_row_to_response(row, base_url) for row in rows]


@app.get("/health", summary="Health check", tags=["System"])
async def health_check() -> dict:
    """Returns 200 OK when the API is reachable and DB pool is connected."""
    try:
        await database.fetch_one("SELECT 1")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    return {"status": "ok", "database": db_status}


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    """Serve the Command & Control Dashboard frontend."""
    return FileResponse("index_v2.html", media_type="text/html")


@app.get("/app.js", include_in_schema=False)
async def dashboard_js() -> FileResponse:
    """Serve the dashboard JavaScript bundle.
    Without this explicit route, FastAPI returns 404 for /app.js because
    only /snapshots/ is mounted as a static directory.
    """
    return FileResponse("app.js", media_type="application/javascript")


@app.get("/app_v2.js", include_in_schema=False)
async def dashboard_js_v2() -> FileResponse:
    """Serve the v2 dashboard JavaScript bundle."""
    return FileResponse("app_v2.js", media_type="application/javascript")

