# IBVAP -- Intelligent Border Video Analytics Platform

An AI-powered border surveillance system prototype featuring an Edge AI inference pipeline (YOLOv8 + OpenCV), a FastAPI + PostgreSQL backend for alert ingestion, a real-time Command & Control Dashboard, and a **Blockchain-secured audit trail** (Ganache + Web3.py) for tamper-proof event logging.

---

## Architecture Overview

```
+--------------------------------------------------------+
|  EDGE LAYER (edge_pipeline.py)                         |
|  - Webcam capture (OpenCV)                             |
|  - Smart frame skipping & YOLOv8 detection             |
|  - Sends Base64 JPEG + metadata to API on anomaly      |
+--------------------------+-----------------------------+
                           | HTTP POST /api/events
+--------------------------v-----------------------------+
|  BACKEND LAYER (main.py - FastAPI + PostgreSQL)        |
|  - Saves JPEGs to local filesystem (/snapshots)        |
|  - Stores event metadata & image paths in PostgreSQL   |
|  - Serves API endpoints & static dashboard assets      |
+--------+-----------------+-----------------------------+
         | GET /api/events | SQL poll (psycopg2)
+--------v--------+  +-----v------------------------------+
|  C&C DASHBOARD  |  |  BLOCKCHAIN AUDITOR                |
|  (index_v2.html |  |  (blockchain_auditor.py)            |
|   + app_v2.js)  |  |  - Polls events table every 3s     |
|  - Dark-mode    |  |  - SHA-256 hashes each event       |
|    military UI  |  |  - Anchors hash on Ganache (0-ETH  |
|  - ON-CHAIN     |  |    tx with data field)              |
|    SECURED      |  |  - Writes audit_ledger.json        |
|    badge per    |  +-------------+-----------------------+
|    event card   |                | JSON-RPC (port 8545)
+-----------------+  +-------------v-----------------------+
                     |  GANACHE (Ethereum local testnet)    |
                     |  Docker: trufflesuite/ganache:latest |
                     |  Deterministic wallet, networkId    |
                     |  5777                                |
                     +--------------------------------------+
```

---

## Prerequisites

Before installing, ensure you have the following installed on your system:

1. **Python 3.11+**: [Download Python](https://www.python.org/downloads/)
2. **Docker Desktop**: [Download Docker](https://www.docker.com/products/docker-desktop/) (required for PostgreSQL 15 + Ganache blockchain node)
3. **Webcam**: System camera for local video capture simulation

---

## Installation & Setup Instructions

Follow these step-by-step instructions to get IBVAP up and running locally.

### Step 1: Clone the Repository & Navigate to Workspace

```bash
git clone <repository-url>
cd prototype
```

### Step 2: Create and Activate a Virtual Environment (Recommended)

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all required Python packages (FastAPI, OpenCV, Ultralytics YOLOv8, SQLAlchemy, asyncpg, Web3, psycopg2, etc.):

```bash
pip install -r requirements.txt
```

> **Note:** ultralytics>=8.3.0 is pinned to avoid compatibility issues with PyTorch 2.6+.

### Step 4: Environment Configuration

Create a .env file from the provided .env.example:

**On Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**On Linux / macOS:**
```bash
cp .env.example .env
```

Default configuration:
```env
DATABASE_URL=postgresql+asyncpg://ibvap_user:ibvap_secret@localhost:5432/ibvap_db
SNAPSHOTS_DIR=./snapshots
```

---

## Running the System

To run the complete platform, open **four separate terminal windows**:

### Terminal 1: Launch PostgreSQL + Ganache Containers

Start both the PostgreSQL 15 database and the Ganache Ethereum testnet using Docker Compose:

```bash
docker compose up -d
```

Verify that both containers are running:
```bash
docker ps
```
*(Look for `ibvap_postgres` with status Up (healthy) and `ibvap_blockchain` with status Up)*

### Terminal 2: Start the FastAPI Backend Server

Run the FastAPI backend server with Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Once running, access the following URLs in your web browser:
- **C&C Dashboard (v2):** [http://localhost:8000/](http://localhost:8000/) -- serves the blockchain-secured `index_v2.html`
- **Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### Terminal 3: Start the Edge AI Inference Pipeline

Run the edge pipeline script to start webcam video capture and object detection:

```bash
python edge_pipeline.py
```

* **First Run Note:** YOLOv8 model weights (yolov8n.pt, ~6.2 MB) will download automatically on first run.
* **Controls:** Press **q** in the OpenCV video feed window to stop the edge pipeline cleanly.

### Terminal 4: Start the Blockchain Audit Trail Daemon

Run the standalone blockchain auditor microservice:

```bash
python blockchain_auditor.py
```

This daemon:
- Polls the PostgreSQL `events` table every 3 seconds for new rows
- Computes a SHA-256 hash of each event's canonical fields
- Anchors each hash on the local Ganache Ethereum node via a 0-ETH transaction
- Appends audit receipts (event ID, blockchain tx hash, stored hash) to `audit_ledger.json`

> **Note:** The auditor runs independently of the backend and dashboard. It can be started or stopped at any time without affecting the core API.

---

## Project Structure

```
prototype/
|-- docker-compose.yml       # PostgreSQL 15 + Ganache container definitions
|-- main.py                  # FastAPI backend & database setup
|-- edge_pipeline.py         # YOLOv8 webcam inference pipeline
|-- edge_pipeline_phone.py   # Mobile camera variant of edge pipeline
|-- blockchain_auditor.py    # Standalone blockchain audit trail daemon
|-- index.html               # v1 Dashboard (original, untouched)
|-- app.js                   # v1 Dashboard JS (original, untouched)
|-- index_v2.html            # v2 Dashboard -- blockchain-secured UI
|-- app_v2.js                # v2 Dashboard JS -- new event types + on-chain badge
|-- audit_ledger.json        # Auto-generated blockchain audit receipts (gitignored)
|-- requirements.txt         # Python project dependencies
|-- smoke_test.py            # Quick API smoke test script
|-- .env.example             # Environment variable template
|-- .gitignore               # Git ignore rules
|-- README.md                # Project documentation
```

---

## Stopping & Cleaning Up

### Stop the Blockchain Auditor
Press Ctrl + C in Terminal 4.

### Stop the Edge Pipeline
Press **q** inside the OpenCV video window, or press Ctrl + C in Terminal 3.

### Stop the Backend Server
Press Ctrl + C in Terminal 2.

### Stop the Docker Containers (PostgreSQL + Ganache)
In Terminal 1:
```bash
docker compose down
```

To remove stored database volumes as well:
```bash
docker compose down -v
```

---

## License

MIT License -- Built for research and prototyping.