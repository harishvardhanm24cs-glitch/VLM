# 🛡️ AI Surveillance & Multi-Modal Threat Detection Platform

An advanced, edge-to-cloud computer vision and intelligent surveillance platform built to provide real-time object tracking, hierarchical military vehicle classification, Automated Number Plate Recognition (ANPR), and Vision-Language Model (VLM) behavior analysis.

## 🌟 Key Features

*   **Real-Time Edge Pipeline (YOLOv8 & ByteTrack):** Processes live CCTV feeds and high-resolution video streams at the edge, utilizing YOLOv8 for sub-millisecond object detection and ByteTrack for robust temporal bounding box tracking.
*   **Hierarchical Vehicle Classification:** 
    *   *Level 1:* Binary Classification (MobileNetV2) instantly separates Civilian/Normal vehicles from Military vehicles.
    *   *Level 2:* 10-Class Subtype Classification identifies specific military hardware (Tanks, APCs, Prime Movers, etc.).
*   **Qwen Vision-Language Model (VLM) Integration:** Analyzes cropped frames of individuals to intelligently assess "objects in hand" and calculate dynamic threat levels based on behavioral context.
*   **Automated Number Plate Recognition (ANPR):** Rapid OCR integration to log and track vehicle plates in real-time.
*   **Blockchain Audit Ledger:** Cryptographically secures threat events and detection metadata to prevent tampering, storing immutable JSON logs.
*   **React/Vite Operations Dashboard:** A stunning, dark-mode real-time UI that ingests WebSocket events from the FastAPI backend to visualize tracking data, alert statuses, and live video feeds simultaneously.

## 🏗️ System Architecture

The platform operates across four primary operational layers:

1.  **Edge Detection (`CCTV/prototype/edge_pipeline.py`)**: Connects directly to the video source. Performs all initial object detection, tracking, cropping, and lightweight model inference (YOLO + Hierarchical Classifiers).
2.  **Event Bridge (`L4/bridge.py`)**: Acts as a message broker and orchestrator. It receives raw telemetry from the Edge, enriches it via the heavy VLM pipeline, and forwards the payload.
3.  **Unified Server (`backend/unified_server.py`)**: A FastAPI server that ingests enriched events from the bridge and edge stats, broadcasting them over high-speed WebSockets to connected clients.
4.  **Operations Dashboard (`dashboard/`)**: A responsive React front-end that translates raw JSON telemetry into actionable security alerts, timelines, and statistics.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Node.js 18+
*   PyTorch (CUDA configured for GPU acceleration)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/harishvardhanm24cs-glitch/VLM.git
    cd VLM
    ```

2.  **Install Python Dependencies:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Install Dashboard Dependencies:**
    ```bash
    cd dashboard
    npm install
    cd ..
    ```

### Running the Platform

The entire platform (Edge Pipeline, Bridge, Unified API, VLM Backend, and React UI) can be booted simultaneously using the master orchestrator script:

```bash
python start.py
```

*This will automatically open the beautiful React dashboard at `http://localhost:5173`.*

To gracefully shutdown all distributed processes, run:

```bash
python stop.py
```

## 🛠️ ML Training & Models

The repository contains scripts for retraining the classification layers:
*   `training/train_vehicle_classifier.py`: Trains the binary Civilian vs. Military classifier. Features automated class balancing (`WeightedRandomSampler`) and fast validation capabilities.
*   *Note: Pre-trained weights (`*.pt`), the enormous `gadiya` dataset, and raw video sources are excluded from this repository via `.gitignore` to preserve Git health and storage limits.*

## 🔒 Blockchain Ledger
Events flagged with a threat level > 0 or specific military classifications are permanently logged using a mock SHA-256 blockchain ledger in `CCTV/prototype/blockchain_auditor.py`. The state is saved locally to `audit_ledger.json`.
