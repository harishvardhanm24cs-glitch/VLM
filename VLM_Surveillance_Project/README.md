# VLM Surveillance Project

A 4-layer (L1 -> L2 -> L3 -> L4) CCTV/VLM surveillance system.

## Architecture

This project is built modularly in phases:
- **L1**: Frame/Pixel Analysis (OpenCV + NumPy)
- **L2**: Object Detection (Ultralytics YOLO)
- **Tracking**: Object Tracking
- **L3**: Human Tracking & Behaviour Analysis (Python Rules)
- **VLM**: Higher-level visual reasoning (Qwen2.5-VL)
- **L4**: Dashboard / UI (Streamlit)

## Getting Started

1. Create a virtual environment using Python 3.11:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the system:
   ```bash
   python main.py
   ```
