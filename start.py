import os
import sys
import time
import json
import socket
import subprocess
import threading
from pathlib import Path

# Load .env manually to avoid extra dependencies if possible
ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
STATUS_FILE = ROOT_DIR / "system_status.json"

status = {
    "PostgreSQL": "OFFLINE",
    "CCTV Edge": "OFFLINE",
    "Bridge": "OFFLINE",
    "VLM": "OFFLINE",
    "FastAPI": "OFFLINE",
    "Blockchain Auditor": "OFFLINE"
}

def write_status():
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)

def get_python_exe():
    # Prefer the VLM_Surveillance_Project venv as it has all dependencies installed
    venv_path = ROOT_DIR / "VLM_Surveillance_Project" / ".venv" / "Scripts" / "python.exe"
    if venv_path.exists():
        return str(venv_path)
    return sys.executable

PYTHON_EXE = get_python_exe()

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_process(name, cmd, cwd, status_key):
    print(f"[*] Starting {name}...")
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, shell=True)
        status[status_key] = "ONLINE"
        write_status()
        return proc
    except Exception as e:
        status[status_key] = f"FAILED: {str(e)}"
        write_status()
        print(f"[!] Failed to start {name}: {e}")
        return None

def main():
    print("========================================")
    print("Starting Integrated Surveillance System")
    print("========================================")
    write_status()

    # 1. Start Database / Docker
    if not is_port_open(5433):
        print("[*] PostgreSQL not running on 5433. Starting Docker containers...")
        subprocess.run("docker-compose up -d", cwd=str(ROOT_DIR / "CCTV" / "prototype"), shell=True)
        time.sleep(3)
        if is_port_open(5433):
            status["PostgreSQL"] = "ONLINE"
        else:
            status["PostgreSQL"] = "FAILED: Port 5433 closed"
    else:
        print("[*] PostgreSQL already running.")
        status["PostgreSQL"] = "ONLINE"
    
    write_status()

    processes = []
    
    # 2. Start Unified FastAPI Backend
    backend_cmd = f'"{PYTHON_EXE}" VLM_Surveillance_Project/backend/unified_server.py'
    p = start_process("Unified FastAPI", backend_cmd, str(ROOT_DIR), "FastAPI")
    if p: processes.append(p)
    
    # Wait for backend to be ready
    print("[*] Waiting for FastAPI to bind to port 8000...")
    for _ in range(15):
        if is_port_open(8000):
            break
        time.sleep(1)
        
    # 3. Start CCTV Edge Pipeline
    edge_cmd = f'"{PYTHON_EXE}" CCTV/prototype/edge_pipeline.py'
    p = start_process("CCTV Edge", edge_cmd, str(ROOT_DIR), "CCTV Edge")
    if p: processes.append(p)

    # 4. Start Bridge
    bridge_cmd = f'"{PYTHON_EXE}" VLM_Surveillance_Project/bridge.py'
    p = start_process("Bridge Pipeline", bridge_cmd, str(ROOT_DIR), "Bridge")
    if p: processes.append(p)

    # 5. Start Blockchain Auditor
    auditor_cmd = f'"{PYTHON_EXE}" CCTV/prototype/blockchain_auditor.py'
    p = start_process("Blockchain Auditor", auditor_cmd, str(ROOT_DIR), "Blockchain Auditor")
    if p: processes.append(p)

    # 6. Start React Dashboard
    npm_cmd = "npm run dev"
    p = start_process("React Dashboard", npm_cmd, str(ROOT_DIR / "VLM_Surveillance_Project" / "dashboard"), "React")
    if p: processes.append(p)
    
    # VLM pipeline is run by FastAPI, we can assume ONLINE if FastAPI is up
    status["VLM"] = "ONLINE"
    write_status()
    
    print("\n========================================")
    print("ALL SERVICES STARTED")
    print("Dashboard available at: http://localhost:5173")
    print("API available at: http://localhost:8000")
    print("Press Ctrl+C to cleanly shut down all services.")
    print("========================================\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Shutting down Integrated Surveillance System...")
        for proc in processes:
            try:
                proc.terminate()
            except:
                pass
        
        print("[*] All processes terminated. Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    main()
