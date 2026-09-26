import sys
import time
import json
import socket
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
STATUS_FILE = ROOT_DIR / "system_status.json"

status = {
    "PostgreSQL": "OFFLINE",
    "CCTV Edge": "OFFLINE",
    "Bridge": "OFFLINE",
    "VLM": "OFFLINE",
    "FastAPI": "OFFLINE",
    "Blockchain Auditor": "OFFLINE",
}


def write_status():
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)


def get_python_exe():

    venv_path = (
        ROOT_DIR / "VLM_Surveillance_Project" / ".venv" / "Scripts" / "python.exe"
    )
    if venv_path.exists():
        return str(venv_path)
    return sys.executable


PYTHON_EXE = get_python_exe()


def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_process(name, cmd, cwd, status_key):
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, shell=True)
        status[status_key] = "ONLINE"
        write_status()
        return proc
    except Exception as e:
        status[status_key] = f"FAILED: {str(e)}"
        write_status()
        return None


def main():
    write_status()

    if not is_port_open(5433):
        subprocess.run(
            "docker-compose up -d", cwd=str(ROOT_DIR / "CCTV" / "prototype"), shell=True
        )
        time.sleep(3)
        if is_port_open(5433):
            status["PostgreSQL"] = "ONLINE"
        else:
            status["PostgreSQL"] = "FAILED: Port 5433 closed"
    else:
        status["PostgreSQL"] = "ONLINE"

    write_status()

    processes = []

    backend_cmd = f'"{PYTHON_EXE}" VLM_Surveillance_Project/backend/unified_server.py'
    p = start_process("Unified FastAPI", backend_cmd, str(ROOT_DIR), "FastAPI")
    if p:
        processes.append(p)

    for _ in range(15):
        if is_port_open(8000):
            break
        time.sleep(1)

    edge_cmd = f'"{PYTHON_EXE}" CCTV/prototype/edge_pipeline.py'
    p = start_process("CCTV Edge", edge_cmd, str(ROOT_DIR), "CCTV Edge")
    if p:
        processes.append(p)

    bridge_cmd = f'"{PYTHON_EXE}" VLM_Surveillance_Project/bridge.py'
    p = start_process("Bridge Pipeline", bridge_cmd, str(ROOT_DIR), "Bridge")
    if p:
        processes.append(p)

    auditor_cmd = f'"{PYTHON_EXE}" CCTV/prototype/blockchain_auditor.py'
    p = start_process(
        "Blockchain Auditor", auditor_cmd, str(ROOT_DIR), "Blockchain Auditor"
    )
    if p:
        processes.append(p)

    npm_cmd = "npm run dev"
    p = start_process(
        "React Dashboard",
        npm_cmd,
        str(ROOT_DIR / "VLM_Surveillance_Project" / "dashboard"),
        "React",
    )
    if p:
        processes.append(p)

    status["VLM"] = "ONLINE"
    write_status()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for proc in processes:
            try:
                proc.terminate()
            except:
                pass

        sys.exit(0)


if __name__ == "__main__":
    main()
