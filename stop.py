import subprocess


def kill_process_by_arg(arg):
    cmd = f"wmic process where \"commandline like '%{arg}%' and name like '%python%'\" call terminate"
    subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True
    )


def main():

    kill_process_by_arg("unified_server.py")
    kill_process_by_arg("edge_pipeline.py")
    kill_process_by_arg("bridge.py")
    kill_process_by_arg("blockchain_auditor.py")

    subprocess.run(
        "taskkill /IM node.exe /F",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True,
    )

    try:
        import json
        from pathlib import Path

        status_file = Path(__file__).parent / "system_status.json"
        if status_file.exists():
            status = {
                "PostgreSQL": "OFFLINE",
                "CCTV Edge": "OFFLINE",
                "Bridge": "OFFLINE",
                "VLM": "OFFLINE",
                "FastAPI": "OFFLINE",
                "React": "OFFLINE",
                "Blockchain Auditor": "OFFLINE",
            }
            with open(status_file, "w") as f:
                json.dump(status, f)
    except:
        pass


if __name__ == "__main__":
    main()
