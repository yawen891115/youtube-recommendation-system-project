"""Start temporary local services, exercise them, and shut them down."""

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from smoke_test import get, run


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    ports = [free_port() for _ in range(3)]
    api, ui, tracking = ports
    environment = os.environ | {
        "API_BASE_URL": f"http://127.0.0.1:{api}",
        "MLFLOW_DISABLE_TELEMETRY": "true",
        "MLFLOW_DISABLE_AGENT_HINT": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }
    commands = [
        [sys.executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", str(api)],
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.address=127.0.0.1",
            f"--server.port={ui}",
            "--browser.gatherUsageStats=false",
        ],
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            "sqlite:///experiments/mlflow.db",
            "--host",
            "127.0.0.1",
            "--port",
            str(tracking),
        ],
    ]
    processes = []
    with tempfile.TemporaryDirectory() as directory:
        logs = []
        try:
            for index, command in enumerate(commands):
                log = open(Path(directory) / f"service-{index}.log", "w+")
                logs.append(log)
                processes.append(
                    subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                )
            for port, path in [(api, "/health"), (ui, "/_stcore/health"), (tracking, "/health")]:
                deadline = time.monotonic() + 60
                while True:
                    try:
                        get(f"http://127.0.0.1:{port}{path}")
                        break
                    except (OSError, ValueError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                f"Service on port {port} failed to become healthy"
                            ) from None
                        time.sleep(0.25)
            run(api=f"http://127.0.0.1:{api}")
            assert "<html" in get(f"http://127.0.0.1:{ui}").lower()
            experiments = get(
                f"http://127.0.0.1:{tracking}/api/2.0/mlflow/experiments/search",
                {"max_results": 20},
            )
            assert any(
                row["name"] == "youtube-shorts-chronological" for row in experiments["experiments"]
            )
            print("Local API, Streamlit page and MLflow experiment database: passed")
        except Exception:
            for log in logs:
                log.flush()
                log.seek(0)
                print(log.read()[-6000:])
            raise
        finally:
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()
