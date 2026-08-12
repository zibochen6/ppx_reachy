from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "reachy-service.sh"


def test_service_script_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_service_status_with_isolated_state_does_not_touch_processes(tmp_path) -> None:
    env = {
        **os.environ,
        "CHAIHUO_SERVICE_STATE_DIR": str(tmp_path),
        "CHAIHUO_SERVICE_HEALTH_URL": "http://127.0.0.1:1/healthz",
    }
    result = subprocess.run(
        ["bash", str(SCRIPT), "status"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == 1
    assert "服务未运行" in result.stdout
    assert not (tmp_path / "service.pid").exists()


def test_service_refuses_to_stop_pid_with_wrong_identity(tmp_path) -> None:
    process = subprocess.Popen(["sleep", "10"])
    try:
        (tmp_path / "service.pid").write_text(f"{process.pid}\n", encoding="utf-8")
        env = {
            **os.environ,
            "CHAIHUO_SERVICE_STATE_DIR": str(tmp_path),
            "CHAIHUO_SERVICE_HEALTH_URL": "http://127.0.0.1:1/healthz",
            "REACHY_DAEMON_STATE_FILE": str(tmp_path / "daemon.json"),
        }
        result = subprocess.run(
            ["bash", str(SCRIPT), "stop"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        assert result.returncode == 0
        assert process.poll() is None
        assert not (tmp_path / "service.pid").exists()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_service_start_and_graceful_stop_with_fake_dashboard(tmp_path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import http.server
import os
import sys

if len(sys.argv) > 1 and sys.argv[1] == "-c":
    raise SystemExit(1)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/healthz" else 404)
        self.end_headers()
    def log_message(self, *_args):
        pass

server = http.server.HTTPServer(("127.0.0.1", int(os.environ["REACHY_DASHBOARD_PORT"])), Handler)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    server.server_close()
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "CHAIHUO_SERVICE_STATE_DIR": str(tmp_path / "state"),
        "CHAIHUO_SERVICE_PYTHON": str(fake_python),
        "CHAIHUO_SERVICE_START_TIMEOUT_S": "5",
        "CHAIHUO_SERVICE_STOP_TIMEOUT_S": "5",
        "REACHY_DASHBOARD_PORT": str(port),
        "REACHY_DAEMON_STATE_FILE": str(tmp_path / "daemon.json"),
    }

    started = subprocess.run(
        ["bash", str(SCRIPT), "start"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    try:
        assert started.returncode == 0, started.stdout + started.stderr
        assert "服务启动成功" in started.stdout
        stopped = subprocess.run(
            ["bash", str(SCRIPT), "stop"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=12,
        )
        assert stopped.returncode == 0, stopped.stdout + stopped.stderr
        assert "服务已关闭" in stopped.stdout
        assert not (tmp_path / "state" / "service.pid").exists()
    finally:
        pid_file = tmp_path / "state" / "service.pid"
        if pid_file.exists():
            os.kill(int(pid_file.read_text().strip()), 9)
