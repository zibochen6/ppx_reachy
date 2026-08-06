"""Safe lifecycle helpers for the project-owned Reachy daemon.

The daemon is an independently running process, so a PID alone is not an
ownership proof.  This module records a small, atomically-written manifest and
requires the live command line to match it before attempting termination.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def state_path(value: str | os.PathLike[str] | None = None) -> Path:
    path = Path(value or "state/daemon.json")
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def command_line(pid: int) -> str:
    """Return a process command line without raising on a stale PID."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def command_fingerprint(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def write_state(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    target = state_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def read_state(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(state_path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def remove_state(path: str | os.PathLike[str]) -> None:
    try:
        state_path(path).unlink()
    except FileNotFoundError:
        pass


def owned_process(state: dict[str, Any] | None) -> bool:
    """Validate PID, daemon identity, and recorded command fingerprint."""
    if not state or not isinstance(state.get("pid"), int):
        return False
    pid = int(state["pid"])
    if pid <= 0:
        return False
    live_command = command_line(pid)
    if "reachy-mini-daemon" not in live_command:
        return False
    recorded = str(state.get("command_fingerprint") or "")
    if recorded and recorded != command_fingerprint(live_command):
        return False
    serial = str(state.get("serial_port") or "")
    if serial and serial not in live_command:
        return False
    simulation = bool(state.get("simulation"))
    if simulation != ("--sim" in live_command):
        return False
    if bool(state.get("no_media")) != ("--no-media" in live_command):
        return False
    return True


def terminate_owned_state(path: str | os.PathLike[str]) -> bool:
    """Terminate a daemon only when the project manifest still owns it."""
    state = read_state(path)
    if not owned_process(state):
        return False
    pid = int(state["pid"])
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        remove_state(path)
        return True
    except OSError:
        return False
    remove_state(path)
    return True


def make_state(
    process: Any,
    command: list[str],
    *,
    host: str,
    port: int,
    serial_port: str = "",
    simulation: bool = False,
    no_media: bool = False,
) -> dict[str, Any]:
    command_text = command_line(int(process.pid)) or " ".join(command)
    return {
        "pid": int(process.pid),
        "started_at": time.time(),
        "host": host,
        "port": port,
        "serial_port": serial_port,
        "simulation": simulation,
        "no_media": no_media,
        "command": command_text,
        "command_fingerprint": command_fingerprint(command_text),
    }
