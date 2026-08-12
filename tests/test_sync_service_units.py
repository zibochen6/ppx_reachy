"""Portable text-level checks of the systemd units (no systemd needed).

Parses the unit files in deploy/ and asserts the schedule / command /
user settings that make the journal auto-sync service stable and
unattended.  Runs on macOS too.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_UNIT = REPO_ROOT / "deploy" / "chaihuo-journal-sync.service"
TIMER_UNIT = REPO_ROOT / "deploy" / "chaihuo-journal-sync.timer"


def _ini_section(text: str, section: str) -> dict[str, str]:
    result: dict[str, str] = {}
    in_section = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_section = line[1:-1] == section
            continue
        if in_section and "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def test_service_unit_declares_oneshot_and_sync_command() -> None:
    text = SERVICE_UNIT.read_text(encoding="utf-8")
    service = _ini_section(text, "Service")
    assert service["Type"] == "oneshot"  # timer retries next tick, no Restart loop
    assert service["User"] == "recomputer"
    assert service["WorkingDirectory"] == "/home/recomputer/chaihuo_reachy"
    assert service["ExecStart"].endswith("chaihuo-reachy sync-journals")
    assert "Restart=" not in service  # 401 entries must not trigger restart loops
    assert int(service["TimeoutStartSec"]) >= 900  # first run downloads the corpus


def test_timer_unit_declares_boot_and_periodic_schedule() -> None:
    text = TIMER_UNIT.read_text(encoding="utf-8")
    timer = _ini_section(text, "Timer")
    install = _ini_section(text, "Install")
    assert timer["OnBootSec"] == "2min"
    assert timer["OnUnitActiveSec"] == "4h"
    assert timer["Persistent"] == "true"  # missed ticks catch up at next boot
    assert install["WantedBy"] == "timers.target"
