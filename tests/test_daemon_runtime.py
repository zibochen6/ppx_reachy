from __future__ import annotations

from chaihuo_reachy import daemon_runtime


def test_state_round_trip_and_validated_ownership(monkeypatch, tmp_path) -> None:
    state_file = tmp_path / "state" / "daemon.json"
    command = "/opt/bin/reachy-mini-daemon --serialport /dev/cu.usbmodem1 --no-media"
    monkeypatch.setattr(daemon_runtime, "command_line", lambda _pid: command)
    payload = {
        "pid": 1234,
        "serial_port": "/dev/cu.usbmodem1",
        "simulation": False,
        "no_media": True,
        "command_fingerprint": daemon_runtime.command_fingerprint(command),
    }

    daemon_runtime.write_state(state_file, payload)

    assert daemon_runtime.read_state(state_file) == payload
    assert daemon_runtime.owned_process(payload)


def test_mismatched_command_is_never_owned(monkeypatch) -> None:
    monkeypatch.setattr(
        daemon_runtime,
        "command_line",
        lambda _pid: "/opt/bin/reachy-mini-daemon --sim",
    )
    state = {
        "pid": 1234,
        "simulation": False,
        "no_media": False,
        "command_fingerprint": daemon_runtime.command_fingerprint("different"),
    }

    assert not daemon_runtime.owned_process(state)
