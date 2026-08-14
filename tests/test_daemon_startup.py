from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from chaihuo_reachy import main as main_module
from chaihuo_reachy.config import Config


def test_shutdown_backstop_force_exits_once_after_delay(monkeypatch) -> None:
    """Ctrl+C must never leave the XMOS/port occupied: the backstop
    daemon thread force-exits the process if cleanup hangs."""
    calls: list[int] = []
    monkeypatch.setattr(
        main_module, "_SHUTDOWN_BACKSTOP_ARMED", threading.Event()
    )
    main_module._arm_shutdown_backstop(delay_s=0.05, force_exit=calls.append)
    # Idempotent: a second arm does not start another force-exit.
    main_module._arm_shutdown_backstop(delay_s=0.05, force_exit=calls.append)
    time.sleep(0.15)
    assert calls == [1]


def test_startup_clears_any_persisted_official_app(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        "reachy_mini.daemon.startup_app_config.set_startup_app", calls.append
    )
    main_module._clear_persisted_startup_app()
    assert calls == [None]


def test_spawn_uses_real_hardware_mode_by_default(monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(main_module, "_clear_persisted_startup_app", lambda: None)
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/reachy-mini-daemon")
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda command, **_kwargs: commands.append(command) or SimpleNamespace(),
    )

    main_module._spawn_sdk_daemon_process(Config(daemon_serial_port="/dev/cu.usbmodem-test"))
    main_module._spawn_sdk_daemon_process(Config(daemon_simulation=True))

    assert commands == [
        [
            "/usr/local/bin/reachy-mini-daemon",
            "--autostart",
            "--headless",
            "--no-wake-up-on-start",
            "--goto-sleep-on-stop",
            "--serialport",
            "/dev/cu.usbmodem-test",
        ],
        [
            "/usr/local/bin/reachy-mini-daemon",
            "--autostart",
            "--headless",
            "--no-wake-up-on-start",
            "--goto-sleep-on-stop",
            "--sim",
        ],
    ]


def test_spawn_passes_no_media_to_prevent_direct_backend_conflict(monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(main_module, "_clear_persisted_startup_app", lambda: None)
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/reachy-mini-daemon")
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda command, **_kwargs: commands.append(command) or SimpleNamespace(),
    )

    main_module._spawn_sdk_daemon_process(
        Config(media_backend="no_media", daemon_serial_port="/dev/cu.usbmodem-test")
    )

    assert commands[0][-1] == "--no-media"


def test_spawn_finds_daemon_beside_venv_python_when_path_is_minimal(
    monkeypatch, tmp_path
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    real_python = tmp_path / "python3-real"
    real_python.touch()
    python = bin_dir / "python3"
    daemon = bin_dir / "reachy-mini-daemon"
    python.symlink_to(real_python)
    daemon.touch(mode=0o755)
    commands: list[list[str]] = []
    monkeypatch.setattr(main_module, "_clear_persisted_startup_app", lambda: None)

    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(main_module.sys, "executable", str(python))
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda command, **_kwargs: commands.append(command) or SimpleNamespace(),
    )

    main_module._spawn_sdk_daemon_process(Config(daemon_simulation=True))

    assert commands == [[
        str(daemon),
        "--autostart",
        "--headless",
        "--no-wake-up-on-start",
        "--goto-sleep-on-stop",
        "--sim",
    ]]


def test_resolve_serial_recovers_only_unique_candidate(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "missing"
    candidate = tmp_path / "cu.usbmodem-unique"
    candidate.touch()
    monkeypatch.setattr(main_module.Path, "glob", lambda _self, _pattern: [candidate])

    assert main_module._resolve_daemon_serial_port(
        Config(daemon_serial_port=str(missing))
    ) == str(candidate)


def test_resolve_serial_rejects_ambiguous_candidates(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "missing"
    first = tmp_path / "cu.usbmodem-a"
    second = tmp_path / "cu.usbmodem-b"
    monkeypatch.setattr(main_module.Path, "glob", lambda _self, _pattern: [first, second])

    with pytest.raises(RuntimeError, match="多个候选串口"):
        main_module._resolve_daemon_serial_port(Config(daemon_serial_port=str(missing)))


@pytest.mark.asyncio
async def test_auto_does_not_replace_unhealthy_external_daemon(monkeypatch) -> None:
    async def unhealthy(*_args, **_kwargs):
        raise main_module._DaemonStartupError("机器人硬件未就绪")

    monkeypatch.setattr(main_module, "_find_reachy_daemon", unhealthy)
    monkeypatch.setattr(
        main_module,
        "_spawn_sdk_daemon_process",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )

    result = await main_module._try_connect_daemon(
        Config(daemon_mode="auto", daemon_host="localhost")
    )

    assert result[:4] == (None, None, None, None)
    assert result[5]["robot_status"] == "degraded"
    assert result[5]["daemon_error"] == "机器人硬件未就绪"


@pytest.mark.asyncio
async def test_connect_mode_never_spawns(monkeypatch) -> None:
    async def missing(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main_module, "_find_reachy_daemon", missing)
    monkeypatch.setattr(
        main_module,
        "_spawn_sdk_daemon_process",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )

    result = await main_module._try_connect_daemon(
        Config(daemon_mode="connect", daemon_host="localhost")
    )

    assert result[5]["daemon_error"] == "connect 模式下未发现健康 daemon"


@pytest.mark.asyncio
async def test_auto_degrades_when_an_external_process_owns_daemon_port(monkeypatch) -> None:
    async def missing(*_args, **_kwargs):
        return None

    async def occupied(_cfg):
        return True

    monkeypatch.setattr(main_module, "_find_reachy_daemon", missing)
    monkeypatch.setattr(main_module, "_daemon_port_is_occupied", occupied)
    monkeypatch.setattr(
        main_module,
        "_spawn_sdk_daemon_process",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )

    result = await main_module._try_connect_daemon(
        Config(daemon_mode="auto", daemon_host="localhost")
    )

    assert "端口已被外部或异常进程占用" in result[5]["daemon_error"]


@pytest.mark.asyncio
async def test_running_daemon_without_ready_backend_is_rejected(monkeypatch) -> None:
    import httpx

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"state": "running", "backend_status": {"ready": False}}

    class Client:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, _url):
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", Client)

    error = await main_module._daemon_backend_error(
        "localhost", 8000, Config(daemon_simulation=False)
    )

    assert "ready=false" in error


@pytest.mark.asyncio
async def test_spawned_daemon_waits_through_transient_not_ready_state(monkeypatch) -> None:
    class Process:
        returncode = None

        def poll(self):
            return None

    reachy = object()
    health_results = iter(["机器人硬件未就绪 (backend_status.ready=false)", ""])

    connect_calls = 0

    async def connect(*_args, **_kwargs):
        nonlocal connect_calls
        connect_calls += 1
        return reachy

    async def health(*_args, **_kwargs):
        return next(health_results)

    monkeypatch.setattr(main_module, "_connect_reachy_once", connect)
    monkeypatch.setattr(main_module, "_daemon_backend_error", health)
    monkeypatch.setattr(main_module, "_DAEMON_POLL_INTERVAL_S", 0.0)

    connected = await main_module._find_reachy_daemon(
        object,
        Config(daemon_host="localhost"),
        timeout_s=0.1,
        daemon_process=Process(),
    )

    assert connected is reachy
    assert connect_calls == 1


@pytest.mark.asyncio
async def test_live_controller_is_accepted_when_sdk_ready_flag_is_stale(monkeypatch) -> None:
    import httpx

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "state": "running",
                "simulation_enabled": False,
                "backend_status": {
                    "ready": False,
                    "error": None,
                    "motor_control_mode": "enabled",
                    "control_loop_stats": {
                        "mean_control_loop_frequency": 48.0,
                        "nb_error": 0,
                    },
                },
            }

    class Client:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, _url):
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", Client)

    assert await main_module._daemon_backend_error(
        "localhost", 8000, Config(daemon_simulation=False)
    ) == ""


@pytest.mark.asyncio
async def test_ctrl_c_during_startup_stops_the_owned_daemon(monkeypatch, tmp_path) -> None:
    events: list[str] = []

    class Process:
        pid = 4321
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("terminate")

        def wait(self, _timeout=None):
            events.append("wait")
            self.returncode = 0
            return 0

    process = Process()
    started = asyncio.Event()

    async def blocked_start(cfg):
        setattr(cfg, "_chaihuo_starting_daemon_process", process)
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(main_module, "_try_connect_daemon_impl", blocked_start)
    cfg = Config(daemon_state_file=str(tmp_path / "daemon.json"))
    task = asyncio.create_task(main_module._try_connect_daemon(cfg))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["terminate", "wait"]
