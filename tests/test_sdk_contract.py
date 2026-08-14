from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[1]


def test_package_pins_sdk_and_has_only_one_public_cli() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["requires-python"] == ">=3.10,<3.11"
    assert "reachy-mini==1.9.0rc1" in metadata["project"]["dependencies"]
    assert "numpy>=2.2.5,<2.3" in metadata["project"]["dependencies"]
    assert "onnxruntime==1.23.2" in metadata["project"]["dependencies"]
    assert any(
        "torch-2.8.0-cp310-cp310-linux_aarch64.whl" in dependency
        and "sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"
        in dependency
        for dependency in metadata["project"]["optional-dependencies"]["jetson"]
    )
    assert metadata["project"]["scripts"] == {
        "chaihuo-reachy": "chaihuo_reachy.main:main"
    }
    assert "entry-points" not in metadata["project"]


def test_reachy_app_wrapper_is_removed() -> None:
    assert not (ROOT / "src/chaihuo_reachy/reachy_app.py").exists()


@pytest.mark.asyncio
async def test_daemon_spawn_timeout_polls_same_process_until_ready(monkeypatch) -> None:
    import reachy_mini
    from chaihuo_reachy import main as main_module
    from chaihuo_reachy.config import Config

    calls: list[dict] = []

    class FakeProcess:
        pid = 4242
        returncode = None

        def poll(self):
            return None

    class FakeReachy:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)
            # Existing-daemon probe fails; the readiness poll after the
            # explicitly owned SDK daemon starts succeeds.
            if len(calls) == 1:
                raise ConnectionError("still starting")
            self.media_manager = object()
            self.client = SimpleNamespace(host="localhost")

        def enable_motors(self) -> None:
            return None

        def wake_up(self) -> None:
            return None

    monkeypatch.setattr(reachy_mini, "ReachyMini", FakeReachy)
    process = FakeProcess()
    monkeypatch.setattr(
        main_module,
        "_spawn_sdk_daemon_process",
        lambda *_args: process,
    )
    monkeypatch.setattr(
        main_module,
        "_resolve_daemon_serial_port",
        lambda _cfg: "/dev/cu.usbmodem-test",
    )
    async def port_available(_cfg):
        return False
    monkeypatch.setattr(main_module, "_daemon_port_is_occupied", port_available)
    monkeypatch.setattr(
        "chaihuo_reachy.backends.factory.create_audio_backend",
        lambda _cfg, _manager: SimpleNamespace(backend_name="fake-audio"),
    )
    monkeypatch.setattr(
        "chaihuo_reachy.backends.factory.create_camera_backend",
        lambda _cfg, _manager: SimpleNamespace(backend_name="fake-camera"),
    )

    cfg = Config(
        daemon_host="localhost",
        media_backend="no_media",
        wobbling_enabled=False,
    )
    reachy, audio, camera, motion, beat_dance, status = await main_module._try_connect_daemon(cfg)

    assert reachy is not None
    assert audio.backend_name == "fake-audio"
    assert camera.backend_name == "fake-camera"
    assert motion is not None
    assert status["robot_ready"] is True
    assert reachy._chaihuo_daemon_process is process
    assert [call["spawn_daemon"] for call in calls] == [False, False]


@pytest.mark.asyncio
async def test_main_wake_enables_motors_before_wake_up() -> None:
    from chaihuo_reachy import main as main_module

    calls: list[str] = []
    reachy = SimpleNamespace(
        enable_motors=lambda: calls.append("enable_motors"),
        wake_up=lambda: calls.append("wake_up"),
    )

    assert await main_module._wake_up_reachy(reachy, attempts=1)
    assert calls == ["enable_motors", "wake_up"]


@pytest.mark.asyncio
async def test_owned_daemon_shutdown_lets_daemon_sleep_then_stops_process() -> None:
    from chaihuo_reachy import main as main_module
    from chaihuo_reachy.config import Config

    events: list[str] = []

    class FakeProcess:
        pid = 4242
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            events.append("terminate")

        def wait(self, _timeout=None) -> int:
            events.append("wait")
            self.returncode = 0
            return 0

        def kill(self) -> None:
            events.append("kill")
            self.returncode = -9

    class FakeMedia:
        def close(self) -> None:
            events.append("media_close")

    process = FakeProcess()
    reachy = SimpleNamespace(
        goto_sleep=lambda: events.append("sleep"),
        media_manager=FakeMedia(),
        _chaihuo_daemon_process=process,
    )

    assert await main_module._sleep_reachy_on_shutdown(reachy, Config())
    await main_module._close_reachy_runtime(reachy)

    # The daemon performs its own single goto_sleep during SIGTERM shutdown.
    # Calling through the client as well produces a duplicate stop prompt.
    assert events == ["terminate", "wait"]

    # The outer startup guard may run after run_dashboard's own finally.
    # Cleanup is intentionally idempotent.
    assert await main_module._sleep_reachy_on_shutdown(reachy, Config())
    await main_module._close_reachy_runtime(reachy)
    assert events == ["terminate", "wait"]


@pytest.mark.asyncio
async def test_concurrent_external_shutdown_requests_send_one_sleep_command() -> None:
    from chaihuo_reachy import main as main_module
    from chaihuo_reachy.config import Config

    calls: list[str] = []

    def goto_sleep() -> None:
        calls.append("sleep")
        time.sleep(0.02)

    reachy = SimpleNamespace(goto_sleep=goto_sleep)
    await asyncio.gather(
        main_module._sleep_reachy_on_shutdown(reachy, Config()),
        main_module._sleep_reachy_on_shutdown(reachy, Config()),
    )

    assert calls == ["sleep"]


@pytest.mark.asyncio
async def test_dashboard_initialization_error_still_cleans_owned_daemon(
    monkeypatch,
) -> None:
    from chaihuo_reachy import main as main_module
    from chaihuo_reachy.config import Config

    events: list[str] = []

    class FakeProcess:
        pid = 4242
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            events.append("terminate")

        def wait(self, _timeout=None) -> int:
            events.append("wait")
            self.returncode = 0
            return 0

        def kill(self) -> None:
            raise AssertionError("graceful termination should succeed")

    class FakeMedia:
        def close(self) -> None:
            events.append("media_close")

    reachy = SimpleNamespace(
        goto_sleep=lambda: events.append("sleep"),
        media_manager=FakeMedia(),
        _chaihuo_daemon_process=FakeProcess(),
    )

    async def fake_connect(_cfg):
        return (
            reachy,
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            None,  # beat_dance
            {"sdk_connected": True},
        )

    async def broken_dashboard(*_args, **_kwargs):
        raise RuntimeError("audio initialization failed")

    monkeypatch.setattr(main_module, "_try_connect_daemon", fake_connect)
    monkeypatch.setattr(main_module, "run_dashboard", broken_dashboard)

    with pytest.raises(RuntimeError, match="audio initialization failed"):
        await main_module._start_dashboard(Config())

    assert events == ["terminate", "wait"]
