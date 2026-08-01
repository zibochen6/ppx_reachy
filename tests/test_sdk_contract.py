from __future__ import annotations

import inspect
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]


def test_package_declares_sdk_dependency_and_app_entrypoint() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert any(dep.startswith("reachy-mini>=1.8.0") for dep in metadata["project"]["dependencies"])
    assert metadata["project"]["entry-points"]["reachy_mini_apps"]["chaihuo_reachy"].endswith(
        ":ChaihuoReachyApp"
    )


def test_sdk_app_contract_when_dependency_is_installed() -> None:
    pytest.importorskip("reachy_mini")
    from reachy_mini import ReachyMiniApp
    from chaihuo_reachy.reachy_app import ChaihuoReachyApp

    assert issubclass(ChaihuoReachyApp, ReachyMiniApp)
    assert ChaihuoReachyApp.request_media_backend in ("no_media", "local")
    assert list(inspect.signature(ChaihuoReachyApp.run).parameters) == [
        "self",
        "reachy_mini",
        "stop_event",
    ]


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

        def wake_up(self) -> None:
            return None

    monkeypatch.setattr(reachy_mini, "ReachyMini", FakeReachy)
    process = FakeProcess()
    monkeypatch.setattr(
        main_module,
        "_spawn_sdk_daemon_process",
        lambda: process,
    )
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
    reachy, audio, camera, motion, status = await main_module._try_connect_daemon(cfg)

    assert reachy is not None
    assert audio.backend_name == "fake-audio"
    assert camera.backend_name == "fake-camera"
    assert motion is not None
    assert status["robot_ready"] is True
    assert reachy._chaihuo_daemon_process is process
    assert [call["spawn_daemon"] for call in calls] == [False, False]


@pytest.mark.asyncio
async def test_owned_daemon_shutdown_sleeps_robot_then_stops_process() -> None:
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

    assert events == ["sleep", "media_close", "terminate", "wait"]

    # The outer startup guard may run after run_dashboard's own finally.
    # Cleanup is intentionally idempotent.
    assert await main_module._sleep_reachy_on_shutdown(reachy, Config())
    await main_module._close_reachy_runtime(reachy)
    assert events == ["sleep", "media_close", "terminate", "wait"]


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
            {"sdk_connected": True},
        )

    async def broken_dashboard(*_args, **_kwargs):
        raise RuntimeError("audio initialization failed")

    monkeypatch.setattr(main_module, "_try_connect_daemon", fake_connect)
    monkeypatch.setattr(main_module, "run_dashboard", broken_dashboard)

    with pytest.raises(RuntimeError, match="audio initialization failed"):
        await main_module._start_dashboard(Config())

    assert events == ["sleep", "media_close", "terminate", "wait"]
