"""Reachy Mini daemon application shell.

The SDK owns daemon/robot lifecycle and media (camera + audio via GStreamer).
On startup the robot wakes up (stands up, spreads antennas), on shutdown it
goes to sleep. Motion control (dance, gestures, poses) is exposed through the
Dashboard WebSocket and LLM function calling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading

from reachy_mini import ReachyMini, ReachyMiniApp

from chaihuo_reachy.config import load_config
from chaihuo_reachy.main import run_dashboard, setup_logging

logger = logging.getLogger("chaihuo_reachy.sdk")


class ChaihuoReachyApp(ReachyMiniApp):
    """Daemon-discoverable production entry point.

    Uses LOCAL media backend so the SDK's MediaManager provides camera
    frames (via daemon IPC) and audio capture/playback (via GStreamer).
    """

    request_media_backend: str | None = "local"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Called by the SDK framework after daemon connection is established.

        Args:
            reachy_mini: Connected robot instance with media_manager ready.
            stop_event: Set by the framework when the app should shut down.
        """
        config = load_config(os.environ.get("CHAIHUO_CONFIG"))

        # ── Step 1: Wake up the robot ──────────────────────────────
        if config.auto_wake_up:
            logger.info("🤖 小柴正在站起来...")
            try:
                reachy_mini.wake_up()
                logger.info("✅ 小柴已就绪 — 头部归位，天线展开")
            except Exception:
                logger.warning("⚠️  wake_up 失败，继续启动（可能无硬件连接）")

        # ── Step 2: Build SDK status for Dashboard ────────────────
        status: dict[str, object] = {
            "sdk_connected": True,
            "mode": "reachy_daemon_app",
            "media_backend": "sdk_gstreamer",
            "robot_ready": True,
            "daemon_host": os.environ.get("REACHY_DAEMON_HOST", "reachy-mini.local"),
            "daemon_port": int(os.environ.get("REACHY_DAEMON_PORT", "8000")),
            "robot_class": type(reachy_mini).__name__,
        }
        logger.info("Reachy SDK connected: %s", status)

        # ── Step 3: Run the Dashboard + engine ────────────────────
        try:
            asyncio.run(_run_app(config, reachy_mini, status, stop_event))
        except KeyboardInterrupt:
            logger.info("收到中断信号")

        # ── Step 4: Clean shutdown ────────────────────────────────
        finally:
            if config.auto_sleep:
                logger.info("🛌 小柴准备休眠...")
                try:
                    reachy_mini.goto_sleep()
                    logger.info("😴 小柴已休眠")
                except Exception:
                    logger.warning("⚠️  goto_sleep 失败")
            reachy_mini.media_manager.close()
            logger.info("SDK media manager closed")


async def _run_app(
    config,
    reachy_mini: ReachyMini,
    status: dict[str, object],
    stop_event: threading.Event,
) -> None:
    """Async entry point — wires up backends and starts the Dashboard."""
    from chaihuo_reachy.backends.factory import (
        create_audio_backend,
        create_camera_backend,
    )
    from chaihuo_reachy.motion import MotionController

    # Create SDK backends
    audio_backend = create_audio_backend(config, reachy_mini.media_manager)
    camera_backend = create_camera_backend(config, reachy_mini.media_manager)
    motion = MotionController(reachy_mini, config) if config.dance_enabled else None

    logger.info(
        "Backends: audio=%s, camera=%s, motion=%s",
        type(audio_backend).__name__,
        type(camera_backend).__name__,
        type(motion).__name__ if motion else "disabled",
    )

    await run_dashboard(
        config,
        sdk_status=status,
        stop_event=stop_event,
        audio_backend=audio_backend,
        camera_backend=camera_backend,
        reachy=reachy_mini,
        motion=motion,
    )


def main() -> None:
    """CLI entry point: connect to daemon outside app discovery."""
    setup_logging(os.environ.get("REACHY_LOG_LEVEL", "INFO").upper() == "DEBUG")
    app = ChaihuoReachyApp()
    try:
        app.wrapped_run(
            host=os.environ.get("REACHY_DAEMON_HOST", "reachy-mini.local"),
            port=int(os.environ.get("REACHY_DAEMON_PORT", "8000")),
        )
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
