from __future__ import annotations

import asyncio
import platform
import time

import cv2
import numpy as np
import pytest

from chaihuo_reachy.config import Config
from chaihuo_reachy.engine import ConversationEngine
from chaihuo_reachy.main import _MJPEGStream
from chaihuo_reachy.camera import _avfoundation_video_devices, find_reachy_camera


class FakeCamera:
    def __init__(self) -> None:
        self.capture_count = 0
        self.closed = False

    @property
    def is_active(self) -> bool:
        return not self.closed

    @property
    def backend_name(self) -> str:
        return "fake-camera"

    def capture_jpeg(self, quality: int = 85) -> bytes:
        self.capture_count += 1
        return b"\xff\xd8" + str(self.capture_count).encode()

    def close(self) -> None:
        self.closed = True


def test_macos_camera_resolution_never_falls_back_to_an_index(monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        "chaihuo_reachy.camera._find_named_reachy_camera_macos", lambda: None
    )
    with pytest.raises(RuntimeError, match="拒绝回退到 Mac 前置摄像头"):
        find_reachy_camera("auto")


def test_macos_camera_resolution_uses_exact_reachy_name(monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        "chaihuo_reachy.camera._find_named_reachy_camera_macos",
        lambda: "avfoundation:Reachy Mini Camera",
    )
    assert find_reachy_camera("auto") == "avfoundation:Reachy Mini Camera"


def test_avfoundation_parser_maps_exact_video_names(monkeypatch) -> None:
    stderr = """[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] Reachy Mini Camera
[AVFoundation indev @ 0x1] [1] MacBook Pro的相机
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Pro麦克风
"""
    monkeypatch.setattr("chaihuo_reachy.camera.shutil.which", lambda _: "ffmpeg")
    monkeypatch.setattr(
        "chaihuo_reachy.camera.subprocess.run",
        lambda *args, **kwargs: type("Result", (), {"stderr": stderr})(),
    )
    assert _avfoundation_video_devices() == {
        "Reachy Mini Camera": 0,
        "MacBook Pro的相机": 1,
    }


@pytest.mark.asyncio
async def test_camera_service_supplies_a_frame_newer_than_request() -> None:
    camera = FakeCamera()
    stream = _MJPEGStream(camera_backend=camera, fps=30)
    assert stream.start() is True
    try:
        await asyncio.sleep(0.04)
        before = camera.capture_count
        frame = await stream.capture_fresh(timeout_s=0.3)
        assert frame is not None
        assert camera.capture_count > before
        assert stream.frame_age_s is not None
    finally:
        stream.stop()


@pytest.mark.asyncio
async def test_engine_does_not_reopen_or_bypass_shared_camera_service() -> None:
    camera = FakeCamera()
    engine = ConversationEngine(Config(), camera_backend=camera)

    async def unavailable_service() -> None:
        return None

    engine.set_camera_snapshot_provider(unavailable_service)
    result = await engine._tool_take_photo()
    assert "暂时看不到前面" in result
    assert "画面" not in result
    assert camera.capture_count == 0


@pytest.mark.asyncio
async def test_vision_question_captures_and_rejects_a_black_frame() -> None:
    engine = ConversationEngine(Config())
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    encoded, jpeg = cv2.imencode(".jpg", black)
    assert encoded
    captures = 0

    async def capture_black_frame() -> bytes:
        nonlocal captures
        captures += 1
        return jpeg.tobytes()

    engine.set_camera_snapshot_provider(capture_black_frame)
    result = await engine.process_text("你能看到什么？")

    assert captures == 1
    assert result["intent"] == "front_camera"
    assert "光线太暗" in result["reply"]
    assert all(word not in result["reply"] for word in ("拍照", "照片", "画面", "摄像头"))
