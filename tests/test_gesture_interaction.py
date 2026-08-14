from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaihuo_reachy.gesture_interaction import GestureInteractionController
from chaihuo_reachy.hand_pose import HandLandmark, HandPoseResult


OPEN = [
    (.50, .78),
    (.42, .68), (.35, .61), (.28, .54), (.21, .47),
    (.40, .59), (.39, .46), (.38, .33), (.37, .20),
    (.48, .56), (.48, .42), (.48, .28), (.48, .14),
    (.56, .58), (.57, .45), (.58, .32), (.59, .19),
    (.64, .62), (.67, .51), (.70, .40), (.73, .29),
]


def result(points: list[tuple[float, float]]) -> HandPoseResult:
    return HandPoseResult(tuple(HandLandmark(x, y, .99) for x, y in points), backend="fake")


def fist() -> list[tuple[float, float]]:
    points = list(OPEN)
    for chain in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        mcp = points[chain[0]]
        points[chain[1]] = (mcp[0], mcp[1] - .08)
        points[chain[2]] = (mcp[0] + .07, mcp[1] - .06)
        points[chain[3]] = (.51, .61)
    return points


class Frames:
    def get_bgr_frame(self):
        return np.zeros((224, 224, 3), dtype=np.uint8)


class Motion:
    def __init__(self) -> None:
        self.targets: list[dict] = []
        self.neutral_count = 0

    def set_realtime_target(self, **target) -> None:
        self.targets.append(target)

    async def return_to_neutral(self, duration: float) -> None:
        self.neutral_count += 1


class Audio:
    def __init__(self) -> None:
        self.stops = 0
        self.played = 0

    def stop_playback(self) -> None:
        self.stops += 1

    def set_output_sample_rate(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    async def play(self, pcm: bytes) -> None:
        self.played += len(pcm)


class CountingRng:
    def __init__(self) -> None:
        self.calls = 0

    def choice(self, values):
        self.calls += 1
        return values[0]


def write_music(path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\0\0" * 800)


@pytest.mark.asyncio
async def test_debounce_single_trigger_preemption_and_lost_timeout(tmp_path: Path) -> None:
    write_music(tmp_path / "happy.wav")
    config = SimpleNamespace(
        gesture_keypoint_confidence=.15,
        gesture_confirmation_ms=300,
        gesture_lost_timeout_ms=800,
        gesture_tracking_alpha=.35,
        gesture_tracking_deadzone=.06,
        gesture_head_yaw_max_deg=20,
        gesture_head_pitch_max_deg=20,
        gesture_body_yaw_max_deg=30,
        gesture_tracking_max_step_deg=2,
        dance_music_dir=str(tmp_path),
    )
    motion, audio, rng = Motion(), Audio(), CountingRng()
    controller = GestureInteractionController(
        config,
        frame_source=Frames(),
        motion=motion,
        audio=audio,
        pose_factory=lambda **_: np.eye(4),
        rng=rng,  # type: ignore[arg-type]
    )
    controller._state = "SEARCHING"

    await controller._handle_pose(result(OPEN), 1.0)
    assert controller.state == "SEARCHING"
    await controller._handle_pose(result(OPEN), 1.31)
    assert controller.state == "TRACKING"

    await controller._handle_pose(result(fist()), 1.4)
    await controller._handle_pose(result(fist()), 1.71)
    assert controller.state == "DANCING"
    assert rng.calls == 1
    await asyncio.sleep(.03)
    assert controller._dance_task is not None

    await controller._handle_pose(result(fist()), 2.2)
    assert rng.calls == 1

    await controller._handle_pose(result(OPEN), 2.3)
    await controller._handle_pose(result(OPEN), 2.61)
    assert controller.state == "TRACKING"
    assert controller._dance_task is None
    assert audio.stops >= 1

    partial = list(fist())
    partial[5:9] = OPEN[5:9]
    await controller._handle_pose(result(partial), 3.0)
    assert controller.state == "TRACKING"
    await controller._handle_pose(result(partial), 3.5)
    assert controller.state == "LOST"
    assert motion.neutral_count == 1
