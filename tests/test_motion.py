"""Tests for MotionController fault tolerance (SDK interpolation race)."""

from __future__ import annotations

import math
import time

import pytest

from chaihuo_reachy.motion import MotionController


class _FlakyReachy:
    """goto_target that fails the first N calls with the SDK race error."""

    def __init__(self, fail_first: int = 1) -> None:
        self.calls = 0
        self.fail_first = fail_first
        self.set_targets: list[dict] = []
        self.goto_targets: list[dict] = []
        self.client = _FakeClient()

    def goto_target(self, **kwargs) -> None:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise Exception("Task failed with error: time value is out of range [0,1]")
        # Real SDK calls block until their trajectory completes.  A tiny wait
        # keeps the talk-motion worker realistic without slowing unit tests.
        time.sleep(min(float(kwargs.get("duration", 0.0)), 0.01))
        self.goto_targets.append(kwargs)

    def set_target(self, **kwargs) -> None:
        self.set_targets.append(kwargs)


class _FakeClient:
    def __init__(self, *, delay_s: float = 0.0, error: str = "") -> None:
        self.commands = []
        self.delay_s = delay_s
        self.error = error

    def send_command(self, command) -> None:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error:
            raise RuntimeError(self.error)
        self.commands.append(command)


@pytest.mark.asyncio
async def test_safe_goto_retries_once_and_succeeds() -> None:
    reachy = _FlakyReachy(fail_first=1)
    motion = MotionController(reachy)  # type: ignore[arg-type]
    ok = await motion._safe_goto(head=None, duration=0.5)
    assert ok is True
    assert reachy.calls == 2  # first failed, retry succeeded


@pytest.mark.asyncio
async def test_safe_goto_gives_up_and_reports_skipped() -> None:
    reachy = _FlakyReachy(fail_first=99)  # always fails
    motion = MotionController(reachy)  # type: ignore[arg-type]
    ok = await motion._safe_goto(head=None, duration=0.5)
    assert ok is False
    assert reachy.calls == 2  # original + one retry


@pytest.mark.asyncio
async def test_safe_goto_propagates_unrelated_errors() -> None:
    class _BrokenReachy:
        def goto_target(self, **kwargs) -> None:
            raise RuntimeError("motor disconnected")

    motion = MotionController(_BrokenReachy())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="motor disconnected"):
        await motion._safe_goto(head=None, duration=0.5)


@pytest.mark.asyncio
async def test_dance_continues_and_counts_skipped_steps() -> None:
    # Fail every goto: all choreography steps + the neutral pose are skipped.
    reachy = _FlakyReachy(fail_first=99)
    motion = MotionController(reachy)  # type: ignore[arg-type]
    summary = await motion.dance("happy", beat_s=0.1)
    assert summary["style"] == "happy"
    assert summary["skipped"] >= 9  # 8 choreography steps + 1 neutral pose


@pytest.mark.asyncio
async def test_dance_succeeds_without_skips_when_reachy_is_healthy() -> None:
    reachy = _FlakyReachy(fail_first=0)
    motion = MotionController(reachy)  # type: ignore[arg-type]
    summary = await motion.dance("happy", beat_s=0.1)
    assert summary["skipped"] == 0
    assert summary["duration"] >= 0


@pytest.mark.asyncio
async def test_dance_respects_duration_limit() -> None:
    reachy = _FlakyReachy(fail_first=0)
    motion = MotionController(reachy)  # type: ignore[arg-type]
    summary = await motion.dance("happy", duration_s=0.4, beat_s=0.1)
    assert summary["duration"] < 1.0  # stopped near the requested length


@pytest.mark.asyncio
async def test_dance_random_selects_a_known_style() -> None:
    reachy = _FlakyReachy(fail_first=0)
    motion = MotionController(reachy)  # type: ignore[arg-type]
    summary = await motion.dance("random", beat_s=0.1)
    assert summary["style"] in {"happy", "swing", "robot"}


# ── Continuous speech offsets during actual speaker playback ───────────


def _speech_pcm(seconds: float = 0.5, sr: int = 16_000) -> bytes:
    samples = int(seconds * sr)
    return b"".join(
        int(10_000 * math.sin(2 * math.pi * 220 * index / sr)).to_bytes(
            2, "little", signed=True
        )
        for index in range(samples)
    )


def _wait_stopped(motion: MotionController, timeout: float = 1.5) -> None:
    deadline = time.monotonic() + timeout
    while motion.is_talk_shaking and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not motion.is_talk_shaking


def test_talk_motion_uses_speech_offsets_and_never_goto_target() -> None:
    reachy = _FlakyReachy(fail_first=0)
    motion = MotionController(reachy)  # type: ignore[arg-type]
    motion.start_talk_motion()
    motion.feed_talk_audio(_speech_pcm(), 16_000)
    deadline = time.monotonic() + 1.0
    while len(reachy.client.commands) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    motion.stop_talk_motion(immediate=True)
    _wait_stopped(motion)

    assert motion.is_wobbling  # self-healed at speech start
    assert not reachy.goto_targets
    # Importing the full SDK protocol package can take most of this timeout on
    # Jetson; two real-time frames are sufficient to prove the offsets path.
    assert len(reachy.client.commands) >= 2
    nonzero = [cmd.offsets for cmd in reachy.client.commands if any(cmd.offsets)]
    assert nonzero
    assert max(abs(values[5]) for values in nonzero) <= math.radians(20.0) * 1.1
    assert max(abs(values[4]) for values in nonzero) <= math.radians(4.0)
    assert max(abs(values[3]) for values in nonzero) <= math.radians(3.0)
    assert reachy.client.commands[-1].offsets == [0.0] * 6


def test_talk_motion_graceful_stop_eases_to_exact_zero() -> None:
    reachy = _FlakyReachy(fail_first=0)
    motion = MotionController(reachy)  # type: ignore[arg-type]
    motion.start_talk_motion()
    motion.feed_talk_audio(_speech_pcm(0.25), 16_000)
    time.sleep(0.18)
    before = len(reachy.client.commands)
    motion.stop_talk_motion()
    _wait_stopped(motion)

    tail = [cmd.offsets for cmd in reachy.client.commands[before:]]
    assert len(tail) >= 5
    assert tail[-1] == [0.0] * 6
    assert motion.talk_motion_envelope == 0.0


def test_talk_motion_queue_drops_oldest_without_blocking_producer() -> None:
    reachy = _FlakyReachy(fail_first=0)
    reachy.client = _FakeClient(delay_s=0.08)
    motion = MotionController(reachy)  # type: ignore[arg-type]
    motion.start_talk_motion()
    started = time.monotonic()
    for _ in range(40):
        motion.feed_talk_audio(_speech_pcm(0.05), 16_000)
    elapsed = time.monotonic() - started
    motion.stop_talk_motion(immediate=True)
    _wait_stopped(motion)

    assert elapsed < 0.2
    assert motion.talk_motion_dropped_frames > 0
    assert reachy.client.commands[-1].offsets == [0.0] * 6


def test_talk_motion_command_error_stops_and_reports_without_goto_fallback() -> None:
    reachy = _FlakyReachy(fail_first=0)
    reachy.client = _FakeClient(error="daemon disconnected")
    motion = MotionController(reachy)  # type: ignore[arg-type]
    motion.start_talk_motion()
    motion.feed_talk_audio(_speech_pcm(0.1), 16_000)
    _wait_stopped(motion)

    assert "daemon disconnected" in motion.talk_motion_error
    assert not reachy.goto_targets
