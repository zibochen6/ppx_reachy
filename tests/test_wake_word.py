"""Tests for the local KWS wake-word path (sherpa-onnx KeywordSpotter)."""

from __future__ import annotations

import asyncio
import time

import pytest

from chaihuo_reachy.config import Config
from chaihuo_reachy.engine import ConversationEngine
from chaihuo_reachy.wake_word import WakeWordDetector


def test_default_kws_tuning_accepts_normal_near_field_speech() -> None:
    cfg = Config()
    assert cfg.kws_threshold == 0.20
    assert cfg.kws_score == 1.5


class _FakeStream:
    def __init__(self, spotter: "_FakeSpotter") -> None:
        self.spotter = spotter
        self.pending = 0

    def accept_waveform(self, _sample_rate: int, _samples) -> None:
        self.pending += 1

    def input_finished(self) -> None:
        pass


class _FakeSpotter:
    """KeywordSpotter stand-in: reports a hit on the Nth fed chunk."""

    def __init__(self, hit_after: int = 3, keyword: str = "皮皮虾") -> None:
        self.hit_after = hit_after
        self.keyword = keyword
        self.chunks_seen = 0
        self.reset_calls = 0

    def create_stream(self, _keywords: str | None = None) -> _FakeStream:
        return _FakeStream(self)

    def is_ready(self, _stream: _FakeStream) -> bool:
        return _stream.pending > 0

    def decode_stream(self, stream: _FakeStream) -> None:
        stream.pending = 0

    def get_result(self, stream: _FakeStream) -> str:
        del stream  # unused
        self.chunks_seen += 1
        if self.chunks_seen >= self.hit_after:
            return self.keyword
        return ""

    def reset_stream(self, _stream: _FakeStream) -> None:
        self.reset_calls += 1


def _detector(spotter: _FakeSpotter, *, threshold: float = 0.35) -> WakeWordDetector:
    return WakeWordDetector(Config(kws_threshold=threshold), spotter=spotter)


def test_detector_reports_hit_and_enters_cooldown() -> None:
    spotter = _FakeSpotter(hit_after=3)
    d = _detector(spotter)
    assert d.hit(b"\0" * 320) is None
    assert d.hit(b"\0" * 320) is None
    assert d.hit(b"\0" * 320) == "皮皮虾"
    assert spotter.reset_calls == 1
    # Cooldown: the next feed must not re-trigger immediately.
    assert d.hit(b"\0" * 320) is None
    assert d.hit(b"\0" * 320) is None


def test_detector_reset_drops_state() -> None:
    spotter = _FakeSpotter(hit_after=1)
    d = _detector(spotter)
    d.hit(b"\0" * 320)  # hit → cooldown starts
    d.reset()
    # Cooldown is unaffected by reset; after it expires the next hit works.
    assert d.hit(b"\0" * 320) is None
    d._cooldown_until = 0.0  # simulate cooldown expiry
    assert d.hit(b"\0" * 320) == "皮皮虾"


def test_needs_wake_word_follows_window_and_engine() -> None:
    engine = ConversationEngine(Config(wake_engine="local"))
    engine._wake = _detector(_FakeSpotter())
    assert engine._needs_wake_word()  # window not open yet

    engine._wake_word_active_until = time.monotonic() + 30.0
    assert not engine._needs_wake_word()  # follow-up window open

    engine.config.wake_engine = "cloud"
    assert not engine._needs_wake_word()

    engine._wake = None
    engine.config.wake_engine = "local"
    assert not engine._needs_wake_word()


@pytest.mark.asyncio
async def test_wait_for_wake_word_returns_preroll_after_hit() -> None:
    engine = ConversationEngine(
        Config(wake_engine="local", wake_listen_timeout_s=0.2)
    )

    class _FakeWake:
        def __init__(self) -> None:
            self.count = 0
            self.reset_called = False

        def hit(self, _chunk: bytes) -> str | None:
            self.count += 1
            return "皮皮虾" if self.count >= 2 else None

        def reset(self) -> None:
            self.reset_called = True

    engine._wake = _FakeWake()

    async def capture():
        for _ in range(4):
            yield b"\0" * 320
            await asyncio.sleep(0)

    pre_roll = await engine._wait_for_wake_word(capture().__aiter__())
    # All chunks fed before (and including) the hit are preserved.
    assert len(pre_roll) == 2
    assert engine._wake.reset_called


@pytest.mark.asyncio
async def test_wait_for_wake_word_times_out_empty() -> None:
    engine = ConversationEngine(
        Config(wake_engine="local", wake_listen_timeout_s=0.02)
    )
    engine._wake = _FakeWakeNoHit()

    async def capture():
        while True:
            yield b"\0" * 320
            await asyncio.sleep(0)

    assert await engine._wait_for_wake_word(capture().__aiter__()) == []
    assert engine._last_asr_end_reason != "wake_word_timeout"  # set by caller


class _FakeWakeNoHit:
    def hit(self, _chunk: bytes) -> str | None:
        return None

    def reset(self) -> None:
        pass
