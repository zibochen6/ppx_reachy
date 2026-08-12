from __future__ import annotations

import math

import numpy as np

from chaihuo_reachy.speech_motion import HOP_S, ContinuousSpeechMotion


def _tone(seconds: float, *, sr: int = 16_000, amplitude: float = 0.35) -> bytes:
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    samples = np.sin(2.0 * np.pi * 220.0 * t) * amplitude
    return (samples * 32767.0).astype(np.int16).tobytes()


def _silence(seconds: float, *, sr: int = 16_000) -> bytes:
    return np.zeros(int(seconds * sr), dtype=np.int16).tobytes()


def test_pcm_envelope_attacks_within_roughly_80ms() -> None:
    motion = ContinuousSpeechMotion()
    frames = motion.feed_pcm(_tone(0.10), 16_000)
    assert len(frames) == 2
    assert frames[0].voiced
    assert 0.35 < frames[0].envelope < 0.65
    assert frames[1].envelope > frames[0].envelope


def test_offsets_are_multi_axis_bounded_and_smooth_per_hop() -> None:
    motion = ContinuousSpeechMotion()
    frames = motion.feed_pcm(_tone(4.0), 16_000)
    offsets = np.asarray([frame.offsets for frame in frames])
    assert len(frames) == round(4.0 / HOP_S)
    assert np.ptp(np.degrees(offsets[:, 5])) >= 28.0
    assert np.ptp(np.degrees(offsets[:, 4])) >= 3.0
    assert np.ptp(np.degrees(offsets[:, 3])) >= 2.0
    assert np.max(np.abs(np.degrees(offsets[:, 5]))) <= 20.0 + 1e-6
    assert np.max(np.abs(np.degrees(offsets[:, 4]))) <= 4.0 + 1e-6
    assert np.max(np.abs(np.degrees(offsets[:, 3]))) <= 3.0 + 1e-6
    assert np.max(np.abs(np.diff(np.degrees(offsets[:, 5])))) <= 3.8 + 1e-6
    assert np.max(np.abs(offsets[:, :3])) <= 0.003 + 1e-9


def test_continuous_phase_has_live_speed_and_light_asymmetry() -> None:
    motion = ContinuousSpeechMotion()
    frames = motion.feed_pcm(_tone(5.0), 16_000)
    yaw = np.asarray([frame.offsets[5] for frame in frames])
    signs = np.signbit(yaw)
    zero_crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
    cycles = zero_crossings / 2.0
    assert 4.2 <= cycles <= 7.0  # about 0.9–1.3 Hz, allowing startup slew
    assert not math.isclose(abs(float(yaw.max())), abs(float(yaw.min())), rel_tol=0.001)


def test_short_pause_keeps_breathing_floor_then_recovers() -> None:
    motion = ContinuousSpeechMotion()
    motion.feed_pcm(_tone(0.5), 16_000)
    pause = motion.feed_pcm(_silence(0.4), 16_000)
    assert pause[-1].envelope >= 0.1
    assert any(
        any(abs(value) > 1e-5 for value in frame.offsets) for frame in pause[-3:]
    )
    resumed = motion.feed_pcm(_tone(0.15), 16_000)
    assert resumed[-1].envelope > pause[-1].envelope


def test_graceful_end_is_monotonic_and_exactly_zero() -> None:
    motion = ContinuousSpeechMotion()
    motion.feed_pcm(_tone(0.5), 16_000)
    decay = motion.graceful_decay(0.30)
    magnitudes = [np.linalg.norm(frame.offsets) for frame in decay]
    assert len(decay) == 6
    assert all(b <= a + 1e-12 for a, b in zip(magnitudes, magnitudes[1:]))
    assert decay[-1].offsets == (0.0,) * 6
    assert motion.last_offsets == (0.0,) * 6


def test_invalid_or_partial_pcm_never_emits_a_partial_hop() -> None:
    motion = ContinuousSpeechMotion()
    assert motion.feed_pcm(b"", 16_000) == []
    assert motion.feed_pcm(_tone(0.02), 16_000) == []
    assert len(motion.feed_pcm(_tone(0.03), 16_000)) == 1
