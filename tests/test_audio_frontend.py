from __future__ import annotations

import numpy as np

from chaihuo_reachy.audio_frontend import (
    DirectionGate,
    SpeechEndpoint,
    circular_distance_deg,
    pcm16_rms,
)


def _pcm(level: float, samples: int = 1600) -> bytes:
    return np.full(samples, int(level * 32767), dtype=np.int16).tobytes()


def test_pcm16_rms_is_true_root_mean_square() -> None:
    assert abs(pcm16_rms(_pcm(0.25)) - 0.25) < 0.001


def test_circular_angle_distance_wraps_at_zero() -> None:
    assert circular_distance_deg(350, 10) == 20
    assert circular_distance_deg(10, 350) == 20


def test_direction_gate_mutes_only_after_persistent_mismatch() -> None:
    gate = DirectionGate(tolerance_deg=35, mismatch_ms=800)
    gate.lock(350)
    assert gate.accepts(10, now=0.0)
    assert gate.accepts(90, now=0.1)
    assert not gate.accepts(90, now=0.91)
    assert gate.accepts(5, now=1.0)


def test_endpoint_forces_turn_end_after_silence() -> None:
    endpoint = SpeechEndpoint(
        on_threshold=0.60,
        off_threshold=0.35,
        min_speech_ms=200,
        silence_ms=700,
        max_utterance_s=15,
    )
    endpoint.vad.probability = lambda _pcm: 0.9  # type: ignore[method-assign]
    endpoint.update(_pcm(0.1), now=0.0)
    started = endpoint.update(_pcm(0.1), now=0.21)
    assert started.speech
    endpoint.vad.probability = lambda _pcm: 0.1  # type: ignore[method-assign]
    assert not endpoint.update(_pcm(0.0), now=0.3).endpoint
    ended = endpoint.update(_pcm(0.0), now=1.01)
    assert ended.endpoint and ended.endpoint_reason == "local_silence"
