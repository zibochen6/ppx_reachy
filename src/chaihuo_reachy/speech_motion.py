"""Continuous, audio-reactive motion frames for natural conversation.

The generator consumes the PCM that is actually being written to the speaker,
not TTS callbacks that may run many seconds ahead.  It produces 6-DoF offsets
for Reachy Mini's daemon-native ``speech_offsets`` composition path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HOP_S = 0.05
_ZERO_OFFSETS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SpeechMotionFrame:
    """One 50 ms motion hop synchronized to speaker PCM."""

    offsets: tuple[float, float, float, float, float, float]
    envelope: float
    loudness: float
    voiced: bool


class ContinuousSpeechMotion:
    """Generate smooth multi-axis speech motion with continuous phase.

    Yaw is the dominant motion. Pitch, roll and millimetre-scale translation
    run at independent frequencies so the result does not look periodic or
    robotic. Loudness controls speed and amplitude; short pauses retain a
    small breathing floor instead of freezing at an endpoint.
    """

    def __init__(
        self,
        *,
        yaw_max_deg: float = 20.0,
        pitch_max_deg: float = 3.2,
        roll_max_deg: float = 2.0,
        master_gain: float = 1.0,
        breath_floor: float = 0.10,
    ) -> None:
        self.yaw_max_deg = max(0.0, min(24.0, float(yaw_max_deg)))
        self.pitch_max_deg = max(0.0, min(8.0, float(pitch_max_deg)))
        self.roll_max_deg = max(0.0, min(6.0, float(roll_max_deg)))
        self.master_gain = max(0.0, min(1.5, float(master_gain)))
        self.breath_floor = max(0.0, min(0.2, float(breath_floor)))

        self._sample_rate = 16_000
        self._carry = np.zeros(0, dtype=np.float32)
        self._envelope = 0.0
        self._loudness = 0.0
        self._voiced = False
        self._voice_on_count = 0
        self._voice_off_count = 0
        # Stable non-zero phases prevent every utterance starting centered.
        self._yaw_phase = 0.63
        self._pitch_phase = 2.17
        self._roll_phase = 4.02
        self._x_phase = 1.31
        self._y_phase = 3.44
        self._z_phase = 5.10
        self._slow_phase = 0.0
        self._last_offsets = _ZERO_OFFSETS

    @property
    def envelope(self) -> float:
        return self._envelope

    @property
    def last_offsets(self) -> tuple[float, float, float, float, float, float]:
        return self._last_offsets

    def reset(self) -> None:
        """Reset audio/envelope state while preserving natural phase."""
        self._carry = np.zeros(0, dtype=np.float32)
        self._envelope = 0.0
        self._loudness = 0.0
        self._voiced = False
        self._voice_on_count = 0
        self._voice_off_count = 0
        self._last_offsets = _ZERO_OFFSETS

    def feed_pcm(self, pcm: bytes, sample_rate: int) -> list[SpeechMotionFrame]:
        """Convert int16 mono PCM into one frame per 50 ms hop."""
        if sample_rate <= 0 or not pcm:
            return []
        if sample_rate != self._sample_rate:
            self._sample_rate = int(sample_rate)
            self._carry = np.zeros(0, dtype=np.float32)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._carry = (
            np.concatenate((self._carry, samples)) if self._carry.size else samples
        )
        hop = max(1, round(self._sample_rate * HOP_S))
        frames: list[SpeechMotionFrame] = []
        while self._carry.size >= hop:
            block = self._carry[:hop]
            self._carry = self._carry[hop:]
            frames.append(self._next_frame(block))
        return frames

    def graceful_decay(self, duration_s: float = 0.30) -> list[SpeechMotionFrame]:
        """Ease the most recent pose to zero without snapping."""
        count = max(1, round(max(HOP_S, duration_s) / HOP_S))
        start = self._last_offsets
        start_env = self._envelope
        frames: list[SpeechMotionFrame] = []
        for index in range(1, count + 1):
            progress = index / count
            # Cosine ease preserves velocity continuity at both ends.
            gain = 0.5 * (1.0 + math.cos(math.pi * progress))
            offsets = tuple(value * gain for value in start)
            frames.append(
                SpeechMotionFrame(
                    offsets=offsets,  # type: ignore[arg-type]
                    envelope=start_env * gain,
                    loudness=0.0,
                    voiced=False,
                )
            )
        self._envelope = 0.0
        self._loudness = 0.0
        self._voiced = False
        self._last_offsets = _ZERO_OFFSETS
        return frames

    def _next_frame(self, block: np.ndarray) -> SpeechMotionFrame:
        rms = float(np.sqrt(np.mean(block * block) + 1e-12))
        db = 20.0 * math.log10(rms + 1e-12)
        # Normalize ordinary TTS speech (-46..-16 dBFS) into 0..1.
        loudness = max(0.0, min(1.0, (db + 46.0) / 30.0)) ** 0.85

        if db >= -43.0:
            self._voice_on_count += 1
            self._voice_off_count = 0
            if self._voice_on_count >= 1:
                self._voiced = True
        elif db <= -49.0:
            self._voice_off_count += 1
            self._voice_on_count = 0
            if self._voice_off_count >= 3:  # 150 ms release hysteresis
                self._voiced = False

        target = 1.0 if self._voiced else self.breath_floor
        tau = 0.08 if target > self._envelope else 0.25
        alpha = 1.0 - math.exp(-HOP_S / tau)
        self._envelope += alpha * (target - self._envelope)
        self._loudness += 0.35 * (loudness - self._loudness)

        level = self._loudness
        # Audible cadence: around 0.9–1.3 full yaw cycles per second.
        yaw_hz = 0.90 + 0.34 * level + 0.05 * math.sin(self._slow_phase)
        self._yaw_phase += 2.0 * math.pi * yaw_hz * HOP_S
        self._pitch_phase += 2.0 * math.pi * (1.55 + 0.20 * level) * HOP_S
        self._roll_phase += 2.0 * math.pi * (1.02 + 0.12 * level) * HOP_S
        self._x_phase += 2.0 * math.pi * 0.37 * HOP_S
        self._y_phase += 2.0 * math.pi * 0.43 * HOP_S
        self._z_phase += 2.0 * math.pi * 0.29 * HOP_S
        self._slow_phase += 2.0 * math.pi * 0.11 * HOP_S

        envelope = self._envelope * self.master_gain
        emphasis = 0.72 + 0.28 * level
        asymmetry = 1.0 + 0.08 * math.sin(self._slow_phase)
        yaw_wave = math.sin(self._yaw_phase)
        if yaw_wave < 0.0:
            asymmetry *= 0.95

        yaw_limit = math.radians(self.yaw_max_deg)
        pitch_limit = math.radians(self.pitch_max_deg)
        roll_limit = math.radians(self.roll_max_deg)
        yaw = yaw_limit * envelope * emphasis * asymmetry * yaw_wave
        pitch = (
            pitch_limit * envelope * (0.48 + 0.52 * level) * math.sin(self._pitch_phase)
        )
        roll = (
            roll_limit * envelope * (0.55 + 0.45 * level) * math.sin(self._roll_phase)
        )
        # Millimetre-scale translations add life without moving the body.
        x = 0.0022 * envelope * math.sin(self._x_phase)
        y = 0.0030 * envelope * math.sin(self._y_phase)
        z = 0.0018 * envelope * math.sin(self._z_phase)

        raw_offsets = (
            max(-0.0025, min(0.0025, x)),
            max(-0.0030, min(0.0030, y)),
            max(-0.0025, min(0.0025, z)),
            max(-roll_limit, min(roll_limit, roll)),
            max(-pitch_limit, min(pitch_limit, pitch)),
            max(-yaw_limit, min(yaw_limit, yaw)),
        )
        # Bound per-hop angular changes as a final safety/smoothness layer.
        # The phase keeps advancing behind this limiter, so endpoints reverse
        # continuously without the zero-velocity dwell caused by goto_target.
        slew = (
            0.0008,
            0.0008,
            0.0008,
            math.radians(0.8),
            math.radians(1.0),
            math.radians(3.5),
        )
        offsets = tuple(
            previous + max(-limit, min(limit, target - previous))
            for target, previous, limit in zip(raw_offsets, self._last_offsets, slew)
        )
        self._last_offsets = offsets
        return SpeechMotionFrame(
            offsets=offsets,
            envelope=min(1.0, self._envelope),
            loudness=level,
            voiced=self._voiced,
        )
