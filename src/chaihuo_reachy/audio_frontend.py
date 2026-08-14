"""Local speech gating and endpointing for noisy Reachy deployments.

The XVF3800 remains responsible for AEC/beamforming/noise suppression.  This
module adds the missing conversational policy layer: calibrated levels,
optional Silero VAD, a wake-direction gate and a deterministic endpoint.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def pcm16_rms(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, dtype=np.int16)
    if not samples.size:
        return 0.0
    x = samples.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def circular_distance_deg(a: float, b: float) -> float:
    """Smallest angular distance on a circle, in degrees."""
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


class SileroVAD:
    """Small stateful ONNX wrapper; unavailable models fail closed to fallback."""

    def __init__(self, model_path: str, sample_rate: int = 16000) -> None:
        self.available = False
        self._session = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(sample_rate, dtype=np.int64)
        path = Path(model_path)
        if not path.is_file():
            return
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self.available = True
        except Exception:
            self._session = None

    def reset(self) -> None:
        self._state.fill(0)

    def probability(self, pcm: bytes) -> float | None:
        if self._session is None:
            return None
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if not samples.size:
            return 0.0
        # Current Silero streaming exports accept arbitrary multiples of a
        # frame; pad very short backend chunks to the canonical 512 samples.
        if len(samples) < 512:
            samples = np.pad(samples, (0, 512 - len(samples)))
        try:
            output, state = self._session.run(
                None,
                {
                    "input": samples.reshape(1, -1),
                    "state": self._state,
                    "sr": self._sr,
                },
            )
            self._state = state
            return max(0.0, min(1.0, float(np.asarray(output).squeeze())))
        except Exception:
            self.available = False
            self._session = None
            return None


@dataclass
class FrontendFrame:
    rms: float
    dbfs: float
    snr_db: float
    vad_probability: float
    speech: bool
    endpoint: bool
    endpoint_reason: str = ""


class SpeechEndpoint:
    """Hysteretic VAD plus minimum speech, silence and hard-duration limits."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        model_path: str = "",
        on_threshold: float = 0.60,
        off_threshold: float = 0.35,
        min_speech_ms: int = 200,
        silence_ms: int = 700,
        max_utterance_s: float = 15.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.min_speech_s = max(0.0, min_speech_ms / 1000.0)
        self.silence_s = max(0.1, silence_ms / 1000.0)
        self.max_utterance_s = max(1.0, max_utterance_s)
        self.vad = SileroVAD(model_path, sample_rate)
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self.noise_rms = 0.006
        self.speaking = False
        self.started_at: float | None = None
        self.candidate_at: float | None = None
        self.silence_at: float | None = None

    def update(self, pcm: bytes, *, now: float | None = None) -> FrontendFrame:
        now = time.monotonic() if now is None else now
        rms = pcm16_rms(pcm)
        dbfs = 20.0 * math.log10(max(rms, 1e-8))
        snr_db = 20.0 * math.log10(max(rms, 1e-8) / max(self.noise_rms, 1e-5))
        probability = self.vad.probability(pcm)
        if probability is None:
            # Energy fallback is intentionally conservative and adaptive. It
            # is not a replacement for Silero, but keeps endpointing available
            # when the optional model has not yet been deployed.
            probability = 1.0 / (1.0 + math.exp(-(snr_db - 7.0) / 2.0))
            if rms < 0.008:
                probability *= rms / 0.008

        threshold = self.off_threshold if self.speaking else self.on_threshold
        voiced = probability >= threshold
        if not self.speaking and not voiced:
            self.noise_rms = 0.98 * self.noise_rms + 0.02 * rms

        if voiced:
            self.silence_at = None
            if self.candidate_at is None:
                self.candidate_at = now
            if not self.speaking and now - self.candidate_at >= self.min_speech_s:
                self.speaking = True
                self.started_at = self.candidate_at
        else:
            self.candidate_at = None
            if self.speaking and self.silence_at is None:
                self.silence_at = now

        endpoint = False
        reason = ""
        if self.speaking and self.started_at is not None:
            if now - self.started_at >= self.max_utterance_s:
                endpoint, reason = True, "max_utterance"
            elif self.silence_at is not None and now - self.silence_at >= self.silence_s:
                endpoint, reason = True, "local_silence"
        return FrontendFrame(
            rms=rms,
            dbfs=dbfs,
            snr_db=snr_db,
            vad_probability=float(probability),
            speech=self.speaking,
            endpoint=endpoint,
            endpoint_reason=reason,
        )


class DirectionGate:
    """Mute persistent off-axis speech relative to the wake-word direction."""

    def __init__(self, tolerance_deg: float = 35.0, mismatch_ms: int = 800) -> None:
        self.tolerance_deg = max(0.0, min(180.0, tolerance_deg))
        self.mismatch_s = max(0.0, mismatch_ms / 1000.0)
        self.locked_doa: float | None = None
        self._mismatch_at: float | None = None
        self.muted = False

    def lock(self, doa: float | None) -> None:
        self.locked_doa = None if doa is None else float(doa) % 360.0
        self._mismatch_at = None
        self.muted = False

    def accepts(self, doa: float | None, *, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if self.locked_doa is None or doa is None:
            self._mismatch_at = None
            self.muted = False
            return True
        if circular_distance_deg(self.locked_doa, doa) <= self.tolerance_deg:
            self._mismatch_at = None
            self.muted = False
            return True
        if self._mismatch_at is None:
            self._mismatch_at = now
        self.muted = now - self._mismatch_at >= self.mismatch_s
        return not self.muted
