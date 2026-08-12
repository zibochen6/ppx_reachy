"""SDK audio backend — capture and playback via GStreamerAudio.

Uses the SDK's MediaManager for audio I/O. Capture is pull-based
(get_audio_sample() polled in a loop). Playback is push-based
(push_audio_sample()). This replaces the callback-based sounddevice
RawStream with a simpler async paradigm.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, AsyncIterator, Callable

import numpy as np

from chaihuo_reachy.backends.interfaces import MAX_PLAYBACK_GAIN

if TYPE_CHECKING:
    from reachy_mini.media.media_manager import MediaManager

logger = logging.getLogger("chaihuo_reachy.backends.sdk_audio")

# Default output sample rate (Bailian TTS uses 24000 by default)
_DEFAULT_OUTPUT_SR = 24000


class SdkAudioIO:
    """Audio backend using Reachy SDK's MediaManager GStreamer pipelines.

    Audio specs: 16 kHz, 2 channels (stereo with XMOS AEC on channel 0).
    Capture returns mono int16 PCM (channel 0 only). Playback accepts
    mono int16 PCM and duplicates to stereo before pushing.
    """

    backend_name = "sdk_gstreamer"

    def __init__(
        self,
        media_manager: MediaManager,
        sr: int = 16000,
        chunk_ms: int = 100,
        input_channel: int = 1,
    ) -> None:
        self._mm = media_manager
        self._sr = sr
        self._chunk_frames = int(sr * chunk_ms / 1000)
        self._chunk_bytes = self._chunk_frames * 2  # int16 = 2 bytes/sample
        self._input_channel = input_channel  # 0=raw mic, 1=AEC processed

        self._volume = 2.0  # Gain for Reachy Mini speaker
        self._play_rms = 0.0
        self._playback_observer: Callable[[bytes, int], None] | None = None
        self._output_sr = _DEFAULT_OUTPUT_SR
        self._playing = False
        self._capture_task: asyncio.Task | None = None
        self._playback_event = threading.Event()
        self._playback_pending = (
            False  # True while audio has been pushed and not yet drained
        )
        self._playback_deadline = 0.0
        self._capture_rms: float = 0.0  # Smoothed mic input level for diagnostics
        self._last_sample_at: float = 0.0  # monotonic time of last captured sample

        # Resampling state (simple linear interpolation)
        self._resample_buf: np.ndarray | None = None  # leftover float32

    # ── Lifecycle ────────────────────────────────────────────────────

    async def open(self) -> None:
        """Start SDK recording + playback pipelines."""
        self._mm.start_recording()
        self._mm.start_playing()
        self._playing = True
        logger.info("SDK audio opened (GStreamer pipelines started)")

    async def close(self) -> None:
        """Stop SDK pipelines."""
        if self._playing:
            self._mm.stop_playing()
            self._playing = False
        self._mm.stop_recording()
        logger.info("SDK audio closed")

    # ── Capture ──────────────────────────────────────────────────────

    async def start_capture(self) -> AsyncIterator[bytes]:
        """Async generator: poll get_audio_sample(), accumulate, yield chunks.

        The SDK returns float32 (samples, 2) at ~16kHz. We take the configured
        input channel, accumulate into a buffer, and yield int16 PCM chunks.
        """
        buf = bytearray()
        poll_s = 0.005  # 200 Hz poll rate
        ch = getattr(self, "_input_channel", 0)  # 0=raw, 1=AEC
        import math as _math

        while True:
            try:
                sample = self._mm.get_audio_sample()
                if sample is not None and sample.size > 0:
                    self._last_sample_at = time.monotonic()
                    # float32 (samples, 2) → int16 mono (selected channel)
                    ch_idx = min(ch, sample.shape[1] - 1) if sample.ndim > 1 else 0
                    if sample.ndim > 1:
                        mono = sample[:, ch_idx].astype(np.float32)
                    else:
                        mono = sample.astype(np.float32)

                    # Update smoothed capture RMS for diagnostics
                    rms = (
                        float(_math.sqrt(float(_math.sqrt((mono * mono).mean()))))
                        if mono.size
                        else 0.0
                    )
                    self._capture_rms = 0.9 * self._capture_rms + 0.1 * rms

                    i16 = (
                        (mono * 32767.0).clip(-32768, 32767).astype(np.int16)
                    ).tobytes()
                    buf.extend(i16)

                    # Yield complete chunks
                    while len(buf) >= self._chunk_bytes:
                        yield bytes(buf[: self._chunk_bytes])
                        del buf[: self._chunk_bytes]

                await asyncio.sleep(poll_s)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("SDK audio capture error, continuing")
                await asyncio.sleep(0.1)

    # ── Playback ─────────────────────────────────────────────────────

    async def play(self, pcm: bytes) -> None:
        """Queue PCM int16 mono for playback via push_audio_sample.

        Performs volume gain, tanh soft-clip, int16→float32 conversion,
        and resampling from output_sr to 16000 if needed.
        """
        if not self._playing:
            return

        try:
            # int16 → float32 normalized
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

            # Apply gain + soft-clip
            samples = samples * self._volume
            samples = np.tanh(samples)
            samples = np.clip(samples, -0.99, 0.99)

            # Resample if needed
            if self._output_sr != self._sr and self._output_sr > 0:
                samples = self._resample(samples, self._output_sr, self._sr)

            # Convert mono to stereo (duplicate to both channels)
            stereo = np.column_stack([samples, samples]).astype(np.float32)

            self._mm.push_audio_sample(stereo)
            self._playback_pending = True
            # The SDK does not expose a queue-drained notification. Track the
            # submitted PCM duration and retain a small hardware tail window.
            now = time.monotonic()
            self._playback_deadline = max(now, self._playback_deadline) + (
                len(samples) / self._sr
            )

            # Update RMS
            rms = float(np.sqrt(np.mean(samples**2)))
            self._play_rms = 0.9 * self._play_rms + 0.1 * rms
            observer = self._playback_observer
            if observer is not None:
                try:
                    observer(
                        (samples * 32767.0)
                        .clip(-32768, 32767)
                        .astype(np.int16)
                        .tobytes(),
                        self._sr,
                    )
                except Exception:
                    logger.warning("playback observer failed", exc_info=True)

        except Exception:
            logger.exception("SDK audio playback error")

    def stop_playback(self) -> None:
        """Immediately clear playback queue (barge-in)."""
        # MediaManager 1.9 has no clear_player(). Restarting its output
        # pipeline is the documented stop mechanism and drops queued PCM.
        # Never let an interrupt failure take down the LLM turn.
        try:
            self._mm.stop_playing()
            if self._playing:
                self._mm.start_playing()
        except Exception:
            logger.warning("SDK playback stop failed", exc_info=True)
        self._play_rms = 0.0
        self._playback_pending = False
        self._playback_deadline = 0.0

    def mark_playback_done(self) -> None:
        """Signal end of playback — no more audio will be queued."""
        self._playback_event.set()
        if self._playback_pending:
            self._playback_deadline = (
                max(self._playback_deadline, time.monotonic()) + 0.3
            )

    @property
    def is_playing(self) -> bool:
        """True while audio has been queued and not yet marked as drained."""
        if self._playback_pending and time.monotonic() >= self._playback_deadline:
            self._playback_pending = False
        return self._playback_pending

    @property
    def capture_rms(self) -> float:
        """Smoothed RMS (0..1) of microphone input — for level diagnostics."""
        return self._capture_rms

    @property
    def last_sample_age_s(self) -> float:
        """Seconds since the last captured audio sample (inf if never)."""
        if not self._last_sample_at:
            return float("inf")
        return time.monotonic() - self._last_sample_at

    # ── Volume ───────────────────────────────────────────────────────

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, val: float) -> None:
        self._volume = max(0.0, min(MAX_PLAYBACK_GAIN, float(val)))

    def play_rms(self) -> float:
        return self._play_rms

    def set_playback_observer(
        self, callback: Callable[[bytes, int], None] | None
    ) -> None:
        """Install a best-effort observer for SDK playback PCM."""
        self._playback_observer = callback

    # ── Sample rate ──────────────────────────────────────────────────

    def set_output_sample_rate(self, sr: int) -> None:
        """Set the sample rate of incoming playback PCM for resampling."""
        self._output_sr = sr

    # ── Metadata ─────────────────────────────────────────────────────

    @property
    def resolved_info(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "sample_rate": self._sr,
            "input_channels": (self._mm.get_input_channels() if self._mm.audio else 2),
            "output_channels": (
                self._mm.get_output_channels() if self._mm.audio else 2
            ),
            "output_sample_rate": self._output_sr,
            "volume": self._volume,
        }

    # ── SDK extensions ───────────────────────────────────────────────

    @property
    def supports_doa(self) -> bool:
        return True

    def get_doa(self) -> float | None:
        """Direction of Arrival from the ReSpeaker mic array."""
        result = self._mm.get_DoA()
        if result is None:
            return None
        angle, _speech_detected = result
        return float(angle)

    @property
    def supports_wobbling(self) -> bool:
        return True

    def enable_wobbling(self) -> None:
        """Enable audio-reactive head movement (handled by MotionController)."""
        pass  # Wobbling is controlled via ReachyMini, not the audio backend

    def disable_wobbling(self) -> None:
        pass

    # ── Internal ─────────────────────────────────────────────────────

    def _resample(
        self,
        samples: np.ndarray,
        from_sr: int,
        to_sr: int,
    ) -> np.ndarray:
        """Simple linear-interpolation resampling."""
        if from_sr == to_sr:
            return samples

        ratio = to_sr / from_sr
        n_out = int(len(samples) * ratio)
        if n_out == 0:
            return np.array([], dtype=np.float32)

        # Combine with leftover from previous call
        if self._resample_buf is not None:
            samples = np.concatenate([self._resample_buf, samples])
            self._resample_buf = None

        xp = np.arange(len(samples))
        x = np.linspace(0, len(samples) - 1, n_out)
        out = np.interp(x, xp, samples).astype(np.float32)
        return out
