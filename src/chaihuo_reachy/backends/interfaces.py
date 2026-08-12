"""Abstract backend protocols for camera and audio.

Uses typing.Protocol for structural subtyping — any class that satisfies
the interface is a valid backend, no inheritance required.
"""

from __future__ import annotations

from typing import AsyncIterator, Callable, Protocol, runtime_checkable

import numpy as np


# The Dashboard's 0-100 scale maps quadratically to PCM gain. This keeps
# lower listening levels controllable while 100% reaches +12 dB over 50%.
MAX_PLAYBACK_GAIN = 8.0

PlaybackObserver = Callable[[bytes, int], None]


def playback_gain_from_percent(percent: int | float) -> float:
    """Convert a Dashboard volume value to a bounded PCM gain."""
    normalized = max(0.0, min(100.0, float(percent))) / 100.0
    return MAX_PLAYBACK_GAIN * normalized * normalized


def playback_percent_from_gain(gain: float) -> int:
    """Convert a PCM gain to the Dashboard's 0-100 scale."""
    from math import sqrt

    normalized = max(0.0, min(MAX_PLAYBACK_GAIN, float(gain))) / MAX_PLAYBACK_GAIN
    return round(sqrt(normalized) * 100)


@runtime_checkable
class CameraBackend(Protocol):
    """Abstract camera backend — provides BGR frames and JPEG encoding."""

    @property
    def is_active(self) -> bool:
        """True if the camera is connected and delivering frames."""
        ...

    def read(self) -> np.ndarray | None:
        """Return latest BGR frame as (H, W, 3) uint8 array, or None."""
        ...

    def capture_jpeg(self, quality: int = 85) -> bytes | None:
        """Return JPEG-encoded bytes of current frame, or None."""
        ...

    def close(self) -> None:
        """Release camera resources."""
        ...

    @property
    def backend_name(self) -> str:
        """Human-readable backend identifier."""
        ...


@runtime_checkable
class AudioBackend(Protocol):
    """Abstract audio backend — capture + playback.

    Capture is exposed as an async generator yielding PCM int16 mono chunks.
    Playback accepts PCM int16 bytes and queues them for output.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def open(self) -> None:
        """Start the audio device (recording + playback)."""
        ...

    async def close(self) -> None:
        """Stop and release the audio device."""
        ...

    # ── Capture ───────────────────────────────────────────────────────

    async def start_capture(self) -> AsyncIterator[bytes]:
        """Async generator yielding PCM int16 mono chunks.

        Each chunk is ``chunk_ms`` worth of audio at the device sample rate.
        """
        ...

    # ── Playback ──────────────────────────────────────────────────────

    async def play(self, pcm: bytes) -> None:
        """Queue PCM int16 mono bytes for playback."""
        ...

    def stop_playback(self) -> None:
        """Immediately clear the playback queue (barge-in)."""
        ...

    def mark_playback_done(self) -> None:
        """Signal that no more audio will be queued (drain and stop)."""
        ...

    @property
    def is_playing(self) -> bool:
        """True while audio is being output."""
        ...

    # ── Volume ────────────────────────────────────────────────────────

    @property
    def volume(self) -> float:
        """Current playback gain (0.0 - 8.0)."""
        ...

    @volume.setter
    def volume(self, val: float) -> None: ...

    def play_rms(self) -> float:
        """Smoothed RMS of the output signal."""
        ...

    def set_playback_observer(self, callback: PlaybackObserver | None) -> None:
        """Observe post-gain mono PCM at the real speaker-consumption point.

        The callback runs on an audio thread and must return immediately.
        """
        ...

    # ── Sample rate ───────────────────────────────────────────────────

    def set_output_sample_rate(self, sr: int) -> None:
        """Configure the expected output sample rate for resampling."""
        ...

    # ── Metadata ──────────────────────────────────────────────────────

    @property
    def resolved_info(self) -> dict[str, object]:
        """Device info dict for status/diagnostics."""
        ...

    # ── Optional SDK extensions ───────────────────────────────────────

    @property
    def supports_doa(self) -> bool:
        """Whether Direction of Arrival (mic array) is available."""
        ...

    def get_doa(self) -> float | None:
        """Return DoA angle in radians, or None if unavailable."""
        ...

    @property
    def supports_wobbling(self) -> bool:
        """Whether audio-reactive head wobbling is supported."""
        ...

    def enable_wobbling(self) -> None: ...

    def disable_wobbling(self) -> None: ...

    @property
    def backend_name(self) -> str:
        """Human-readable backend identifier."""
        ...
