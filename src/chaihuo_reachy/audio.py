"""Full-duplex audio I/O — single sounddevice RawStream for mic + speaker.

Design (shared by Mac dev and Jetson prod):
  - macOS: default audio device, typically built-in mic/speaker or USB headset
  - Jetson: Reachy Mini USB sound card (XMOS XVF3800), natively 2-in/2-out
  - Uses ONE duplex stream: the sound card breaks if separate input+output
    streams are opened independently (verified on Reachy Mini hardware).

The callback simultaneously captures mic PCM and feeds the speaker from an
internal playback buffer.  Resampling is handled by the caller — this module
just moves int16 PCM at the device sample rate.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from numbers import Integral
from typing import AsyncIterator, Callable

import numpy as np
import sounddevice as sd

from chaihuo_reachy.backends.interfaces import MAX_PLAYBACK_GAIN

logger = logging.getLogger("chaihuo_reachy.audio")

_REACHY_DEVICE_NAME = "reachy mini audio"


class AudioDeviceResolutionError(RuntimeError):
    """Raised when a requested duplex audio device is absent or ambiguous."""


@dataclass(frozen=True)
class AudioDeviceInfo:
    """Authoritative PortAudio selection reported by runtime diagnostics."""

    requested_selector: str
    input_index: int
    output_index: int
    input_name: str
    output_name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    backend: str = "sounddevice-duplex"
    stream_input_channels: int | None = None
    stream_output_channels: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_selector": self.requested_selector,
            "input": {
                "index": self.input_index,
                "name": self.input_name,
                "max_channels": self.max_input_channels,
                "stream_channels": self.stream_input_channels,
            },
            "output": {
                "index": self.output_index,
                "name": self.output_name,
                "max_channels": self.max_output_channels,
                "stream_channels": self.stream_output_channels,
            },
            "default_sample_rate": self.default_sample_rate,
            "backend": self.backend,
        }


def _normalized_device_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower()).rstrip(":")


def _device_table(devices: object) -> list[dict[str, object]]:
    return [dict(device) for device in devices]  # type: ignore[arg-type]


def _candidate_summary(devices: list[dict[str, object]]) -> str:
    if not devices:
        return "(no PortAudio devices)"
    return "; ".join(
        f"[{idx}] {device.get('name', '?')} "
        f"(in={int(device.get('max_input_channels', 0))}, "
        f"out={int(device.get('max_output_channels', 0))})"
        for idx, device in enumerate(devices)
    )


def _validate_index(
    devices: list[dict[str, object]], index: int, role: str
) -> dict[str, object]:
    if index < 0 or index >= len(devices):
        raise AudioDeviceResolutionError(
            f"Audio {role} device index {index} is invalid. Available: "
            f"{_candidate_summary(devices)}"
        )
    return devices[index]


def _build_info(
    selector: object,
    devices: list[dict[str, object]],
    input_index: int,
    output_index: int,
) -> AudioDeviceInfo:
    input_device = _validate_index(devices, input_index, "input")
    output_device = _validate_index(devices, output_index, "output")
    in_channels = int(input_device.get("max_input_channels", 0))
    out_channels = int(output_device.get("max_output_channels", 0))
    if in_channels < 1 or out_channels < 1:
        raise AudioDeviceResolutionError(
            f"Audio selector {selector!r} is not full duplex: input "
            f"[{input_index}] has {in_channels} channel(s), output "
            f"[{output_index}] has {out_channels} channel(s). Available: "
            f"{_candidate_summary(devices)}"
        )
    return AudioDeviceInfo(
        requested_selector="auto" if selector is None else str(selector),
        input_index=input_index,
        output_index=output_index,
        input_name=str(input_device.get("name", "?")),
        output_name=str(output_device.get("name", "?")),
        max_input_channels=in_channels,
        max_output_channels=out_channels,
        default_sample_rate=float(
            input_device.get("default_samplerate")
            or output_device.get("default_samplerate")
            or 0
        ),
    )


def resolve_audio_device(
    selector: str | int | None = "auto",
    *,
    devices: object | None = None,
    default_device: object | None = None,
) -> AudioDeviceInfo:
    """Resolve and validate the one duplex device used by the application.

    Omitted/``auto`` selection is intentionally Reachy-only.  A developer who
    wants the Mac/system devices must opt in with ``selector="default"``.
    ``devices`` and ``default_device`` are injectable for hardware-free tests.
    """
    table = _device_table(sd.query_devices() if devices is None else devices)
    normalized_selector = _normalized_device_name(selector or "auto")

    if normalized_selector in ("", "auto"):
        candidates = [
            index
            for index, device in enumerate(table)
            if _REACHY_DEVICE_NAME in _normalized_device_name(device.get("name", ""))
            and "camera" not in _normalized_device_name(device.get("name", ""))
            and int(device.get("max_input_channels", 0)) > 0
            and int(device.get("max_output_channels", 0)) > 0
        ]
        exact = [
            index
            for index in candidates
            if _normalized_device_name(table[index].get("name", ""))
            == _REACHY_DEVICE_NAME
        ]
        if len(exact) == 1:
            candidates = exact
        if len(candidates) != 1:
            reason = (
                "not found" if not candidates else f"ambiguous indexes {candidates}"
            )
            raise AudioDeviceResolutionError(
                "Reachy Mini full-duplex audio device was " + reason + ". "
                "Check the USB connection, set REACHY_AUDIO_DEVICE to a unique "
                "name/index, or explicitly use REACHY_AUDIO_DEVICE=default for "
                "local Mac audio. Available: " + _candidate_summary(table)
            )
        return _build_info("auto", table, candidates[0], candidates[0])

    if normalized_selector == "default":
        defaults = sd.default.device if default_device is None else default_device
        try:
            input_index, output_index = (int(defaults[0]), int(defaults[1]))  # type: ignore[index]
        except (TypeError, ValueError, IndexError) as exc:
            raise AudioDeviceResolutionError(
                f"PortAudio default devices are unavailable: {defaults!r}. "
                f"Available: {_candidate_summary(table)}"
            ) from exc
        return _build_info("default", table, input_index, output_index)

    explicit_index: int | None = None
    if isinstance(selector, Integral):
        explicit_index = int(selector)
    elif isinstance(selector, str) and selector.strip().lstrip("+-").isdigit():
        explicit_index = int(selector.strip())
    if explicit_index is not None:
        return _build_info(selector, table, explicit_index, explicit_index)

    matches = [
        index
        for index, device in enumerate(table)
        if normalized_selector in _normalized_device_name(device.get("name", ""))
    ]
    exact = [
        index
        for index in matches
        if _normalized_device_name(table[index].get("name", "")) == normalized_selector
    ]
    if len(exact) == 1:
        matches = exact
    if len(matches) != 1:
        reason = "not found" if not matches else f"ambiguous indexes {matches}"
        raise AudioDeviceResolutionError(
            f"Audio selector {selector!r} was {reason}. Choose a unique full-duplex "
            f"device name/index. Available: {_candidate_summary(table)}"
        )
    return _build_info(selector, table, matches[0], matches[0])


def _want_alsa_backend(info: AudioDeviceInfo) -> bool:
    """True when we must bypass PortAudio for this device.

    The Reachy Mini XMOS (XVF3800) card returns silent (all-zero) capture in
    mmap mode, which is what PortAudio/sounddevice uses by default — arecord
    (rw mode) records fine.  Force pyalsaaudio rw-mode PCM for that card.
    Explicit opt-in via REACHY_AUDIO_BACKEND=alsa also works.
    """
    import os

    forced = os.environ.get("REACHY_AUDIO_BACKEND", "").strip().lower()
    if forced == "alsa":
        return True
    if forced == "sounddevice":
        return False
    name = f"{info.input_name} {info.output_name}".lower()
    return "reachy mini audio" in name


class DuplexAudioIO:
    """Full-duplex audio: single RawStream, mic in + speaker out.

    Usage::

        audio = DuplexAudioIO(sr=16000)
        # Capture loop (blocks until cancelled):
        async for chunk in audio.start_capture():
            process(chunk)  # PCM int16 bytes
        # Playback (from any coroutine):
        await audio.play(pcm_bytes)
        audio.mark_playback_done()
    """

    def __init__(
        self,
        device: str | int | None = "auto",
        sr: int = 16000,
        chunk_ms: int = 100,
        input_channel: int = 0,
    ) -> None:
        self.resolved_info = resolve_audio_device(device)
        self.input_device = self.resolved_info.input_index
        self.output_device = self.resolved_info.output_index
        self.input_sr = sr
        self.output_sr = sr
        self._chunk_frames = int(sr * chunk_ms / 1000)
        self._input_channel = (
            input_channel if input_channel != 0 else 1
        )  # default to AEC channel

        self._duplex_stream: sd.RawStream | None = None
        self._in_ch = 1
        self._out_ch = 1
        self._loop: asyncio.AbstractEventLoop | None = None
        self._in_queue: asyncio.Queue[bytes] | None = None

        # Playback buffer (thread-safe)
        self._playback_lock = threading.Lock()
        self._playback_buffer = bytearray()
        self._playback_done = False
        self._playback_sample_rate: int = sr  # source rate for resampling
        self._capture_event = asyncio.Event()
        self._volume: float = 2.0  # Default gain for Reachy Mini speaker
        self._play_rms: float = 0.0
        self._playback_observer: Callable[[bytes, int], None] | None = None
        self._capture_rms: float = 0.0  # Smoothed input level for diagnostics
        self._mic_gain: float = 1.0  # Unity gain; clipping degrades ASR accuracy
        self._capture_rms_last_emit = 0.0  # Throttle RMS log emission

        # ALSA direct backend (Reachy Mini XMOS: PortAudio/mmap capture is
        # silent on this card, so we fall back to pyalsaaudio rw-mode PCM,
        # the same path arecord uses and which is verified to produce audio).
        self._alsa: bool = _want_alsa_backend(self.resolved_info)
        self._alsa_stop = threading.Event()  # capture stop
        self._alsa_playback_stop = threading.Event()  # playback stop (dance music etc.)
        self._alsa_threads: list[threading.Thread] = []
        self._alsa_playback_pcm: object | None = None

        # Monotonic time of the last captured sample — the engine's
        # "no samples flowing" watchdog reads this via last_sample_age_s.
        self._last_sample_at: float = 0.0

    # ── Capture ────────────────────────────────────────────────────────
    async def start_capture(self) -> AsyncIterator[bytes]:
        """Yield PCM int16 audio chunks from the microphone."""
        self._loop = asyncio.get_running_loop()
        self._in_queue = asyncio.Queue(maxsize=64)
        self._open_duplex()
        try:
            while True:
                chunk = await self._in_queue.get()
                yield chunk
        except asyncio.CancelledError:
            pass
        finally:
            self._close_duplex()

    # ── Playback ───────────────────────────────────────────────────────
    def set_output_sample_rate(self, sr: int) -> None:
        """Set the sample rate of the PCM data being fed to play()."""
        self._playback_sample_rate = sr

    def _ensure_playback_thread(self) -> None:
        """Start the ALSA playback thread if it is not running.

        The capture stream (and its playback thread) is torn down at the end
        of every listen turn, but dance music / TTS can be queued right
        after — so a queued ``play()`` re-opens a dedicated playback PCM.
        """
        if not self._alsa or self._alsa_playback_stop.is_set():
            return
        if any(
            t.name == "alsa-playback" and t.is_alive()
            for t in getattr(self, "_alsa_threads", ())
        ):
            return
        import alsaaudio

        name = getattr(self.resolved_info, "input_name", "") or ""
        m = re.search(r"\(hw:\d+,\d+\)", name)
        device = (
            m.group(0).strip("()")
            if m
            else os.environ.get("REACHY_ALSA_DEVICE", "default")
        )
        try:
            pcm = alsaaudio.PCM(
                alsaaudio.PCM_PLAYBACK, alsaaudio.PCM_NORMAL, device=device
            )
            pcm.setchannels(2)
            pcm.setrate(self.input_sr)
            pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
            pcm.setperiodsize(self._chunk_frames)
        except Exception as exc:
            logger.warning("无法打开独立播放通道: %r", exc)
            return
        self._alsa_playback_pcm = pcm
        thread = threading.Thread(
            target=self._alsa_playback_loop,
            args=(pcm,),
            name="alsa-playback",
            daemon=True,
        )
        thread.start()
        self._alsa_threads.append(thread)
        logger.info("Playback thread started (no active capture)")

    async def play(self, pcm: bytes) -> None:
        """Queue PCM audio for playback. Resamples if needed."""
        self._ensure_playback_thread()
        if self._playback_sample_rate != self.output_sr:
            pcm = self._resample_pcm(pcm, self._playback_sample_rate, self.output_sr)
        with self._playback_lock:
            self._playback_buffer.extend(pcm)
            self._playback_done = False

    def mark_playback_done(self) -> None:
        """Signal that no more audio will be queued."""
        self._playback_done = True

    def stop_playback(self) -> None:
        """Immediately stop playback (barge-in)."""
        with self._playback_lock:
            self._playback_buffer.clear()
            self._playback_done = True

    @property
    def is_playing(self) -> bool:
        """True while audio is still in the playback buffer."""
        with self._playback_lock:
            return len(self._playback_buffer) > 0

    @property
    def volume(self) -> float:
        """Current playback volume as gain multiplier (0.0 - 8.0)."""
        return getattr(self, "_volume", 1.5)

    @volume.setter
    def volume(self, val: float) -> None:
        """Set playback volume as PCM gain multiplier (0.0 - 8.0)."""
        self._volume = max(0.0, min(MAX_PLAYBACK_GAIN, float(val)))

    def play_rms(self) -> float:
        """Smoothed RMS (0..1) of speaker output — for speech-timed animation."""
        return getattr(self, "_play_rms", 0.0)

    def set_playback_observer(
        self, callback: Callable[[bytes, int], None] | None
    ) -> None:
        """Install a non-blocking observer for PCM consumed by the speaker."""
        self._playback_observer = callback

    def _notify_playback_observer(self, pcm: bytes) -> None:
        callback = self._playback_observer
        if callback is None:
            return
        try:
            callback(pcm, self.input_sr)
        except Exception:
            # Never let cosmetic motion break the real-time audio thread.
            logger.warning("playback observer failed", exc_info=True)

    @property
    def capture_rms(self) -> float:
        """Smoothed RMS (0..1) of microphone input — for level diagnostics."""
        return getattr(self, "_capture_rms", 0.0)

    @property
    def last_sample_age_s(self) -> float:
        """Seconds since the last captured sample (inf if none yet)."""
        if not getattr(self, "_last_sample_at", 0.0):
            return float("inf")
        return time.monotonic() - self._last_sample_at

    @property
    def mic_gain(self) -> float:
        """Current microphone preamp gain (1.0 = unity, up to 10.0)."""
        return getattr(self, "_mic_gain", 1.0)

    @mic_gain.setter
    def mic_gain(self, val: float) -> None:
        self._mic_gain = max(0.1, min(10.0, float(val)))

    # ── Lifecycle ──────────────────────────────────────────────────────
    async def open(self) -> None:
        """Initialize the audio device (AudioBackend compat)."""
        pass  # Duplex stream opens on first capture

    async def close(self) -> None:
        self._close_duplex(keep_playback=False)

    # ── Backend interface compat ──────────────────────────────────────

    @property
    def backend_name(self) -> str:
        return "alsa_duplex" if self._alsa else "sounddevice_duplex"

    @property
    def supports_doa(self) -> bool:
        return False

    def get_doa(self) -> float | None:
        return None

    @property
    def supports_wobbling(self) -> bool:
        return False

    def enable_wobbling(self) -> None:
        pass

    def disable_wobbling(self) -> None:
        pass

    @property
    def resolved_info_dict(self) -> dict[str, object]:
        """Device info as a dict (AudioBackend compat)."""
        return self.resolved_info.to_dict()

    # ── Duplex stream ──────────────────────────────────────────────────
    def _open_duplex(self) -> None:
        """Open the full-duplex stream.

        Reachy Mini XMOS card: pyalsaaudio rw-mode PCM (mmap/PortAudio
        capture is silent on this card).  Otherwise sounddevice RawStream,
        trying 2-channel input first (echo-cancelled channel), then mono.
        """
        if self._alsa:
            self._open_alsa_duplex()
            return
        last_err: Exception | None = None
        channel_options = [
            (in_ch, out_ch)
            for in_ch, out_ch in ((2, 2), (2, 1), (1, 2), (1, 1))
            if in_ch <= self.resolved_info.max_input_channels
            and out_ch <= self.resolved_info.max_output_channels
        ]
        for in_ch, out_ch in channel_options:
            try:
                stream = sd.RawStream(
                    samplerate=self.input_sr,
                    blocksize=self._chunk_frames,
                    device=(self.input_device, self.output_device),
                    channels=(in_ch, out_ch),
                    dtype="int16",
                    callback=self._duplex_cb,
                )
                stream.start()
            except Exception as e:
                last_err = e
                continue
            self._in_ch, self._out_ch = in_ch, out_ch
            self._duplex_stream = stream
            self.resolved_info = replace(
                self.resolved_info,
                stream_input_channels=in_ch,
                stream_output_channels=out_ch,
            )
            logger.info(
                "Duplex stream: input=[%d] %s output=[%d] %s "
                "sr=%d ch=(%d,%d) input_channel=%d chunk=%d backend=%s",
                self.resolved_info.input_index,
                self.resolved_info.input_name,
                self.resolved_info.output_index,
                self.resolved_info.output_name,
                self.input_sr,
                in_ch,
                out_ch,
                self._input_channel,
                self._chunk_frames,
                self.resolved_info.backend,
            )
            return
        raise RuntimeError(f"Cannot open duplex stream: {last_err!r}")

    def _close_duplex(self, keep_playback: bool = True) -> None:
        """Stop the capture side; playback survives by default.

        The ASR/listen loop closes the capture stream at the end of every
        turn, but the playback thread must keep running so dance music and
        TTS queued outside a capture session are still heard.  Only a full
        ``close()`` stops playback (``keep_playback=False``).
        """
        if self._alsa:
            self._alsa_stop.set()
            for t in list(self._alsa_threads):
                if t.name == "alsa-playback" and keep_playback:
                    continue
                t.join(timeout=3.0)
                if t in self._alsa_threads:
                    self._alsa_threads.remove(t)
            for pcm in getattr(self, "_alsa_pcms", ()):
                try:
                    pcm.close()
                except Exception:
                    pass
            self._alsa_pcms = ()
            self._duplex_stream = None
            if not keep_playback:
                self._alsa_playback_stop.set()
                for t in list(self._alsa_threads):
                    t.join(timeout=3.0)
                self._alsa_threads = []
                # Drop the playback PCM explicitly — otherwise the ALSA
                # device stays open until GC, and the next launch fails
                # with "device busy" on the XMOS card.
                pb = getattr(self, "_alsa_playback_pcm", None)
                if pb is not None:
                    try:
                        pb.close()
                    except Exception:
                        pass
                self._alsa_playback_pcm = None
            return
        if self._duplex_stream is not None:
            try:
                self._duplex_stream.stop()
                self._duplex_stream.close()
            except Exception:
                pass
            self._duplex_stream = None

    # ── ALSA direct backend ────────────────────────────────────────────
    def _open_alsa_duplex(self) -> None:
        """Open pyalsaaudio capture+playback PCMs on the same card, rw mode.

        arecord --mmap on the XMOS card returns silence, so we deliberately
        use default (rw) access — verified to produce real mic audio.
        """
        import alsaaudio  # type: ignore[import-not-found]

        name = getattr(self.resolved_info, "input_name", "") or ""
        m = re.search(r"\(hw:\d+,\d+\)", name)
        device = (
            m.group(0).strip("()")
            if m
            else os.environ.get("REACHY_ALSA_DEVICE", "default")
        )

        self._alsa_stop.clear()
        self._in_ch = 2  # XMOS is natively 2-in; pick the AEC channel below
        self._out_ch = 2  # duplicate mono to both speaker channels
        self._duplex_stream = None  # not a PortAudio stream; threads own PCMs

        def open_pcm(direction: int) -> "alsaaudio.PCM":
            pcm = alsaaudio.PCM(direction, alsaaudio.PCM_NORMAL, device=device)
            pcm.setchannels(2)  # native card channels — mono requests fail on hw:
            pcm.setrate(self.input_sr)
            pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
            pcm.setperiodsize(self._chunk_frames)
            return pcm

        capture = open_pcm(alsaaudio.PCM_CAPTURE)
        self._alsa_pcms = (capture,)
        threads = [
            threading.Thread(
                target=self._alsa_capture_loop,
                args=(capture,),
                name="alsa-capture",
                daemon=True,
            ),
        ]
        # Reuse a surviving playback thread (dance music / TTS keep it alive
        # across capture teardown) instead of re-opening the card.
        if not any(
            t.name == "alsa-playback" and t.is_alive()
            for t in getattr(self, "_alsa_threads", ())
        ):
            playback = open_pcm(alsaaudio.PCM_PLAYBACK)
            self._alsa_playback_pcm = playback
            self._alsa_playback_stop.clear()
            threads.append(
                threading.Thread(
                    target=self._alsa_playback_loop,
                    args=(playback,),
                    name="alsa-playback",
                    daemon=True,
                )
            )
        else:
            logger.info("复用已有播放线程（capture 重建）")
        for t in threads:
            t.start()
        self._alsa_threads.extend(threads)
        self.resolved_info = replace(
            self.resolved_info,
            stream_input_channels=self._in_ch,
            stream_output_channels=self._out_ch,
            backend="alsa-direct",
        )
        logger.info(
            "Duplex stream (ALSA): device=%s sr=%d ch=(%d,%d) input_channel=%d "
            "chunk=%d backend=alsa-direct",
            device,
            self.input_sr,
            self._in_ch,
            self._out_ch,
            self._input_channel,
            self._chunk_frames,
        )

    def _alsa_capture_loop(self, pcm: "alsaaudio.PCM") -> None:
        """Read mic PCM in rw mode and push into the asyncio queue.

        The whole body is exception-guarded so this thread can never die
        (a dead capture thread leaves the engine waiting on an empty queue
        with no logs).  Consecutive read failures throttle to 1 Hz so a
        wedged XMOS card does not spin the CPU.
        """
        errors = 0
        while not self._alsa_stop.is_set():
            try:
                length, data = pcm.read()
                if length > 0:
                    self._last_sample_at = time.monotonic()
                if length <= 0:
                    time.sleep(0.01)
                    continue
                errors = 0
                if length < self._chunk_frames:
                    data = data + b"\x00" * ((self._chunk_frames - length) * 4)
                x = np.frombuffer(data, dtype=np.int16)
                if self._in_ch > 1:
                    m = x.reshape(-1, self._in_ch)
                    ch = self._input_channel if self._input_channel < self._in_ch else 0
                    x = m[:, ch].copy()
                # Apply microphone preamp gain with soft clipping (mirror _duplex_cb)
                mic_gain = getattr(self, "_mic_gain", 1.0)
                if mic_gain != 1.0:
                    x_float = x.astype(np.float32) * mic_gain
                    x_float = np.tanh(x_float / 28000.0) * 28000.0
                    x = np.clip(x_float, -32768, 32767).astype(np.int16)
                buf = x.tobytes()
                rms = (
                    float(np.sqrt(np.mean(x.astype(np.float32) ** 2))) / 32768.0
                    if x.size
                    else 0.0
                )
                prev = getattr(self, "_capture_rms", 0.0)
                self._capture_rms = prev + 0.3 * (rms - prev)
                if self._loop is not None and self._in_queue is not None:
                    self._loop.call_soon_threadsafe(self._safe_put, buf)
            except Exception as exc:
                errors += 1
                if errors in (1, 10, 50):
                    logger.warning("alsa capture read error x%d: %r", errors, exc)
                time.sleep(1.0 if errors >= 50 else 0.05)
                continue

    def _alsa_playback_loop(self, pcm: "alsaaudio.PCM") -> None:
        """Drain the playback buffer into the speaker (rw mode).

        Runs independently of the capture stream: it keeps writing silence
        while idle so the XMOS playback pipe stays open, and survives
        capture teardown (dance music / TTS queued outside a listen turn
        must still be heard).  Only a full ``close()`` stops it.
        """
        need = self._chunk_frames * 2  # int16 mono bytes per period
        period_s = need / 2 / self.input_sr  # wall-clock seconds per period
        errors = 0
        while not self._alsa_playback_stop.is_set():
            try:
                with self._playback_lock:
                    n = min(need, len(self._playback_buffer))
                    mono = bytes(self._playback_buffer[:n])
                    del self._playback_buffer[:n]
                if n > 0:
                    s = np.frombuffer(mono, dtype=np.int16).astype(np.float32) / 32768.0
                    vol = getattr(self, "_volume", 1.0)
                    s = s * vol
                    s = np.tanh(s)  # soft clip
                    mono = (
                        (np.clip(s, -0.99, 0.99) * 32767.0).astype(np.int16).tobytes()
                    )
                    rms = float(np.sqrt(np.mean(s * s))) if s.size else 0.0
                else:
                    rms = 0.0
                self._play_rms = getattr(self, "_play_rms", 0.0)
                self._play_rms += 0.3 * (rms - self._play_rms)
                if n < need:
                    mono += b"\x00" * (need - n)
                self._notify_playback_observer(mono)
                # Duplicate mono to both speaker channels (2-out card)
                y = np.frombuffer(mono, dtype=np.int16)
                out = np.repeat(y[:, None], 2, axis=1).tobytes()
                write_start = time.monotonic()
                pcm.write(out)
                # Enforce a real-time cadence: if the XMOS ALSA buffer is
                # large, write() returns without blocking and the thread
                # would drain the whole playback buffer instantly — the
                # prefill cushion vanishes and music stutters 1s-on/1s-off.
                # Top the period up to wall-clock so consumption == playback.
                spent = time.monotonic() - write_start
                if spent < period_s:
                    time.sleep(period_s - spent)
                errors = 0
            except Exception as exc:
                errors += 1
                if errors in (1, 10, 50):
                    logger.warning("alsa playback write error x%d: %r", errors, exc)
                time.sleep(1.0 if errors >= 50 else 0.05)

    def _duplex_cb(self, indata, outdata, frames, time_info, status) -> None:
        """PortAudio callback — runs on a real-time thread."""
        if status:
            logger.debug("duplex status: %s", status)

        # ── Capture ──
        try:
            self._last_sample_at = time.monotonic()
            x = np.frombuffer(bytes(indata), dtype=np.int16)
            if self._in_ch > 1:
                m = x.reshape(-1, self._in_ch)
                ch = self._input_channel if self._input_channel < self._in_ch else 0
                x = m[:, ch].copy()
            # Apply microphone preamp gain with soft clipping
            mic_gain = getattr(self, "_mic_gain", 1.0)
            if mic_gain != 1.0:
                x_float = x.astype(np.float32)
                x_float = x_float * mic_gain
                # Soft clip to prevent hard distortion at high gain
                x_float = np.tanh(x_float / 28000.0) * 28000.0
                x = np.clip(x_float, -32768, 32767).astype(np.int16)
            buf = x.tobytes()
            # Smoothed capture RMS for diagnostics
            rms = (
                float(np.sqrt(np.mean(x.astype(np.float32) ** 2))) / 32768.0
                if x.size
                else 0.0
            )
            prev = getattr(self, "_capture_rms", 0.0)
            self._capture_rms = prev + 0.3 * (rms - prev)
            if self._loop is not None and self._in_queue is not None:
                self._loop.call_soon_threadsafe(self._safe_put, buf)
        except Exception:
            logger.debug("duplex capture error", exc_info=True)

        # ── Playback ──
        try:
            need_mono = frames * 2  # int16 mono bytes
            with self._playback_lock:
                n = min(need_mono, len(self._playback_buffer))
                mono = bytes(self._playback_buffer[:n])
                del self._playback_buffer[:n]

            if n > 0:
                s = np.frombuffer(mono, dtype=np.int16).astype(np.float32) / 32768.0
                # Apply the Dashboard-controlled PCM gain before limiting.
                vol = getattr(self, "_volume", 1.0)
                s = s * vol
                # Soft clip to prevent harsh distortion
                s = np.tanh(s)
                mono = (np.clip(s, -0.99, 0.99) * 32767.0).astype(np.int16).tobytes()
                rms = float(np.sqrt(np.mean(s * s))) if s.size else 0.0
            else:
                rms = 0.0
            self._play_rms = getattr(self, "_play_rms", 0.0)
            self._play_rms += 0.3 * (rms - self._play_rms)

            if n < need_mono:
                mono += b"\x00" * (need_mono - n)
            self._notify_playback_observer(mono)
            if self._out_ch == 1:
                out = mono
            else:
                y = np.frombuffer(mono, dtype=np.int16)
                out = np.repeat(y[:, None], self._out_ch, axis=1).tobytes()
            outdata[: len(out)] = out
        except Exception:
            logger.debug("duplex playback error", exc_info=True)
            outdata[:] = b"\x00" * len(outdata)

    def _safe_put(self, buf: bytes) -> None:
        """Thread-safe queue insertion from the audio callback."""
        try:
            self._in_queue.put_nowait(buf)
        except asyncio.QueueFull:
            pass

    @staticmethod
    def _resample_pcm(pcm: bytes, src_sr: int, dst_sr: int) -> bytes:
        """Simple linear resampling for PCM int16."""
        if src_sr == dst_sr:
            return pcm
        ratio = dst_sr / src_sr
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        n_out = int(len(samples) * ratio)
        indices = np.linspace(0, len(samples) - 1, n_out)
        resampled = np.interp(indices, np.arange(len(samples)), samples)
        return resampled.astype(np.int16).tobytes()
