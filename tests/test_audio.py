from __future__ import annotations

import threading
import types

import numpy as np
import pytest

from chaihuo_reachy.audio import AudioDeviceResolutionError, resolve_audio_device
from chaihuo_reachy.backends.interfaces import (
    playback_gain_from_percent,
    playback_percent_from_gain,
)


DEVICES = [
    {
        "name": "Reachy Mini Audio",
        "max_input_channels": 2,
        "max_output_channels": 2,
        "default_samplerate": 16000,
    },
    {
        "name": "MacBook Pro Microphone",
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_samplerate": 48000,
    },
    {
        "name": "MacBook Pro Speakers",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000,
    },
    {
        "name": "Reachy Mini Camera: USB Audio",
        "max_input_channels": 2,
        "max_output_channels": 2,
        "default_samplerate": 48000,
    },
]


def test_dashboard_volume_mapping_preserves_default_and_expands_maximum() -> None:
    assert playback_gain_from_percent(0) == 0.0
    assert playback_gain_from_percent(50) == 2.0
    assert playback_gain_from_percent(100) == 8.0
    assert playback_percent_from_gain(2.0) == 50
    assert playback_percent_from_gain(8.0) == 100


def test_auto_selects_unique_reachy_duplex_and_rejects_camera_audio() -> None:
    info = resolve_audio_device("auto", devices=DEVICES)
    assert info.input_index == info.output_index == 0
    assert info.input_name == "Reachy Mini Audio"
    assert info.max_input_channels == info.max_output_channels == 2


def test_omitted_selector_does_not_fall_back_to_mac_defaults() -> None:
    with pytest.raises(AudioDeviceResolutionError, match="REACHY_AUDIO_DEVICE=default"):
        resolve_audio_device(None, devices=DEVICES[1:], default_device=(0, 1))


def test_explicit_default_allows_separate_system_devices() -> None:
    info = resolve_audio_device("default", devices=DEVICES, default_device=(1, 2))
    assert info.input_name == "MacBook Pro Microphone"
    assert info.output_name == "MacBook Pro Speakers"


def test_explicit_index_must_be_full_duplex() -> None:
    with pytest.raises(AudioDeviceResolutionError, match="not full duplex"):
        resolve_audio_device(1, devices=DEVICES)


def test_ambiguous_reachy_candidates_fail_with_indexes() -> None:
    ambiguous = [DEVICES[0], {**DEVICES[0], "name": "Reachy Mini Audio: USB"}]
    # Exact normalized name wins over a substring candidate.
    assert resolve_audio_device("auto", devices=ambiguous).input_index == 0
    truly_ambiguous = [
        {**DEVICES[0], "name": "USB Reachy Mini Audio A"},
        {**DEVICES[0], "name": "USB Reachy Mini Audio B"},
    ]
    with pytest.raises(AudioDeviceResolutionError, match="ambiguous indexes"):
        resolve_audio_device("auto", devices=truly_ambiguous)


class _FakePCM:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _make_alsa_io() -> tuple:
    from chaihuo_reachy.audio import DuplexAudioIO

    audio = object.__new__(DuplexAudioIO)
    audio._alsa = True
    audio._alsa_stop = threading.Event()
    audio._alsa_playback_stop = threading.Event()
    audio._alsa_threads = []
    audio._alsa_pcms = ()
    audio._duplex_stream = None
    audio._alsa_playback_pcm = _FakePCM()
    return audio, audio._alsa_playback_pcm


def test_alsa_full_close_releases_playback_pcm() -> None:
    audio, pcm = _make_alsa_io()
    audio._close_duplex(keep_playback=False)
    assert pcm.closed  # else the XMOS device stays busy for the next launch


def test_alsa_capture_teardown_keeps_playback_pcm() -> None:
    audio, pcm = _make_alsa_io()
    audio._close_duplex(keep_playback=True)  # end of a listen turn
    assert not pcm.closed  # dance music / TTS outside a turn must keep playing
    ambiguous = [DEVICES[0], {**DEVICES[0], "name": "Reachy Mini Audio: USB"}]
    # Exact normalized name wins over a substring candidate.
    assert resolve_audio_device("auto", devices=ambiguous).input_index == 0
    truly_ambiguous = [
        {**DEVICES[0], "name": "USB Reachy Mini Audio A"},
        {**DEVICES[0], "name": "USB Reachy Mini Audio B"},
    ]
    with pytest.raises(AudioDeviceResolutionError, match="ambiguous indexes"):
        resolve_audio_device("auto", devices=truly_ambiguous)


def test_playback_observer_receives_consumed_pcm_and_can_be_unregistered() -> None:
    from chaihuo_reachy.audio import DuplexAudioIO

    audio = object.__new__(DuplexAudioIO)
    audio.input_sr = 16_000
    received: list[tuple[bytes, int]] = []
    audio.set_playback_observer(lambda pcm, sr: received.append((pcm, sr)))
    audio._notify_playback_observer(b"post-gain")
    assert received == [(b"post-gain", 16_000)]

    audio.set_playback_observer(None)
    audio._notify_playback_observer(b"ignored")
    assert received == [(b"post-gain", 16_000)]


def test_playback_observer_failure_never_escapes_audio_thread() -> None:
    from chaihuo_reachy.audio import DuplexAudioIO

    audio = object.__new__(DuplexAudioIO)
    audio.input_sr = 48_000

    def broken(_pcm: bytes, _sr: int) -> None:
        raise RuntimeError("motion overloaded")

    audio.set_playback_observer(broken)
    audio._notify_playback_observer(b"pcm")


class _FakeOutputStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _make_sounddevice_io() -> tuple:
    from chaihuo_reachy.audio import DuplexAudioIO

    audio = object.__new__(DuplexAudioIO)
    audio._alsa = False
    audio._duplex_stream = None
    audio._output_stream = None
    audio._out_ch = 1
    audio.input_sr = 16_000
    audio._chunk_frames = 1600
    audio.output_device = 0
    return audio


def test_playback_only_stream_opens_when_no_capture_session(monkeypatch) -> None:
    # Dance music / TTS queued while idle (no listen turn) must still be
    # heard — the duplex stream is closed between turns, so a dedicated
    # output stream takes over.  (Previously the buffer grew in silence.)
    from chaihuo_reachy import audio as audio_mod

    monkeypatch.setattr(audio_mod, "sd", types.SimpleNamespace(OutputStream=_FakeOutputStream))

    audio = _make_sounddevice_io()
    audio._ensure_output_stream()
    stream = audio._output_stream
    assert isinstance(stream, _FakeOutputStream)
    assert stream.started
    assert stream.kwargs["samplerate"] == 16_000
    assert stream.kwargs["device"] == 0
    assert stream.kwargs["callback"] == audio._output_cb

    # Already open → no-op; a live duplex session → no-op either.
    audio._ensure_output_stream()
    assert audio._output_stream is stream
    audio._duplex_stream = object()
    audio._output_stream = None
    audio._ensure_output_stream()
    assert audio._output_stream is None


def test_playback_only_stream_closes_on_full_shutdown() -> None:
    audio = _make_sounddevice_io()
    audio._output_stream = _FakeOutputStream()
    audio._close_duplex(keep_playback=False)
    assert audio._output_stream is None  # device released for the next launch


def test_output_stream_callback_drains_playback_buffer() -> None:
    from chaihuo_reachy.audio import DuplexAudioIO

    audio = _make_sounddevice_io()
    audio._playback_lock = threading.Lock()
    audio._playback_done = False
    audio._play_rms = 0.0
    audio._volume = 1.0
    observed: list[tuple[bytes, int]] = []
    audio.set_playback_observer(lambda pcm, sr: observed.append((pcm, sr)))

    samples = np.full(1600, 1000, dtype=np.int16)
    audio._playback_buffer = bytearray(samples.tobytes())
    outdata = np.zeros(1600, dtype=np.int16)
    audio._output_cb(outdata, 1600, None, None)

    assert len(audio._playback_buffer) == 0  # full chunk consumed
    expected = (np.clip(np.tanh(samples.astype(np.float32) / 32768.0), -0.99, 0.99) * 32767.0).astype(np.int16)
    assert outdata.tobytes() == expected.tobytes()
    assert observed == [(expected.tobytes(), 16_000)]  # observer sees post-gain PCM
