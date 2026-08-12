"""Tests for the dance backing-track helpers (music.py)."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from chaihuo_reachy.music import detect_bpm, load_audio, read_track, resolve_track


def _make_wav(path: Path, sr: int = 16000, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = b"".join(struct.pack("<h", 0) for _ in range(int(sr * seconds)))
        wf.writeframes(frames)


def _beat_pulse_pcm(sr: int, bpm: float, seconds: float) -> bytes:
    """Synthesize a click track with one short pulse per beat."""
    n = int(sr * seconds)
    x = np.zeros(n, dtype=np.int16)
    step = int(sr * 60.0 / bpm)
    for t in range(0, n, step):
        x[t : t + int(sr * 0.05)] = 8000
    return x.tobytes()


def test_detect_bpm_finds_known_tempo() -> None:
    sr = 16000
    pcm = _beat_pulse_pcm(sr, bpm=120.0, seconds=5.0)
    assert detect_bpm(pcm, sr) == pytest.approx(120.0, abs=3)


def test_detect_bpm_handles_slower_tempo() -> None:
    sr = 16000
    pcm = _beat_pulse_pcm(sr, bpm=90.0, seconds=6.0)
    assert detect_bpm(pcm, sr) == pytest.approx(90.0, abs=3)


def test_detect_bpm_returns_none_on_silence() -> None:
    assert detect_bpm(b"\x00" * (16000 * 4 * 2), 16000) is None


def test_resolve_track_prefers_style_then_any(tmp_path) -> None:
    _make_wav(tmp_path / "happy.wav", seconds=0.3)
    _make_wav(tmp_path / "other.wav", seconds=0.3)
    assert resolve_track(tmp_path, "happy").name == "happy.wav"
    # No robot track → falls back to any wav in the dir.
    assert resolve_track(tmp_path, "robot").name in ("happy.wav", "other.wav")
    assert resolve_track(tmp_path / "missing", "happy") is None


def test_read_track_rejects_non_mono(tmp_path) -> None:
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 1600 * 4)
    assert read_track(path) is None
    assert read_track(tmp_path / "nope.wav") is None


def test_load_audio_wav_and_missing(tmp_path) -> None:
    path = tmp_path / "beat.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    info = load_audio(path)
    assert info is not None
    sr, pcm = info
    assert sr == 16000
    assert len(pcm) == 3200
    assert load_audio(tmp_path / "missing.wav") is None


def test_load_audio_mp3_falls_back_to_wav(tmp_path) -> None:
    # miniaudio may be missing in some dev environments: the sibling .wav
    # fallback must kick in.
    import sys

    wav = tmp_path / "beat.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    mp3 = tmp_path / "beat.mp3"
    mp3.write_bytes(b"not-a-real-mp3")

    monkeypatch_import = pytest.MonkeyPatch()
    monkeypatch_import.setitem(sys.modules, "miniaudio", None)
    try:
        info = load_audio(mp3)
    finally:
        monkeypatch_import.undo()
    assert info is not None  # fell back to beat.wav
    assert info[0] == 16000
