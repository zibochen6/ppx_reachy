"""Tests for the infinite beat-synced dance (pure math + fake robot)."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from chaihuo_reachy.beat_dance import (
    BeatDanceController,
    BeatMotionSynthesizer,
    BeatSyncOscillator,
    _clamp,
    _compute_smooth_energy_buckets,
    _load_timeline,
)

PITCH_BASE_MAX = 22.0  # pitch soft cap


def test_oscillator_phase_is_pure_function_of_elapsed() -> None:
    osc = BeatSyncOscillator(freq=2.0, phase_offset=math.pi / 4)
    t = 1.234
    expected = math.sin(2 * math.pi * 2.0 * t + math.pi / 4)
    assert osc.advance(t) == pytest.approx(2 * math.pi * 2.0 * t + math.pi / 4)
    assert osc.value() == pytest.approx(expected)
    # Re-advancing to the same time yields the same value (no accumulated drift)
    osc2 = BeatSyncOscillator(freq=2.0, phase_offset=math.pi / 4)
    osc2.advance(t)
    assert osc2.value() == pytest.approx(expected)


def test_energy_bucket_short_segments_merge() -> None:
    # 10 beats: FADE x3, MID x1, PEAK x6 — the lone MID must merge into a neighbour
    beats = [
        {"rms": 0.05, "strength": 0.1},   # FADE
        {"rms": 0.05, "strength": 0.1},   # FADE
        {"rms": 0.05, "strength": 0.1},   # FADE
        {"rms": 0.9, "strength": 0.8},    # PEAK (short)
        {"rms": 0.9, "strength": 0.8},    # PEAK
        {"rms": 0.9, "strength": 0.8},    # PEAK
        {"rms": 0.9, "strength": 0.8},    # PEAK
        {"rms": 0.9, "strength": 0.8},    # PEAK
        {"rms": 0.9, "strength": 0.8},    # PEAK
        {"rms": 0.9, "strength": 0.8},    # PEAK
    ]
    buckets = _compute_smooth_energy_buckets(beats, min_segment_beats=4)
    assert len(buckets) == 10
    # All PEAK beats stay PEAK; the boundary has no isolated 1-beat segment
    assert buckets.count("PEAK") >= 6


def test_load_timeline_smooths_buckets(tmp_path) -> None:
    beats = [
        {"time": i * 0.476, "strength": 0.5, "energy_bucket": "MID"}
        for i in range(12)
    ]
    beats[3]["energy_bucket"] = "PEAK"  # isolated short segment
    path = tmp_path / "tl.json"
    path.write_text(json.dumps({
        "tempo": 126.0,
        "duration": 12.0,
        "beats": beats,
    }), encoding="utf-8")
    tl = _load_timeline(path)
    assert tl["tempo"] == 126.0
    assert len(tl["beats"]) == 12
    assert tl["beats"][3]["energy_bucket"] != "PEAK"  # merged away


def _beat(energy: str = "MID", strength: float = 0.5) -> dict:
    return {"time": 0.0, "strength": strength, "energy_bucket": energy}


def test_synthesizer_outputs_respect_soft_caps() -> None:
    synth = BeatMotionSynthesizer(beat_interval=0.476)
    synth.on_run_start()
    for t in np.linspace(0.0, 5.0, 60):
        pose, antennas, body_yaw, status = synth.synthesize(t, _beat("PEAK", 0.9), 0.0)
        # pose is a 4x4 homogeneous head matrix
        assert pose.shape == (4, 4)
        assert len(antennas) == 2
        assert abs(status["head_yaw_deg"]) <= 18.0 + 1e-6
        assert abs(status["body_yaw_deg"]) <= 25.0 + 1e-6
        assert abs(status["left_antenna_deg"]) <= 28.0 + 1e-6
        assert abs(status["right_antenna_deg"]) <= 28.0 + 1e-6
        assert -8.0 <= status["z_mm"] <= 16.0


def test_synthesizer_bucket_switch_triggers_crossfade() -> None:
    synth = BeatMotionSynthesizer(beat_interval=0.476)
    synth.on_run_start()
    synth.synthesize(0.1, _beat("MID", 0.5), 0.0)
    synth.synthesize(0.3, _beat("PEAK", 0.9), 0.0)
    assert synth._crossfade.active
    assert synth._crossfade.progress == 0.0
    # after 2 beats the crossfade finishes
    synth.synthesize(0.3 + 2.0 * 0.476 + 0.01, _beat("PEAK", 0.9), 0.0)
    assert not synth._crossfade.active


def test_clamp_bounds() -> None:
    assert _clamp(5.0, 0.0, 1.0) == 1.0
    assert _clamp(-1.0, 0.0, 1.0) == 0.0
    assert _clamp(0.5, 0.0, 1.0) == 0.5


class _FakeReachy:
    def __init__(self) -> None:
        self.targets: list[tuple] = []
        self.neutral_calls = 0

    def set_target(self, head=None, antennas=None, body_yaw=None) -> None:
        self.targets.append((head, antennas, body_yaw))

    def goto_target(self, head=None, antennas=None, body_yaw=None, duration=0.8) -> None:
        self.neutral_calls += 1


def _make_timeline(tmp_path, beats=8) -> None:
    path = tmp_path / "beat_timeline.json"
    path.write_text(json.dumps({
        "tempo": 120.0,
        "duration": 4.0,
        "beats": [
            {"time": i * 0.5, "strength": 0.5, "energy_bucket": "MID"}
            for i in range(beats)
        ],
    }), encoding="utf-8")
    return path


def _make_music(tmp_path) -> None:
    import wave

    with wave.open(str(tmp_path / "beat.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)


def test_controller_start_stop(tmp_path) -> None:
    from chaihuo_reachy.config import Config

    timeline = _make_timeline(tmp_path)
    _make_music(tmp_path)
    reachy = _FakeReachy()
    cfg = Config(
        beat_timeline_path=str(timeline),
        beat_music_path=str(tmp_path / "beat.wav"),
    )
    ctl = BeatDanceController(reachy, cfg)
    info = ctl.start()
    assert info is not None
    sr, pcm = info
    assert sr == 16000
    assert len(pcm) > 0
    time.sleep(0.15)  # let the 50 Hz thread emit a few targets
    assert ctl.is_active
    assert len(reachy.targets) > 0
    pose, antennas, body_yaw = reachy.targets[-1]
    assert pose.shape == (4, 4)
    assert len(antennas) == 2
    ctl.stop()
    assert not ctl.is_active
    assert reachy.neutral_calls == 1  # returned to neutral


def test_controller_start_without_timeline_returns_none(tmp_path) -> None:
    from chaihuo_reachy.config import Config

    reachy = _FakeReachy()
    cfg = Config(
        beat_timeline_path=str(tmp_path / "missing.json"),
        beat_music_path=str(tmp_path / "beat.wav"),
    )
    ctl = BeatDanceController(reachy, cfg)
    assert ctl.start() is None
    assert not ctl.is_active


def test_synthesizer_motif_variation() -> None:
    """Different 4-beat groups must lead different DOFs (not just yaw/antenna)."""
    synth = BeatMotionSynthesizer(beat_interval=0.476)
    synth.on_run_start()
    beat = _beat("PEAK", 0.9)
    # capture peak |yaw| and |roll| across one full 4-beat group cycle
    yaw_peaks: list[float] = []
    roll_peaks: list[float] = []
    pitch_peaks: list[float] = []
    for group_start in (0.0, 4 * 0.476, 8 * 0.476, 12 * 0.476):
        ymax = rmax = pmax = 0.0
        for k in range(40):
            t = group_start + k * 0.02
            _, _, _, st = synth.synthesize(t, beat, group_start)
            ymax = max(ymax, abs(st["head_yaw_deg"]))
            rmax = max(rmax, abs(st["head_roll_deg"]))
            pmax = max(pmax, abs(st["head_pitch_deg"]))
        yaw_peaks.append(ymax)
        roll_peaks.append(rmax)
        pitch_peaks.append(pmax)
    # The dominant DOF changes across groups: yaw peaks differ between the
    # yaw-led group and the roll-led group; roll becomes active somewhere.
    assert max(yaw_peaks) > min(yaw_peaks) + 2.0, f"yaw monotone: {yaw_peaks}"
    assert max(roll_peaks) > 4.0, f"roll never active: {roll_peaks}"
    assert max(pitch_peaks) > 4.0, f"pitch never active: {pitch_peaks}"


def test_synthesizer_roll_respects_cap() -> None:
    synth = BeatMotionSynthesizer(beat_interval=0.476)
    synth.on_run_start()
    for t in np.linspace(0.0, 6.0, 80):
        _, _, _, st = synth.synthesize(t, _beat("PEAK", 0.9), 0.0)
        assert abs(st["head_roll_deg"]) <= 18.0 + 1e-6
        assert -8.0 <= st["z_mm"] <= 16.0
