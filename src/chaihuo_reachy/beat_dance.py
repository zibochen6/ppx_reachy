"""Infinite beat-synced dance — ported from reachy_mini_dance_by_beat.

Plays ``beat.mp3`` while the robot dances to the beat forever (loop).
Music is fed through ``DuplexAudioIO.play()`` by the engine; this module
owns the motion side: a 50 Hz wall-clock thread drives eight phase-locked
sine oscillators (phase = 2π·freq·elapsed, no drift), an energy-bucket
system classifies each beat (FADE/LOW/MID/TILT/PEAK), and every tick sends
a real-time ``robot.set_target(...)``.

The pure math lives in :class:`BeatMotionSynthesizer` so it is unit
testable without hardware; :class:`BeatDanceController` is the thin
threaded driver.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chaihuo_reachy.config import Config
from chaihuo_reachy.music import load_audio

logger = logging.getLogger("chaihuo_reachy.beat_dance")

CONTROL_PERIOD_S = 0.02  # 50 Hz motion loop

# 能量桶 → 动作样式映射（保留源项目定义）
BUCKET_STYLE_MAP: dict[str, str] = {
    "LOW": "bounce", "MID": "groove", "TILT": "sway",
    "PEAK": "groove", "FADE": "bounce",
    "intro": "bounce", "normal": "groove", "buildup": "sway",
    "climax": "groove", "outro": "bounce",
}

BUCKET_SECTION_LABEL: dict[str, str] = {
    "LOW": "intro", "MID": "normal", "TILT": "buildup",
    "PEAK": "climax", "FADE": "outro",
}

SECTION_TO_BUCKET: dict[str, str] = {
    "intro": "LOW", "normal": "MID", "buildup": "TILT",
    "climax": "PEAK", "outro": "FADE",
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _compute_smooth_energy_buckets(beats: list, min_segment_beats: int = 4) -> list:
    """Compute each beat's energy bucket, merging segments shorter than
    ``min_segment_beats`` into their longer neighbour (de-jitters)."""
    if not beats:
        return []

    raw = []
    for b in beats:
        rms = b.get("rms", 0.0)
        strength = b.get("strength", 0.5)
        if rms > 0.7 or strength > 0.75:
            bucket = "PEAK"
        elif rms > 0.5:
            bucket = "MID"
        elif rms > 0.35 or strength > 0.65:
            bucket = "TILT"
        elif rms > 0.2:
            bucket = "LOW"
        else:
            bucket = "FADE"
        raw.append(bucket)

    def build_segments(buckets):
        segs = []
        i = 0
        while i < len(buckets):
            j = i
            while j < len(buckets) and buckets[j] == buckets[i]:
                j += 1
            segs.append((buckets[i], i, j))
            i = j
        return segs

    segs = build_segments(raw)
    if not segs:
        return raw

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(segs):
            bucket, start, end = segs[i]
            length = end - start
            if length < min_segment_beats and len(segs) > 1:
                prev = segs[i - 1] if i > 0 else None
                next_s = segs[i + 1] if i + 1 < len(segs) else None
                target = None
                if prev and next_s:
                    target = prev if (prev[2] - prev[1]) >= (next_s[2] - next_s[1]) else next_s
                elif prev:
                    target = prev
                elif next_s:
                    target = next_s
                if target:
                    new_bucket = target[0]
                    if prev and next_s and prev[0] == next_s[0]:
                        segs[i - 1] = (new_bucket, prev[1], next_s[2])
                        segs.pop(i)
                        segs.pop(i)
                    elif prev:
                        segs[i - 1] = (new_bucket, prev[1], end)
                        segs.pop(i)
                    elif next_s:
                        segs[i + 1] = (new_bucket, start, next_s[2])
                        segs.pop(i)
                    changed = True
                    break
            i += 1

    result = []
    for bucket, start, end in segs:
        result.extend([bucket] * (end - start))
    return result


def _load_timeline(path: Path) -> dict:
    """Load the beat timeline; re-smooth energy buckets when present."""
    with open(path, "r", encoding="utf-8") as f:
        tl = json.load(f)
    beats = tl.get("beats", [])
    if beats and "energy_bucket" in beats[0]:
        buckets = _compute_smooth_energy_buckets(beats, min_segment_beats=4)
        for i, b in enumerate(beats):
            b["energy_bucket"] = buckets[i] if i < len(buckets) else "MID"
    return tl


def _energy_bucket_for_beat(beat: dict) -> str:
    bucket = beat.get("energy_bucket", "")
    if bucket in BUCKET_STYLE_MAP:
        return bucket
    section = beat.get("section", "intro")
    return SECTION_TO_BUCKET.get(section, "MID")


@dataclass
class CrossfadeState:
    """Bucket-switch interpolation (cosine eased)."""
    from_yaw: float = 0.0
    from_pitch: float = 0.0
    from_roll: float = 0.0
    from_ant_l: float = 0.0
    from_ant_r: float = 0.0
    from_body: float = 0.0
    from_body_dec: float = 0.0
    to_yaw: float = 0.0
    to_pitch: float = 0.0
    to_ant_l: float = 0.0
    to_ant_r: float = 0.0
    to_body: float = 0.0
    progress: float = 1.0  # 0=from, 1=to
    beats_remaining: float = 0.0
    active: bool = False


@dataclass
class BeatSyncOscillator:
    """Phase = 2π·freq·elapsed + offset — wall-clock driven, no drift."""

    freq: float = 1.0
    phase_offset: float = 0.0
    _phase: float = 0.0

    def advance(self, elapsed: float) -> float:
        self._phase = 2 * math.pi * self.freq * elapsed + self.phase_offset
        return self._phase

    def value(self) -> float:
        return math.sin(self._phase)


# 能量桶参数表: (osc_freq, yaw_base, pitch_base, roll_base, ant_base, body_base, z_amp_mm)
# 每桶动作签名不同：FADE 慢摆 / LOW 弹跳 / MID 律动 / TILT 侧倾 / PEAK 全身大开
# body_base 大幅提高（底盘摆动是律动感核心），PEAK 顶满 BODY_SOFT_CAP
BUCKET_PARAMS: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "FADE": (0.5, 12,  8,  6, 22, 16,  2),
    "LOW":  (0.8, 14, 10,  8, 23, 19,  3),
    "MID":  (1.5, 14, 12, 10, 20, 22,  4),
    "TILT": (1.5, 12, 12, 14, 18, 26,  5),
    "PEAK": (2.0, 16, 14, 16, 18, 30,  6),
}

BUCKET_AMP_TARGET: dict[str, float] = {
    "FADE": 0.85, "LOW": 0.9, "MID": 0.95, "TILT": 1.0, "PEAK": 1.25,
}

# 4 拍一组的动作重心（yaw, pitch, roll 权重）—— 避免一直摇头晃天线
GROUP_MOTIFS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.45, 0.35),  # 摇头主导
    (0.45, 1.0, 0.5),   # 点头主导
    (0.45, 0.5, 1.0),   # 侧倾主导
    (0.75, 0.75, 0.6),  # 全身综合
)

# 安全软上限（body 提到 25°：底盘摆动是律动感核心；SDK 无硬限制，
# 电机机械范围允许；其余保持源项目收紧值）
YAW_SOFT_CAP = 18.0
ROLL_SOFT_CAP = 18.0
BODY_SOFT_CAP = 25.0
ANT_SOFT_CAP = 28.0


class BeatMotionSynthesizer:
    """Pure-math pose generator (no hardware, no threads — unit-testable).

    Stateful per run: oscillators, crossfade, amplitude/frequency smoothing.
    ``synthesize(elapsed, beat, beat_time)`` returns
    ``(head_pose, antennas_rad, body_yaw_rad, status_dict)``.
    """

    def __init__(self, beat_interval: float) -> None:
        self._beat_interval = beat_interval
        self._bucket_transition_beats = 2.0
        self._osc_yaw = BeatSyncOscillator(freq=2.0, phase_offset=0.0)
        self._osc_yaw_dec = BeatSyncOscillator(freq=4.0, phase_offset=math.pi / 4)
        self._osc_pitch = BeatSyncOscillator(freq=2.0, phase_offset=math.pi / 4)
        self._osc_pitch_dec = BeatSyncOscillator(freq=4.0, phase_offset=0.0)
        self._osc_roll = BeatSyncOscillator(freq=2.0, phase_offset=math.pi / 2)
        self._osc_ant_l = BeatSyncOscillator(freq=2.0, phase_offset=0.0)
        self._osc_ant_r = BeatSyncOscillator(freq=2.0, phase_offset=math.pi / 2)
        self._osc_body = BeatSyncOscillator(freq=0.3, phase_offset=0.0)
        self._osc_body_dec = BeatSyncOscillator(freq=0.6, phase_offset=math.pi / 3)
        self._crossfade = CrossfadeState()
        self._prev_bucket = "MID"
        self._bucket_transition_start = 0.0
        self._prev_amplitude_scale = 0.0
        # per-run smoothed state
        self.smooth_strength = 0.3
        self.amplitude_scale = 0.0
        self.cur_osc_freq = 0.5
        self.target_scale = 0.0
        self.loop_count = 0

    # ── Lifecycle ─────────────────────────────────────────────────────
    def on_run_start(self) -> None:
        """Reset cross-run state (called at dance start and every loop)."""
        self._prev_bucket = ""  # sentinel → first frame fades in smoothly
        self._crossfade.active = False
        self._prev_amplitude_scale = self.amplitude_scale
        self.smooth_strength = 0.3

    def on_loop_restart(self) -> None:
        self.loop_count += 1
        self.on_run_start()

    # ── Synthesis ─────────────────────────────────────────────────────
    def _trigger_crossfade(self, elapsed: float) -> None:
        self._crossfade.from_yaw = self._osc_yaw.value()
        self._crossfade.from_pitch = self._osc_pitch.value()
        self._crossfade.from_roll = self._osc_roll.value()
        self._crossfade.from_ant_l = self._osc_ant_l.value()
        self._crossfade.from_ant_r = self._osc_ant_r.value()
        self._crossfade.from_body = self._osc_body.value()
        self._crossfade.from_body_dec = self._osc_body_dec.value()
        self._crossfade.active = True
        self._crossfade.progress = 0.0
        self._crossfade.beats_remaining = self._bucket_transition_beats
        self._bucket_transition_start = elapsed

    def _apply_crossfade(self, elapsed: float) -> float:
        """Cosine-eased progress (0→1); 1.0 means the fade is done."""
        if not self._crossfade.active:
            return 1.0
        t = elapsed - self._bucket_transition_start
        raw = min(t / (self._beat_interval * self._bucket_transition_beats), 1.0)
        if raw >= 1.0:
            self._crossfade.active = False
            return 1.0
        return 0.5 - 0.5 * math.cos(math.pi * raw)

    def synthesize(
        self, elapsed: float, beat: dict[str, Any], beat_time: float
    ) -> tuple[Any, np.ndarray, float, dict[str, float]]:
        """One 50 Hz tick: compute and return (pose, antennas, body_yaw, status)."""
        beat_elapsed = elapsed - beat_time
        beat_phase = _clamp(
            beat_elapsed / self._beat_interval if self._beat_interval > 0 else 0.0,
            0.0, 1.0,
        )

        raw_strength = beat.get("strength", 0.5)
        self.smooth_strength = self.smooth_strength + 0.1 * (raw_strength - self.smooth_strength)

        energy_bucket = _energy_bucket_for_beat(beat)
        section_label = BUCKET_SECTION_LABEL.get(energy_bucket, "normal")
        params = BUCKET_PARAMS.get(energy_bucket, BUCKET_PARAMS["MID"])
        (osc_freq, yaw_base, pitch_base, roll_base, ant_base, body_base, z_amp) = params

        # 4-beat motif: which DOF leads this group (keeps the dance varied)
        group = int(beat_time / self._beat_interval) % len(GROUP_MOTIFS) \
            if self._beat_interval > 0 else 0
        w_yaw, w_pitch, w_roll = GROUP_MOTIFS[group]

        # Bucket switch → capture from-pose, set oscillator freqs, jump amplitude
        if energy_bucket != self._prev_bucket:
            self._trigger_crossfade(elapsed)
            logger.info(
                "[beat-dance] %s → %s (%s)",
                self._prev_bucket or "(start)", energy_bucket, section_label,
            )
            self._prev_bucket = energy_bucket
            self._osc_yaw.freq = self.cur_osc_freq
            self._osc_yaw_dec.freq = self.cur_osc_freq * 2
            self._osc_pitch.freq = self.cur_osc_freq
            self._osc_pitch_dec.freq = self.cur_osc_freq * 2
            self._osc_roll.freq = self.cur_osc_freq
            self._osc_ant_l.freq = self.cur_osc_freq * 2
            self._osc_ant_r.freq = self.cur_osc_freq * 2
            self._osc_body.freq = 0.3
            self.target_scale = BUCKET_AMP_TARGET.get(energy_bucket, 0.5)
            self.amplitude_scale = self.target_scale

        # amplitude_scale exponential approach to target (~2 beats)
        if abs(self.amplitude_scale - self.target_scale) > 0.003:
            t = elapsed - self._bucket_transition_start
            tau = 2.0 * self._beat_interval
            self.amplitude_scale += (
                (self.target_scale - self.amplitude_scale) * (1.0 - math.exp(-t / tau))
            )
            self.amplitude_scale = _clamp(self.amplitude_scale, 0.0, 1.5)

        # osc frequency smooth approach (~3 beats)
        if abs(self.cur_osc_freq - osc_freq) > 0.01:
            t = elapsed - self._bucket_transition_start
            tau = 3.0 * self._beat_interval
            self.cur_osc_freq += (osc_freq - self.cur_osc_freq) * (1.0 - math.exp(-t / tau))

        self._osc_yaw.advance(elapsed)
        self._osc_yaw_dec.advance(elapsed)
        self._osc_pitch.advance(elapsed)
        self._osc_pitch_dec.advance(elapsed)
        self._osc_roll.advance(elapsed)
        self._osc_ant_l.advance(elapsed)
        self._osc_ant_r.advance(elapsed)
        self._osc_body.advance(elapsed)
        self._osc_body_dec.advance(elapsed)

        yaw_raw = self._osc_yaw.value()
        pitch_raw = self._osc_pitch.value()
        roll_raw = self._osc_roll.value()
        ant_l_val = self._osc_ant_l.value()
        ant_r_val = self._osc_ant_r.value()
        body_raw = self._osc_body.value()

        ease = self._apply_crossfade(elapsed)
        if ease < 1.0:
            yaw_raw = self._crossfade.from_yaw + (yaw_raw - self._crossfade.from_yaw) * ease
            pitch_raw = self._crossfade.from_pitch + (pitch_raw - self._crossfade.from_pitch) * ease
            roll_raw = self._crossfade.from_roll + (roll_raw - self._crossfade.from_roll) * ease
            ant_l_val = self._crossfade.from_ant_l + (ant_l_val - self._crossfade.from_ant_l) * ease
            ant_r_val = self._crossfade.from_ant_r + (ant_r_val - self._crossfade.from_ant_r) * ease
            body_raw = self._crossfade.from_body + (body_raw - self._crossfade.from_body) * ease

        amp_mult = self.amplitude_scale * (0.45 + 0.30 * self.smooth_strength)
        amp_mult = _clamp(amp_mult, 0.65, 1.8)  # 整体幅度上限提高，高潮更猛

        # 4-beat motif weights make the leading DOF vary over time
        yaw_deg_raw = yaw_raw * yaw_base * amp_mult * w_yaw
        pitch_deg_raw = pitch_raw * pitch_base * amp_mult * w_pitch
        roll_deg_raw = roll_raw * roll_base * amp_mult * w_roll
        ant_l_deg_raw = ant_l_val * ant_base * amp_mult
        ant_r_deg_raw = ant_r_val * ant_base * amp_mult

        # Body: slow sway + a beat-synced pulse (per-beat body bump)
        body_pulse = math.sin(2 * math.pi * beat_phase) * self.smooth_strength
        body_deg_raw = body_raw * body_base * amp_mult + body_pulse * body_base * 0.6

        sub_sin = (
            math.sin(2 * math.pi * osc_freq * 2 * elapsed)
            * 0.1 * self.smooth_strength * self.amplitude_scale
        )
        ant_l_deg_raw += sub_sin * ant_base
        ant_r_deg_raw -= sub_sin * ant_base

        pulse = self.smooth_strength * math.exp(-4.0 * beat_phase)

        yaw_deg = _clamp(yaw_deg_raw, -YAW_SOFT_CAP, YAW_SOFT_CAP)
        pitch_deg = _clamp(pitch_deg_raw, -18.0, 22.0)
        roll_deg = _clamp(roll_deg_raw, -ROLL_SOFT_CAP, ROLL_SOFT_CAP)
        ant_l_deg = _clamp(ant_l_deg_raw, -ANT_SOFT_CAP, ANT_SOFT_CAP)
        ant_r_deg = _clamp(ant_r_deg_raw, -ANT_SOFT_CAP, ANT_SOFT_CAP)
        body_deg = _clamp(body_deg_raw, -BODY_SOFT_CAP, BODY_SOFT_CAP)
        z_mm = pulse * z_amp

        from reachy_mini.utils import create_head_pose

        pose = create_head_pose(
            z=_clamp(z_mm, -8.0, 16.0),
            roll=_clamp(roll_deg, -18.0, 18.0),
            pitch=pitch_deg,
            yaw=yaw_deg,
            mm=True, degrees=True,
        )
        antennas = np.deg2rad([ant_r_deg, ant_l_deg])
        body_yaw_rad = math.radians(body_deg)
        status = {
            "pulse": pulse,
            "phase": beat_phase,
            "head_yaw_deg": yaw_deg,
            "head_pitch_deg": pitch_deg,
            "head_roll_deg": roll_deg,
            "left_antenna_deg": ant_l_deg,
            "right_antenna_deg": ant_r_deg,
            "body_yaw_deg": body_deg,
            "z_mm": z_mm,
            "mode_label": f"{section_label} / {energy_bucket}",
        }
        return pose, antennas, body_yaw_rad, status


class BeatDanceController:
    """50 Hz threaded driver: timeline → synthesizer → ``set_target``.

    Music is NOT played here — the engine feeds it through ``audio.play()``
    starting from the same wall-clock origin (``self._loop_origin``) so
    motion and music stay phase-locked and loop together.
    """

    def __init__(self, reachy: Any, cfg: Config) -> None:
        self._reachy = reachy
        self._timeline_path = Path(cfg.beat_timeline_path)
        self._music_path = Path(cfg.beat_music_path)
        self._timeline: dict | None = None
        self._beat_times: np.ndarray = np.array([])
        self._beats: list = []
        self._beat_interval: float = 0.5
        self._timeline_duration: float = 0.0
        self._tempo: float = 120.0
        self._loop_origin: float = 0.0
        self._loop_started_at: float = 0.0
        self._loop_count: int = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._status: dict[str, Any] = {"active": False, "elapsed": 0.0,
                                        "beat_index": 0, "mode_label": "Idle",
                                        "loop_count": 0, "uptime_s": 0.0}
        self._synthesizer: BeatMotionSynthesizer | None = None
        self._sent_errors = 0

    # ── Public API ────────────────────────────────────────────────────
    def load_timeline(self) -> bool:
        if self._timeline is not None:
            return True
        if not self._timeline_path.is_file():
            logger.warning("[beat-dance] timeline 缺失: %s", self._timeline_path)
            return False
        try:
            self._timeline = _load_timeline(self._timeline_path)
        except Exception:
            logger.warning("[beat-dance] timeline 解析失败", exc_info=True)
            return False
        self._beats = self._timeline.get("beats", [])
        self._beat_times = np.array([b["time"] for b in self._beats])
        self._timeline_duration = float(self._timeline.get("duration") or 0.0)
        self._tempo = float(self._timeline.get("tempo") or 120.0)
        self._beat_interval = 60.0 / self._tempo if self._tempo > 0 else 0.5
        logger.info(
            "[beat-dance] timeline loaded %d beats, %.1fs, %.1f BPM",
            len(self._beats), self._timeline_duration, self._tempo,
        )
        return True

    def music_info(self) -> tuple[int, bytes] | None:
        return load_audio(self._music_path)

    @property
    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> tuple[int, bytes] | None:
        """Start the dance loop. Returns (sr, pcm) for the engine to feed,
        or None when the timeline/music is unavailable or already running."""
        if self.is_active:
            return self.music_info()
        if not self.load_timeline():
            return None
        info = self.music_info()
        if info is None:
            logger.warning("[beat-dance] 音乐不可用: %s", self._music_path)
            return None
        self._stop_event.clear()
        self._loop_origin = time.monotonic()
        self._loop_started_at = self._loop_origin
        self._synthesizer = BeatMotionSynthesizer(self._beat_interval)
        self._synthesizer.on_run_start()
        self._thread = threading.Thread(
            target=self._control_loop, name="beat-dance", daemon=True
        )
        self._thread.start()
        with self._status_lock:
            self._status["active"] = True
        logger.info("[beat-dance] 开始无限节拍连跳 (%.1f BPM)", self._tempo)
        return info

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._status_lock:
            self._status["active"] = False
        self._move_to_neutral()
        logger.info("[beat-dance] 停止，回中立")

    def status(self) -> dict[str, Any]:
        with self._status_lock:
            return dict(self._status)

    # ── Internals ─────────────────────────────────────────────────────
    def _move_to_neutral(self) -> None:
        try:
            from reachy_mini.utils import create_head_pose

            self._reachy.goto_target(
                head=create_head_pose(),
                antennas=np.deg2rad([0.0, 0.0]),
                body_yaw=0.0,
                duration=0.8,
            )
        except Exception:
            logger.debug("[beat-dance] 回中立失败", exc_info=True)

    def _send_target(self, pose: Any, antennas: np.ndarray, body_yaw_rad: float) -> None:
        try:
            self._reachy.set_target(head=pose, antennas=antennas, body_yaw=body_yaw_rad)
            self._sent_errors = 0
        except TimeoutError:
            self._sent_errors += 1
            if self._sent_errors <= 3:
                logger.warning("[beat-dance] set_target 超时")
        except Exception:
            self._sent_errors += 1
            if self._sent_errors <= 3:
                logger.debug("[beat-dance] set_target 失败", exc_info=True)

    def _control_loop(self) -> None:
        assert self._synthesizer is not None
        synth = self._synthesizer
        last = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            elapsed = now - self._loop_origin

            if elapsed >= self._timeline_duration and self._timeline_duration > 0:
                # seamless loop: reset the wall-clock origin (source behavior)
                self._loop_origin = time.monotonic()
                synth.on_loop_restart()
                self._loop_count += 1
                logger.info("[beat-dance] loop %d starts", self._loop_count)
                continue

            idx = int(np.searchsorted(self._beat_times, elapsed, side="right")) - 1
            idx = max(0, min(idx, len(self._beats) - 1))
            beat = self._beats[idx]
            try:
                pose, antennas, body_yaw_rad, status = synth.synthesize(
                    elapsed, beat, float(self._beat_times[idx])
                )
                self._send_target(pose, antennas, body_yaw_rad)
                with self._status_lock:
                    self._status.update(status)
                    self._status["elapsed"] = elapsed
                    self._status["beat_index"] = idx
                    self._status["loop_count"] = self._loop_count
                    self._status["uptime_s"] = now - self._loop_started_at
            except Exception:
                logger.debug("[beat-dance] 合成失败", exc_info=True)

            self._stop_event.wait(CONTROL_PERIOD_S)

        with self._status_lock:
            self._status["active"] = False
            self._status["mode_label"] = "Finished"
