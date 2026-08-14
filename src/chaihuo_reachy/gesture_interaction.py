"""Preemptible open-palm tracking / fist dancing interaction mode."""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .hand_pose import (
    ActiveHandSelector,
    HandPoseBackend,
    HandPoseResult,
    classify_gesture,
    create_hand_pose_backend,
)
from .motion import DANCE_CHOREOGRAPHIES
from .music import STYLE_BPM, read_track

logger = logging.getLogger("chaihuo_reachy.gesture_interaction")


class FrameSource(Protocol):
    def get_bgr_frame(self) -> np.ndarray | None: ...


class GestureInteractionController:
    """Own inference, debouncing, tracking, music, and realtime motion."""

    def __init__(
        self,
        config: Any,
        *,
        frame_source: FrameSource,
        motion: Any,
        audio: Any,
        backend_factory: Callable[[Any], HandPoseBackend] = create_hand_pose_backend,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
        pose_factory: Callable[..., np.ndarray] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self._frames = frame_source
        self._motion = motion
        self._audio = audio
        self._backend_factory = backend_factory
        self._status_callback = status_callback
        self._pose_factory = pose_factory
        self._rng = rng or random.Random()
        self._backend: HandPoseBackend | None = None
        self._enabled = False
        self._task: asyncio.Task[None] | None = None
        self._dance_task: asyncio.Task[None] | None = None
        self._tracking_task: asyncio.Task[None] | None = None
        self._selector = ActiveHandSelector()
        self._generation = 0
        self._state = "DISABLED"
        self._gesture = "OTHER"
        self._finger_states = {name: False for name in ("thumb", "index", "middle", "ring", "pinky")}
        self._pose: HandPoseResult | None = None
        self._candidate = "OTHER"
        self._candidate_since = 0.0
        self._last_seen = 0.0
        self._started_at = 0.0
        self._inference_count = 0
        self._last_latency_ms = 0.0
        self._dance_style = ""
        self._filtered_center: tuple[float, float] | None = None
        self._tracking_center = (0.5, 0.5)
        self._head_yaw = 0.0
        self._head_pitch = 0.0
        self._body_yaw = 0.0
        self._last_publish = 0.0
        self._error = ""

    @property
    def active(self) -> bool:
        return self._enabled

    @property
    def state(self) -> str:
        return self._state

    async def start(self) -> dict[str, Any]:
        if self.active:
            return self.status()
        if self._motion is None:
            raise RuntimeError("机器人运动控制未就绪")
        if self._audio is None:
            raise RuntimeError("机器人音频输出未就绪")
        if self._frames.get_bgr_frame() is None:
            raise RuntimeError("Reachy 前置摄像头尚未提供可用 BGR 帧")
        self._backend = await asyncio.to_thread(self._backend_factory, self.config)
        self._selector.reset()
        self._generation += 1
        self._state = "SEARCHING"
        self._gesture = "OTHER"
        self._candidate = "OTHER"
        self._candidate_since = 0.0
        self._last_seen = 0.0
        self._started_at = time.monotonic()
        self._inference_count = 0
        self._error = ""
        self._enabled = True
        self._task = asyncio.create_task(self._run(), name="gesture-interaction")
        self._publish(force=True)
        return self.status()

    async def stop(self) -> None:
        self._enabled = False
        self._generation += 1
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._stop_tracking()
        await self._stop_dance(neutral=False)
        if self._backend is not None:
            await asyncio.to_thread(self._backend.close)
            self._backend = None
        await self._neutral()
        self._selector.reset()
        self._pose = None
        self._state = "DISABLED"
        self._gesture = "OTHER"
        self._dance_style = ""
        self._publish(force=True)

    async def _run(self) -> None:
        interval = 1.0 / max(1.0, float(self.config.gesture_inference_fps))
        try:
            while True:
                tick = time.monotonic()
                frame = self._frames.get_bgr_frame()
                if frame is None or self._backend is None:
                    await self._handle_missing(tick)
                else:
                    candidates = await asyncio.to_thread(self._backend.infer, frame)
                    self._inference_count += 1
                    pose = self._selector.select(candidates)
                    if pose is None:
                        await self._handle_missing(tick)
                    else:
                        await self._handle_pose(pose, tick)
                self._publish()
                await asyncio.sleep(max(0.0, interval - (time.monotonic() - tick)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error = str(exc)
            logger.exception("手势交互循环失败")
            self._generation += 1
            await self._stop_tracking()
            await self._stop_dance(neutral=False)
            self._state = "LOST"
            self._publish(force=True)

    async def _handle_pose(self, pose: HandPoseResult, now: float) -> None:
        self._pose = pose
        self._last_latency_ms = pose.latency_ms
        gesture, fingers = classify_gesture(
            pose, min_confidence=float(self.config.gesture_keypoint_confidence)
        )
        self._finger_states = fingers
        if gesture == "OTHER":
            self._candidate = "OTHER"
            self._candidate_since = now
            await self._handle_missing(now)
            return
        self._last_seen = now
        if gesture != self._candidate:
            self._candidate = gesture
            self._candidate_since = now
            return
        confirmation_s = float(self.config.gesture_confirmation_ms) / 1000.0
        if now - self._candidate_since < confirmation_s:
            return
        if gesture != self._gesture:
            self._gesture = gesture
            if gesture == "OPEN_PALM":
                await self._enter_tracking(pose)
            elif gesture == "FIST":
                await self._enter_dancing()
        elif gesture == "OPEN_PALM" and self._state == "TRACKING":
            self._tracking_center = pose.palm_center

    async def _handle_missing(self, now: float) -> None:
        if self._last_seen <= 0.0:
            self._state = "SEARCHING"
            return
        if now - self._last_seen < float(self.config.gesture_lost_timeout_ms) / 1000.0:
            return
        if self._state != "LOST":
            self._generation += 1
            await self._stop_tracking()
            await self._stop_dance(neutral=False)
            await self._neutral()
            self._state = "LOST"
            self._gesture = "OTHER"
            self._candidate = "OTHER"
            self._pose = None
            self._selector.reset()

    async def _enter_tracking(self, pose: HandPoseResult) -> None:
        self._generation += 1
        await self._stop_tracking()
        await self._stop_dance(neutral=False)
        self._state = "TRACKING"
        self._dance_style = ""
        self._filtered_center = pose.palm_center
        self._tracking_center = pose.palm_center
        generation = self._generation
        self._tracking_task = asyncio.create_task(
            self._tracking_loop(generation), name="gesture-tracking"
        )

    async def _enter_dancing(self) -> None:
        if self._state == "DANCING" and self._dance_task is not None:
            return
        pairs = self._available_dances()
        if not pairs:
            self._error = "没有找到成对的预设舞蹈与音乐"
            return
        self._generation += 1
        await self._stop_tracking()
        await self._stop_dance(neutral=False)
        style, track = self._rng.choice(pairs)
        self._dance_style = style
        self._state = "DANCING"
        generation = self._generation
        self._dance_task = asyncio.create_task(
            self._dance_loop(style, track, generation),
            name=f"gesture-dance-{style}",
        )

    def _available_dances(self) -> list[tuple[str, Path]]:
        music_dir = Path(self.config.dance_music_dir)
        return [
            (style, music_dir / f"{style}.wav")
            for style in DANCE_CHOREOGRAPHIES
            if (music_dir / f"{style}.wav").is_file()
        ]

    async def _stop_dance(self, *, neutral: bool) -> None:
        task, self._dance_task = self._dance_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        stop = getattr(self._audio, "stop_playback", None)
        if callable(stop):
            stop()
        if neutral:
            await self._neutral()

    async def _stop_tracking(self) -> None:
        task, self._tracking_task = self._tracking_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _tracking_loop(self, generation: int) -> None:
        while generation == self._generation and self._state == "TRACKING":
            self._track(self._tracking_center)
            await asyncio.sleep(0.025)

    async def _dance_loop(self, style: str, track: Path, generation: int) -> None:
        info = read_track(track)
        if info is None:
            self._error = f"舞蹈音乐无法读取: {track}"
            return
        sr, frames = info
        self._audio.set_output_sample_rate(sr)
        music_task = asyncio.create_task(
            self._music_loop(sr, frames, generation), name="gesture-dance-music"
        )
        beat_s = 60.0 / STYLE_BPM.get(style, 120.0)
        choreography = DANCE_CHOREOGRAPHIES[style]
        previous = choreography[-1][0]
        try:
            while generation == self._generation:
                for target, beats in choreography:
                    if generation != self._generation:
                        return
                    duration = beats * beat_s
                    started = time.monotonic()
                    while generation == self._generation:
                        progress = min(1.0, (time.monotonic() - started) / duration)
                        eased = 0.5 - 0.5 * math.cos(math.pi * progress)
                        self._motion.set_realtime_target(
                            **self._interpolate_target(previous, target, eased)
                        )
                        if progress >= 1.0:
                            break
                        await asyncio.sleep(0.025)
                    previous = target
        except asyncio.CancelledError:
            raise
        finally:
            music_task.cancel()
            try:
                await music_task
            except asyncio.CancelledError:
                pass

    async def _music_loop(self, sr: int, frames: bytes, generation: int) -> None:
        chunk_size = max(2, int(sr * 2 * 0.1))
        while generation == self._generation:
            for start in range(0, len(frames), chunk_size):
                if generation != self._generation:
                    return
                await self._audio.play(frames[start : start + chunk_size])
                await asyncio.sleep(len(frames[start : start + chunk_size]) / 2 / sr)

    @staticmethod
    def _interpolate_target(
        source: dict[str, Any], target: dict[str, Any], progress: float
    ) -> dict[str, Any]:
        progress = float(np.clip(progress, 0.0, 1.0))
        head = (1.0 - progress) * np.asarray(source["head"]) + progress * np.asarray(
            target["head"]
        )
        # Project the blended 3x3 block back onto SO(3).
        u, _, vh = np.linalg.svd(head[:3, :3])
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vh
        head[:3, :3] = rotation
        antennas = (
            (1.0 - progress) * np.asarray(source["antennas"], dtype=np.float64)
            + progress * np.asarray(target["antennas"], dtype=np.float64)
        ).tolist()
        body_yaw = (1.0 - progress) * float(source["body_yaw"]) + progress * float(
            target["body_yaw"]
        )
        return {"head": head, "antennas": antennas, "body_yaw": body_yaw}

    def _track(self, center: tuple[float, float]) -> None:
        alpha = float(self.config.gesture_tracking_alpha)
        if self._filtered_center is None:
            self._filtered_center = center
        else:
            self._filtered_center = (
                alpha * center[0] + (1.0 - alpha) * self._filtered_center[0],
                alpha * center[1] + (1.0 - alpha) * self._filtered_center[1],
            )
        error_x = self._filtered_center[0] - 0.5
        error_y = self._filtered_center[1] - 0.5
        deadzone = float(self.config.gesture_tracking_deadzone)
        if abs(error_x) < deadzone:
            error_x = 0.0
        if abs(error_y) < deadzone:
            error_y = 0.0
        max_head_yaw = float(self.config.gesture_head_yaw_max_deg)
        max_head_pitch = float(self.config.gesture_head_pitch_max_deg)
        max_body = float(self.config.gesture_body_yaw_max_deg)
        target_head_yaw = float(np.clip(-error_x * max_head_yaw * 2.0, -max_head_yaw, max_head_yaw))
        target_head_pitch = float(np.clip(error_y * max_head_pitch * 2.0, -max_head_pitch, max_head_pitch))
        target_body = float(np.clip(-error_x * max_body * 2.0, -max_body, max_body))
        step = float(self.config.gesture_tracking_max_step_deg)
        self._head_yaw += float(np.clip(target_head_yaw - self._head_yaw, -step, step))
        self._head_pitch += float(np.clip(target_head_pitch - self._head_pitch, -step, step))
        self._body_yaw += float(np.clip(target_body - self._body_yaw, -step, step))
        pose_factory = self._pose_factory
        if pose_factory is None:
            from reachy_mini.utils import create_head_pose

            pose_factory = create_head_pose
        head = pose_factory(
            yaw=self._head_yaw,
            pitch=self._head_pitch,
            degrees=True,
        )
        self._motion.set_realtime_target(
            head=head,
            body_yaw=math.radians(self._body_yaw),
        )

    async def _neutral(self) -> None:
        self._filtered_center = None
        self._head_yaw = self._head_pitch = self._body_yaw = 0.0
        neutral = getattr(self._motion, "return_to_neutral", None)
        if callable(neutral):
            try:
                await neutral(duration=0.5)
            except Exception:
                logger.debug("手势模式回中立失败", exc_info=True)

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = max(1e-6, now - self._started_at) if self._started_at else 0.0
        return {
            "type": "gesture_status",
            "active": self.active,
            "state": self._state,
            "gesture": self._gesture,
            "finger_states": dict(self._finger_states),
            "confidence": round(self._pose.confidence, 4) if self._pose else 0.0,
            "backend": self._pose.backend if self._pose else getattr(self._backend, "name", ""),
            "fps": round(self._inference_count / elapsed, 1) if elapsed else 0.0,
            "latency_ms": round(self._last_latency_ms, 1),
            "landmarks": [point.public() for point in self._pose.landmarks] if self._pose else [],
            "dance_style": self._dance_style,
            "error": self._error,
        }

    def _publish(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_publish < 0.1:
            return
        self._last_publish = now
        if self._status_callback is not None:
            self._status_callback(self.status())
