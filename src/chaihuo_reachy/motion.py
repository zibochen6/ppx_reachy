"""Robot motion control — gestures, dance sequences, and pose management.

Wraps ReachyMini's movement APIs into high-level action primitives
that can be triggered from Dashboard buttons or LLM function calls.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import random
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from .speech_motion import HOP_S, ContinuousSpeechMotion

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

    from .config import Config

logger = logging.getLogger("chaihuo_reachy.motion")

# ── Standard poses ──────────────────────────────────────────────────────────

# Head tilted slightly (nod-ready)
HEAD_NEUTRAL = np.eye(4)

# Head looking up ~15°
HEAD_LOOK_UP = np.eye(4)
HEAD_LOOK_UP[:3, :3] = np.array(
    [
        [0.966, 0, 0.259],
        [0, 1, 0],
        [-0.259, 0, 0.966],
    ]
)

# Head looking down ~15°
HEAD_LOOK_DOWN = np.eye(4)
HEAD_LOOK_DOWN[:3, :3] = np.array(
    [
        [0.966, 0, -0.259],
        [0, 1, 0],
        [0.259, 0, 0.966],
    ]
)

# Head tilted left ~20°
HEAD_TILT_LEFT = np.eye(4)
HEAD_TILT_LEFT[:3, :3] = np.array(
    [
        [1, 0, 0],
        [0, 0.940, -0.342],
        [0, 0.342, 0.940],
    ]
)

# Head tilted right ~20°
HEAD_TILT_RIGHT = np.eye(4)
HEAD_TILT_RIGHT[:3, :3] = np.array(
    [
        [1, 0, 0],
        [0, 0.940, 0.342],
        [0, -0.342, 0.940],
    ]
)

# ── Antenna positions ───────────────────────────────────────────────────────

ANTENNA_NEUTRAL = [-0.1745, 0.1745]  # ~10° outward
ANTENNA_UP = [-1.5, 1.5]  # raised high
ANTENNA_WAVE_LEFT_UP = [-2.0, 0.1745]  # left up, right neutral
ANTENNA_WAVE_RIGHT_UP = [-0.1745, 2.0]  # right up, left neutral
ANTENNA_CROSSED = [1.5, -1.5]  # crossed inward

# ── Dance choreographies (data-driven) ──────────────────────────────────────
#
# Each style is a list of (targets, beats) steps.  ``targets`` are
# goto_target kwargs (head/antennas/body_yaw); ``beats`` is how many
# musical beats the pose holds.  With ``beat_s`` paced from the backing
# track's BPM, the sequence loops until the track ends — steps land on
# the beat, and the whole dance spans the music.
DANCE_CHOREOGRAPHIES: dict[str, list[tuple[dict, int]]] = {
    # 欢快: 快速左右摇摆 + 天线挥动 + 身体转
    "happy": [
        ({"head": HEAD_TILT_LEFT, "antennas": ANTENNA_UP, "body_yaw": 0.25}, 1),
        ({"head": HEAD_TILT_RIGHT, "antennas": ANTENNA_NEUTRAL, "body_yaw": -0.25}, 1),
        ({"head": HEAD_LOOK_UP, "antennas": ANTENNA_WAVE_LEFT_UP, "body_yaw": 0.0}, 2),
        (
            {
                "head": HEAD_LOOK_DOWN,
                "antennas": ANTENNA_WAVE_RIGHT_UP,
                "body_yaw": 0.0,
            },
            2,
        ),
        ({"head": HEAD_NEUTRAL, "antennas": ANTENNA_CROSSED, "body_yaw": 0.0}, 1),
        ({"head": HEAD_LOOK_UP, "antennas": ANTENNA_UP, "body_yaw": 0.25}, 2),
        (
            {
                "head": HEAD_TILT_LEFT,
                "antennas": ANTENNA_WAVE_LEFT_UP,
                "body_yaw": -0.25,
            },
            1,
        ),
        (
            {
                "head": HEAD_TILT_RIGHT,
                "antennas": ANTENNA_WAVE_RIGHT_UP,
                "body_yaw": 0.25,
            },
            1,
        ),
    ],
    # 摇摆: 慢速左右摆 + 天线交替 + 抬头上扬
    "swing": [
        ({"head": HEAD_TILT_LEFT, "body_yaw": 0.3}, 2),
        ({"head": HEAD_TILT_RIGHT, "body_yaw": -0.3}, 2),
        ({"head": HEAD_LOOK_UP, "antennas": ANTENNA_UP, "body_yaw": 0.0}, 4),
        ({"head": HEAD_NEUTRAL, "antennas": ANTENNA_WAVE_LEFT_UP, "body_yaw": 0.2}, 2),
        (
            {"head": HEAD_NEUTRAL, "antennas": ANTENNA_WAVE_RIGHT_UP, "body_yaw": -0.2},
            2,
        ),
        ({"head": HEAD_LOOK_DOWN, "antennas": ANTENNA_NEUTRAL, "body_yaw": 0.0}, 4),
    ],
    # 机械舞: 顿挫短促, 一拍一个造型
    "robot": [
        ({"head": HEAD_LOOK_UP, "antennas": ANTENNA_NEUTRAL, "body_yaw": 0.0}, 1),
        ({"head": HEAD_TILT_LEFT, "antennas": ANTENNA_UP, "body_yaw": 0.0}, 1),
        ({"head": HEAD_TILT_RIGHT, "antennas": ANTENNA_UP, "body_yaw": 0.0}, 1),
        ({"head": HEAD_LOOK_DOWN, "antennas": ANTENNA_CROSSED, "body_yaw": 0.0}, 2),
        ({"head": HEAD_NEUTRAL, "antennas": ANTENNA_WAVE_LEFT_UP, "body_yaw": 0.15}, 1),
        (
            {
                "head": HEAD_NEUTRAL,
                "antennas": ANTENNA_WAVE_RIGHT_UP,
                "body_yaw": -0.15,
            },
            1,
        ),
    ],
}

# Reachy Mini SDK daemon (<= 1.9.0) has a timing race in its interpolation
# loop: on the last tick, ``t`` can exceed the requested duration (event-loop
# scheduling delay), so ``t / duration > 1`` and ``time_trajectory()`` raises
# "time value is out of range [0,1]".  Short, rapid goto_target calls (dances)
# hit this most often.  We retry such failures and skip the step if they
# persist instead of aborting the whole sequence.
_REACHY_TIME_RANGE_ERROR = "time value is out of range"


class MotionController:
    """High-level motion control for Reachy Mini.

    Provides gesture primitives, dance sequences, and pose management.
    All movement methods are async and safe to call concurrently (they
    internally serialize via asyncio locks).

    Usage::

        motion = MotionController(reachy)
        await motion.dance("happy")
        await motion.nod(times=3)
    """

    def __init__(self, reachy: ReachyMini, config: Config | None = None) -> None:
        self._reachy = reachy
        self._lock = asyncio.Lock()
        self._wobbling = False
        self._speech_motion = ContinuousSpeechMotion(
            yaw_max_deg=getattr(config, "talk_motion_yaw_max_deg", 20.0),
            pitch_max_deg=getattr(config, "talk_motion_pitch_max_deg", 3.2),
            roll_max_deg=getattr(config, "talk_motion_roll_max_deg", 2.0),
            master_gain=getattr(config, "talk_motion_gain", 1.0),
            breath_floor=0.12,
        )
        # The speaker callback only performs a non-blocking put.  A small
        # drop-oldest queue keeps physical motion aligned to sound even if the
        # daemon briefly stalls; stale gestures are never replayed later.
        self._talk_motion_queue: queue.Queue[tuple[bytes, int] | None] = queue.Queue(
            maxsize=8
        )
        self._talk_motion_lock = threading.Lock()
        self._speech_command_lock = threading.Lock()
        self._talk_motion_stop = threading.Event()
        self._talk_motion_thread: threading.Thread | None = None
        self._talk_motion_active = False
        self._talk_motion_generation = 0
        self._talk_motion_envelope = 0.0
        self._talk_motion_error = ""
        self._talk_motion_dropped_frames = 0

    # ── Tolerant goto wrapper ───────────────────────────────────────────

    async def _safe_goto(self, *, retries: int = 1, **kwargs) -> bool:
        """goto_target with retries against the SDK interpolation race.

        Runs the blocking SDK call in a worker thread so the event loop
        stays free — otherwise every step stalls music playback and ASR
        polling for the duration of the round-trip.

        Returns True when the movement was applied; False when it had to be
        skipped after retries (the sequence continues, the step is dropped).
        Unrelated errors (real hardware faults) propagate.
        """
        for attempt in range(retries + 1):
            try:
                await asyncio.to_thread(self._reachy.goto_target, **kwargs)
                return True
            except Exception as exc:
                if _REACHY_TIME_RANGE_ERROR not in str(exc):
                    raise  # real failure — do not swallow
                if attempt < retries:
                    logger.warning(
                        "⏳ goto 插值竞态(%s)，重试 %d/%d", exc, attempt + 1, retries
                    )
                    await asyncio.sleep(0.12)
                    continue
                logger.warning("goto 插值竞态重试耗尽，跳过该动作")
                return False
        return False

    # ── Basic gestures ──────────────────────────────────────────────────

    async def nod(self, times: int = 2) -> None:
        """Nod head up and down (affirmation gesture)."""
        async with self._lock:
            logger.info("🙆 点头 x%d", times)
            for _ in range(times):
                await self._safe_goto(head=HEAD_LOOK_UP, duration=0.3)
                await asyncio.sleep(0.3)
                await self._safe_goto(head=HEAD_LOOK_DOWN, duration=0.3)
                await asyncio.sleep(0.3)
            # Return to neutral
            await self._safe_goto(head=HEAD_NEUTRAL, duration=0.3)

    async def shake_head(self, times: int = 2) -> None:
        """Shake head left and right (negation gesture)."""
        async with self._lock:
            logger.info("🙅 摇头 x%d", times)
            for _ in range(times):
                await self._safe_goto(head=HEAD_TILT_LEFT, duration=0.25)
                await asyncio.sleep(0.25)
                await self._safe_goto(head=HEAD_TILT_RIGHT, duration=0.25)
                await asyncio.sleep(0.25)
            await self._safe_goto(head=HEAD_NEUTRAL, duration=0.3)

    async def wave_antenna(self, side: str = "both") -> None:
        """Wave antennas — like a friendly greeting.

        Args:
            side: "left", "right", or "both"
        """
        async with self._lock:
            logger.info("🐜 挥天线: %s", side)
            waves = 3

            if side in ("left", "both"):
                for _ in range(waves):
                    self._reachy.set_target(antennas=[-2.5, ANTENNA_NEUTRAL[1]])
                    await asyncio.sleep(0.15)
                    self._reachy.set_target(antennas=[-0.5, ANTENNA_NEUTRAL[1]])
                    await asyncio.sleep(0.15)

            if side in ("right", "both"):
                for _ in range(waves):
                    self._reachy.set_target(antennas=[ANTENNA_NEUTRAL[0], 2.5])
                    await asyncio.sleep(0.15)
                    self._reachy.set_target(antennas=[ANTENNA_NEUTRAL[0], 0.5])
                    await asyncio.sleep(0.15)

            # Return to neutral
            self._reachy.set_target(antennas=ANTENNA_NEUTRAL)

    async def look_left_right(self) -> None:
        """Look left, then right — scanning gesture."""
        async with self._lock:
            for yaw in [0.5, -0.5, 0.0]:
                await self._safe_goto(head=HEAD_NEUTRAL, body_yaw=yaw, duration=0.5)
                await asyncio.sleep(0.6)

    # ── Dance sequences ─────────────────────────────────────────────────

    async def dance(
        self,
        style: str = "happy",
        *,
        duration_s: float | None = None,
        beat_s: float = 0.5,
    ) -> dict[str, object]:
        """Perform a dance sequence, looping until the music ends.

        Args:
            style: "happy", "swing", "robot", or "random"
            duration_s: total dance length; None = exactly one pass of the
                choreography (used when no backing track exists)
            beat_s: length of one musical beat (from the track's BPM).
                Each step holds ``beats * beat_s`` seconds, so poses land
                on the beat.

        Returns:
            Summary dict: ``{"style", "skipped", "duration"}`` — steps
            dropped by the SDK interpolation race (``_safe_goto``) are
            counted, never fatal; the sequence always completes.
        """
        if style == "random":
            style = random.choice(list(DANCE_CHOREOGRAPHIES))
        choreo = DANCE_CHOREOGRAPHIES.get(style, DANCE_CHOREOGRAPHIES["happy"])

        async with self._lock:
            logger.info(
                "💃 跳舞: %s (拍长 %.2fs, 时长 %s)",
                style,
                beat_s,
                f"{duration_s:.0f}s" if duration_s else "一轮",
            )
            skipped = 0
            start = time.monotonic()
            step_goto_s = 0.3  # each pose's goto duration
            while duration_s is None or (time.monotonic() - start) < duration_s:
                for targets, beats in choreo:
                    if (
                        duration_s is not None
                        and (time.monotonic() - start) >= duration_s
                    ):
                        break
                    if not await self._safe_goto(duration=step_goto_s, **targets):
                        skipped += 1
                    await asyncio.sleep(max(0.0, beats * beat_s - step_goto_s))
                if duration_s is None:
                    break  # no backing track: exactly one pass of the choreography

            # Return to neutral after dance
            if not await self._safe_goto(
                head=HEAD_NEUTRAL,
                antennas=ANTENNA_NEUTRAL,
                duration=0.5,
            ):
                skipped += 1

            return {
                "style": style,
                "skipped": skipped,
                "duration": round(time.monotonic() - start, 1),
            }

    # ── Pose control ────────────────────────────────────────────────────

    async def set_pose(
        self,
        head: np.ndarray | None = None,
        antennas: list[float] | None = None,
        body_yaw: float = 0.0,
    ) -> None:
        """Immediately set a target pose."""
        self._reachy.set_target(head=head, antennas=antennas, body_yaw=body_yaw)

    async def goto_pose(
        self,
        head: np.ndarray | None = None,
        antennas: list[float] | None = None,
        body_yaw: float = 0.0,
        duration: float = 1.0,
    ) -> None:
        """Smoothly move to a target pose."""
        async with self._lock:
            await self._safe_goto(
                head=head,
                antennas=antennas,
                body_yaw=body_yaw,
                duration=duration,
            )

    async def look_at(self, x: float, y: float, z: float) -> None:
        """Look at a 3D point in world coordinates."""
        async with self._lock:
            self._reachy.look_at_world(
                x=x, y=y, z=z, duration=1.0, perform_movement=True
            )

    async def wake_up(self) -> None:
        """Wake the robot up — head up, antennas spread."""
        async with self._lock:
            logger.info("🤖 站起来")
            self._reachy.wake_up()

    async def sleep(self) -> None:
        """Put the robot to sleep — head down, antennas folded."""
        async with self._lock:
            logger.info("😴 休眠")
            self._reachy.goto_sleep()

    async def play_move(self, move_name: str) -> None:
        """Play a pre-recorded named move.

        Note: Move files need to be loaded via ReachyMini SDK's Move loader.
        For now, this is a placeholder for future move library support.
        """
        logger.info("▶️ 播放动作: %s (not yet implemented)", move_name)

    # ── Continuous speech motion ────────────────────────────────────────
    # Speaker PCM drives daemon-native 6DoF speech offsets.  Unlike repeated
    # goto_target trajectories, this path never brakes to zero at every
    # left/right endpoint, so the neck follows one continuous phase curve.

    def enable_wobbling(self) -> None:
        """Enable audio-driven speech motion."""
        if not self._wobbling:
            self._wobbling = True
            logger.info("🔊 连续说话动作已启用（speech_offsets）")

    def disable_wobbling(self) -> None:
        """Disable talk-shake mode."""
        if self._wobbling:
            self._wobbling = False
            self.stop_talk_motion(immediate=True)
            logger.info("🔇 连续说话动作已禁用")

    def start_talk_motion(self) -> None:
        """Start a fresh speech motion stream before audible PCM arrives."""
        if not self._wobbling:
            self.enable_wobbling()
        if self._reachy is None:
            self._record_talk_motion_error("Reachy connection is unavailable")
            return

        previous = self._talk_motion_thread
        if previous and previous.is_alive():
            self.stop_talk_motion(immediate=True)
            previous.join(timeout=0.25)

        self._clear_talk_motion_queue()
        self._speech_motion.reset()
        self._talk_motion_stop.clear()
        with self._talk_motion_lock:
            self._talk_motion_generation += 1
            generation = self._talk_motion_generation
            self._talk_motion_active = True
            self._talk_motion_envelope = 0.0
            self._talk_motion_error = ""
        thread = threading.Thread(
            target=self._talk_motion_loop,
            args=(generation,),
            name="speech-offsets",
            daemon=True,
        )
        self._talk_motion_thread = thread
        thread.start()

    def feed_talk_audio(self, pcm: bytes, sample_rate: int) -> None:
        """Non-blockingly enqueue the PCM currently reaching the speaker."""
        if not pcm:
            return
        with self._talk_motion_lock:
            active = self._talk_motion_active
        if not active:
            return
        item = (bytes(pcm), int(sample_rate))
        try:
            self._talk_motion_queue.put_nowait(item)
        except queue.Full:
            try:
                self._talk_motion_queue.get_nowait()
            except queue.Empty:
                pass
            with self._talk_motion_lock:
                self._talk_motion_dropped_frames += 1
            try:
                self._talk_motion_queue.put_nowait(item)
            except queue.Full:
                # Another producer cannot exist in normal operation, but the
                # callback must still never block the speaker thread.
                pass

    def stop_talk_motion(self, *, immediate: bool = False) -> None:
        """Stop speech motion, immediately for takeover or smoothly normally."""
        with self._talk_motion_lock:
            was_running = self._talk_motion_active or bool(
                self._talk_motion_thread and self._talk_motion_thread.is_alive()
            )
            self._talk_motion_active = False
            if immediate:
                self._talk_motion_generation += 1
                self._talk_motion_envelope = 0.0
        if not was_running:
            if immediate:
                self._send_zero_offsets()
            return
        self._clear_talk_motion_queue()
        if immediate:
            self._talk_motion_stop.set()
            self._send_zero_offsets()
        else:
            try:
                self._talk_motion_queue.put_nowait(None)
            except queue.Full:
                pass

    # Compatibility for dance/older callers while they migrate to the new
    # name.  It intentionally delegates to offsets, never to goto_target.
    def stop_talk_shake(self, *, immediate: bool = False) -> None:
        self.stop_talk_motion(immediate=immediate)

    def _talk_motion_loop(self, generation: int) -> None:
        graceful = False
        try:
            while not self._talk_motion_stop.is_set():
                try:
                    item = self._talk_motion_queue.get(timeout=0.10)
                except queue.Empty:
                    continue
                if item is None:
                    graceful = True
                    break
                pcm, sample_rate = item
                for frame in self._speech_motion.feed_pcm(pcm, sample_rate):
                    if self._talk_motion_stop.is_set():
                        break
                    with self._talk_motion_lock:
                        if generation != self._talk_motion_generation:
                            return
                        self._talk_motion_envelope = frame.envelope
                    if not self._send_speech_offsets(frame.offsets, generation):
                        return
                    if self._talk_motion_stop.wait(HOP_S):
                        break

            if graceful and not self._talk_motion_stop.is_set():
                for frame in self._speech_motion.graceful_decay(0.30):
                    with self._talk_motion_lock:
                        if generation != self._talk_motion_generation:
                            return
                        self._talk_motion_envelope = frame.envelope
                    if not self._send_speech_offsets(frame.offsets, generation):
                        return
                    if self._talk_motion_stop.wait(HOP_S):
                        return
                self._send_speech_offsets((0.0,) * 6, generation)
        finally:
            with self._talk_motion_lock:
                if generation == self._talk_motion_generation:
                    self._talk_motion_active = False
                    self._talk_motion_envelope = 0.0
                if threading.current_thread() is self._talk_motion_thread:
                    self._talk_motion_thread = None

    def _send_speech_offsets(
        self, offsets: tuple[float, float, float, float, float, float], generation: int
    ) -> bool:
        try:
            from reachy_mini.io.protocol import SetSpeechOffsetsCmd

            # Serialize the final generation check with send.  Otherwise an
            # immediate stop could clear offsets and a previously validated
            # worker command could race in afterwards, leaving the head up.
            with self._speech_command_lock:
                with self._talk_motion_lock:
                    if generation != self._talk_motion_generation:
                        return False
                self._reachy.client.send_command(
                    SetSpeechOffsetsCmd(offsets=[float(value) for value in offsets])
                )
            return True
        except Exception as exc:
            self._record_talk_motion_error(str(exc))
            with self._talk_motion_lock:
                if generation == self._talk_motion_generation:
                    self._talk_motion_active = False
                    self._talk_motion_generation += 1
            self._talk_motion_stop.set()
            logger.exception("speech_offsets command failed; speech motion stopped")
            return False

    def _send_zero_offsets(self) -> None:
        if self._reachy is None:
            return
        try:
            from reachy_mini.io.protocol import SetSpeechOffsetsCmd

            with self._speech_command_lock:
                self._reachy.client.send_command(SetSpeechOffsetsCmd(offsets=[0.0] * 6))
        except Exception as exc:
            self._record_talk_motion_error(str(exc))
            logger.exception("failed to clear speech_offsets")

    def _clear_talk_motion_queue(self) -> None:
        while True:
            try:
                self._talk_motion_queue.get_nowait()
            except queue.Empty:
                return

    def _record_talk_motion_error(self, message: str) -> None:
        with self._talk_motion_lock:
            self._talk_motion_error = message

    @property
    def is_wobbling(self) -> bool:
        return self._wobbling

    @property
    def is_talk_shaking(self) -> bool:
        """Whether speech motion is active or returning to neutral."""
        with self._talk_motion_lock:
            return self._talk_motion_active or bool(
                self._talk_motion_thread and self._talk_motion_thread.is_alive()
            )

    @property
    def talk_motion_backend(self) -> str:
        return "speech_offsets"

    @property
    def talk_motion_envelope(self) -> float:
        with self._talk_motion_lock:
            return self._talk_motion_envelope

    @property
    def talk_motion_error(self) -> str:
        with self._talk_motion_lock:
            return self._talk_motion_error

    @property
    def talk_motion_dropped_frames(self) -> int:
        with self._talk_motion_lock:
            return self._talk_motion_dropped_frames

    # ── Status ──────────────────────────────────────────────────────────

    @property
    def is_busy(self) -> bool:
        """Whether a motion is currently being executed."""
        return self._lock.locked()
