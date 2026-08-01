"""Robot motion control — gestures, dance sequences, and pose management.

Wraps ReachyMini's movement APIs into high-level action primitives
that can be triggered from Dashboard buttons or LLM function calls.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

logger = logging.getLogger("chaihuo_reachy.motion")

# ── Standard poses ──────────────────────────────────────────────────────────

# Head tilted slightly (nod-ready)
HEAD_NEUTRAL = np.eye(4)

# Head looking up ~15°
HEAD_LOOK_UP = np.eye(4)
HEAD_LOOK_UP[:3, :3] = np.array([
    [0.966, 0, 0.259],
    [0, 1, 0],
    [-0.259, 0, 0.966],
])

# Head looking down ~15°
HEAD_LOOK_DOWN = np.eye(4)
HEAD_LOOK_DOWN[:3, :3] = np.array([
    [0.966, 0, -0.259],
    [0, 1, 0],
    [0.259, 0, 0.966],
])

# Head tilted left ~20°
HEAD_TILT_LEFT = np.eye(4)
HEAD_TILT_LEFT[:3, :3] = np.array([
    [1, 0, 0],
    [0, 0.940, -0.342],
    [0, 0.342, 0.940],
])

# Head tilted right ~20°
HEAD_TILT_RIGHT = np.eye(4)
HEAD_TILT_RIGHT[:3, :3] = np.array([
    [1, 0, 0],
    [0, 0.940, 0.342],
    [0, -0.342, 0.940],
])

# ── Antenna positions ───────────────────────────────────────────────────────

ANTENNA_NEUTRAL = [-0.1745, 0.1745]     # ~10° outward
ANTENNA_UP = [-1.5, 1.5]                 # raised high
ANTENNA_WAVE_LEFT_UP = [-2.0, 0.1745]    # left up, right neutral
ANTENNA_WAVE_RIGHT_UP = [-0.1745, 2.0]   # right up, left neutral
ANTENNA_CROSSED = [1.5, -1.5]            # crossed inward

# ── Dance movement parameters ───────────────────────────────────────────────

# How many "steps" per dance style
DANCE_STEPS = {
    "happy": 8,
    "shy": 4,
    "energetic": 12,
}


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

    def __init__(self, reachy: ReachyMini) -> None:
        self._reachy = reachy
        self._lock = asyncio.Lock()
        self._wobbling = False

    # ── Basic gestures ──────────────────────────────────────────────────

    async def nod(self, times: int = 2) -> None:
        """Nod head up and down (affirmation gesture)."""
        async with self._lock:
            logger.info("🙆 点头 x%d", times)
            for _ in range(times):
                self._reachy.goto_target(
                    head=HEAD_LOOK_UP, duration=0.25
                )
                await asyncio.sleep(0.3)
                self._reachy.goto_target(
                    head=HEAD_LOOK_DOWN, duration=0.25
                )
                await asyncio.sleep(0.3)
            # Return to neutral
            self._reachy.goto_target(head=HEAD_NEUTRAL, duration=0.3)

    async def shake_head(self, times: int = 2) -> None:
        """Shake head left and right (negation gesture)."""
        async with self._lock:
            logger.info("🙅 摇头 x%d", times)
            for _ in range(times):
                self._reachy.goto_target(
                    head=HEAD_TILT_LEFT, duration=0.2
                )
                await asyncio.sleep(0.25)
                self._reachy.goto_target(
                    head=HEAD_TILT_RIGHT, duration=0.2
                )
                await asyncio.sleep(0.25)
            self._reachy.goto_target(head=HEAD_NEUTRAL, duration=0.3)

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
                    self._reachy.set_target(
                        antennas=[-2.5, ANTENNA_NEUTRAL[1]]
                    )
                    await asyncio.sleep(0.15)
                    self._reachy.set_target(
                        antennas=[-0.5, ANTENNA_NEUTRAL[1]]
                    )
                    await asyncio.sleep(0.15)

            if side in ("right", "both"):
                for _ in range(waves):
                    self._reachy.set_target(
                        antennas=[ANTENNA_NEUTRAL[0], 2.5]
                    )
                    await asyncio.sleep(0.15)
                    self._reachy.set_target(
                        antennas=[ANTENNA_NEUTRAL[0], 0.5]
                    )
                    await asyncio.sleep(0.15)

            # Return to neutral
            self._reachy.set_target(antennas=ANTENNA_NEUTRAL)

    async def look_left_right(self) -> None:
        """Look left, then right — scanning gesture."""
        async with self._lock:
            for yaw in [0.5, -0.5, 0.0]:
                self._reachy.goto_target(
                    head=HEAD_NEUTRAL, body_yaw=yaw, duration=0.5
                )
                await asyncio.sleep(0.6)

    # ── Dance sequences ─────────────────────────────────────────────────

    async def dance(self, style: str = "happy") -> None:
        """Perform a dance sequence.

        Args:
            style: "happy", "shy", "energetic", or "random"
        """
        if style == "random":
            style = random.choice(["happy", "shy", "energetic"])

        async with self._lock:
            logger.info("💃 跳舞: %s", style)

            if style == "happy":
                await self._dance_happy()
            elif style == "shy":
                await self._dance_shy()
            elif style == "energetic":
                await self._dance_energetic()
            else:
                logger.warning("Unknown dance style: %s", style)

            # Return to neutral after dance
            self._reachy.goto_target(
                head=HEAD_NEUTRAL,
                antennas=ANTENNA_NEUTRAL,
                duration=0.5,
            )

    async def _dance_happy(self) -> None:
        """Happy dance: alternating head tilts + antenna waves."""
        for i in range(4):
            # Tilt head
            pose = HEAD_TILT_LEFT if i % 2 == 0 else HEAD_TILT_RIGHT
            self._reachy.goto_target(
                head=pose,
                antennas=ANTENNA_UP if i % 2 == 0 else ANTENNA_NEUTRAL,
                body_yaw=0.2 if i % 2 == 0 else -0.2,
                duration=0.4,
            )
            await asyncio.sleep(0.45)
        # Wiggle antennas
        for _ in range(2):
            self._reachy.set_target(antennas=[-2.0, 2.0])
            await asyncio.sleep(0.15)
            self._reachy.set_target(antennas=[2.0, -2.0])
            await asyncio.sleep(0.15)

    async def _dance_shy(self) -> None:
        """Shy dance: gentle head nods + small antenna movements."""
        for _ in range(2):
            self._reachy.goto_target(
                head=HEAD_LOOK_DOWN, duration=0.5
            )
            await asyncio.sleep(0.55)
            self._reachy.goto_target(
                head=HEAD_NEUTRAL, duration=0.5
            )
            await asyncio.sleep(0.55)
        # Slow antenna wave
        for _ in range(2):
            self._reachy.goto_target(
                antennas=[-1.0, 0.1745], duration=0.5
            )
            await asyncio.sleep(0.55)
            self._reachy.goto_target(
                antennas=[-0.1745, 1.0], duration=0.5
            )
            await asyncio.sleep(0.55)

    async def _dance_energetic(self) -> None:
        """Energetic dance: fast head + body + antenna combo."""
        moves = [
            (HEAD_LOOK_UP, ANTENNA_UP, 0.3),
            (HEAD_TILT_LEFT, [-2.0, 0.5], -0.3),
            (HEAD_TILT_RIGHT, [0.5, 2.0], 0.3),
            (HEAD_LOOK_DOWN, ANTENNA_WAVE_LEFT_UP, 0.0),
            (HEAD_LOOK_UP, ANTENNA_WAVE_RIGHT_UP, -0.3),
            (HEAD_TILT_LEFT, ANTENNA_UP, 0.3),
        ]
        for head, antennas, yaw in moves:
            self._reachy.goto_target(
                head=head,
                antennas=antennas,
                body_yaw=yaw,
                duration=0.25,
            )
            await asyncio.sleep(0.3)

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
            self._reachy.goto_target(
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

    # ── Wobbling ────────────────────────────────────────────────────────

    def enable_wobbling(self) -> None:
        """Enable audio-reactive head wobbling (syncs with TTS playback)."""
        if not self._wobbling:
            self._reachy.enable_wobbling()
            self._wobbling = True
            logger.info("🔊 头部晃动已启用")

    def disable_wobbling(self) -> None:
        """Disable audio-reactive head wobbling."""
        if self._wobbling:
            self._reachy.disable_wobbling()
            self._wobbling = False
            logger.info("🔇 头部晃动已禁用")

    @property
    def is_wobbling(self) -> bool:
        return self._wobbling

    # ── Status ──────────────────────────────────────────────────────────

    @property
    def is_busy(self) -> bool:
        """Whether a motion is currently being executed."""
        return self._lock.locked()
