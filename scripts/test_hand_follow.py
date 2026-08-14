#!/usr/bin/env python3
"""Standalone Reachy Mini camera + Core ML hand-follow test for macOS.

No Dashboard, ASR, TTS, wake-word detector, or conversation engine is started.
The script uses the Reachy SDK MediaManager camera, draws all 21 landmarks,
and lets a debounced OPEN_PALM pose own head/body-yaw tracking.

Keys:
  q / Esc  quit and return to neutral
  space    pause/resume motor tracking (inference keeps running)
  r        return to neutral immediately
"""

from __future__ import annotations

import argparse
import asyncio
import math
import signal
import socket
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chaihuo_reachy.backends.sdk_camera import SdkCamera
from chaihuo_reachy.config import load_config
from chaihuo_reachy.hand_pose import (
    HAND_CONNECTIONS,
    ActiveHandSelector,
    CoreMLHandPoseBackend,
    HandPoseResult,
    classify_gesture,
)
from chaihuo_reachy.main import (
    _resolve_daemon_serial_port,
    _spawn_sdk_daemon_process,
    _terminate_owned_daemon,
)
from chaihuo_reachy.motion import ANTENNA_NEUTRAL, HEAD_NEUTRAL


class ShutdownController:
    """Turn SIGINT into a cooperative stop without cancelling cleanup awaits."""

    def __init__(self) -> None:
        self.requested = threading.Event()
        self.cleaning = threading.Event()

    def handle_signal(self, signum: int, _frame: Any) -> None:
        if self.cleaning.is_set():
            print("\n正在让 Reachy Mini 休眠并释放资源，请稍候…", flush=True)
            return
        if not self.requested.is_set():
            print("\n收到 Ctrl+C，正在安全退出…", flush=True)
            self.requested.set()


def daemon_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


async def wait_for_daemon(host: str, port: int, process: Any, timeout: float = 35) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Reachy daemon 提前退出，code={process.returncode}")
        try:
            async with httpx.AsyncClient(timeout=0.6) as client:
                response = await client.get(f"http://{host}:{port}/api/daemon/status")
            payload = response.json()
            if response.status_code == 200 and payload.get("state") == "running":
                return
            last_error = str(payload.get("error") or payload.get("state") or "未就绪")
        except Exception as exc:
            last_error = str(exc)
        await asyncio.sleep(0.4)
    raise RuntimeError(f"等待 Reachy daemon 超时：{last_error}")


async def wake_and_stand(reachy: Any, motor_settle_s: float = 0.2) -> None:
    """Enable motor torque before asking a daemon-started-asleep robot to stand."""
    await asyncio.to_thread(reachy.enable_motors)
    await asyncio.sleep(motor_settle_s)
    await asyncio.to_thread(reachy.wake_up)


async def sleep_and_disable_motors(reachy: Any) -> None:
    """Move to the resting pose and always release motor torque afterwards."""
    try:
        await asyncio.to_thread(reachy.goto_sleep)
    finally:
        await asyncio.to_thread(reachy.disable_motors)


def draw_pose(frame: np.ndarray, pose: HandPoseResult) -> None:
    height, width = frame.shape[:2]
    for source, target in HAND_CONNECTIONS:
        a, b = pose.landmarks[source], pose.landmarks[target]
        if min(a.confidence, b.confidence) <= 0:
            continue
        cv2.line(
            frame,
            (int(a.x * width), int(a.y * height)),
            (int(b.x * width), int(b.y * height)),
            (60, 230, 80),
            3,
            cv2.LINE_AA,
        )
    for point in pose.landmarks:
        if point.confidence <= 0:
            continue
        cv2.circle(
            frame,
            (int(point.x * width), int(point.y * height)),
            6,
            (0, 210, 255),
            -1,
            cv2.LINE_AA,
        )


def make_display_frame(frame: np.ndarray) -> np.ndarray:
    """Return an independent, C-contiguous frame that OpenCV can draw on.

    Reachy's SDK camera may expose its memory-mapped image as a read-only
    NumPy view.  Inference only reads that view, but OpenCV drawing functions
    require a writable output array.
    """
    return np.array(frame, copy=True, order="C")


class FollowController:
    def __init__(self, reachy: Any, args: argparse.Namespace) -> None:
        self.reachy = reachy
        self.args = args
        self.enabled = True
        self.filtered_center: tuple[float, float] | None = None
        self.head_yaw = 0.0
        self.head_pitch = 0.0
        self.body_yaw = 0.0

    def update(self, center: tuple[float, float]) -> None:
        alpha = self.args.filter_alpha
        if self.filtered_center is None:
            self.filtered_center = center
        else:
            self.filtered_center = (
                alpha * center[0] + (1 - alpha) * self.filtered_center[0],
                alpha * center[1] + (1 - alpha) * self.filtered_center[1],
            )
        error_x = self.filtered_center[0] - 0.5
        error_y = self.filtered_center[1] - 0.5
        if abs(error_x) < self.args.deadzone:
            error_x = 0.0
        if abs(error_y) < self.args.deadzone:
            error_y = 0.0

        target_yaw = float(
            np.clip(-2 * error_x * self.args.head_yaw, -self.args.head_yaw, self.args.head_yaw)
        )
        target_pitch = float(
            np.clip(2 * error_y * self.args.head_pitch, -self.args.head_pitch, self.args.head_pitch)
        )
        target_body = float(
            np.clip(-2 * error_x * self.args.body_yaw, -self.args.body_yaw, self.args.body_yaw)
        )
        step = self.args.max_step
        self.head_yaw += float(np.clip(target_yaw - self.head_yaw, -step, step))
        self.head_pitch += float(np.clip(target_pitch - self.head_pitch, -step, step))
        self.body_yaw += float(np.clip(target_body - self.body_yaw, -step, step))

        from reachy_mini.utils import create_head_pose

        head = create_head_pose(
            yaw=self.head_yaw,
            pitch=self.head_pitch,
            degrees=True,
        )
        self.reachy.set_target(head=head, body_yaw=math.radians(self.body_yaw))

    async def neutral(self) -> None:
        self.filtered_center = None
        self.head_yaw = self.head_pitch = self.body_yaw = 0.0
        await asyncio.to_thread(
            self.reachy.goto_target,
            head=HEAD_NEUTRAL,
            antennas=ANTENNA_NEUTRAL,
            body_yaw=0.0,
            duration=0.5,
        )


async def run(
    args: argparse.Namespace,
    shutdown: ShutdownController | None = None,
) -> None:
    from reachy_mini import ReachyMini

    shutdown = shutdown or ShutdownController()
    cfg = load_config()
    cfg.daemon_host = "127.0.0.1"
    cfg.media_backend = "local"
    cfg.daemon_simulation = False
    daemon_process = None
    reachy = None
    camera = None
    controller = None

    try:
        if not daemon_port_open(cfg.daemon_host, cfg.daemon_port):
            serial_port = _resolve_daemon_serial_port(cfg)
            daemon_process = _spawn_sdk_daemon_process(cfg, serial_port)
            print(f"已启动 Reachy daemon，serial={serial_port}")
        else:
            print("复用当前本机 Reachy daemon（退出时不会终止它）")
        await wait_for_daemon(
            cfg.daemon_host, cfg.daemon_port, daemon_process, args.daemon_timeout
        )
        if shutdown.requested.is_set():
            return

        reachy = await asyncio.to_thread(
            ReachyMini,
            host=cfg.daemon_host,
            port=cfg.daemon_port,
            connection_mode="localhost_only",
            spawn_daemon=False,
            use_sim=False,
            timeout=15.0,
            automatic_body_yaw=False,
            media_backend="local",
        )
        if shutdown.requested.is_set():
            return
        print("启用电机并让 Reachy Mini 站起来…")
        await wake_and_stand(reachy)
        print("Reachy Mini 已完成站立动作")
        if shutdown.requested.is_set():
            return

        camera = SdkCamera(reachy.media_manager)
        if not camera.is_active:
            raise RuntimeError("Reachy SDK 摄像头未就绪")
        backend = CoreMLHandPoseBackend(args.model, threshold=args.confidence)
        selector = ActiveHandSelector()
        controller = FollowController(reachy, args)

        state = "SEARCHING"
        gesture = "OTHER"
        candidate = "OTHER"
        candidate_since = time.monotonic()
        last_valid = 0.0
        last_neutral = 0.0
        inference_count = 0
        started = time.monotonic()
        last_frame_at = time.monotonic()

        print("已启动：张掌保持 300 ms 开始跟随；q/Esc 退出，空格暂停，r 回中。")
        while not shutdown.requested.is_set():
            tick = time.monotonic()
            frame = camera.read()
            if frame is None:
                if tick - last_frame_at > 2.0:
                    raise RuntimeError("Reachy SDK 摄像头超过 2 秒没有新帧")
                await asyncio.sleep(0.01)
                continue
            last_frame_at = tick
            candidates = backend.infer(frame)
            inference_count += 1
            pose = selector.select(candidates)
            display_frame = make_display_frame(frame)
            raw_gesture = "OTHER"
            fingers: dict[str, bool] = {}
            if pose is not None:
                raw_gesture, fingers = classify_gesture(
                    pose, min_confidence=args.confidence
                )
                draw_pose(display_frame, pose)
                if raw_gesture != "OTHER":
                    last_valid = tick

            if raw_gesture != candidate:
                candidate = raw_gesture
                candidate_since = tick
            elif (
                raw_gesture in {"OPEN_PALM", "FIST"}
                and tick - candidate_since >= args.confirm_ms / 1000.0
            ):
                gesture = raw_gesture

            if (
                pose is not None
                and gesture == "OPEN_PALM"
                and raw_gesture == "OPEN_PALM"
                and controller.enabled
            ):
                state = "TRACKING"
                controller.update(pose.palm_center)
            elif gesture == "FIST" and raw_gesture == "FIST":
                state = "FIST (本测试不跳舞)"
            elif last_valid <= 0 or tick - last_valid >= args.lost_ms / 1000.0:
                state = "SEARCHING"
                gesture = "OTHER"
                selector.reset()
                if tick - last_neutral >= 1.0:
                    await controller.neutral()
                    last_neutral = tick

            fps = inference_count / max(0.001, tick - started)
            finger_count = sum(fingers.values()) if fingers else 0
            color = (60, 230, 80) if state == "TRACKING" else (0, 200, 255)
            cv2.putText(
                display_frame,
                f"{state} | raw={raw_gesture} fingers={finger_count}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"CoreML {pose.latency_ms if pose else 0:.1f} ms | {fps:.1f} FPS | motors={'ON' if controller.enabled else 'PAUSED'}",
                (24, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            display = cv2.resize(display_frame, (960, 540))
            cv2.imshow("Reachy Mini Hand Follow Test", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                controller.enabled = not controller.enabled
                if not controller.enabled:
                    await controller.neutral()
            elif key == ord("r"):
                await controller.neutral()
            await asyncio.sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - tick)))
    finally:
        shutdown.cleaning.set()
        cv2.destroyAllWindows()
        # A daemon started by this script owns the safe shutdown sequence:
        # SIGTERM makes it sleep the robot, disable torque, and release media.
        if daemon_process is not None and not args.keep_awake:
            print("正在让 Reachy Mini 休眠并停止本次 daemon（约 5 秒）…")
            try:
                await _terminate_owned_daemon(
                    daemon_process, state_file=cfg.daemon_state_file
                )
            except Exception as exc:
                print(f"警告：daemon 正常停止失败，将在关闭连接后重试：{exc}")
            else:
                daemon_process = None
                print("Reachy Mini 已休眠，电机和 daemon 资源已释放")
        elif reachy is not None and not args.keep_awake:
            # A reused daemon must stay alive, so release this client's motors
            # explicitly after the sleep trajectory completes.
            try:
                print("正在让 Reachy Mini 休眠并关闭电机（约 5 秒）…")
                await sleep_and_disable_motors(reachy)
                print("Reachy Mini 已休眠，电机已关闭")
            except Exception as exc:
                print(f"警告：自动休眠失败，但已尝试关闭电机：{exc}")
        elif controller is not None:
            try:
                await controller.neutral()
            except Exception:
                print("警告：退出回中失败")
        if reachy is not None:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(reachy.media_manager.close), timeout=5.0
                )
            except Exception:
                print("警告：摄像头媒体资源关闭超时或失败")
            try:
                reachy.client.disconnect()
            except Exception:
                pass
        if daemon_process is not None:
            await _terminate_owned_daemon(
                daemon_process, state_file=cfg.daemon_state_file
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reachy Mini 独立手掌实时识别与跟随测试（不启动语音服务）"
    )
    parser.add_argument(
        "--model", default="models/hand_pose/hand_pose_resnet18.mlpackage"
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--confirm-ms", type=int, default=300)
    parser.add_argument("--lost-ms", type=int, default=800)
    parser.add_argument("--deadzone", type=float, default=0.06)
    parser.add_argument("--filter-alpha", type=float, default=0.35)
    parser.add_argument("--max-step", type=float, default=2.0)
    parser.add_argument("--head-yaw", type=float, default=20.0)
    parser.add_argument("--head-pitch", type=float, default=20.0)
    parser.add_argument("--body-yaw", type=float, default=30.0)
    parser.add_argument("--daemon-timeout", type=float, default=35.0)
    parser.add_argument("--keep-awake", action="store_true")
    args = parser.parse_args()
    if not Path(args.model).exists():
        parser.error(f"Core ML 模型不存在: {args.model}")
    return args


def main() -> None:
    shutdown = ShutdownController()
    previous_sigint = signal.signal(signal.SIGINT, shutdown.handle_signal)
    try:
        asyncio.run(run(parse_args(), shutdown))
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    main()
