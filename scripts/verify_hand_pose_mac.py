#!/usr/bin/env python3
"""Verify Core ML hand pose using only the Reachy Mini USB camera on macOS.

This diagnostic never sends motor or audio commands. It deliberately resolves
the camera by the stable AVFoundation name, so camera index 0 (the Mac camera)
cannot be selected accidentally.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter

import cv2

from chaihuo_reachy.camera import Camera, find_reachy_camera
from chaihuo_reachy.hand_pose import (
    HAND_CONNECTIONS,
    CoreMLHandPoseBackend,
    classify_gesture,
)
from chaihuo_reachy.main import _MJPEGStream


def draw_pose(frame, pose, gesture: str) -> None:
    height, width = frame.shape[:2]
    for source, target in HAND_CONNECTIONS:
        a, b = pose.landmarks[source], pose.landmarks[target]
        cv2.line(
            frame,
            (int(a.x * width), int(a.y * height)),
            (int(b.x * width), int(b.y * height)),
            (0, 255, 0),
            3,
        )
    for point in pose.landmarks:
        cv2.circle(
            frame,
            (int(point.x * width), int(point.y * height)),
            6,
            (0, 200, 255),
            -1,
        )
    cv2.putText(
        frame,
        gesture,
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument(
        "--model", default="models/hand_pose/hand_pose_resnet18.mlpackage"
    )
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    selector = find_reachy_camera("auto")
    if selector != "avfoundation:Reachy Mini Camera":
        raise RuntimeError(
            f"拒绝启动：未解析到 Reachy Mini 稳定设备名，实际为 {selector!r}"
        )
    camera = Camera(device=selector, width=1920, height=1080)
    camera.open()
    hub = _MJPEGStream(camera_backend=camera, fps=15)
    if not hub.start():
        camera.close()
        raise RuntimeError("Reachy Mini 摄像头无法启动")
    backend = CoreMLHandPoseBackend(args.model, threshold=0.15)

    started = time.monotonic()
    counts: Counter[str] = Counter()
    stable: list[str] = []
    latencies: list[float] = []
    candidate = "NONE"
    candidate_since = started
    try:
        while time.monotonic() - started < args.seconds:
            tick = time.monotonic()
            frame = hub.get_bgr_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            poses = backend.infer(frame)
            gesture = "NONE"
            if poses:
                pose = poses[0]
                latencies.append(pose.latency_ms)
                gesture, _ = classify_gesture(pose, min_confidence=0.15)
                draw_pose(frame, pose, gesture)
            counts[gesture] += 1
            if gesture != candidate:
                candidate, candidate_since = gesture, tick
            elif (
                gesture in {"OPEN_PALM", "FIST"}
                and tick - candidate_since >= 0.3
                and (not stable or stable[-1] != gesture)
            ):
                stable.append(gesture)
                print(f"稳定姿态: {gesture}")
            if not args.no_window:
                cv2.imshow("Reachy Mini Hand Pose (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(max(0.0, 1.0 / 15.0 - (time.monotonic() - tick)))
    finally:
        hub.stop()
        camera.close()
        cv2.destroyAllWindows()

    mean_latency = statistics.mean(latencies) if latencies else 0.0
    print(f"camera={selector}")
    print(f"frames={sum(counts.values())}, gestures={dict(counts)}")
    print(f"stable_sequence={stable}")
    print(f"coreml_mean_ms={mean_latency:.2f}")
    if "OPEN_PALM" not in stable or "FIST" not in stable:
        raise SystemExit("未同时验证到稳定张掌和握拳，请面对 Reachy 镜头重试")


if __name__ == "__main__":
    main()
