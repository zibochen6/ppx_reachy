#!/usr/bin/env python3
"""Validate the Python 3.10 CUDA/TensorRT hand-pose runtime on Jetson."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from chaihuo_reachy.hand_pose import (
    HAND_CONNECTIONS,
    TensorRTHandPoseBackend,
    decode_hand_poses,
)


OPEN_HAND = (
    (.50, .78),
    (.42, .68), (.35, .61), (.28, .54), (.21, .47),
    (.40, .59), (.39, .46), (.38, .33), (.37, .20),
    (.48, .56), (.48, .42), (.48, .28), (.48, .14),
    (.56, .58), (.57, .45), (.58, .32), (.59, .19),
    (.64, .62), (.67, .51), (.70, .40), (.73, .29),
)


def verify_decoder() -> int:
    """Exercise CMAP/PAF parsing independently of camera content."""
    size = 56
    cmap = np.zeros((1, 21, size, size), dtype=np.float32)
    paf = np.zeros((1, 40, size, size), dtype=np.float32)
    points = [
        (int(round(x * (size - 1))), int(round(y * (size - 1))))
        for x, y in OPEN_HAND
    ]
    for part, (x, y) in enumerate(points):
        cmap[0, part, y, x] = 0.99
    for edge, (source, target) in enumerate(HAND_CONNECTIONS):
        x1, y1 = points[source]
        x2, y2 = points[target]
        length = max(1e-6, math.hypot(x2 - x1, y2 - y1))
        cv2.line(
            paf[0, 2 * edge],
            (x1, y1),
            (x2, y2),
            (x2 - x1) / length,
            2,
        )
        cv2.line(
            paf[0, 2 * edge + 1],
            (x1, y1),
            (x2, y2),
            (y2 - y1) / length,
            2,
        )
    hands = decode_hand_poses(cmap, paf, 0.15)
    if not hands or len(hands[0]) != 21:
        raise RuntimeError("CMAP/PAF decoder did not reconstruct one 21-point hand")
    return len(hands[0])


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("models/hand_pose/hand_pose_resnet18_fp16.engine"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--min-fps", type=float, default=10.0)
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Python 3.10 required, got {sys.version.split()[0]}")

    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false; CPU fallback is forbidden")
    tensor = torch.arange(4096, device="cuda", dtype=torch.float32)
    tensor_result = float((tensor.square().mean()).item())

    decoder_points = verify_decoder()
    backend = TensorRTHandPoseBackend(str(args.engine))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        for _ in range(max(0, args.warmup)):
            backend.infer(frame)
        timings_ms: list[float] = []
        for _ in range(max(1, args.iterations)):
            started = time.perf_counter()
            backend.infer(frame)
            timings_ms.append((time.perf_counter() - started) * 1000.0)
    finally:
        backend.close()

    average_ms = statistics.fmean(timings_ms)
    p95_ms = percentile_95(timings_ms)
    fps = 1000.0 / average_ms
    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "tensorrt": trt.__version__,
        "cuda_available": True,
        "cuda_tensor_result": tensor_result,
        "engine": str(args.engine),
        "backend": backend.name,
        "decoder_landmarks": decoder_points,
        "iterations": len(timings_ms),
        "average_latency_ms": round(average_ms, 3),
        "p95_latency_ms": round(p95_ms, 3),
        "fps": round(fps, 3),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if fps < args.min_fps:
        raise RuntimeError(f"TensorRT hand-pose throughput {fps:.2f} FPS < {args.min_fps:.2f}")


if __name__ == "__main__":
    main()
