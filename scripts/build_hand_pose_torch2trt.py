#!/usr/bin/env python3
"""Build NVIDIA trt_pose_hand with the project's official torch2trt route."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/hand_pose/hand_pose_resnet18_torch2trt.pth"),
    )
    args = parser.parse_args()

    import torch
    import trt_pose.models
    from torch2trt import torch2trt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = trt_pose.models.resnet18_baseline_att(
        21, 40, pretrained=False
    ).cuda().eval()
    state = torch.load(args.weights, map_location="cuda", weights_only=True)
    model.load_state_dict(state)
    sample = torch.zeros((1, 3, 224, 224), device="cuda")
    converted = torch2trt(
        model,
        [sample],
        fp16_mode=True,
        max_workspace_size=1 << 30,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted.state_dict(), args.output)
    print(f"torch2trt FP16 module: {args.output}")


if __name__ == "__main__":
    main()
