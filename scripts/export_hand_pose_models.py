#!/usr/bin/env python3
"""Export NVIDIA trt_pose_hand ResNet18 weights for both supported platforms."""

from __future__ import annotations

import argparse
from pathlib import Path


def load_model(weights: Path):
    import torch
    import trt_pose.models

    # The complete checkpoint is loaded immediately below, so downloading
    # torchvision's ImageNet weights here only adds an unrelated dependency.
    model = trt_pose.models.resnet18_baseline_att(21, 40, pretrained=False).eval()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/hand_pose"))
    parser.add_argument("--skip-coreml", action="store_true")
    args = parser.parse_args()

    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.weights)
    sample = torch.zeros(1, 3, 224, 224)
    with torch.inference_mode():
        traced = torch.jit.trace(model, sample)
        reference = model(sample)

    ts_path = args.output_dir / "hand_pose_resnet18.ts"
    onnx_path = args.output_dir / "hand_pose_resnet18.onnx"
    traced.save(str(ts_path))
    torch.onnx.export(
        model,
        sample,
        str(onnx_path),
        input_names=["input"],
        output_names=["cmap", "paf"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"TorchScript: {ts_path}")
    print(f"ONNX:       {onnx_path}")

    if not args.skip_coreml:
        import coremltools as ct

        coreml_path = args.output_dir / "hand_pose_resnet18.mlpackage"
        converted = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="input", shape=sample.shape)],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS13,
        )
        converted.save(str(coreml_path))
        prediction = converted.predict({"input": sample.numpy()})
        arrays = list(prediction.values())
        cmap = next(value for value in arrays if 21 in value.shape)
        error = (torch.from_numpy(cmap).float() - reference[0].float()).abs()
        print(f"Core ML:    {coreml_path}")
        print(f"CMAP error: mean={error.mean().item():.6f}, max={error.max().item():.6f}")
        if error.max().item() > 0.08:
            raise RuntimeError("Core ML consistency threshold failed; use the MPS fallback")


if __name__ == "__main__":
    main()
