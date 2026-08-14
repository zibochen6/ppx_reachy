#!/usr/bin/env python3
"""Build a fixed-shape FP16 TensorRT engine on the target Jetson."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("models/hand_pose/hand_pose_resnet18_fp16.engine"),
    )
    parser.add_argument("--workspace-mib", type=int, default=1024)
    args = parser.parse_args()

    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
    external_data = Path(f"{args.onnx}.data")
    if not external_data.is_file():
        raise FileNotFoundError(
            f"ONNX external weights not found next to model: {external_data}"
        )

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser_ = trt.OnnxParser(network, logger)
    # parse_from_file resolves ONNX external-data tensors relative to the
    # model file.  parse(read_bytes()) silently loses that base directory.
    if not parser_.parse_from_file(str(args.onnx.resolve())):
        errors = "\n".join(str(parser_.get_error(i)) for i in range(parser_.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errors}")
    if network.num_inputs != 1 or network.num_outputs != 2:
        raise RuntimeError(
            "ONNX I/O mismatch: "
            f"expected 1 input/2 outputs, got {network.num_inputs}/{network.num_outputs}"
        )
    input_shape = tuple(network.get_input(0).shape)
    output_shapes = {
        tuple(network.get_output(index).shape) for index in range(network.num_outputs)
    }
    if input_shape != (1, 3, 224, 224) or output_shapes != {
        (1, 21, 56, 56),
        (1, 40, 56, 56),
    }:
        raise RuntimeError(
            f"ONNX I/O mismatch: input={input_shape}, outputs={sorted(output_shapes)}"
        )
    if not builder.platform_has_fast_fp16:
        raise RuntimeError("目标 Jetson 不支持快速 FP16")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024 * 1024
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(serialized)
    print(f"TensorRT FP16 engine: {args.engine}")


if __name__ == "__main__":
    main()
