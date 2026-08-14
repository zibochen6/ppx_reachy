"""Single-hand 21-keypoint pose inference and gesture geometry.

The public contract deliberately contains landmarks, never a detection box.
Backends may return more than one skeleton, but :class:`ActiveHandSelector`
keeps exactly one stable subject for the interaction controller.
"""

from __future__ import annotations

import abc
import ctypes
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

HAND_KEYPOINT_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
FINGER_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


@dataclass(frozen=True)
class HandLandmark:
    x: float
    y: float
    confidence: float = 1.0

    def public(self) -> dict[str, float]:
        return {
            "x": round(float(self.x), 5),
            "y": round(float(self.y), 5),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class HandPoseResult:
    landmarks: tuple[HandLandmark, ...]
    timestamp: float = field(default_factory=time.monotonic)
    confidence: float = 0.0
    backend: str = "unknown"
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if len(self.landmarks) != 21:
            raise ValueError("a hand pose must contain exactly 21 landmarks")
        if not self.confidence:
            self.confidence = float(np.mean([p.confidence for p in self.landmarks]))

    @property
    def palm_center(self) -> tuple[float, float]:
        points = [self.landmarks[index] for index in (0, 5, 9, 13, 17)]
        return (
            float(np.mean([point.x for point in points])),
            float(np.mean([point.y for point in points])),
        )

    @property
    def scale(self) -> float:
        center = np.array(self.palm_center)
        return max(
            1e-6,
            max(
                float(np.linalg.norm(np.array((point.x, point.y)) - center))
                for point in self.landmarks
            ),
        )


class HandPoseBackend(abc.ABC):
    """Platform inference backend returning zero or more 21-point skeletons."""

    name = "unknown"

    @abc.abstractmethod
    def infer(self, frame: np.ndarray) -> list[HandPoseResult]:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _angle(a: HandLandmark, b: HandLandmark, c: HandLandmark) -> float:
    ba = np.array((a.x - b.x, a.y - b.y), dtype=np.float32)
    bc = np.array((c.x - b.x, c.y - b.y), dtype=np.float32)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def classify_fingers(
    landmarks: Sequence[HandLandmark], *, min_confidence: float = 0.35
) -> dict[str, bool]:
    """Return per-finger extension state using scale/rotation invariant geometry."""
    if len(landmarks) != 21:
        raise ValueError("expected 21 hand landmarks")
    palm_ids = (0, 5, 9, 13, 17)
    if min(landmarks[index].confidence for index in palm_ids) < min_confidence:
        return {name: False for name in FINGER_CHAINS}
    palm = np.mean(
        np.array([(landmarks[index].x, landmarks[index].y) for index in palm_ids]),
        axis=0,
    )
    palm_scale = max(
        1e-6,
        float(
            np.linalg.norm(
                np.array((landmarks[5].x, landmarks[5].y))
                - np.array((landmarks[17].x, landmarks[17].y))
            )
        ),
    )
    states: dict[str, bool] = {}
    for name, (mcp_i, pip_i, dip_i, tip_i) in FINGER_CHAINS.items():
        mcp, pip, dip, tip = (landmarks[i] for i in (mcp_i, pip_i, dip_i, tip_i))
        confident = min(point.confidence for point in (mcp, pip, dip, tip)) >= min_confidence
        straight = _angle(mcp, pip, dip) > 145.0 and _angle(pip, dip, tip) > 140.0
        tip_distance = float(np.linalg.norm(np.array((tip.x, tip.y)) - palm)) / palm_scale
        joint_distance = float(np.linalg.norm(np.array((pip.x, pip.y)) - palm)) / palm_scale
        # The thumb is shorter and its apparent angle changes more under palm roll.
        distance_ratio = 1.12 if name == "thumb" else 1.32
        states[name] = bool(confident and straight and tip_distance > joint_distance * distance_ratio)
    return states


def classify_gesture(
    pose: HandPoseResult, *, min_confidence: float = 0.35
) -> tuple[str, dict[str, bool]]:
    if pose.confidence < min_confidence or min(
        point.confidence for point in pose.landmarks
    ) < min_confidence:
        return "OTHER", {name: False for name in FINGER_CHAINS}
    states = classify_fingers(pose.landmarks, min_confidence=min_confidence)
    extended = sum(states.values())
    if extended == 5:
        return "OPEN_PALM", states
    if extended == 0:
        # A straight-but-low-confidence skeleton must not be called a fist.
        palm = np.array(pose.palm_center)
        scale = max(pose.scale, 1e-6)
        tips = [pose.landmarks[index] for index in (4, 8, 12, 16, 20)]
        compact = float(np.mean([
            np.linalg.norm(np.array((tip.x, tip.y)) - palm) / scale for tip in tips
        ])) < 0.95
        if compact:
            return "FIST", states
    return "OTHER", states


class ActiveHandSelector:
    """Lock one skeleton by wrist proximity and normalized skeleton scale."""

    def __init__(self, max_normalized_jump: float = 2.2) -> None:
        self._previous: HandPoseResult | None = None
        self._max_jump = max_normalized_jump

    def reset(self) -> None:
        self._previous = None

    def select(self, candidates: Sequence[HandPoseResult]) -> HandPoseResult | None:
        valid = [candidate for candidate in candidates if candidate.confidence > 0.0]
        if not valid:
            return None
        if self._previous is None:
            chosen = max(valid, key=lambda item: item.scale * item.confidence)
        else:
            previous_wrist = np.array(
                (self._previous.landmarks[0].x, self._previous.landmarks[0].y)
            )
            previous_scale = max(self._previous.scale, 1e-6)
            scored: list[tuple[float, HandPoseResult]] = []
            for candidate in valid:
                wrist = np.array((candidate.landmarks[0].x, candidate.landmarks[0].y))
                jump = float(np.linalg.norm(wrist - previous_wrist)) / previous_scale
                scale_change = abs(math.log(max(candidate.scale, 1e-6) / previous_scale))
                scored.append((jump + 0.35 * scale_change, candidate))
            score, chosen = min(scored, key=lambda item: item[0])
            if score > self._max_jump:
                return None
        self._previous = chosen
        return chosen


def _heatmap_peaks(cmap: np.ndarray, threshold: float) -> tuple[HandLandmark, ...] | None:
    """Portable single-skeleton CMAP decoder used when PAF parsing is unavailable.

    The interaction supports one hand only, so the highest peak per part is the
    intended portable path. Jetson may optionally use trt_pose's PAF parser.
    """
    if cmap.ndim == 4:
        cmap = cmap[0]
    if cmap.shape[0] != 21 and cmap.shape[-1] == 21:
        cmap = np.moveaxis(cmap, -1, 0)
    if cmap.shape[0] != 21:
        raise RuntimeError(f"unexpected CMAP shape: {cmap.shape}")
    result: list[HandLandmark] = []
    for heatmap in cmap:
        _, confidence, _, location = cv2.minMaxLoc(heatmap.astype(np.float32))
        if confidence < threshold:
            return None
        x = (location[0] + 0.5) / heatmap.shape[1]
        y = (location[1] + 0.5) / heatmap.shape[0]
        result.append(HandLandmark(x, y, float(confidence)))
    return tuple(result)


def decode_hand_poses(
    cmap: np.ndarray,
    paf: np.ndarray | None,
    threshold: float,
    *,
    max_peaks: int = 8,
) -> list[tuple[HandLandmark, ...]]:
    """Associate CMAP peaks with PAF vectors into one or more hand skeletons."""
    if cmap.ndim == 4:
        cmap = cmap[0]
    if cmap.shape[0] != 21 and cmap.shape[-1] == 21:
        cmap = np.moveaxis(cmap, -1, 0)
    if paf is None:
        fallback = _heatmap_peaks(cmap, threshold)
        return [fallback] if fallback else []
    if paf.ndim == 4:
        paf = paf[0]
    if paf.shape[0] != 40 and paf.shape[-1] == 40:
        paf = np.moveaxis(paf, -1, 0)
    if cmap.shape[0] != 21 or paf.shape[0] < 40:
        raise RuntimeError(f"unexpected pose output shapes: cmap={cmap.shape}, paf={paf.shape}")

    peaks: list[list[tuple[float, float, float]]] = []
    for heatmap in cmap:
        source = heatmap.astype(np.float32)
        mask = (source >= cv2.dilate(source, np.ones((3, 3), np.uint8))) & (
            source >= threshold
        )
        ys, xs = np.where(mask)
        found = sorted(
            ((float(x), float(y), float(source[y, x])) for x, y in zip(xs, ys)),
            key=lambda item: item[2],
            reverse=True,
        )[:max_peaks]
        peaks.append(found)

    nodes = [(part, index) for part, found in enumerate(peaks) for index in range(len(found))]
    parent = {node: node for node in nodes}
    parts = {node: {node[0]} for node in nodes}

    def root(node: tuple[int, int]) -> tuple[int, int]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    links: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
    height, width = cmap.shape[1:]
    for edge, (source_part, target_part) in enumerate(HAND_CONNECTIONS):
        field_x, field_y = paf[2 * edge], paf[2 * edge + 1]
        for source_index, source in enumerate(peaks[source_part]):
            for target_index, target in enumerate(peaks[target_part]):
                dx, dy = target[0] - source[0], target[1] - source[1]
                length = math.hypot(dx, dy)
                if length < 1.0:
                    continue
                unit_x, unit_y = dx / length, dy / length
                samples = 10
                xs = np.clip(np.rint(np.linspace(source[0], target[0], samples)).astype(int), 0, width - 1)
                ys = np.clip(np.rint(np.linspace(source[1], target[1], samples)).astype(int), 0, height - 1)
                alignment = field_x[ys, xs] * unit_x + field_y[ys, xs] * unit_y
                positive = alignment > 0.03
                if positive.mean() >= 0.6:
                    score = float(alignment[positive].mean()) + 0.05 * (source[2] + target[2])
                    links.append((score, (source_part, source_index), (target_part, target_index)))

    for _, left, right in sorted(links, reverse=True):
        left_root, right_root = root(left), root(right)
        if left_root == right_root or parts[left_root] & parts[right_root]:
            continue
        parent[right_root] = left_root
        parts[left_root].update(parts.pop(right_root))

    components: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for node in nodes:
        components.setdefault(root(node), []).append(node)
    skeletons: list[tuple[HandLandmark, ...]] = []
    for component in components.values():
        if len(component) < 12:
            continue
        resolved: list[HandLandmark | None] = [None] * 21
        for part, index in component:
            x, y, confidence = peaks[part][index]
            resolved[part] = HandLandmark(
                (x + 0.5) / width, (y + 0.5) / height, confidence
            )
        visible = [point for point in resolved if point is not None]
        center_x = float(np.mean([point.x for point in visible]))
        center_y = float(np.mean([point.y for point in visible]))
        skeletons.append(
            tuple(
                point if point is not None else HandLandmark(center_x, center_y, 0.0)
                for point in resolved
            )
        )
    if not skeletons:
        fallback = _heatmap_peaks(cmap, threshold)
        if fallback:
            skeletons.append(fallback)
    return skeletons


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.transpose(tensor, (2, 0, 1))[None]


class CoreMLHandPoseBackend(HandPoseBackend):
    name = "coreml"

    def __init__(self, model_path: str, *, threshold: float = 0.15) -> None:
        try:
            import coremltools as ct
        except ImportError as exc:
            raise RuntimeError("coremltools 未安装") from exc
        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(f"Core ML 手势模型不存在: {path}")
        self._model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
        self._input_name = self._model.get_spec().description.input[0].name
        self._threshold = threshold

    def infer(self, frame: np.ndarray) -> list[HandPoseResult]:
        started = time.perf_counter()
        outputs = self._model.predict({self._input_name: preprocess_frame(frame)})
        arrays = [np.asarray(value) for value in outputs.values() if isinstance(value, np.ndarray)]
        cmap = next((value for value in arrays if 21 in value.shape), None)
        paf = next((value for value in arrays if 40 in value.shape), None)
        if cmap is None:
            raise RuntimeError("Core ML 输出中没有 21 通道 CMAP")
        return [HandPoseResult(
            landmarks,
            confidence=float(np.mean([point.confidence for point in landmarks])),
            backend=self.name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        ) for landmarks in decode_hand_poses(cmap, paf, self._threshold)]


class TorchHandPoseBackend(HandPoseBackend):
    name = "mps"

    def __init__(self, model_path: str, *, threshold: float = 0.15) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch 未安装") from exc
        if not torch.backends.mps.is_available():
            raise RuntimeError("当前 Mac 没有可用的 PyTorch MPS 后端")
        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(f"TorchScript 手势模型不存在: {path}")
        self._torch = torch
        self._model = torch.jit.load(str(path), map_location="mps").eval()
        self._threshold = threshold

    def infer(self, frame: np.ndarray) -> list[HandPoseResult]:
        started = time.perf_counter()
        tensor = self._torch.from_numpy(preprocess_frame(frame)).to("mps")
        with self._torch.inference_mode():
            outputs = self._model(tensor)
        cmap = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        paf = outputs[1] if isinstance(outputs, (tuple, list)) and len(outputs) > 1 else None
        decoded = decode_hand_poses(
            cmap.detach().float().cpu().numpy(),
            paf.detach().float().cpu().numpy() if paf is not None else None,
            self._threshold,
        )
        return [HandPoseResult(
            landmarks,
            confidence=float(np.mean([point.confidence for point in landmarks])),
            backend=self.name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        ) for landmarks in decoded]


class TensorRTHandPoseBackend(HandPoseBackend):
    name = "tensorrt_fp16"

    def __init__(self, engine_path: str, *, threshold: float = 0.15) -> None:
        if sys.version_info[:2] != (3, 10):
            raise RuntimeError(
                "Jetson TensorRT 手势后端要求 Python 3.10，"
                f"当前为 {sys.version_info.major}.{sys.version_info.minor}"
            )
        try:
            import tensorrt as trt
            import torch
        except ImportError as exc:
            raise RuntimeError("Jetson TensorRT/PyTorch CUDA 运行时未安装") from exc
        trt_version = tuple(
            int(part) for part in str(trt.__version__).split(".")[:2]
        )
        if trt_version != (10, 3):
            raise RuntimeError(
                f"TensorRT 版本不兼容：需要 10.3.x，当前为 {trt.__version__}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；禁止手势推理静默回退 CPU")
        path = Path(engine_path)
        if not path.exists():
            raise RuntimeError(f"TensorRT 手势引擎不存在: {path}")
        self._trt, self._torch = trt, torch
        logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(logger)
        self._engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if self._engine is None:
            raise RuntimeError("TensorRT engine 与当前 JetPack/TensorRT 不兼容")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("TensorRT execution context 创建失败")
        self._threshold = threshold
        input_names = [
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        ]
        if len(input_names) != 1:
            raise RuntimeError(
                f"TensorRT engine I/O 不兼容：需要 1 个输入，实际 {len(input_names)} 个"
            )
        self._input_name = input_names[0]
        self._output_names = [
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(index))
            == trt.TensorIOMode.OUTPUT
        ]
        if len(self._output_names) != 2:
            raise RuntimeError(
                f"TensorRT engine I/O 不兼容：需要 2 个输出，实际 {len(self._output_names)} 个"
            )
        if not self._context.set_input_shape(self._input_name, (1, 3, 224, 224)):
            raise RuntimeError("TensorRT engine 不接受固定输入 1x3x224x224")
        input_shape = tuple(self._context.get_tensor_shape(self._input_name))
        output_shapes = {
            tuple(self._context.get_tensor_shape(name)) for name in self._output_names
        }
        expected_outputs = {(1, 21, 56, 56), (1, 40, 56, 56)}
        if input_shape != (1, 3, 224, 224) or output_shapes != expected_outputs:
            raise RuntimeError(
                "TensorRT engine I/O 规格不兼容："
                f"input={input_shape}, outputs={sorted(output_shapes)}"
            )
        self._input_dtype = (
            torch.float16
            if self._engine.get_tensor_dtype(self._input_name) == trt.float16
            else torch.float32
        )

    def infer(self, frame: np.ndarray) -> list[HandPoseResult]:
        started = time.perf_counter()
        torch = self._torch
        preprocessed = np.ascontiguousarray(preprocess_frame(frame))
        # The JetPack 6.2 JPL PyTorch wheel is built against NumPy 1.x while
        # the application uses NumPy 2.2.  The buffer protocol avoids the
        # unavailable torch.from_numpy C-API without changing inference.
        tensor = (
            torch.frombuffer(memoryview(preprocessed), dtype=torch.float32)
            .reshape(preprocessed.shape)
            .to(device="cuda", dtype=self._input_dtype)
            .contiguous()
        )
        outputs: dict[str, Any] = {}
        self._context.set_tensor_address(self._input_name, tensor.data_ptr())
        for name in self._output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            dtype = torch.float16 if self._engine.get_tensor_dtype(name) == self._trt.float16 else torch.float32
            outputs[name] = torch.empty(shape, device="cuda", dtype=dtype)
            self._context.set_tensor_address(name, outputs[name].data_ptr())
        if not self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT 手势推理执行失败")
        torch.cuda.current_stream().synchronize()
        cmap_tensor = next((value for value in outputs.values() if 21 in value.shape), None)
        paf_tensor = next((value for value in outputs.values() if 40 in value.shape), None)
        if cmap_tensor is None:
            raise RuntimeError("TensorRT 输出中没有 21 通道 CMAP")
        decoded = decode_hand_poses(
            _tensor_to_numpy_float32(cmap_tensor),
            _tensor_to_numpy_float32(paf_tensor) if paf_tensor is not None else None,
            self._threshold,
        )
        return [HandPoseResult(
            landmarks,
            confidence=float(np.mean([point.confidence for point in landmarks])),
            backend=self.name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        ) for landmarks in decoded]


def _tensor_to_numpy_float32(tensor: Any) -> np.ndarray:
    """Copy a torch tensor to NumPy without torch's NumPy C-API bridge."""
    cpu_tensor = tensor.detach().float().cpu().contiguous()
    result = np.empty(tuple(cpu_tensor.shape), dtype=np.float32)
    ctypes.memmove(result.ctypes.data, cpu_tensor.data_ptr(), result.nbytes)
    return result


class Torch2TRTHandPoseBackend(HandPoseBackend):
    name = "torch2trt_fp16"

    def __init__(self, model_path: str, *, threshold: float = 0.15) -> None:
        try:
            import torch
            from torch2trt import TRTModule
        except ImportError as exc:
            raise RuntimeError("Jetson torch2trt 运行时未安装") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；禁止手势推理静默回退 CPU")
        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(f"torch2trt 手势模型不存在: {path}")
        self._torch = torch
        self._model = TRTModule()
        self._model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
        self._model.eval()
        self._threshold = threshold

    def infer(self, frame: np.ndarray) -> list[HandPoseResult]:
        started = time.perf_counter()
        # The official torch2trt conversion keeps the network input binding
        # in FP32 while selecting FP16 kernels internally.
        tensor = self._torch.from_numpy(preprocess_frame(frame)).cuda()
        with self._torch.inference_mode():
            outputs = self._model(tensor)
        cmap = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        paf = outputs[1] if isinstance(outputs, (tuple, list)) and len(outputs) > 1 else None
        decoded = decode_hand_poses(
            cmap.float().cpu().numpy(),
            paf.float().cpu().numpy() if paf is not None else None,
            self._threshold,
        )
        return [HandPoseResult(
            landmarks,
            confidence=float(np.mean([point.confidence for point in landmarks])),
            backend=self.name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        ) for landmarks in decoded]


def create_hand_pose_backend(config: Any) -> HandPoseBackend:
    requested = str(getattr(config, "gesture_backend", "auto") or "auto").lower()
    system = platform.system()
    if requested == "auto":
        requested = "coreml" if system == "Darwin" else "tensorrt"
    if requested == "coreml":
        try:
            return CoreMLHandPoseBackend(
                config.gesture_coreml_model_path,
                threshold=config.gesture_keypoint_confidence,
            )
        except RuntimeError:
            if system != "Darwin":
                raise
            return TorchHandPoseBackend(
                config.gesture_torchscript_model_path,
                threshold=config.gesture_keypoint_confidence,
            )
    if requested == "mps":
        return TorchHandPoseBackend(
            config.gesture_torchscript_model_path,
            threshold=config.gesture_keypoint_confidence,
        )
    if requested == "tensorrt":
        # Jetson production mode is strict: an incompatible/missing engine or
        # unavailable CUDA must surface to the Web UI, never fall back.
        return TensorRTHandPoseBackend(
            config.gesture_tensorrt_engine_path,
            threshold=config.gesture_keypoint_confidence,
        )
    raise RuntimeError(f"未知手势推理后端: {requested}")
