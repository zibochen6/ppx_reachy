from __future__ import annotations

import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import chaihuo_reachy.hand_pose as hand_pose_module
from chaihuo_reachy.hand_pose import (
    ActiveHandSelector,
    HAND_CONNECTIONS,
    HandLandmark,
    HandPoseResult,
    classify_gesture,
    create_hand_pose_backend,
    decode_hand_poses,
)


_OPEN = [
    (.50, .78),
    (.42, .68), (.35, .61), (.28, .54), (.21, .47),
    (.40, .59), (.39, .46), (.38, .33), (.37, .20),
    (.48, .56), (.48, .42), (.48, .28), (.48, .14),
    (.56, .58), (.57, .45), (.58, .32), (.59, .19),
    (.64, .62), (.67, .51), (.70, .40), (.73, .29),
]


def pose(points: list[tuple[float, float]], *, backend: str = "test") -> HandPoseResult:
    return HandPoseResult(
        tuple(HandLandmark(x, y, 0.99) for x, y in points),
        backend=backend,
    )


def fist_points() -> list[tuple[float, float]]:
    points = list(_OPEN)
    for chain in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        mcp = points[chain[0]]
        points[chain[1]] = (mcp[0], mcp[1] - .08)
        points[chain[2]] = (mcp[0] + .07, mcp[1] - .06)
        points[chain[3]] = (.51, .61)
    return points


def transformed(
    points: list[tuple[float, float]], *, scale: float, angle: float
) -> list[tuple[float, float]]:
    cosine, sine = math.cos(angle), math.sin(angle)
    result = []
    for x, y in points:
        x, y = (x - .5) * scale, (y - .5) * scale
        result.append((.5 + x * cosine - y * sine, .5 + x * sine + y * cosine))
    return result


def test_open_palm_is_rotation_and_scale_invariant() -> None:
    for scale, angle in ((1.0, 0.0), (.6, .8), (1.2, -1.1)):
        gesture, fingers = classify_gesture(pose(transformed(_OPEN, scale=scale, angle=angle)))
        assert gesture == "OPEN_PALM"
        assert all(fingers.values())


def test_fist_and_partial_finger_combinations() -> None:
    closed = fist_points()
    gesture, fingers = classify_gesture(pose(closed))
    assert gesture == "FIST"
    assert not any(fingers.values())

    one_finger = list(closed)
    one_finger[5:9] = _OPEN[5:9]
    gesture, fingers = classify_gesture(pose(one_finger))
    assert gesture == "OTHER"
    assert fingers["index"]

    two_fingers = list(one_finger)
    two_fingers[9:13] = _OPEN[9:13]
    gesture, fingers = classify_gesture(pose(two_fingers))
    assert gesture == "OTHER"
    assert fingers["index"] and fingers["middle"]


def test_active_hand_locks_first_large_skeleton_and_ignores_second() -> None:
    selector = ActiveHandSelector()
    large = pose(transformed(_OPEN, scale=1.0, angle=0.0), backend="large")
    small = pose(transformed(_OPEN, scale=.5, angle=0.0), backend="small")
    assert selector.select([small, large]).backend == "large"

    moved_large = pose(
        [(x + .015, y) for x, y in transformed(_OPEN, scale=1.0, angle=0.0)],
        backend="large-moved",
    )
    other = pose(
        [(x - .28, y) for x, y in transformed(_OPEN, scale=.8, angle=0.0)],
        backend="other",
    )
    assert selector.select([other, moved_large]).backend == "large-moved"


def test_paf_decoder_returns_two_skeletons_without_detection_boxes() -> None:
    size = 64
    cmap = np.zeros((21, size, size), np.float32)
    paf = np.zeros((40, size, size), np.float32)
    hands = []
    for center_x in (.27, .73):
        hand = [
            (
                int(round((center_x + (x - .5) * .42) * (size - 1))),
                int(round((.52 + (y - .5) * .72) * (size - 1))),
            )
            for x, y in _OPEN
        ]
        hands.append(hand)
        for part, (x, y) in enumerate(hand):
            cmap[part, y, x] = .99
    for edge, (source, target) in enumerate(HAND_CONNECTIONS):
        for hand in hands:
            x1, y1 = hand[source]
            x2, y2 = hand[target]
            length = max(1e-6, math.hypot(x2 - x1, y2 - y1))
            cv2.line(paf[2 * edge], (x1, y1), (x2, y2), (x2 - x1) / length, 2)
            cv2.line(paf[2 * edge + 1], (x1, y1), (x2, y2), (y2 - y1) / length, 2)
    decoded = decode_hand_poses(cmap, paf, .15)
    assert len(decoded) == 2
    assert all(len(hand) == 21 for hand in decoded)


def test_tensorrt_backend_rejects_non_python_310_before_loading_runtime(
    monkeypatch,
) -> None:
    class FakeVersion(tuple):
        major = 3
        minor = 11

    monkeypatch.setattr(
        hand_pose_module.sys, "version_info", FakeVersion((3, 11, 0))
    )
    with pytest.raises(RuntimeError, match="Python 3.10"):
        hand_pose_module.TensorRTHandPoseBackend("missing.engine")


def test_tensorrt_request_never_falls_back_to_torch2trt(
    monkeypatch, tmp_path
) -> None:
    fallback = tmp_path / "legacy.pth"
    fallback.write_bytes(b"legacy")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("engine incompatible")

    monkeypatch.setattr(hand_pose_module, "TensorRTHandPoseBackend", unavailable)
    config = SimpleNamespace(
        gesture_backend="tensorrt",
        gesture_tensorrt_engine_path="missing.engine",
        gesture_torch2trt_model_path=str(fallback),
        gesture_keypoint_confidence=0.15,
    )
    with pytest.raises(RuntimeError, match="engine incompatible"):
        create_hand_pose_backend(config)
