import importlib.util
import asyncio
from pathlib import Path

import cv2
import numpy as np


_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "test_hand_follow.py"
_SPEC = importlib.util.spec_from_file_location("test_hand_follow_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
make_display_frame = _MODULE.make_display_frame
ShutdownController = _MODULE.ShutdownController
sleep_and_disable_motors = _MODULE.sleep_and_disable_motors
wake_and_stand = _MODULE.wake_and_stand


def test_wake_and_stand_enables_motors_before_wake_up() -> None:
    calls: list[str] = []

    class FakeReachy:
        def enable_motors(self) -> None:
            calls.append("enable_motors")

        def wake_up(self) -> None:
            calls.append("wake_up")

    asyncio.run(wake_and_stand(FakeReachy(), motor_settle_s=0))

    assert calls == ["enable_motors", "wake_up"]


def test_sleep_and_disable_motors_releases_torque_after_sleep() -> None:
    calls: list[str] = []

    class FakeReachy:
        def goto_sleep(self) -> None:
            calls.append("goto_sleep")

        def disable_motors(self) -> None:
            calls.append("disable_motors")

    asyncio.run(sleep_and_disable_motors(FakeReachy()))

    assert calls == ["goto_sleep", "disable_motors"]


def test_sleep_failure_still_disables_motors() -> None:
    calls: list[str] = []

    class FakeReachy:
        def goto_sleep(self) -> None:
            calls.append("goto_sleep")
            raise RuntimeError("sleep failed")

        def disable_motors(self) -> None:
            calls.append("disable_motors")

    try:
        asyncio.run(sleep_and_disable_motors(FakeReachy()))
    except RuntimeError as exc:
        assert str(exc) == "sleep failed"
    else:
        raise AssertionError("sleep failure should be propagated")

    assert calls == ["goto_sleep", "disable_motors"]


def test_repeated_sigint_does_not_raise_or_cancel_cleanup() -> None:
    shutdown = ShutdownController()

    shutdown.handle_signal(2, None)
    shutdown.cleaning.set()
    shutdown.handle_signal(2, None)

    assert shutdown.requested.is_set()


def test_make_display_frame_copies_readonly_sdk_buffer() -> None:
    sdk_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    sdk_frame.setflags(write=False)

    display_frame = make_display_frame(sdk_frame)

    assert display_frame.flags.writeable
    assert display_frame.flags.c_contiguous
    assert not np.shares_memory(sdk_frame, display_frame)

    cv2.putText(
        display_frame,
        "OK",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    assert display_frame.any()
