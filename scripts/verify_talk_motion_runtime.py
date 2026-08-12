#!/usr/bin/env python3
"""Verify that physical talk motion spans the real speaker playback window."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import urllib.request
from typing import Any


def _json_request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def _axis_range(samples: list[dict[str, Any]], axis: str) -> float:
    values = [float(sample[axis]) for sample in samples]
    return max(values) - min(values)


def _largest_pose_jump(samples: list[dict[str, Any]]) -> float:
    return max(
        (
            math.sqrt(
                sum(
                    (float(current[axis]) - float(previous[axis])) ** 2
                    for axis in ("yaw", "pitch", "roll")
                )
            )
            * 0.05
            / max(0.001, float(current["t"]) - float(previous["t"]))
            for previous, current in zip(samples, samples[1:])
        ),
        default=0.0,
    )


def _longest_still_window(samples: list[dict[str, Any]]) -> float:
    """Longest audible interval with under 0.25 degree of angular change."""
    longest = 0.0
    began: float | None = None
    for previous, current in zip(samples, samples[1:]):
        delta = math.sqrt(
            sum(
                (float(current[axis]) - float(previous[axis])) ** 2
                for axis in ("yaw", "pitch", "roll")
            )
        )
        elapsed = max(0.001, float(current["t"]) - float(previous["t"]))
        if delta < 5.0 * elapsed:  # equivalent to <0.25° per 50 ms
            began = float(previous["t"]) if began is None else began
            longest = max(longest, float(current["t"]) - began)
        else:
            began = None
    return longest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", default="http://127.0.0.1:8640")
    parser.add_argument("--daemon", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=150.0)
    args = parser.parse_args()

    initial = _json_request(f"{args.dashboard}/status")
    daemon_before = _json_request(f"{args.daemon}/api/daemon/status")
    before_errors = int(
        daemon_before["backend_status"]["control_loop_stats"]["nb_error"]
    )
    if not initial.get("robot_ready"):
        raise RuntimeError(f"robot is not ready: {initial}")

    reply: dict[str, Any] = {}
    request_error: list[str] = []

    def send_chat() -> None:
        nonlocal reply
        try:
            reply = _json_request(
                f"{args.dashboard}/debug/chat",
                {
                    "text": (
                        "不要调用工具。请一字不改地朗读下面这段话，回答只能包含这段话："
                        "今天我真的很开心。清晨醒来时，阳光刚好落在窗边，空气清清爽爽。"
                        "出门以后，我遇见了几位热情的朋友，大家一路聊天，也分享了有趣的故事。"
                        "现在能站在这里和你说话，我觉得很幸福。"
                    )
                },
            )
        except Exception as exc:  # reported in the final structured result
            request_error.append(repr(exc))

    request_thread = threading.Thread(target=send_chat, daemon=True)
    status_stop = threading.Event()
    latest_status = initial

    def poll_status() -> None:
        nonlocal latest_status
        while not status_stop.is_set():
            try:
                latest_status = _json_request(f"{args.dashboard}/status")
            except Exception:
                pass
            status_stop.wait(0.03)

    status_thread = threading.Thread(target=poll_status, daemon=True)
    status_thread.start()
    request_thread.start()
    deadline = time.monotonic() + args.timeout
    speaker_samples: list[dict[str, Any]] = []
    saw_speaker = False
    last_speaker_at = 0.0
    final_status = initial

    while time.monotonic() < deadline:
        final_status = latest_status
        now = time.monotonic()
        if final_status.get("speech_audio_playing"):
            state = _json_request(f"{args.daemon}/api/state/full")
            pose = state["head_pose"]
            speaker_samples.append(
                {
                    "t": now,
                    # Daemon XYZRPY state endpoints report radians.
                    "yaw": math.degrees(float(pose["yaw"])),
                    "pitch": math.degrees(float(pose["pitch"])),
                    "roll": math.degrees(float(pose["roll"])),
                    "body_yaw": math.degrees(float(state["body_yaw"])),
                    "motion": bool(final_status.get("talk_motion_active")),
                    "backend": str(final_status.get("talk_motion_backend", "")),
                    "motion_error": str(final_status.get("talk_motion_error", "")),
                }
            )
            saw_speaker = True
            last_speaker_at = now
        if (
            saw_speaker
            and not request_thread.is_alive()
            and not final_status.get("speech_audio_playing")
            and now - last_speaker_at >= 0.5
        ):
            break
        time.sleep(0.04)

    request_thread.join(timeout=1)
    status_stop.set()
    status_thread.join(timeout=1)
    daemon_after = _json_request(f"{args.daemon}/api/daemon/status")
    after_errors = int(
        daemon_after["backend_status"]["control_loop_stats"]["nb_error"]
    )

    if len(speaker_samples) >= 9:
        third = len(speaker_samples) // 3
        sections = [
            speaker_samples[:third],
            speaker_samples[third : 2 * third],
            speaker_samples[2 * third :],
        ]
    else:
        sections = [[], [], []]
    section_metrics = [
        {
            "yaw_range_deg": round(_axis_range(section, "yaw"), 3),
            "pitch_range_deg": round(_axis_range(section, "pitch"), 3),
            "roll_range_deg": round(_axis_range(section, "roll"), 3),
        }
        if section
        else {"yaw_range_deg": 0.0, "pitch_range_deg": 0.0, "roll_range_deg": 0.0}
        for section in sections
    ]
    duration = (
        speaker_samples[-1]["t"] - speaker_samples[0]["t"]
        if len(speaker_samples) > 1
        else 0.0
    )
    all_motion_active = bool(speaker_samples) and all(
        sample["motion"] for sample in speaker_samples
    )
    yaw_each_third = all(metric["yaw_range_deg"] >= 28.0 for metric in section_metrics)
    pitch_natural = all(3.0 <= metric["pitch_range_deg"] <= 8.0 for metric in section_metrics)
    roll_natural = all(2.0 <= metric["roll_range_deg"] <= 6.0 for metric in section_metrics)
    body_range = _axis_range(speaker_samples, "body_yaw") if speaker_samples else 0.0
    max_jump = _largest_pose_jump(speaker_samples)
    longest_still = _longest_still_window(speaker_samples)
    backend_ok = bool(speaker_samples) and all(
        sample["backend"] == "speech_offsets" for sample in speaker_samples
    )
    no_motion_error = bool(speaker_samples) and all(
        not sample["motion_error"] for sample in speaker_samples
    )
    mean_hz = float(
        daemon_after["backend_status"]["control_loop_stats"][
            "mean_control_loop_frequency"
        ]
    )
    passed = all(
        (
            not request_error,
            not request_thread.is_alive(),
            15.0 <= duration <= 25.0,
            len(speaker_samples) >= 180,
            all_motion_active,
            backend_ok,
            no_motion_error,
            yaw_each_third,
            pitch_natural,
            roll_natural,
            body_range <= 2.0,
            longest_still <= 0.20,
            max_jump <= 4.0,
            not final_status.get("barge_in_occurred"),
            45.0 <= mean_hz <= 55.0,
            before_errors == after_errors,
        )
    )
    result = {
        "passed": passed,
        "speaker_duration_s": round(duration, 2),
        "speaker_samples": len(speaker_samples),
        "all_speaker_samples_had_talk_motion": all_motion_active,
        "all_samples_used_speech_offsets": backend_ok,
        "talk_motion_error_free": no_motion_error,
        "movement_by_audible_third": section_metrics,
        "body_yaw_range_deg": round(body_range, 3),
        "longest_audible_still_window_s": round(longest_still, 3),
        "max_angular_jump_deg_per_sample": round(max_jump, 3),
        "barge_in_occurred": bool(final_status.get("barge_in_occurred")),
        "motor_control_hz": round(mean_hz, 3),
        "motor_errors_before": before_errors,
        "motor_errors_after": after_errors,
        "request_error": request_error,
        "reply_chars": len(str(reply.get("reply", ""))),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
