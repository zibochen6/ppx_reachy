from __future__ import annotations

import pytest

from chaihuo_reachy.backends.sdk_audio import SdkAudioIO
from chaihuo_reachy.config import Config


class _MediaManager19:
    """SDK 1.9-shaped fake: it deliberately has no clear_player()."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.pushed = []

    def start_recording(self) -> None:
        self.calls.append("start_recording")

    def stop_recording(self) -> None:
        self.calls.append("stop_recording")

    def start_playing(self) -> None:
        self.calls.append("start_playing")

    def stop_playing(self) -> None:
        self.calls.append("stop_playing")

    def push_audio_sample(self, sample) -> None:
        self.pushed.append(sample)


@pytest.mark.asyncio
async def test_sdk_19_playback_and_stop_do_not_require_clear_player() -> None:
    media = _MediaManager19()
    audio = SdkAudioIO(media)  # type: ignore[arg-type]

    await audio.open()
    await audio.play(b"\0\0" * 240)
    audio.stop_playback()

    assert len(media.pushed) == 1
    assert media.calls == [
        "start_recording",
        "start_playing",
        "stop_playing",
        "start_playing",
    ]
    assert not audio.is_playing


def test_barge_in_is_disabled_by_default() -> None:
    assert Config().barge_in_enabled is False
