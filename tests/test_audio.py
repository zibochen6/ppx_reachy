from __future__ import annotations

import pytest

from chaihuo_reachy.audio import AudioDeviceResolutionError, resolve_audio_device
from chaihuo_reachy.backends.interfaces import (
    playback_gain_from_percent,
    playback_percent_from_gain,
)


DEVICES = [
    {
        "name": "Reachy Mini Audio",
        "max_input_channels": 2,
        "max_output_channels": 2,
        "default_samplerate": 16000,
    },
    {
        "name": "MacBook Pro Microphone",
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_samplerate": 48000,
    },
    {
        "name": "MacBook Pro Speakers",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000,
    },
    {
        "name": "Reachy Mini Camera: USB Audio",
        "max_input_channels": 2,
        "max_output_channels": 2,
        "default_samplerate": 48000,
    },
]


def test_dashboard_volume_mapping_preserves_default_and_expands_maximum() -> None:
    assert playback_gain_from_percent(0) == 0.0
    assert playback_gain_from_percent(50) == 2.0
    assert playback_gain_from_percent(100) == 8.0
    assert playback_percent_from_gain(2.0) == 50
    assert playback_percent_from_gain(8.0) == 100


def test_auto_selects_unique_reachy_duplex_and_rejects_camera_audio() -> None:
    info = resolve_audio_device("auto", devices=DEVICES)
    assert info.input_index == info.output_index == 0
    assert info.input_name == "Reachy Mini Audio"
    assert info.max_input_channels == info.max_output_channels == 2


def test_omitted_selector_does_not_fall_back_to_mac_defaults() -> None:
    with pytest.raises(AudioDeviceResolutionError, match="REACHY_AUDIO_DEVICE=default"):
        resolve_audio_device(None, devices=DEVICES[1:], default_device=(0, 1))


def test_explicit_default_allows_separate_system_devices() -> None:
    info = resolve_audio_device("default", devices=DEVICES, default_device=(1, 2))
    assert info.input_name == "MacBook Pro Microphone"
    assert info.output_name == "MacBook Pro Speakers"


def test_explicit_index_must_be_full_duplex() -> None:
    with pytest.raises(AudioDeviceResolutionError, match="not full duplex"):
        resolve_audio_device(1, devices=DEVICES)


def test_ambiguous_reachy_candidates_fail_with_indexes() -> None:
    ambiguous = [DEVICES[0], {**DEVICES[0], "name": "Reachy Mini Audio: USB"}]
    # Exact normalized name wins over a substring candidate.
    assert resolve_audio_device("auto", devices=ambiguous).input_index == 0
    truly_ambiguous = [
        {**DEVICES[0], "name": "USB Reachy Mini Audio A"},
        {**DEVICES[0], "name": "USB Reachy Mini Audio B"},
    ]
    with pytest.raises(AudioDeviceResolutionError, match="ambiguous indexes"):
        resolve_audio_device("auto", devices=truly_ambiguous)
