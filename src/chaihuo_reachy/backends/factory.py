"""Backend factory — selects SDK or Direct backends based on runtime context."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reachy_mini.media.media_manager import MediaManager
    from chaihuo_reachy.backends.interfaces import AudioBackend, CameraBackend
    from chaihuo_reachy.config import Config

logger = logging.getLogger("chaihuo_reachy.backends.factory")


def create_camera_backend(
    config: "Config",
    media_manager: "MediaManager | None" = None,
) -> "CameraBackend":
    """Create the appropriate camera backend.

    Returns SdkCamera when media_manager is available and media_backend
    is not explicitly set to "no_media". Otherwise returns the Direct
    camera (OpenCV-based).

    Args:
        config: Application configuration.
        media_manager: SDK MediaManager (available in daemon mode).

    Returns:
        A CameraBackend instance.
    """
    use_sdk = (
        media_manager is not None
        and config.media_backend != "no_media"
    )

    if use_sdk:
        from chaihuo_reachy.backends.sdk_camera import SdkCamera

        logger.info("Camera: using SDK GStreamer backend")
        return SdkCamera(media_manager)
    else:
        from chaihuo_reachy.camera import Camera as DirectCamera

        logger.info("Camera: using direct OpenCV backend")
        camera = DirectCamera(
            device=config.camera_device,
            width=config.camera_width,
            height=config.camera_height,
        )
        camera.open()
        return camera  # type: ignore[return-value]


def create_audio_backend(
    config: "Config",
    media_manager: "MediaManager | None" = None,
) -> "AudioBackend":
    """Create the appropriate audio backend.

    Returns SdkAudioIO when media_manager is available and media_backend
    is not explicitly set to "no_media". Otherwise returns a Direct
    audio backend (sounddevice RawStream duplex).

    Args:
        config: Application configuration.
        media_manager: SDK MediaManager (available in daemon mode).

    Returns:
        An AudioBackend instance.
    """
    use_sdk = (
        media_manager is not None
        and config.media_backend != "no_media"
    )

    if use_sdk:
        from chaihuo_reachy.backends.sdk_audio import SdkAudioIO

        logger.info("Audio: using SDK GStreamer backend")
        return SdkAudioIO(
            media_manager,
            sr=config.audio_sample_rate,
            input_channel=config.audio_input_channel,
        )
    else:
        from chaihuo_reachy.audio import DuplexAudioIO

        logger.info("Audio: using direct sounddevice duplex backend")
        audio = DuplexAudioIO(
            device=config.audio_device,
            sr=config.audio_sample_rate,
            input_channel=config.audio_input_channel,
        )
        audio.mic_gain = config.audio_mic_gain
        logger.info("Audio mic gain: %.1fx", audio.mic_gain)
        return audio  # type: ignore[return-value]
