"""Media backend abstraction layer.

Provides CameraBackend and AudioBackend protocols plus factory functions
that select the right implementation (SDK GStreamer or direct hardware)
based on runtime availability.
"""

from chaihuo_reachy.backends.interfaces import AudioBackend, CameraBackend
from chaihuo_reachy.backends.factory import (
    create_audio_backend,
    create_camera_backend,
)

__all__ = [
    "AudioBackend",
    "CameraBackend",
    "create_audio_backend",
    "create_camera_backend",
]
