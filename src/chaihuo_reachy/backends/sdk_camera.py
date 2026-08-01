"""SDK camera backend — reads BGR frames from the daemon via GStreamer IPC."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from reachy_mini.media.media_manager import MediaManager

logger = logging.getLogger("chaihuo_reachy.backends.sdk_camera")


class SdkCamera:
    """Camera backend using Reachy SDK's MediaManager (GStreamerCamera via IPC).

    The daemon owns the physical USB camera. Frames are delivered as BGR
    numpy arrays through a Unix-domain-socket IPC pipeline. This backend
    simply reads from that pipeline — no thread, no device enumeration.
    """

    backend_name = "sdk_gstreamer"

    def __init__(self, media_manager: MediaManager) -> None:
        self._mm = media_manager
        logger.info(
            "SDK camera: resolution=%s",
            media_manager.camera.resolution if media_manager.camera else "unknown",
        )

    @property
    def is_active(self) -> bool:
        return self._mm.camera is not None

    def read(self) -> np.ndarray | None:
        """Return the latest BGR frame from the daemon IPC pipeline.

        Returns:
            numpy array (H, W, 3) uint8 in BGR order, or None.
        """
        return self._mm.get_frame()

    def capture_jpeg(self, quality: int = 85) -> bytes | None:
        """Capture a frame and encode as JPEG.

        Args:
            quality: JPEG quality 1-100 (default 85).

        Returns:
            JPEG bytes, or None if no frame available.
        """
        frame = self._mm.get_frame()
        if frame is None:
            return None
        ok, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            return None
        return jpeg.tobytes()

    def close(self) -> None:
        """Release camera resources (no-op — SDK manages lifecycle)."""
        pass
