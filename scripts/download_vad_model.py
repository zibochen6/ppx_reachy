"""Install the pinned Silero VAD v6 ONNX model with checksum validation."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.0/src/silero_vad/data/silero_vad.onnx"
SHA256 = "597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004"
TARGET = Path(__file__).parents[1] / "models" / "vad" / "silero_vad.onnx"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".silero-vad-", dir=TARGET.parent)
    try:
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(
            URL, timeout=30
        ) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        digest = hashlib.sha256(Path(temporary).read_bytes()).hexdigest()
        if digest != SHA256:
            raise RuntimeError(f"Silero VAD checksum mismatch: {digest}")
        os.replace(temporary, TARGET)
        print(f"Silero VAD installed: {TARGET}")
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
