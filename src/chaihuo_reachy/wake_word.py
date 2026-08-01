"""Wake word detection — server-side via cloud ASR.

As of 2026-07-30, wake word detection is performed server-side by the
Bailian ASR service.  The microphone is streamed directly to the cloud,
and ``_accept_transcript()`` in ``engine.py`` matches the wake word in
the recognised text.  No local wake-word model (Porcupine, energy-based,
etc.) is used.

This module is kept as a placeholder for future local wake-word engines
(e.g. Sherpa-Onnx) that may be added later.
"""
