"""Local wake word detection via sherpa-onnx KeywordSpotter.

Detects the configured keyword (default "皮皮虾") on-device before the
cloud ASR session is opened, cutting wake latency from 3-8 s (cloud ASR
transcript matching) to ~0.5 s and avoiding false triggers from ambient
noise that the RMS gate alone cannot filter.

Model files live in ``cfg.kws_model_dir`` and are produced by
``scripts/download_kws_model.py`` (encoder/decoder/joiner .onnx,
tokens.txt, keywords.txt, plus a self-check wav).  If the model is
missing or inference is broken (e.g. a broken platform wheel), the
detector raises :class:`WakeWordUnavailableError` so the engine can fall
back to the cloud transcript-matching path instead of hanging forever.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from chaihuo_reachy.config import Config

logger = logging.getLogger("chaihuo_reachy.wake_word")

_SAMPLE_RATE = 16000
_CHUNK_SECONDS = 0.1
_SELFCHECK_DIR = "test_wavs"                # shipped with the KWS model pack
_SELFCHECK_KEYWORDS = "test_wavs/test_keywords.txt"  # built-in keywords for the wavs


class WakeWordUnavailableError(RuntimeError):
    """Raised when local KWS cannot be initialised (missing model, broken wheel)."""


class WakeWordDetector:
    """Local keyword spotting on 16 kHz mono int16 PCM chunks.

    Usage::

        detector = WakeWordDetector(cfg)   # may raise WakeWordUnavailableError
        detector.self_check()              # verify inference works (called by engine)
        for chunk in audio_chunks:         # 16 kHz int16 bytes
            hit = detector.hit(chunk)      # keyword str, or None
    """

    def __init__(self, cfg: Config, *, spotter: object | None = None) -> None:
        self._cfg = cfg
        self._spotter = spotter
        self._stream = None
        self._cooldown_until = 0.0
        self._cooldown_s = 1.0  # suppress repeat triggers right after a hit
        if self._spotter is None:
            self._spotter = self._build_spotter(cfg)

    # ── Construction ───────────────────────────────────────────────────
    @staticmethod
    def _build_spotter(cfg: Config) -> object:
        try:
            import sherpa_onnx  # noqa: F401 — broken wheels fail here loudly
        except Exception as exc:
            raise WakeWordUnavailableError(
                "sherpa-onnx 不可用(import 失败),本地唤醒不可用,"
                f"将回退云端唤醒。原因: {exc}"
            ) from exc

        model_dir = Path(cfg.kws_model_dir)
        required = {
            "tokens.txt": model_dir / "tokens.txt",
            "encoder.onnx": model_dir / "encoder.onnx",
            "decoder.onnx": model_dir / "decoder.onnx",
            "joiner.onnx": model_dir / "joiner.onnx",
            "keywords.txt": model_dir / "keywords.txt",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise WakeWordUnavailableError(
                "本地唤醒模型缺失: models/kws/ 下缺少 "
                + ", ".join(missing)
                + "。请运行 `uv run python scripts/download_kws_model.py`。"
            )

        spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(required["tokens.txt"]),
            encoder=str(required["encoder.onnx"]),
            decoder=str(required["decoder.onnx"]),
            joiner=str(required["joiner.onnx"]),
            keywords_file=str(required["keywords.txt"]),
            num_threads=2,
            provider="cpu",
            keywords_score=cfg.kws_score,
            keywords_threshold=cfg.kws_threshold,
            num_trailing_blanks=2,
        )
        logger.info(
            "KWS 就绪: %s threshold=%.2f score=%.2f",
            model_dir, cfg.kws_threshold, cfg.kws_score,
        )
        return spotter

    # ── Public API ─────────────────────────────────────────────────────
    def hit(self, pcm: bytes) -> str | None:
        """Feed one 16 kHz int16 PCM chunk; return the detected keyword or None."""
        now = time.monotonic()
        if now < self._cooldown_until:
            return None
        if self._stream is None:
            self._stream = self._spotter.create_stream()

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._stream.accept_waveform(_SAMPLE_RATE, samples)
        detected: str | None = None
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            r = self._spotter.get_result(self._stream)
            if r:
                detected = r
                self._spotter.reset_stream(self._stream)
        if detected:
            self._cooldown_until = now + self._cooldown_s
            self._stream = None  # fresh stream for the next utterance
            logger.info("🗣 [KWS] 检测到唤醒词: %s", detected)
        return detected

    def reset(self) -> None:
        """Drop the current stream (e.g. after a conversation turn)."""
        self._stream = None

    def self_check(self) -> None:
        """Verify inference actually produces output.

        Runs the model's own self-test wavs (each contains one of the
        model's built-in keywords) and raises :class:`WakeWordUnavailableError`
        if nothing is detected.  This catches broken platform wheels that
        construct fine but silently produce no output.
        """
        model_dir = Path(self._cfg.kws_model_dir)
        wav_dir = model_dir / _SELFCHECK_DIR
        kw_file = model_dir / _SELFCHECK_KEYWORDS
        wavs = sorted(wav_dir.glob("*.wav")) if wav_dir.is_dir() else []
        if not wavs or not kw_file.is_file():
            logger.warning("KWS 自检音频缺失,跳过自检: %s", model_dir)
            return
        try:
            extra_keywords = kw_file.read_text(encoding="utf-8").strip()
            hits: list[str] = []
            for wav in wavs:
                audio = self._read_wav(wav)
                stream = self._spotter.create_stream(extra_keywords)
                stream.accept_waveform(_SAMPLE_RATE, audio)
                tail = np.zeros(int(0.66 * _SAMPLE_RATE), dtype=np.float32)
                stream.accept_waveform(_SAMPLE_RATE, tail)
                stream.input_finished()
                while self._spotter.is_ready(stream):
                    self._spotter.decode_stream(stream)
                    r = self._spotter.get_result(stream)
                    if r:
                        hits.append(r)
                        self._spotter.reset_stream(stream)
                if hits:
                    break
        except Exception as exc:
            raise WakeWordUnavailableError(f"KWS 自检异常: {exc}") from exc
        if not hits:
            raise WakeWordUnavailableError(
                "KWS 自检失败: 官方自测音频未检测到任何关键词。"
                "sherpa-onnx 推理可能不可用(如损坏的平台 wheel),"
                "将回退云端唤醒。"
            )
        logger.info("✅ KWS 自检通过: %s", hits)

    @staticmethod
    def _read_wav(path: Path) -> np.ndarray:
        import wave

        with wave.open(str(path), "rb") as wf:
            if wf.getframerate() != _SAMPLE_RATE or wf.getnchannels() != 1:
                raise ValueError(f"自检音频必须是 16kHz 单声道: {path}")
            frames = wf.readframes(wf.getnframes())
            return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
