"""TTS client for Bailian — supports Qwen Realtime, streaming, and HTTP models.

Three transports, one public interface (``open/feed/flush/close``):

* **Qwen Realtime (WebSocket)** — ``qwen3-tts-flash-realtime`` via dashscope SDK's
  ``QwenTtsRealtime``. Text appended as LLM produces it; Base64 PCM events arrive.
* **Streaming (WebSocket)** — ``qwen-audio-3.0-tts-plus``, ``cosyvoice-v2`` via
  ``SpeechSynthesizer``. Sentence-level PCM chunks through ``ResultCallback``.
* **HTTP** — ``qwen3-tts-flash`` via ``MultiModalConversation.call()``.
  Each call is one HTTP round-trip returning a WAV URL.

Threading: SDK callbacks fire on internal threads — ``on_audio`` is called from
those threads. Callers bridge to asyncio with ``run_coroutine_threadsafe``.

All SDK calls run via ``loop.run_in_executor`` so they never block the event loop.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
import wave
from typing import Callable

import httpx

from chaihuo_reachy.config import Config

logger = logging.getLogger("chaihuo_reachy.tts")

# Audio callback: (pcm_bytes, sample_rate)
AudioCB = Callable[[bytes, int], None] | None
SentenceCB = Callable[[str, str], None] | None
DoneCB = Callable[[], None] | None

_TTS_PCM_SAMPLE_RATE = 22050  # for streaming models
_QWEN_REALTIME_PCM_SAMPLE_RATE = 24000

_STREAMING_MODELS = {
    "qwen-audio-3.0-tts-plus",
    "qwen-audio-3.0-tts",
    "cosyvoice-v2",
    "cosyvoice-v1",
}


def _is_qwen_realtime_model(model: str) -> bool:
    return model.startswith("qwen3-tts-") and "realtime" in model


def _is_streaming_model(model: str) -> bool:
    return _is_qwen_realtime_model(model) or model in _STREAMING_MODELS or "cosyvoice" in model.lower()


# ── SDK callback classes (lazily built to avoid import errors) ────────────

def _make_streaming_callback(on_audio: AudioCB, on_sentence: SentenceCB, on_done: DoneCB):
    """Build a ResultCallback subclass bound to our callbacks."""
    from dashscope.audio.tts_v2 import ResultCallback

    class _CB(ResultCallback):
        def on_open(self) -> None:
            logger.debug("TTS streaming connected")

        def on_data(self, data: bytes) -> None:
            if on_audio:
                try:
                    on_audio(data, _TTS_PCM_SAMPLE_RATE)
                except Exception:
                    logger.debug("TTS on_audio error", exc_info=True)

        def on_event(self, message: str) -> None:
            if on_sentence is None:
                return
            try:
                msg = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return
            output = msg.get("payload", {}).get("output", {})
            event_type = output.get("type", "")
            text = output.get("original_text", "")
            if event_type:
                try:
                    on_sentence(event_type, text)
                except Exception:
                    logger.debug("TTS on_sentence error", exc_info=True)

        def on_complete(self) -> None:
            logger.debug("TTS synthesis complete")
            if on_done:
                try:
                    on_done()
                except Exception:
                    logger.debug("TTS on_done error", exc_info=True)

        def on_error(self, message: str) -> None:
            logger.error("TTS error: %s", message)

        def on_close(self) -> None:
            logger.debug("TTS streaming closed")

    return _CB


def _make_qwen_realtime_callback(on_audio: AudioCB, on_sentence: SentenceCB, on_done: DoneCB, complete_event: threading.Event):
    """Build a QwenTtsRealtimeCallback subclass."""
    from dashscope.audio.qwen_tts_realtime import QwenTtsRealtimeCallback

    class _CB(QwenTtsRealtimeCallback):
        def on_open(self) -> None:
            logger.debug("Qwen Realtime TTS connected")

        def on_event(self, response: dict) -> None:
            event_type = response.get("type", "")
            if event_type == "response.audio.delta":
                delta = response.get("delta", "")
                if on_audio and delta:
                    try:
                        import base64
                        on_audio(base64.b64decode(delta), _QWEN_REALTIME_PCM_SAMPLE_RATE)
                    except Exception:
                        logger.debug("Qwen TTS audio error", exc_info=True)
            elif event_type == "response.done":
                if on_done:
                    try:
                        on_done()
                    except Exception:
                        logger.debug("Qwen TTS done error", exc_info=True)
                complete_event.set()
            elif event_type == "error":
                logger.error("Qwen Realtime TTS error: %s", response)
                complete_event.set()
            elif on_sentence and event_type:
                try:
                    on_sentence(event_type, "")
                except Exception:
                    logger.debug("Qwen TTS sentence error", exc_info=True)

        def on_close(self, close_status_code, close_msg) -> None:
            logger.debug("Qwen Realtime TTS closed: %s %s", close_status_code, close_msg)
            complete_event.set()

    return _CB


class BailianTTSClient:
    """Bailian TTS client — auto-selects transport by model name.

    Usage::

        tts = BailianTTSClient(cfg, on_audio=play_cb)
        await tts.open()
        await tts.feed("你好，")
        await tts.feed("欢迎来到基地车！")
        await tts.flush()
        await tts.close()

        # One-shot:
        await tts.open()
        await tts.synthesize("你好！")
        await tts.close()
    """

    def __init__(
        self,
        cfg: Config,
        on_audio: AudioCB = None,
        on_sentence: SentenceCB = None,
        on_done: DoneCB = None,
    ) -> None:
        self._cfg = cfg
        self._on_audio = on_audio
        self._on_sentence = on_sentence
        self._on_done = on_done
        self._loop: asyncio.AbstractEventLoop | None = None
        self._streaming = _is_streaming_model(cfg.bailian_tts_model)
        self._qwen_realtime = _is_qwen_realtime_model(cfg.bailian_tts_model)
        self._synthesizer: object | None = None
        self._callback: object | None = None
        self._complete_event = threading.Event()

    # ── Public API ─────────────────────────────────────────────────────
    async def open(self) -> None:
        self._loop = asyncio.get_running_loop()
        if self._streaming:
            if self._qwen_realtime:
                await self._run_sdk(self._open_qwen_realtime)
            else:
                await self._run_sdk(self._open_streaming)

    async def feed(self, text: str) -> None:
        if not text.strip():
            return
        if self._streaming:
            if self._qwen_realtime:
                await self._run_sdk(self._qwen_append_text, text)
            else:
                await self._run_sdk(self._feed_streaming, text)
        else:
            await self._run_sdk(self._synthesize_http, text)

    async def flush(self) -> None:
        if not self._streaming:
            return  # HTTP: each feed() completes before returning
        if self._qwen_realtime:
            await self._run_sdk(self._qwen_finish)
        else:
            await self._run_sdk(self._flush_streaming)

    async def close(self) -> None:
        if self._qwen_realtime and self._synthesizer is not None:
            await self._run_sdk(self._qwen_close)
        self._synthesizer = None
        self._callback = None

    async def synthesize(self, text: str) -> None:
        """One-shot synthesis: feed + flush."""
        await self.feed(text)
        await self.flush()

    # ── Qwen Realtime (qwen3-tts-flash-realtime) ───────────────────────
    def _open_qwen_realtime(self) -> None:
        import dashscope
        from dashscope.audio.qwen_tts_realtime import AudioFormat, QwenTtsRealtime

        dashscope.api_key = self._cfg.bailian_api_key
        self._complete_event.clear()
        cb_cls = _make_qwen_realtime_callback(
            self._on_audio, self._on_sentence, self._on_done, self._complete_event,
        )
        self._callback = cb_cls()
        self._synthesizer = QwenTtsRealtime(
            model=self._cfg.bailian_tts_model,
            callback=self._callback,
            url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        )
        self._synthesizer.connect()
        self._synthesizer.update_session(
            voice=self._cfg.bailian_tts_voice,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode="server_commit",
            volume=self._cfg.tts_volume,
            speech_rate=self._cfg.tts_speech_rate,
            pitch_rate=self._cfg.tts_pitch_rate,
        )
        logger.info("TTS Qwen Realtime ready: %s voice=%s", self._cfg.bailian_tts_model, self._cfg.bailian_tts_voice)

    def _qwen_append_text(self, text: str) -> None:
        assert self._synthesizer is not None
        self._synthesizer.append_text(text)

    def _qwen_finish(self) -> None:
        assert self._synthesizer is not None
        self._synthesizer.finish()
        if not self._complete_event.wait(timeout=60):
            raise TimeoutError("Qwen Realtime TTS did not finish within 60s")

    def _qwen_close(self) -> None:
        assert self._synthesizer is not None
        self._synthesizer.close()

    # ── Streaming (SpeechSynthesizer) ──────────────────────────────────
    def _open_streaming(self) -> None:
        import dashscope
        from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

        dashscope.api_key = self._cfg.bailian_api_key
        host = f"{self._cfg.bailian_workspace_id}.{self._cfg.bailian_region}.maas.aliyuncs.com"
        dashscope.base_websocket_api_url = f"wss://{host}/api-ws/v1/inference"

        cb_cls = _make_streaming_callback(self._on_audio, self._on_sentence, self._on_done)
        self._callback = cb_cls()
        self._synthesizer = SpeechSynthesizer(
            model=self._cfg.bailian_tts_model,
            voice=self._cfg.bailian_tts_voice,
            format=AudioFormat.PCM_22050HZ_MONO_16BIT,
            volume=self._cfg.tts_volume,
            speech_rate=self._cfg.tts_speech_rate,
            pitch_rate=self._cfg.tts_pitch_rate,
            callback=self._callback,
        )
        logger.info("TTS streaming ready: %s voice=%s", self._cfg.bailian_tts_model, self._cfg.bailian_tts_voice)

    def _feed_streaming(self, text: str) -> None:
        assert self._synthesizer is not None
        self._synthesizer.streaming_call(text)

    def _flush_streaming(self) -> None:
        assert self._synthesizer is not None
        self._synthesizer.streaming_complete()

    # ── HTTP (qwen3-tts-flash, non-streaming) ──────────────────────────
    def _synthesize_http(self, text: str) -> None:
        import dashscope
        from dashscope import MultiModalConversation

        dashscope.api_key = self._cfg.bailian_api_key
        host = f"{self._cfg.bailian_workspace_id}.{self._cfg.bailian_region}.maas.aliyuncs.com"
        dashscope.base_http_api_url = f"https://{host}/api/v1"

        try:
            resp = MultiModalConversation.call(
                model=self._cfg.bailian_tts_model,
                text=text,
                voice=self._cfg.bailian_tts_voice,
                stream=False,
            )
        except Exception as e:
            logger.exception("TTS HTTP call failed")
            return

        if resp.status_code != 200:
            logger.error("TTS HTTP %d: %s", resp.status_code, str(resp)[:300])
            return

        try:
            data = json.loads(str(resp))
        except Exception:
            logger.error("TTS response not JSON")
            return

        audio_url = data.get("output", {}).get("audio", {}).get("url", "")
        if not audio_url:
            logger.error("TTS response missing audio URL")
            return

        try:
            with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                wav_bytes = client.get(audio_url).content
        except Exception as e:
            logger.exception("TTS audio download failed")
            return

        try:
            with wave.open(io.BytesIO(wav_bytes)) as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    logger.error("TTS WAV not 16-bit mono")
                    return
                sr = wf.getframerate()
                chunks = []
                while True:
                    frame = wf.readframes(4096)
                    if not frame:
                        break
                    chunks.append(frame)
        except wave.Error as e:
            logger.error("TTS WAV decode error: %s", e)
            return

        for chunk in chunks:
            if self._on_audio:
                try:
                    self._on_audio(chunk, sr)
                except Exception:
                    pass

        if self._on_done:
            try:
                self._on_done()
            except Exception:
                pass

    # ── Internals ──────────────────────────────────────────────────────
    async def _run_sdk(self, fn, *args) -> None:
        assert self._loop is not None
        await self._loop.run_in_executor(None, fn, *args)
