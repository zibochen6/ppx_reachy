"""ASR WebSocket client for Bailian ``qwen3-asr-flash-realtime``.

Wire protocol (client → server):
  1. session.update — configure language, VAD, audio format
  2. input_audio_buffer.append — Base64-encoded PCM16 audio chunks
  3. session.finish — end the session

Server events:
  - session.created / session.updated — handshake
  - input_audio_buffer.speech_started / .speech_stopped — VAD events
  - conversation.item.input_audio_transcription.text — interim (stash)
  - conversation.item.input_audio_transcription.completed — **final**
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

import websockets
from websockets.exceptions import ConnectionClosed

from chaihuo_reachy.config import Config

logger = logging.getLogger("chaihuo_reachy.asr")


@dataclass
class ASRResult:
    """A single recognition result from the ASR service."""

    text: str
    is_final: bool = False
    speech_started: bool = False
    speech_stopped: bool = False
    error: str = ""


class BailianASRClient:
    """Async WebSocket client for Bailian realtime ASR.

    Usage::

        async with BailianASRClient(cfg) as asr:
            await asr.configure()
            await asr.send_audio(pcm_bytes)
            async for result in asr.results():
                if result.is_final:
                    print(f"Final: {result.text}")
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._result_queue: asyncio.Queue[ASRResult] = asyncio.Queue(maxsize=64)
        self._recv_task: asyncio.Task | None = None
        self._close_timeout_s = 0.5
        self._connected = False
        self._session_id: str | None = None
        self._speech_start_time: float | None = None  # monotonic timestamp
        self._send_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def _ws_url(self) -> str:
        host = f"{self._cfg.bailian_workspace_id}.{self._cfg.bailian_region}.maas.aliyuncs.com"
        return f"wss://{host}/api-ws/v1/realtime?model={self._cfg.bailian_asr_model}"

    async def __aenter__(self) -> "BailianASRClient":
        self._ws = await websockets.connect(
            self._ws_url,
            additional_headers={
                "Authorization": f"Bearer {self._cfg.bailian_api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
            ping_interval=20,
            ping_timeout=10,
            close_timeout=self._close_timeout_s,
            max_size=2**24,
        )
        self._connected = True
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("🔗 ASR WebSocket connected")
        return self

    async def __aexit__(self, *args) -> None:
        self._connected = False
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, ConnectionClosed):
                pass
        if self._ws is not None:
            try:
                await asyncio.wait_for(
                    self._ws.close(), timeout=self._close_timeout_s + 0.25
                )
            except asyncio.TimeoutError:
                logger.debug("ASR close handshake timed out — aborting")
                transport = getattr(self._ws, "transport", None)
                if transport is not None:
                    transport.abort()
            self._ws = None

    async def configure(self) -> None:
        """Send session.update to configure ASR."""
        event_id = f"evt_{uuid.uuid4().hex[:8]}"
        msg = {
            "event_id": event_id,
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm",
                "sample_rate": self._cfg.audio_sample_rate,
                "input_audio_transcription": {
                    "language": self._cfg.asr_language,
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self._cfg.asr_vad_threshold,
                    "silence_duration_ms": self._cfg.asr_vad_silence_ms,
                },
            },
        }
        await self._send(msg)

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Send a PCM16 mono audio chunk (Base64-encoded internally)."""
        event_id = f"evt_{uuid.uuid4().hex[:8]}"
        msg = {
            "event_id": event_id,
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm_bytes).decode(),
        }
        await self._send(msg)

    async def finish(self) -> None:
        """End the session gracefully."""
        event_id = f"evt_{uuid.uuid4().hex[:8]}"
        await self._send({"event_id": event_id, "type": "session.finish"})

    async def results(self) -> AsyncIterator[ASRResult]:
        """Yield ASR results as they arrive."""
        while True:
            result = await self._result_queue.get()
            yield result
            if result.is_final or result.error:
                return

    async def _send(self, msg: dict) -> None:
        if self._ws is None:
            raise RuntimeError("ASR client not connected")
        async with self._send_lock:
            await self._ws.send(json.dumps(msg, ensure_ascii=False))

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                await self._dispatch(msg)
        except ConnectionClosed as e:
            logger.debug("ASR WebSocket closed: %s", e)
            if self._connected:
                await self._result_queue.put(ASRResult(text="", error=f"ASR connection closed: {e}"))
        except Exception:
            logger.exception("ASR recv loop error")

    async def _dispatch(self, msg: dict) -> None:
        import time as _time
        msg_type = msg.get("type", "")

        if msg_type == "input_audio_buffer.speech_started":
            self._speech_start_time = _time.monotonic()
            logger.info("🟢 [VAD] speech_started — 检测到语音")
            await self._result_queue.put(
                ASRResult(text="", speech_started=True)
            )
        elif msg_type == "input_audio_buffer.speech_stopped":
            duration_s = ""
            if self._speech_start_time is not None:
                d = _time.monotonic() - self._speech_start_time
                duration_s = f" (持续 {d:.1f}s)"
                self._speech_start_time = None
            logger.info("🛑 [VAD] speech_stopped — 语音结束%s", duration_s)
            await self._result_queue.put(
                ASRResult(text="", speech_stopped=True)
            )
        elif msg_type == "conversation.item.input_audio_transcription.text":
            stash = msg.get("stash", "")
            if stash:
                logger.debug("📝 [ASR] partial: %r", stash)
                await self._result_queue.put(ASRResult(text=stash))
        elif msg_type == "conversation.item.input_audio_transcription.completed":
            transcript = msg.get("transcript", "")
            logger.info("✅ [ASR] FINAL: %r", transcript)
            await self._result_queue.put(
                ASRResult(
                    text=transcript,
                    is_final=True,
                )
            )
        elif msg_type == "session.created":
            self._session_id = msg.get("session", {}).get("id", "?")
            logger.info("🔗 ASR session created: %s", self._session_id)
        elif msg_type == "session.updated":
            logger.info(
                "⚙️  [ASR] configured (VAD=%.1f threshold, %dms silence, lang=%s)",
                self._cfg.asr_vad_threshold,
                self._cfg.asr_vad_silence_ms,
                self._cfg.asr_language,
            )
        elif msg_type == "session.finished":
            logger.info("🔌 ASR session finished")
        elif msg_type == "error":
            error = str(msg.get("error", msg))
            logger.error("❌ ASR error: %s", error)
            await self._result_queue.put(ASRResult(text="", error=error))
        else:
            logger.debug("ASR unhandled: %s", msg_type)
