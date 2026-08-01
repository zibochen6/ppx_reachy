"""Conversation engine: ASR → Memory → (Vision) → LLM → TTS.

The main loop runs one turn at a time:
  1. Listen: mic → Bailian ASR → text
  2. Context: retrieve relevant memories from ChromaDB
  3. Vision: if user asks about visual scene → capture camera → VLM
  4. Think: LLM streaming inference with context
  5. Speak: TTS synthesis → speaker

Barge-in: while TTS is playing, new speech onset cancels playback.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from chaihuo_reachy.bailian import (
    ASRResult,
    BailianASRClient,
    BailianLLMClient,
    BailianTTSClient,
    BailianVLMClient,
)
from chaihuo_reachy.camera import visual_quality_issue
from chaihuo_reachy.config import Config
from chaihuo_reachy.intent import IntentDecision, TurnIntent, classify_intent
from chaihuo_reachy.location import LocationService, create_location_service
from chaihuo_reachy.memory import JournalFetcher
from chaihuo_reachy.memory.store import MemoryStore

if TYPE_CHECKING:
    from chaihuo_reachy.backends.interfaces import AudioBackend, CameraBackend
    from chaihuo_reachy.motion import MotionController

logger = logging.getLogger("chaihuo_reachy.engine")

_JOURNAL_UNKNOWN = "这件事我在基地车日记里没有找到可靠记录，所以我不知道。"

# Common ASR homophone corrections (Chinese: same sound, wrong character)
_ASR_CORRECTIONS: dict[str, str] = {
    "跳个五": "跳个舞",   # 舞/五 both wǔ
    "跳歌舞": "跳个舞",   # segment error
    "跳个无": "跳个舞",
    "跳个午": "跳个舞",
    "个五": "跳个舞",     # VAD missed first char "跳"
    "个舞": "跳个舞",
    "跳五": "跳舞",
    "调个舞": "跳个舞",
}

# Emotion tag regex: [happy], [curious], etc.
_TAG_RE = re.compile(r"\[([a-zA-Z一-鿿_]+)\]")

# A bare wake word is an admitted turn, but it must never reach the LLM.  The
# main turn coroutine handles this sentinel synchronously so the next ASR
# session cannot overlap the canned acknowledgement.
_WAKE_ONLY = "__wake_only__"
_SHORT_ACKS = {
    "嗯嗯",
    "好的",
    "没错",
    "是的",
    "可以",
    "行吧",
}

# Single-character utterances and common ASR noise that should be
# silently ignored (no LLM, no TTS, no canned response).
_NOISE_WORDS = {
    "嗯", "哦", "好", "对", "行", "嘘", "嘿", "啊", "呀",
    "呵", "哎", "喂", "呃", "哈", "嘻", "哟", "噢", "啧",
    "嘿咻", "好嘞",
}

# Regex: pure-noise transcript (single char or noise word, optional punct)
_NOISE_RE = re.compile(
    r"^[" + re.escape("，。！？!?、～~…") + r"]*$"
)
_UNGROUNDED_VISUAL_CLAIM_RE = re.compile(
    r"(?:刚(?:刚)?拍(?:到|了)|镜头(?:里|中)|画面(?:里|中|显示)|"
    r"照片(?:里|中)|摄像头(?:里|中|拍到|显示)|"
    r"(?:我)?(?:正|正在)?(?:用|通过|透过)摄像头(?:在)?(?:看|观察)|"
    r"我(?:现在|目前|这会儿|正|正在)?(?:能|可以)?(?:清楚地)?(?:看到|看见)|"
    r"(?:我)?眼前(?:有|是|出现))"
)
_VISUAL_PREFIX_RE = re.compile(
    r"^\s*(?:车后方\s*/\s*外面画面描述|车后方画面描述|"
    r"外面画面描述|摄像头画面描述|画面描述)\s*[：:]\s*"
)

# Language-lock instructions
_LANGUAGE_LOCK = {
    "zh": (
        "Reply ONLY in Chinese (简体中文). Never switch languages, even if "
        "the visitor speaks another language.\n"
        'Example spoken reply: "欢迎，很高兴见到你。"'
    ),
    "en": (
        "Reply ONLY in English. Never switch languages, even if the visitor "
        "speaks another language.\n"
        'Example spoken reply: "Welcome! Glad you stopped by."'
    ),
}

def _build_system_prompt(
    cfg: Config,
    journal_context: str = "",
    *,
    inject_location: str = "",
) -> str:
    """Build the full prompt from verified program-selected context."""
    from datetime import date, timedelta

    prompt = cfg.system_prompt()
    today = date.today()
    yesterday = today - timedelta(days=1)
    prompt += (
        f"\n\n【当前真实日期】\n"
        f"今天是 {today.isoformat()}（{today.strftime('%Y年%m月%d日')}），"
        f"昨天是 {yesterday.isoformat()}。\n"
        f"用户提到'昨天''今天'等时间词时请以以上日期为准。"
    )

    if inject_location:
        prompt += f"\n\n【当前位置】\n{inject_location}"

    if journal_context:
        prompt += f"\n\n【已验证日记证据】\n{journal_context}"

    # Anti-hallucination rules — always appended, both modes
    prompt += (
        "\n\n⚠️ 诚实规则（最高优先级）：\n"
        "- 如果工具调用返回「无法获取」「暂不可用」「失败」「定位失败」，"
        "你必须直接告诉用户目前获取不到该信息，绝对不要猜测、编造或从对话历史推断！\n"
        "- 用户宁愿听到「抱歉，目前GPS信号不好获取不到位置」"
        "也不愿听到一个随口编造的地点。\n"
        "- 系统提供的【当前位置】如果显示「无法获取」或坐标为空，说明GPS无信号，"
        "直接告诉用户「目前获取不到GPS定位」。\n"
        "- 对于任何你不确定的事实（位置、日期、天气、新闻等），如果工具没有返回结果，"
        "就说「这个我不太确定」，不要瞎编。\n"
        "- 凡是基地车、队员、路线、站点、旅途事件，只能使用本轮【已验证日记证据】；"
        "历史对话和常识都不是证据。\n"
        f"- 日记证据为空或不能支持问题时，只能回答：{_JOURNAL_UNKNOWN}\n"
        "- 当前画面只能依据本轮刚拍摄且质量合格的照片；不要推断人物身份、地点或不可见信息。\n"
        "- 本轮没有提供【本轮实时画面】时，严禁声称「刚拍到」「镜头里」「画面中」有什么。"
    )

    prompt += f"\n\n{_LANGUAGE_LOCK.get(cfg.language, _LANGUAGE_LOCK['zh'])}"
    return prompt


class ConversationEngine:
    """Full voice conversation pipeline: ASR → LLM → TTS.

    Usage::

        engine = ConversationEngine(cfg)
        await engine.start()   # blocks, runs the conversation loop
        # ... Ctrl+C to stop
        await engine.stop()
    """

    def __init__(
        self,
        cfg: Config,
        *,
        audio_backend: AudioBackend | None = None,
        camera_backend: CameraBackend | None = None,
        motion: MotionController | None = None,
    ) -> None:
        self.config = cfg
        self._audio: AudioBackend | None = audio_backend
        self._camera: CameraBackend | None = camera_backend
        self._motion = motion
        self._memory: MemoryStore | None = None
        self._location: LocationService | None = None
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Conversation state
        self._state = "idle"  # idle | listening | thinking | speaking
        self._speaking_lock = asyncio.Lock()  # prevents overlapping TTS
        self._language = cfg.language
        self._conversation_history: list[dict[str, str]] = []
        self._last_activity = 0.0
        self._turn_lock = asyncio.Lock()
        self._current_turn_id = ""
        self._current_sources: list[dict[str, Any]] = []
        self._wake_word_active_until = 0.0
        self._barge_in = True
        self._tts_playing = False
        self._barge_in_occurred = False
        self._barge_in_requested = False  # set by concurrent watcher
        self._listen_not_before = 0.0
        self._pending_playbacks: set[concurrent.futures.Future[None]] = set()

        # Callbacks
        self._on_state_change: Callable[[str], None] | None = None
        self._on_transcript: Callable[[str, bool], None] | None = None
        self._on_llm_token: Callable[[str], None] | None = None
        self._on_emotion: Callable[[str], None] | None = None
        self._on_snapshot: Callable[[bytes, str], None] | None = None  # (jpeg_bytes, label)
        self._on_turn_event: Callable[[dict[str, Any]], None] | None = None
        self._camera_snapshot_provider: (
            Callable[[], bytes | None | Awaitable[bytes | None]] | None
        ) = None
        self._journal_fetcher = JournalFetcher(
            listing_url=cfg.journal_url,
            cache_dir=cfg.journal_cache_dir,
        )

    def on_state_change(self, cb: Callable[[str], None]) -> None:
        self._on_state_change = cb

    def on_transcript(self, cb: Callable[[str, bool], None]) -> None:
        """Callback: (text, is_final) for ASR transcripts."""
        self._on_transcript = cb

    def on_llm_token(self, cb: Callable[[str], None]) -> None:
        """Callback: streaming LLM tokens."""
        self._on_llm_token = cb

    def on_emotion(self, cb: Callable[[str], None]) -> None:
        """Callback: emotion detected from LLM response ([happy], etc.)."""
        self._on_emotion = cb

    def on_snapshot(self, cb: Callable[[bytes, str], None]) -> None:
        """Callback: (jpeg_bytes, label) when a camera snapshot is taken.
        Label is 'rear' for rear-view or 'front' for Reachy camera."""
        self._on_snapshot = cb

    def on_turn_event(self, cb: Callable[[dict[str, Any]], None]) -> None:
        """Receive normalized begin/status/delta/final events for all inputs."""
        self._on_turn_event = cb

    def set_camera_snapshot_provider(
        self,
        provider: Callable[[], bytes | None | Awaitable[bytes | None]] | None,
    ) -> None:
        """Use the camera service's latest fresh frame instead of reopening it."""
        self._camera_snapshot_provider = provider

    @property
    def current_turn_id(self) -> str:
        return self._current_turn_id

    def clear_conversation(self) -> None:
        self._conversation_history.clear()
        self._last_activity = time.monotonic()

    def _emit_turn_event(self, event_type: str, **payload: Any) -> None:
        if self._on_turn_event is None:
            return
        try:
            self._on_turn_event(
                {
                    "type": event_type,
                    "turn_id": self._current_turn_id,
                    **payload,
                }
            )
        except Exception:
            logger.debug("Turn event callback failed", exc_info=True)

    def _begin_turn(
        self, text: str, *, source: str, client_message_id: str = ""
    ) -> str:
        now = time.monotonic()
        if (
            self._conversation_history
            and self._last_activity
            and now - self._last_activity >= self.config.session_reset_idle_s
        ):
            logger.info("Conversation idle for %.1fs; resetting context", now - self._last_activity)
            self._conversation_history.clear()
        self._current_turn_id = uuid.uuid4().hex
        self._current_sources = []
        self._emit_turn_event(
            "turn_begin",
            text=text,
            source=source,
            client_message_id=client_message_id,
        )
        self._emit_turn_event("turn_status", status="thinking")
        return self._current_turn_id

    def set_wake_word_enabled(self, enabled: bool) -> None:
        """Enable or disable wake word at runtime."""
        self.config.enable_wake_word = enabled
        logger.info("Wake word %s", "enabled" if enabled else "disabled")

    # ── Lifecycle ──────────────────────────────────────────────────────
    async def start(self) -> None:
        """Start the engine: open audio, camera, memory, begin conversation."""
        if not self.config.bailian_api_key:
            raise RuntimeError("BAILIAN_API_KEY is required")
        if not self.config.bailian_workspace_id:
            raise RuntimeError("BAILIAN_WORKSPACE_ID is required")

        self._loop = asyncio.get_running_loop()

        # Audio: use injected backend or create from factory
        if self._audio is None:
            from chaihuo_reachy.backends.factory import create_audio_backend
            self._audio = create_audio_backend(self.config)
            logger.info("Audio: auto-created %s", self._audio.backend_name)

        await self._audio.open()

        # Camera: use injected backend or create from factory
        if self._camera is None:
            from chaihuo_reachy.backends.factory import create_camera_backend
            self._camera = create_camera_backend(self.config)
            logger.info("Camera: auto-created %s", self._camera.backend_name)

        # Memory (ChromaDB for journal retrieval)
        try:
            self._memory = MemoryStore(
                persist_dir=self.config.chroma_persist_dir,
                journal_dir=self.config.journal_cache_dir,
            )
            logger.info("Memory store ready: %d documents", self._memory.count())
        except Exception:
            logger.warning("Memory store unavailable — journal recall disabled", exc_info=True)
            self._memory = None

        # Location (GPS / IP geolocation)
        self._location = create_location_service(
            gpsd_enabled=self.config.location_gpsd_enabled,
            gpsd_host=self.config.location_gpsd_host,
            gpsd_port=self.config.location_gpsd_port,
            poll_interval_s=self.config.location_poll_interval_s,
        )
        await self._location.start()

        self._last_activity = time.monotonic()
        self._task = asyncio.create_task(self._conversation_loop())
        logger.info(
            "引擎启动: ASR=%s LLM=%s TTS=%s",
            self.config.bailian_asr_model,
            self.config.bailian_llm_model,
            self.config.bailian_tts_model,
        )

    async def stop(self) -> None:
        """Stop the engine gracefully."""
        if self._task is not None:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        if self._location is not None:
            await self._location.stop()
        if self._audio is not None:
            await self._audio.close()
        if self._camera is not None:
            self._camera.close()
        logger.info("引擎已停止")

    # ── Main loop ──────────────────────────────────────────────────────
    async def _conversation_loop(self) -> None:
        """Main loop: listen → think → speak, repeat."""
        # Startup greeting
        try:
            await self._speak_greeting()
        except Exception:
            logger.exception("startup greeting failed (continuing)")

        while True:
            try:
                await self._run_turn()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("conversation turn error — restarting")
                await asyncio.sleep(1.0)

    async def _run_turn(self) -> None:
        """Serialize listening and response with Dashboard text turns."""
        async with self._turn_lock:
            await self._run_voice_turn()

    async def _run_voice_turn(self) -> None:
        """One full turn: listen → think → speak."""
        # ── 1. Listen ──
        self._set_state("listening")

        try:
            user_text = await asyncio.wait_for(
                self._listen_for_speech(),
                timeout=self.config.asr_turn_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("ASR turn timed out")
            user_text = ""

        if not user_text:
            # During the 30s wake-free window, pause briefly before retrying
            # to avoid hammering the cloud ASR with rapid reconnections.
            if time.monotonic() < self._wake_word_active_until:
                await asyncio.sleep(1.0)
            return

        logger.info("ASR final: %r", user_text)
        user_text = self._accept_transcript(user_text)
        if not user_text:
            self._set_state("idle")
            return

        if user_text == _WAKE_ONLY:
            await self._speak_wake_response()
            self._set_state("idle")
            return

        await self._coordinate_turn(user_text, source="voice", speak=True)
        # Refresh the wake-free window so the user can continue the
        # conversation without saying the wake word on every turn.
        if self.config.enable_wake_word:
            self._wake_word_active_until = time.monotonic() + self.config.wake_word_timeout_s

    # ── ASR: mic → text ────────────────────────────────────────────────
    async def _listen_for_speech(self) -> str:
        """Capture audio and stream to Bailian ASR. Returns final transcript.

        Wake word detection is done server-side: the ASR transcript is
        checked for the wake word by ``_accept_transcript``.  This keeps the
        device simple (no local wake-word model) and lets the cloud ASR
        handle both wake-word spotting and speech recognition in one pass.
        """
        assert self._audio is not None

        # Never start ASR during thinking or speaking — the LLM/TTS phase
        # must fully drain before the microphone is re-enabled.
        while self._state in ("thinking", "speaking"):
            await asyncio.sleep(0.1)

        # Wait for the echo gate (post-playback silence) to expire so that
        # residual TTS output doesn't leak into the ASR session.
        remaining = max(0.0, self._listen_not_before - time.monotonic())
        if remaining > 0.01:
            logger.info("🎤 [聆听] 等待回声门控... (剩余 %.2fs)", remaining)
        await self._wait_for_listening_gate()

        logger.info("🎤 [聆听] 开始录音，建立 ASR 连接...")
        return await self._listen_cloud_asr()

    async def _listen_cloud_asr(self) -> str:
        """Stream mic audio to Bailian realtime ASR via WebSocket.

        Server-side VAD (``turn_detection.server_vad``) detects
        speech-start / speech-stop and the ASR service returns the
        final transcript once the user stops speaking.

        Periodic diagnostic logging helps identify audio-quality issues
        (clipping, low level, dropped frames) without external tools.
        """
        assert self._audio is not None

        audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=256)
        capture_done = asyncio.Event()
        dropped_frames = 0
        speech_start_time: float | None = None
        _last_level_log = 0.0
        _peak_since_last_log = 0.0
        _clip_since_last_log = 0

        async def _capture() -> None:
            nonlocal dropped_frames, _last_level_log, _peak_since_last_log, _clip_since_last_log
            async for chunk in self._audio.start_capture():
                if capture_done.is_set():
                    break
                if self._listening_is_blocked():
                    continue
                try:
                    audio_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    dropped_frames += 1

                # ── Audio quality diagnostics (every ~5 s) ──────────
                now = time.monotonic()
                if now - _last_level_log >= 5.0:
                    _last_level_log = now
                    rms = self._audio.capture_rms if self._audio else 0.0
                    db = round(20.0 * math.log10(max(rms, 1e-8)), 1)
                    logger.info(
                        "🎤 [诊断] 电平 %5.1f dBFS | 丢帧 %d | 削波 %d",
                        db, dropped_frames, _clip_since_last_log,
                    )
                    _peak_since_last_log = 0.0
                    _clip_since_last_log = 0
                else:
                    # Track peak and clipping between log intervals
                    rms = self._audio.capture_rms if self._audio else 0.0
                    if rms > _peak_since_last_log:
                        _peak_since_last_log = rms
                    if rms > 0.98:  # near-digital-full-scale = clipping
                        _clip_since_last_log += 1

        t_connect = time.monotonic()
        capture_task = asyncio.create_task(_capture())

        try:
            async with BailianASRClient(self.config) as asr:
                await asr.configure()
                logger.info(
                    "🎤 [ASR] WebSocket 已连接 (耗时 %.2fs)",
                    time.monotonic() - t_connect,
                )

                final_text = ""
                speech_count = 0

                async def _feed_asr() -> None:
                    while not capture_done.is_set():
                        try:
                            chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                            if self._listening_is_blocked():
                                continue
                            await asr.send_audio(chunk)
                        except asyncio.TimeoutError:
                            continue

                feed_task = asyncio.create_task(_feed_asr())

                try:
                    async for result in asr.results():
                        if result.speech_started:
                            speech_count += 1
                            speech_start_time = time.monotonic()
                            if self._tts_playing and self._barge_in:
                                logger.info("⏸ [打断] 用户开始说话，停止播放")
                                self._audio.stop_playback()
                                self._tts_playing = False
                                self._barge_in_occurred = True

                        if not result.is_final and result.text:
                            if self._on_transcript:
                                self._on_transcript(result.text, False)

                        if result.is_final:
                            duration = ""
                            if speech_start_time is not None:
                                d = time.monotonic() - speech_start_time
                                duration = f" ({d:.1f}s)"
                            logger.info("✅ [最终] %r%s", result.text, duration)
                            if dropped_frames > 0:
                                logger.warning(
                                    "⚠️  音频丢帧: %d 帧 — 可能影响识别",
                                    dropped_frames,
                                )
                            final_text = result.text
                            if self._on_transcript:
                                self._on_transcript(result.text, True)
                            break
                finally:
                    capture_done.set()
                    feed_task.cancel()
                    try:
                        await feed_task
                    except asyncio.CancelledError:
                        pass

                await asr.finish()

                # ── Empty-result diagnostics ─────────────────────────
                if not final_text.strip():
                    rms = self._audio.capture_rms if self._audio else 0.0
                    db = round(20.0 * math.log10(max(rms, 1e-8)), 1)
                    logger.info(
                        "🎤 [ASR] 无识别结果 | 电平 %5.1f dBFS | "
                        "speech_count=%d | dropped=%d — "
                        "%s",
                        db, speech_count, dropped_frames,
                        "可能是无人说话" if speech_count == 0 else "有语音但未识别出文字",
                    )

                return final_text.strip()
        finally:
            capture_done.set()
            capture_task.cancel()
            try:
                await capture_task
            except asyncio.CancelledError:
                pass

    # ── Unified turn coordination ──────────────────────────────────────
    async def _coordinate_turn(
        self,
        text: str,
        *,
        source: str,
        speak: bool,
        client_message_id: str = "",
        image_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """Run voice and Dashboard text through one deterministic pipeline."""
        self._begin_turn(text, source=source, client_message_id=client_message_id)
        decision = classify_intent(text)
        journal_context = ""
        vision_context = ""
        location_context = ""
        reply = ""
        emotion = ""
        error: str | None = None

        try:
            if decision.intent == TurnIntent.AMBIGUOUS_CAMERA:
                reply = "你想让我看 Reachy 面前，还是看基地车外面？"
            elif decision.intent == TurnIntent.FRONT_CAMERA:
                self._emit_turn_event("turn_status", status="capturing")
                reply = await self._tool_take_photo(image_bytes)
                vision_context = reply
            elif decision.intent == TurnIntent.REAR_CAMERA:
                self._emit_turn_event("turn_status", status="capturing")
                reply = await self._tool_capture_rear_view()
                vision_context = reply
            elif decision.intent == TurnIntent.LOCATION:
                self._emit_turn_event("turn_status", status="retrieving")
                reply = await self._tool_get_current_location()
                location_context = reply
            elif decision.intent == TurnIntent.JOURNAL:
                self._emit_turn_event("turn_status", status="retrieving")
                journal_context = await self._verified_journal_context(text)
                if not journal_context:
                    reply = _JOURNAL_UNKNOWN
            elif decision.intent == TurnIntent.MOTION:
                reply = await self._execute_deterministic_motion(text)
            elif (
                decision.intent == TurnIntent.GENERAL
                and re.sub(r"[\s，。！？!?、～~]+", "", text).lower()
                in _SHORT_ACKS
            ):
                reply = "嗯嗯，我在呢。你慢慢说～"

            if not reply:
                system_prompt = _build_system_prompt(
                    self.config,
                    journal_context,
                    inject_location=location_context,
                )
                if vision_context:
                    system_prompt += f"\n\n【本轮实时画面】\n{vision_context}"
                messages: list[dict[str, str]] = [
                    {"role": "system", "content": system_prompt},
                ]
                # Prior assistant replies remain conversational context for
                # general questions, but protected journal facts are rebuilt
                # solely from this turn's verified source material.
                if decision.intent != TurnIntent.JOURNAL:
                    messages.extend(self._conversation_history)
                messages.append({"role": "user", "content": text})

                self._set_state("thinking")
                old_search = self.config.enable_search
                if decision.intent == TurnIntent.JOURNAL:
                    self.config.enable_search = False
                try:
                    if speak and self._audio is not None:
                        reply, emotion = await self._think_and_speak(messages)
                    else:
                        reply, emotion = await self._think_text_only(messages)
                finally:
                    self.config.enable_search = old_search
                if (
                    decision.intent == TurnIntent.GENERAL
                    and not vision_context
                    and _UNGROUNDED_VISUAL_CLAIM_RE.search(reply)
                ):
                    logger.warning(
                        "Blocked ungrounded visual claim in general reply: %s",
                        reply,
                    )
                    reply = (
                        "我这轮没有拍照，所以不能假装看见啦。"
                        "想让我看看 Reachy 面前，还是基地车外面？"
                    )
                elif decision.intent == TurnIntent.JOURNAL and journal_context:
                    # The model summarizes journal prose, but the calendar
                    # calculation is deterministic.  Guard against it
                    # changing "大前天=7月28日" back into a nearby date.
                    reply = _enforce_journal_target_date(
                        text,
                        reply,
                        _extract_target_date(text),
                    )
            elif speak and self._audio is not None:
                self._emit_turn_event("turn_status", status="speaking")
                await self._speak_reply(reply)

            if reply:
                self._record_history(text, reply)
            self._emit_turn_event(
                "turn_final",
                text=reply,
                sources=self._current_sources,
                error=None,
                intent=decision.intent.value,
            )
        except Exception as exc:
            logger.exception("Coordinated turn failed")
            error = str(exc)
            reply = "抱歉，这一轮处理失败了，请稍后再试。"
            self._emit_turn_event(
                "turn_final",
                text=reply,
                sources=self._current_sources,
                error=error,
                intent=decision.intent.value,
            )
        finally:
            self._last_activity = time.monotonic()
            self._set_state("idle")
            self._emit_turn_event("turn_status", status="error" if error else "done")

        return {
            "reply": reply,
            "emotion": emotion,
            "memory_context": journal_context,
            "vision_context": vision_context,
            "intent": decision.intent.value,
            "sources": list(self._current_sources),
            "error": error,
            "turn_id": self._current_turn_id,
        }

    def _record_history(self, user_text: str, reply: str) -> None:
        self._conversation_history.append({"role": "user", "content": user_text})
        self._conversation_history.append({"role": "assistant", "content": reply})
        max_messages = self.config.max_history_turns * 2
        if len(self._conversation_history) > max_messages:
            self._conversation_history = self._conversation_history[-max_messages:]

    async def _think_text_only(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, str]:
        parts: list[str] = []
        async with BailianLLMClient(self.config) as llm:
            async for token in llm.chat_stream(messages):
                parts.append(token)
                if self._on_llm_token:
                    self._on_llm_token(token)
                self._emit_turn_event("chat_message_delta", delta=token)
        raw = "".join(parts)
        emotion = _extract_emotion(raw)
        return _TAG_RE.sub("", raw).strip(), emotion

    async def _verified_journal_context(self, query: str) -> str:
        """Online-check the corpus, revalidate candidates, then return evidence."""
        if self._memory is None:
            return ""
        try:
            await self._journal_fetcher.sync(memory_store=self._memory)
        except Exception:
            health = self._journal_fetcher.health()
            if not health["complete"]:
                logger.warning("No verified journal cache is available", exc_info=True)
                return ""
            logger.warning(
                "Journal directory is partially unavailable; continuing with "
                "%d individually verified cached entries (%d expected)",
                health["complete"],
                health["expected"],
            )

        target_date = _extract_target_date(query)
        results = (
            self._memory.search_by_date(target_date, k=3)
            if target_date
            else self._memory.search(query, k=self.config.memory_top_k)
        )
        if not results:
            return ""

        # search_by_date only returns manifest entries which cover target_date,
        # including multi-day title ranges whose primary ``date`` differs.
        exact_date = bool(target_date and results)
        if not exact_date and float(results[0].get("score") or 0) < self.config.journal_relevance_threshold:
            return ""

        candidate_slugs = [str(r.get("slug") or r.get("id") or "") for r in results]
        candidate_slugs = [slug for slug in candidate_slugs if slug]
        try:
            await self._journal_fetcher.sync(
                memory_store=self._memory,
                refresh_slugs=candidate_slugs,
            )
            results = (
                self._memory.search_by_date(target_date, k=3)
                if target_date
                else self._memory.search(query, k=self.config.memory_top_k)
            )
        except Exception:
            logger.warning(
                "Candidate journal revalidation was partial; using its "
                "individually verified cached copy"
            )

        selected = results[:1] if exact_date else results[:2]
        health = self._journal_fetcher.health()
        cache_note = (
            f"缓存最后完整校验：{health['last_success_at']}。"
            if health.get("failures")
            else "本轮已在线校验官方目录和候选正文。"
        )
        blocks = [cache_note]
        if target_date:
            relative_word = _relative_date_word(query)
            label = f"（用户原话：{relative_word}）" if relative_word else ""
            blocks.append(
                f"程序已确定本题目标日期：{_format_chinese_date(target_date)}{label}。"
                "这是确定性日历计算；回答时不得改成其他日期。"
            )
        for item in selected:
            source = {
                "slug": item.get("slug") or item.get("id"),
                "title": item.get("title", ""),
                "date": item.get("date", ""),
                "url": item.get("source_url", ""),
                "source_updated_at": item.get("source_updated_at", ""),
            }
            self._current_sources.append(source)
            blocks.append(
                f"日期：{source['date'] or '未知'}\n"
                f"标题：{source['title']}\n"
                f"来源：{source['url']}\n"
                f"完整正文：\n{item.get('content', '')}"
            )
        return "\n\n---\n\n".join(blocks)

    async def _execute_deterministic_motion(self, text: str) -> str:
        if "跳" in text:
            return await self._tool_dance("happy")
        if "点头" in text:
            return await self._tool_gesture("nod", 2)
        if "摇头" in text:
            return await self._tool_gesture("shake_head", 2)
        if "挥" in text or "招呼" in text:
            return await self._tool_gesture("wave", 1)
        return await self._tool_pose("sleep" if "睡" in text or "休息" in text else "wake_up")

    # ── Barge-in watcher ──────────────────────────────────────────────

    async def _watch_barge_in(self) -> None:
        """Run concurrently with LLM+TTS: monitor audio input for speech onset.

        Polls ``capture_rms`` every 50ms. When RMS exceeds threshold while
        the engine is speaking, sets ``_barge_in_requested`` so the main
        loop cancels the current reply and re-enters listening.

        Waits a short grace period after speaking starts before enabling
        barge-in, to avoid false triggers from the first few milliseconds
        of TTS output leaking into the mic.
        """
        assert self._audio is not None
        # capture_rms is 0.0–1.0 (normalized to 32768). 0.02 ≈ loud room tone,
        # 0.05 ≈ normal speech from ~1m, 0.10 ≈ close/loud speech.
        threshold = self.config.barge_in_sensitivity  # default 0.035
        holdoff_s = 0.3  # grace period: ignore first 300ms of TTS (speaker echo)

        start = time.monotonic()
        while not self._barge_in_requested:
            try:
                # Grace period — don't barge-in on the first ~300ms of TTS output
                if time.monotonic() - start < holdoff_s:
                    await asyncio.sleep(0.1)
                    continue

                rms = self._audio.capture_rms
                if rms > threshold:
                    self._barge_in_requested = True
                    logger.info(
                        "🗣 [打断] 检测到语音 RMS=%.4f (阈值=%.3f) — 停止当前回答",
                        rms, threshold,
                    )
                    self._audio.stop_playback()
                    self._barge_in_occurred = True
                    return
            except Exception:
                pass
            await asyncio.sleep(0.05)

    # ── LLM → TTS ──────────────────────────────────────────────────────
    async def _think_and_speak(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, str]:
        """Run LLM and stream tokens to TTS. Returns (full_text, emotion)."""
        assert self._audio is not None

        # Set TTS output rate (Qwen Realtime = 24kHz)
        self._audio.set_output_sample_rate(self.config.tts_sample_rate)

        full_parts: list[str] = []
        emotion = ""
        tts_text_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
        llm_done_flag = asyncio.Event()
        tts_done_flag = asyncio.Event()

        # Reset barge-in for this turn (barge-in is disabled for now)
        self._barge_in_requested = False

        async with self._speaking_scope():
            # Barge-in watcher disabled — too prone to TTS echo false triggers.
            # Will re-enable when AEC hardware is properly configured.
            # barge_watcher = asyncio.create_task(self._watch_barge_in())

            async with BailianLLMClient(self.config) as llm:
                # TTS worker
                async def _tts_worker() -> None:
                    tts = BailianTTSClient(
                        self.config,
                        on_audio=self._queue_tts_audio,
                        on_done=tts_done_flag.set,
                    )
                    await tts.open()
                    try:
                        while not llm_done_flag.is_set() or not tts_text_queue.empty():
                            try:
                                text = await asyncio.wait_for(
                                    tts_text_queue.get(), timeout=0.2
                                )
                                await tts.feed(text)
                            except asyncio.TimeoutError:
                                continue
                        await tts.flush()
                    except Exception:
                        logger.exception("TTS worker error")
                        tts_done_flag.set()
                    finally:
                        await tts.close()

                tts_task = asyncio.create_task(_tts_worker())

                try:
                    token_buf = ""

                    async for token in llm.chat_stream(messages):
                        # Check for barge-in on every token
                        if self._barge_in_requested:
                            logger.info("⏸ [打断] 取消 LLM 流 — 用户开始说话")
                            self._audio.stop_playback()
                            llm_done_flag.set()
                            break

                        full_parts.append(token)
                        if self._on_llm_token:
                            self._on_llm_token(token)
                        self._emit_turn_event("chat_message_delta", delta=token)

                        token_buf += token
                        # Feed sentence chunks to TTS
                        if any(
                            token_buf.rstrip().endswith(p)
                            for p in ("。", "！", "？", ".", "!", "?", "\n")
                        ):
                            clean = _TAG_RE.sub("", token_buf.strip())
                            if clean:
                                await tts_text_queue.put(clean)
                            token_buf = ""
                        elif len(token_buf) >= 40:
                            clean = _TAG_RE.sub("", token_buf.strip())
                            if clean:
                                await tts_text_queue.put(clean)
                            token_buf = ""

                    # Flush remaining
                    if token_buf.strip():
                        clean = _TAG_RE.sub("", token_buf.strip())
                        if clean:
                            await tts_text_queue.put(clean)

                    llm_done_flag.set()
                    await tts_task

                except Exception:
                    logger.exception("LLM → TTS pipeline error")
                    llm_done_flag.set()
                    tts_task.cancel()
                    try:
                        await tts_task
                    except asyncio.CancelledError:
                        pass

            # Barge-in watcher disabled
            # barge_watcher.cancel()
            # try: await barge_watcher
            # except asyncio.CancelledError: pass

            full_text = "".join(full_parts)
            emotion = _extract_emotion(full_text)
            clean_text = _TAG_RE.sub("", full_text).strip()

            # Fire emotion callback for Dashboard / LED display
            if emotion and self._on_emotion:
                try:
                    self._on_emotion(emotion)
                except Exception:
                    logger.debug("Emotion callback error", exc_info=True)

            # If barge-in interrupted, return empty so _run_turn skips history
            if self._barge_in_requested:
                logger.info("⏸ [打断] 本轮回答已取消，将重新聆听")
                return "", ""

            return clean_text, emotion

    async def _drain_audio(self) -> None:
        """Keep the duplex stream alive."""
        try:
            async for _ in self._audio.start_capture():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    def _queue_tts_audio(self, pcm: bytes, sample_rate: int) -> None:
        """Bridge sync TTS callback (from SDK thread) → async audio playback."""
        if self._audio is None or self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(
            self._play_tts_audio(pcm, sample_rate), self._loop
        )
        self._pending_playbacks.add(future)
        future.add_done_callback(self._pending_playbacks.discard)

    async def _play_tts_audio(self, pcm: bytes, sample_rate: int) -> None:
        """Queue PCM for playback."""
        assert self._audio is not None
        self._audio.set_output_sample_rate(sample_rate)
        await self._audio.play(pcm)

    async def _wait_for_playback_drain(self) -> None:
        """Wait until all queued audio has been played."""
        assert self._audio is not None
        while any(not future.done() for future in tuple(self._pending_playbacks)):
            await asyncio.sleep(0.01)
        while self._audio.is_playing:
            await asyncio.sleep(0.05)

    def _listening_is_blocked(self) -> bool:
        return (
            self._tts_playing
            or time.monotonic() < self._listen_not_before
            or self._state in ("thinking", "speaking")
        )

    async def _wait_for_listening_gate(self) -> None:
        while True:
            remaining = self._listen_not_before - time.monotonic()
            if not self._tts_playing and remaining <= 0:
                return
            await asyncio.sleep(min(0.05, max(0.01, remaining)))

    @asynccontextmanager
    async def _speaking_scope(self):
        """Own speaking state, duplex playback drain, and echo holdoff.

        Serialised via ``_speaking_lock`` so only one TTS stream plays at a time.
        """
        async with self._speaking_lock:
            assert self._audio is not None
            self._tts_playing = True
            self._barge_in_occurred = False
            self._set_state("speaking")
            drain_task = asyncio.create_task(self._drain_audio())
            try:
                yield
            finally:
                self._audio.mark_playback_done()
                try:
                    await self._wait_for_playback_drain()
                finally:
                    self._tts_playing = False
                    # Skip echo gate when barge-in occurred — user is already talking
                    if not self._barge_in_occurred:
                        self._listen_not_before = max(
                            self._listen_not_before,
                            time.monotonic() + max(0.0, self.config.post_playback_silence_s),
                        )
                    else:
                        logger.debug("Barge-in: skipping echo gate")
                    drain_task.cancel()
                    try:
                        await drain_task
                    except asyncio.CancelledError:
                        pass

    # ── Wake word ──────────────────────────────────────────────────────
    def _accept_transcript(self, text: str) -> str:
        """Decide whether ASR transcript should be processed.

        Returns:
            - "" if too short, noise, or no wake word
            - ``_WAKE_ONLY`` for a bare wake word
            - The utterance text (with wake word stripped) if accepted
        """
        normalized = " ".join(text.split()).strip()

        # ── ASR homophone correction ────────────────────────────────
        # Bailian ASR sometimes picks the wrong character for
        # identical-sounding words (e.g. 舞→五).  Fix before intent
        # classification so the right tool path is selected.
        corrected = _ASR_CORRECTIONS.get(normalized.rstrip("，。！？!?"))
        if corrected:
            logger.info("ASR correction: %r → %r", normalized, corrected)
            normalized = corrected

        # Strip trailing/free-floating punctuation for length & noise checks
        stripped = normalized.strip("，。！？!?、～~… ")

        # ── Noise filter: single-char utterances & ASR junk ─────────
        if not stripped or stripped in _NOISE_WORDS or _NOISE_RE.match(stripped):
            logger.info("Filtering noise transcript: %r", normalized)
            return ""

        if len(stripped) < self.config.asr_min_chars:
            logger.info("Ignoring short transcript: %r", normalized)
            return ""
        if not self.config.enable_wake_word:
            return normalized

        now = time.monotonic()

        # Check for wake word
        remaining = self._strip_wake_word(normalized)
        if remaining is not None:
            self._wake_word_active_until = (
                now + self.config.wake_word_timeout_s
            )
            if not remaining:
                logger.info("Wake word only — scheduling synchronous canned response")
                return _WAKE_ONLY
            logger.info("Wake word stripped, remaining: %r", remaining)
            return remaining

        # No wake word — check if within follow-up window
        if now < self._wake_word_active_until:
            return normalized

        logger.info("Ignoring transcript without wake word: %r", normalized)
        return ""

    def _strip_wake_word(self, normalized: str) -> str | None:
        """Return text after first wake word, or None if no wake word."""
        wake_words = [
            phrase.strip().lower()
            for phrase in self.config.wake_words.split(",")
            if phrase.strip()
        ]
        lowered = normalized.lower()
        for ww in wake_words:
            idx = lowered.find(ww)
            if idx != -1:
                return normalized[idx + len(ww):].strip(" ，,。！？!?")
        return None

    async def _speak_wake_response(self) -> None:
        """Play the canned wake-word acknowledgment."""
        assert self._audio is not None
        text = self.config.wake_response
        logger.info("Wake response: %s", text)

        self._audio.set_output_sample_rate(self.config.tts_sample_rate)
        async with self._speaking_scope():
            try:
                tts = BailianTTSClient(self.config, on_audio=self._queue_tts_audio)
                await tts.open()
                try:
                    await tts.synthesize(text)
                finally:
                    await tts.close()
            except Exception:
                logger.exception("Wake response TTS failed")

    async def _speak_greeting(self) -> None:
        """Play the startup greeting."""
        assert self._audio is not None
        lang = self._language
        text = (
            "嗨，我是皮皮虾！柴火基地车的AI小助手，有什么想聊的随时叫我哦。"
            if lang != "en"
            else "Hi, I'm Pipi Xia, the AI assistant of the Chaihuo MCV! Feel free to chat with me."
        )
        logger.info("Startup greeting: %s", text)

        self._audio.set_output_sample_rate(self.config.tts_sample_rate)
        async with self._speaking_scope():
            try:
                tts = BailianTTSClient(self.config, on_audio=self._queue_tts_audio)
                await tts.open()
                try:
                    await tts.synthesize(text)
                finally:
                    await tts.close()
            except Exception:
                logger.exception("Greeting TTS failed")

    def runtime_status(self) -> dict[str, object]:
        """Sanitized live status used by diagnostics and the Dashboard."""
        now = time.monotonic()
        if self._audio:
            ri = self._audio.resolved_info
            audio = ri if isinstance(ri, dict) else ri.to_dict()
        else:
            audio = None
        # Audio level diagnostics
        # Use play_rms() method (available on all AudioBackend implementations)
        capture_rms = self._audio.play_rms() if self._audio else 0.0
        capture_db = round(20.0 * math.log10(max(capture_rms, 1e-8)), 1)
        location = (
            self._location.latest_position.to_dict()
            if self._location and self._location.latest_position
            else None
        )
        return {
            "state": self._state,
            "language": self._language,
            "audio": audio,
            "audio_level_rms": round(capture_rms, 4),
            "audio_level_db": capture_db,
            "echo_gate_remaining_s": round(max(0.0, self._listen_not_before - now), 3),
            "tts_playing": self._tts_playing,
            "barge_in_occurred": self._barge_in_occurred,
            "wake_word_active": now < self._wake_word_active_until,
            "conversation_turns": len(self._conversation_history) // 2,
            "location": location,
        }

    # ── Debug: text-in / text-out ─────────────────────────────────────
    async def process_text(
        self,
        text: str,
        image_bytes: bytes | None = None,
        *,
        source: str = "dashboard",
        client_message_id: str = "",
    ) -> dict[str, Any]:
        """Process typed input through the exact same coordinator as ASR."""
        normalized = " ".join(text.split()).strip()
        if not normalized:
            return {
                "reply": "",
                "emotion": "",
                "memory_context": "",
                "vision_context": "",
                "intent": TurnIntent.GENERAL.value,
                "sources": [],
                "error": "消息不能为空",
                "turn_id": "",
            }
        async with self._turn_lock:
            return await self._coordinate_turn(
                normalized,
                source=source,
                speak=self._audio is not None,
                client_message_id=client_message_id,
                image_bytes=image_bytes,
            )

    async def _speak_reply(self, text: str) -> None:
        """Speak a text reply via TTS (background, fire-and-forget).

        Uses the same speaking lock and echo gate as voice turns.
        """
        assert self._audio is not None
        self._audio.set_output_sample_rate(self.config.tts_sample_rate)
        async with self._speaking_scope():
            try:
                tts = BailianTTSClient(self.config, on_audio=self._queue_tts_audio)
                await tts.open()
                try:
                    await tts.synthesize(text)
                finally:
                    await tts.close()
            except Exception:
                logger.exception("Debug TTS failed")

    async def _tool_get_current_location(self) -> str:
        """Return the real-time GPS position of the vehicle."""
        if self._location is None:
            return "定位服务未启动，无法获取位置信息。"
        try:
            pos = await self._location.get_position()
            if pos.source == "unavailable":
                return "当前无法获取位置信息。请检查 GPS 设备连接或网络是否正常。"
            logger.info("📍 位置查询: %.6f, %.6f (%s)", pos.lat, pos.lon, pos.source)
            return pos.to_llm_context()
        except Exception:
            logger.exception("Location tool failed")
            return "位置查询失败，请稍后重试。"

    async def _tool_capture_rear_view(self) -> str:
        """Capture rear-view (EZVIZ) snapshot and run VLM to describe."""
        try:
            from chaihuo_reachy.ezviz import capture_rear_view

            jpeg = await capture_rear_view(self.config)
        except Exception:
            logger.exception("Rear-view capture failed")
            return "没有拿到车外后视摄像头画面，请检查萤石设备连接和配置。"

        if self._on_snapshot:
            try:
                self._on_snapshot(jpeg, "rear")
            except Exception:
                logger.debug("Rear-view snapshot callback error", exc_info=True)

        issue = visual_quality_issue(jpeg)
        if issue:
            logger.warning("Rear-view frame rejected: %s", issue)
            return f"我拿到车外画面了，不过{issue}"

        try:
            async with BailianVLMClient(self.config) as vlm:
                description = await vlm.understand(
                    jpeg,
                    question=(
                        "只描述照片中清晰可见的车外场景、人物、车辆和物体。"
                        "不要推断人物身份、具体地点、天气或画面外事实。"
                        "如果画面大部分是暗的、模糊的或者看不清，"
                        "必须直接说'画面太暗，什么都看不见'，不要编造任何内容。"
                        "用亲切自然的中文，60字以内。"
                    ),
                )
            description = _VISUAL_PREFIX_RE.sub("", description).strip()
            if not description:
                raise RuntimeError("VLM returned an empty rear-view description")
            logger.info("Rear view VLM: %s", description)
            return description
        except Exception:
            logger.exception("Rear view VLM failed")
            return "车后方画面已拍摄但AI分析失败。"

    async def _tool_take_photo(self, fallback_image: bytes | None = None) -> str:
        """Capture camera frame and run VLM to describe the scene."""
        jpeg: bytes | None = fallback_image

        if jpeg is None and self._camera_snapshot_provider is not None:
            provided = self._camera_snapshot_provider()
            jpeg = await provided if hasattr(provided, "__await__") else provided

        if (
            jpeg is None
            and self._camera_snapshot_provider is None
            and self._camera is not None
        ):
            jpeg = self._camera.capture_jpeg()

        if jpeg is None:
            return "没有拿到 Reachy 前置摄像头画面。"

        if self._on_snapshot:
            try:
                self._on_snapshot(jpeg, "front")
            except Exception:
                logger.debug("Front snapshot callback error", exc_info=True)

        issue = visual_quality_issue(jpeg)
        if issue:
            logger.warning("Front frame rejected: %s", issue)
            return f"我拍到画面了，不过{issue}可以帮我检查一下 Reachy 的镜头吗？"

        try:
            async with BailianVLMClient(self.config) as vlm:
                description = await vlm.understand(
                    jpeg,
                    question=(
                        "只描述这张照片中清晰可见的人、环境和物体。"
                        "不要猜测或编造人物身份、具体地点、不可见事实。"
                        "如果画面大部分是暗的、模糊的或者看不清，"
                        "必须直接说'画面太暗，什么都看不见'，不要编造任何内容。"
                        "用亲切自然的中文，60字以内。"
                    ),
                )
            description = _VISUAL_PREFIX_RE.sub("", description).strip()
            if not description:
                raise RuntimeError("VLM returned an empty front-view description")
            logger.info("Tool take_photo: %s", description)
            return description
        except Exception:
            logger.exception("Tool take_photo VLM failed")
            return "拍照成功但画面分析失败。"

    # ── Motion tool handlers ─────────────────────────────────────────

    async def _tool_dance(self, style: str = "happy") -> str:
        """Execute a dance via the MotionController."""
        if self._motion is None:
            return "运动控制未启用，无法跳舞。"
        try:
            await self._motion.dance(style)
            return f"跳了一段{style}风格的舞蹈！"
        except Exception as e:
            logger.exception("Dance failed")
            return f"跳舞失败: {e}"

    async def _tool_gesture(self, gesture: str, times: int = 1) -> str:
        """Execute a gesture via the MotionController."""
        if self._motion is None:
            return "运动控制未启用，无法做动作。"
        try:
            if gesture == "nod":
                await self._motion.nod(times=times)
            elif gesture == "shake_head":
                await self._motion.shake_head(times=times)
            elif gesture == "wave":
                await self._motion.wave_antenna("both")
            elif gesture == "look_around":
                await self._motion.look_left_right()
            return f"完成了动作: {gesture}"
        except Exception as e:
            logger.exception("Gesture failed")
            return f"动作执行失败: {e}"

    async def _tool_pose(self, pose: str) -> str:
        """Switch robot pose via the MotionController."""
        if self._motion is None:
            return "运动控制未启用，无法切换姿势。"
        try:
            if pose == "sleep":
                await self._motion.sleep()
                return "机器人已休眠。"
            elif pose == "wake_up":
                await self._motion.wake_up()
                return "机器人已站起来。"
            return f"未知姿势: {pose}"
        except Exception as e:
            logger.exception("Pose change failed")
            return f"姿势切换失败: {e}"

    # ── State ──────────────────────────────────────────────────────────
    def _set_state(self, state: str) -> None:
        self._state = state
        self._last_activity = time.monotonic()
        if self._on_state_change:
            self._on_state_change(state)
        # Play feedback sound asynchronously (non-blocking)
        if self._audio is not None and self.config.feedback_sounds_enabled:
            asyncio.ensure_future(self._play_feedback_sound(state))

    async def _play_feedback_sound(self, state: str) -> None:
        """Play a short synthesized beep for state transitions."""
        import numpy as np
        try:
            sr = 16000
            duration = 0.08
            t = np.linspace(0, duration, int(sr * duration), False)

            if state == "wake_listening":
                # Gentle rising two-tone
                t1 = np.linspace(0, 0.06, int(sr * 0.06), False)
                tone1 = np.sin(2 * np.pi * 800 * t1) * np.linspace(0, 0.5, len(t1))
                tone2 = np.sin(2 * np.pi * 1200 * t1) * np.linspace(0, 0.5, len(t1))
                beep = np.concatenate([tone1, tone2])
            elif state == "listening":
                # Short click — very subtle
                beep = np.sin(2 * np.pi * 1000 * t) * 0.2
            elif state == "speaking":
                # Confirmation chirp
                t2 = np.linspace(0, 0.05, int(sr * 0.05), False)
                beep = np.sin(2 * np.pi * 600 * t2) * 0.3
            elif state == "thinking":
                beep = np.sin(2 * np.pi * 400 * t) * 0.15
            else:
                return  # no sound for idle/other states

            pcm = (beep * 32767 * 0.3).astype(np.int16).tobytes()
            self._audio.set_output_sample_rate(sr)
            await self._audio.play(pcm)
        except Exception:
            logger.debug("Feedback sound skipped", exc_info=True)


def _extract_emotion(text: str) -> str:
    m = _TAG_RE.search(text)
    return m.group(1).lower() if m else ""


def _extract_target_date(query: str) -> str | None:
    """Detect time-based queries and return the target ISO date.

    Supports: 昨天, 今天, 前天, 大前天, 周X, 上/下周X,
    and explicit dates like 7月28日, 2026-07-28, 2026.0728.
    """
    from datetime import date, timedelta
    import re

    today = date.today()
    q = query.lower()

    # Relative days
    relative_days = {
        "大前天": 3, "前天": 2, "昨天": 1,
        "今天": 0, "今日": 0,
        "明天": -1, "后天": -2, "大后天": -3,
    }
    for word, offset in relative_days.items():
        if word in q:
            return (today - timedelta(days=offset)).isoformat()

    # Day of week: 周一/... / 上周一/... / 下周一/...
    weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
    week_match = re.search(r"(上周|下周|周|星期)([一二三四五六日天])", q)
    if week_match:
        prefix = week_match.group(1)
        day_char = week_match.group(2)
        target_wd = weekday_map.get(day_char, 0)
        current_wd = today.weekday()
        this_week_delta = target_wd - current_wd
        if prefix in ("上周",):
            delta = this_week_delta - 7
        elif prefix in ("下周",):
            delta = this_week_delta + 7
        else:
            delta = this_week_delta
            if delta <= 0:
                delta += 7  # Next occurrence
        return (today + timedelta(days=delta)).isoformat()

    # Explicit dates: 7月28日, 7.28, 2026-07-28, 2026.0728
    date_patterns = [
        (r"(\d{4})[.年-](\d{1,2})[.月-](\d{1,2})[日]?", "%Y-%m-%d"),
        (r"(\d{1,2})[.月](\d{1,2})[日]?", "%m-%d"),  # month-day only
    ]
    for pattern, fmt in date_patterns:
        m = re.search(pattern, q)
        if m:
            try:
                if fmt == "%Y-%m-%d":
                    return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
                else:
                    d = date(today.year, int(m.group(1)), int(m.group(2)))
                    return d.isoformat()
            except ValueError:
                pass

    return None


_RELATIVE_DATE_WORDS = (
    "大前天",
    "大后天",
    "前天",
    "昨天",
    "今天",
    "今日",
    "明天",
    "后天",
)


def _relative_date_word(query: str) -> str:
    """Return the longest relative-day word present in *query*."""
    return next((word for word in _RELATIVE_DATE_WORDS if word in query), "")


def _format_chinese_date(date_str: str) -> str:
    """Format an ISO date without leading zeroes for spoken Chinese."""
    from datetime import date

    parsed = date.fromisoformat(date_str)
    return f"{parsed.year}年{parsed.month}月{parsed.day}日"


def _enforce_journal_target_date(
    query: str,
    reply: str,
    target_date: str | None,
) -> str:
    """Keep a model-generated relative-date answer aligned with the calendar.

    Journal prose still comes from the model, while phrases such as
    ``大前天是2026年07月29日`` are corrected deterministically.
    """
    if not target_date:
        return reply
    relative_word = _relative_date_word(query)
    if not relative_word:
        return reply

    correct = _format_chinese_date(target_date)
    date_token = (
        r"20\d{2}\s*(?:年|[-/.])\s*\d{1,2}\s*"
        r"(?:月|[-/.])\s*\d{1,2}\s*日?"
    )
    guarded, replacements = re.subn(
        rf"({re.escape(relative_word)}\s*(?:是|为)?\s*){date_token}",
        rf"\g<1>{correct}",
        reply,
        count=1,
    )
    if replacements:
        return guarded
    return f"{relative_word}是{correct}。{reply}"
