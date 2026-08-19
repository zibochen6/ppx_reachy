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
from collections import deque
from datetime import datetime, timezone
import json
import logging
import math
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import numpy as np

from chaihuo_reachy.bailian import (
    ASRResult,
    BailianASRClient,
    BailianLLMClient,
    BailianTTSClient,
    BailianVLMClient,
)
from chaihuo_reachy.camera import visual_quality_issue
from chaihuo_reachy.config import Config
from chaihuo_reachy.audio_frontend import DirectionGate, SpeechEndpoint
from chaihuo_reachy.intent import (
    IntentDecision,
    TurnIntent,
    classify_intent,
    extract_location_update,
    is_journey_overview_query,
)
from chaihuo_reachy.location import LocationService, create_location_service
from chaihuo_reachy.memory import JournalFetcher
from chaihuo_reachy.memory.store import MemoryStore
from chaihuo_reachy.turn_planner import plan_turn
from chaihuo_reachy.vision import (
    VisualCache,
    VisualObservation,
    parse_vlm_observation,
)

if TYPE_CHECKING:
    from chaihuo_reachy.backends.interfaces import AudioBackend, CameraBackend
    from chaihuo_reachy.motion import MotionController

logger = logging.getLogger("chaihuo_reachy.engine")

_JOURNAL_UNKNOWN = (
    "基地车日记暂时没有记录到这一段，我不能把别处的内容当成我们的经历。"
    "你可以告诉我更具体的地点或日期，我再帮你查一次。"
)

# 跳完舞后的 LLM 接话注入：不透露风格/曲库，表现得像即兴编排
_DANCE_REPLY_INJECTION = (
    "\n\n【舞蹈提示】你刚刚配合音乐完成了一段即兴舞蹈，动作和音乐都已结束。"
    "请在回复里自然地接话（简短、口语化），比如说说自己跳得很尽兴、"
    "欢迎对方再来点歌或指定节奏。绝对不要使用"
    "「我跳了一段XX风格的舞蹈」这类模板话术；"
    "不要提及任何预设风格、舞蹈库、音乐文件或程序编舞的存在，"
    "表现得像你临场即兴编排、随性发挥。"
)

_LOCATION_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_current_location",
        "description": (
            "读取基地车当前定位。精确坐标优先；只有公网IP时只返回城市级位置。"
            "不得根据聊天内容猜测工具未返回的街道、校园或建筑。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "refresh": {
                    "type": "boolean",
                    "description": "是否跳过定位缓存并立即刷新",
                    "default": False,
                }
            },
            "additionalProperties": False,
        },
    },
}

_OBSERVE_SCENE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "observe_scene",
    "description": (
        "只在回答依赖此刻真实可见的人、物体、动作、数量、颜色、文字或环境时调用。"
        "观点、知识、历史描述、日记、代码和方案问题不要调用。未指定方向使用 front；"
        "只有用户明确询问车外、车后、后方或路况时使用 rear。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["front", "rear"],
                "description": "观察 Reachy 面前或基地车后方",
            },
            "focus": {
                "type": "string",
                "description": "为了回答用户，本轮具体需要确认的可见事实",
            },
        },
        "required": ["scope", "focus"],
        "additionalProperties": False,
    },
}

_VISUAL_REFRESH_RE = re.compile(r"现在|此刻|又|变成|变化|还在吗|还有吗|重新|再看")
_VISUAL_REFERENCE_RE = re.compile(r"它|这个|那个|刚才|上面|那上面|其")

# 明确要求"基地车事实证据"的指代词：含这些词的检索失败才拒绝回答；
# 否则（科普/常识问题误入日记检索）降级给 LLM 正常回答。
_JOURNAL_EVIDENCE_TERMS = (
    "基地车",
    "日记",
    "咱们",
    "我们",
    "记不记得",
    "还记得",
    "回忆",
    "旅途",
    "旅程",
    "哪一站",
    "队友",
    "领队",
    "车上",
)


def _requires_journal_evidence(text: str) -> bool:
    return any(term in text for term in _JOURNAL_EVIDENCE_TERMS)


# Common ASR homophone corrections (Chinese: same sound, wrong character)
_ASR_CORRECTIONS: dict[str, str] = {
    "跳个五": "跳个舞",  # 舞/五 both wǔ
    "跳歌舞": "跳个舞",  # segment error
    "跳个无": "跳个舞",
    "跳个午": "跳个舞",
    "个五": "跳个舞",  # VAD missed first char "跳"
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
    "嗯",
    "哦",
    "好",
    "对",
    "行",
    "嘘",
    "嘿",
    "啊",
    "呀",
    "呵",
    "哎",
    "喂",
    "呃",
    "哈",
    "嘻",
    "哟",
    "噢",
    "啧",
    "嘿咻",
    "好嘞",
}

# Regex: pure-noise transcript (single char or noise word, optional punct)
_NOISE_RE = re.compile(r"^[" + re.escape("，。！？!?、～~…") + r"]*$")
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
_VISUAL_MECHANISM_RE = re.compile(
    r"拍(?:到|照|摄|下)|照片|图片|画面|摄像头|镜头|根据(?:这张)?(?:图|照片|图片)"
)


def _natural_visual_quality_error(issue: str) -> str:
    if "暗" in issue:
        return "我这里有点看不清，光线太暗了。"
    if "过曝" in issue:
        return "光线太亮了，我现在看不清。"
    if "挡住" in issue:
        return "我的视线好像被挡住了，可以帮我检查一下吗？"
    if "模糊" in issue or "尺寸太小" in issue:
        return "刚才没看清楚，可以把它靠近一点吗？"
    return "我现在没看清楚，可以再让我看一次吗？"


def _natural_visual_reply(observation: VisualObservation) -> str:
    if not observation.ok:
        return observation.error or "我现在没看清楚，可以再让我看一次吗？"
    facts = observation.facts.strip().rstrip("。")
    reply = facts if facts.startswith(("我看到", "这是", "我数到")) else f"我看到{facts}"
    if observation.uncertainties:
        reply += f"；不过{observation.uncertainties.strip().rstrip('。')}，我还不能完全确定"
    return reply + "。"

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
    include_location: bool = True,
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

    if include_location and inject_location:
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
        "- 【当前位置】或 get_current_location 工具结果是可靠地点事实；"
        "城市级IP定位只能说省市，不能扩写具体校园、街道或建筑。\n"
        "- 对于任何你不确定的事实（位置、日期、天气、新闻等），如果工具没有返回结果，"
        "就说「这个我不太确定」，不要瞎编。\n"
        "- 凡是基地车亲历、队员、路线、站点和旅途事件，只能使用本轮【已验证日记证据】；"
        "组织背景和常识都不能冒充旅途证据。\n"
        "- 日记证据只能支持一部分时，先回答已确认部分，再准确说明缺口；"
        f"完全没有相关证据时可回答：{_JOURNAL_UNKNOWN}\n"
        "- 当前视觉事实只能依据本轮成功的【本轮视觉观察】；不要推断人物身份、地点或不可见信息。\n"
        "- 没有【本轮视觉观察】时，严禁声称自己看到当前人物、物体或环境。\n"
        "- 有可靠视觉观察时要像亲眼看见一样自然回答：场景可说“我看到……”，"
        "识别直接说“这是……”，计数说“我数到……”，评价直接回答。\n"
        "- 成功观察后禁止提及拍照、照片、图片、画面、镜头、摄像头、视觉模型或识别过程；"
        "不要说“根据图片”“画面中显示”“我拍到了”。\n"
        "- 观察不确定时明确说“看起来像……但我不能完全确定”；"
        "观察失败时只自然说明看不清、太暗、视线被挡住或暂时看不到。"
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
        beat_dance: Any | None = None,
    ) -> None:
        self.config = cfg
        self._audio: AudioBackend | None = audio_backend
        self._camera: CameraBackend | None = camera_backend
        self._motion = motion
        self._beat_dance = beat_dance  # BeatDanceController | None
        self._dance_loop_active = False
        self._beat_music_task: asyncio.Task | None = None
        self._external_interaction_active = False
        self._memory: MemoryStore | None = None
        self._location: LocationService | None = None
        self._wake: Any | None = None  # WakeWordDetector (local KWS) or None
        self._task: asyncio.Task | None = None
        self._voice_listen_task: asyncio.Task[str] | None = None
        self._voice_listener_preempted = False
        self._loop: asyncio.AbstractEventLoop | None = None

        # Conversation state
        self._state = "idle"  # idle | listening | thinking | speaking
        self._speaking_lock = asyncio.Lock()  # prevents overlapping TTS
        self._language = cfg.language
        self._conversation_history: list[dict[str, str]] = []
        self._session_location: dict[str, Any] | None = None
        self._last_topic = ""
        self._last_evidence_ids: list[str] = []
        self._last_activity = 0.0
        self._turn_lock = asyncio.Lock()
        self._current_turn_id = ""
        # Set when a dance finishes successfully; consumed by the LLM branch
        # of the same turn to improvise a natural post-dance remark.
        self._pending_dance: dict[str, object] | None = None
        self._current_sources: list[dict[str, Any]] = []
        self._active_search_mode = "off"
        self._last_search_status: dict[str, object] = {
            "mode": "off",
            "used": False,
            "error": "",
            "source_count": 0,
        }
        self._visual_cache: VisualCache | None = None
        self._current_visual_observation: VisualObservation | None = None
        self._active_response_tools: list[dict[str, Any]] = []
        self._active_response_tool_handler: (
            Callable[[str, dict[str, Any]], Awaitable[dict[str, Any] | str]] | None
        ) = None
        self._last_vision_status: dict[str, object] = {
            "policy": cfg.vision_policy,
            "decision": "off",
            "called": False,
            "reason": "",
            "scope": None,
            "observed_at": None,
            "age_s": None,
            "capture_latency_ms": None,
            "vlm_latency_ms": None,
            "total_latency_ms": None,
            "error": "",
        }
        self._wake_word_active_until = 0.0
        self._barge_in = True
        self._tts_playing = False
        # True only after conversational PCM has been handed to the playback
        # path.  Unlike AudioBackend.is_playing, this excludes state-change
        # chirps and other non-speech sounds.
        self._speech_pcm_active = False
        self._playback_observer_registered = False
        self._barge_in_occurred = False
        self._barge_in_requested = False  # set by concurrent watcher
        self._tts_audio_started = asyncio.Event()
        self._tts_generation = 0
        self._active_tts_generation: int | None = None
        self._listen_not_before = 0.0
        self._last_asr_end_reason = ""
        self._audio_frontend_metrics: dict[str, object] = {}
        self._locked_doa: float | None = None
        self._off_axis_ambient = False
        self._silent_timeouts = 0  # consecutive listen timeouts with no mic level
        self._pending_playbacks: set[concurrent.futures.Future[None]] = set()

        # Callbacks
        self._on_state_change: Callable[[str], None] | None = None
        self._on_transcript: Callable[[str, bool], None] | None = None
        self._on_asr_status: Callable[[str], None] | None = None
        self._on_llm_token: Callable[[str], None] | None = None
        self._on_emotion: Callable[[str], None] | None = None
        self._on_snapshot: Callable[[bytes, str], None] | None = (
            None  # (jpeg_bytes, label)
        )
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

    def on_asr_status(self, cb: Callable[[str], None]) -> None:
        """Receive ASR lifecycle status without creating a chat message."""
        self._on_asr_status = cb

    def _emit_asr_status(self, status: str) -> None:
        if self._on_asr_status:
            self._on_asr_status(status)

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
        self._clear_session_state()
        self._last_activity = time.monotonic()

    def _clear_session_state(self) -> None:
        self._conversation_history.clear()
        self._session_location = None
        self._last_topic = ""
        self._last_evidence_ids = []
        self._visual_cache = None
        self._current_visual_observation = None

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
            logger.info(
                "Conversation idle for %.1fs; resetting context",
                now - self._last_activity,
            )
            self._clear_session_state()
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

    async def set_external_interaction(self, active: bool) -> None:
        """Pause/resume conversation while an exclusive controller owns the robot."""
        self._external_interaction_active = bool(active)
        if active:
            self._barge_in_requested = True
            self._tts_generation += 1
            self._active_tts_generation = self._tts_generation
            self._barge_in_occurred = True
            self._speech_pcm_active = False
            await self.stop_beat_dance()
            if self._audio is not None:
                self._audio.stop_playback()
            self._stop_talk_motion_immediately()
            task, self._task = self._task, None
            if task is not None and task is not asyncio.current_task():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self._set_state("gesture")
        else:
            self._barge_in_occurred = False
            self._set_state("idle")
            if self._loop is not None and self._task is None:
                self._task = asyncio.create_task(
                    self._conversation_loop(startup_greeting=False)
                )

    # ── Lifecycle ──────────────────────────────────────────────────────
    async def start(self) -> None:
        """Start the engine: open audio, camera, memory, begin conversation."""
        if not self.config.bailian_api_key:
            raise RuntimeError("BAILIAN_API_KEY is required")
        if self.config.wake_engine == "off":
            self.config.enable_wake_word = False
        if not self.config.bailian_workspace_id:
            raise RuntimeError("BAILIAN_WORKSPACE_ID is required")

        self._loop = asyncio.get_running_loop()

        # Audio: use injected backend or create from factory
        if self._audio is None:
            from chaihuo_reachy.backends.factory import create_audio_backend

            self._audio = create_audio_backend(self.config)
            logger.info("Audio: auto-created %s", self._audio.backend_name)

        await self._audio.open()
        observer_setter = getattr(self._audio, "set_playback_observer", None)
        if callable(observer_setter):
            observer_setter(self._on_speaker_pcm)
            self._playback_observer_registered = True

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
                use_v3=self.config.journal_index_v3_enabled,
            )
            logger.info("Memory store ready: %d documents", self._memory.count())
        except Exception:
            logger.warning(
                "Memory store unavailable — journal recall disabled", exc_info=True
            )
            self._memory = None

        # Location (GPS / IP geolocation)
        self._location = create_location_service(
            gpsd_enabled=self.config.location_gpsd_enabled,
            gpsd_host=self.config.location_gpsd_host,
            gpsd_port=self.config.location_gpsd_port,
            poll_interval_s=self.config.location_poll_interval_s,
            gps_fresh_s=self.config.location_gps_fresh_s,
            browser_fresh_s=self.config.location_browser_fresh_s,
            amap_web_key=self.config.amap_web_key,
            amap_web_private_key=self.config.amap_web_private_key,
            amap_timeout_s=self.config.amap_timeout_s,
            amap_cache_ttl_s=self.config.amap_cache_ttl_s,
            wifi_enabled=self.config.location_wifi_enabled,
            wifi_scan_interval_s=self.config.location_wifi_scan_interval_s,
            wifi_fresh_s=self.config.location_wifi_fresh_s,
        )
        await self._location.start()

        # Local wake word (sherpa-onnx KWS).  Falls back to cloud ASR text
        # matching when the model is missing or inference is broken — never
        # leave the device unwakeable.
        if self.config.enable_wake_word and self.config.wake_engine == "local":
            from chaihuo_reachy.wake_word import (
                WakeWordDetector,
                WakeWordUnavailableError,
            )

            try:
                detector = WakeWordDetector(self.config)
                detector.self_check()
                self._wake = detector
                logger.info("唤醒词引擎: 本地 KWS (%s)", self.config.wake_words)
            except WakeWordUnavailableError as exc:
                logger.warning("本地唤醒不可用，回退云端唤醒: %s", exc)
                self.config.wake_engine = "cloud"

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
        await self.stop_beat_dance()
        self._speech_pcm_active = False
        if self._motion is not None:
            stop_motion = getattr(self._motion, "stop_talk_motion", None)
            if callable(stop_motion):
                stop_motion(immediate=True)
        if self._task is not None:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        if self._location is not None:
            await self._location.stop()
        if self._audio is not None:
            observer_setter = getattr(self._audio, "set_playback_observer", None)
            if callable(observer_setter):
                observer_setter(None)
            self._playback_observer_registered = False
            await self._audio.close()
        if self._camera is not None:
            self._camera.close()
        logger.info("引擎已停止")

    # ── Main loop ──────────────────────────────────────────────────────
    async def _conversation_loop(self, *, startup_greeting: bool = True) -> None:
        """Main loop: listen → think → speak, repeat."""
        # Startup greeting
        if startup_greeting:
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
        if self._external_interaction_active:
            if self._state != "gesture":
                self._set_state("gesture")
            await asyncio.sleep(0.2)
            return
        # Beat-dance loop suspends all voice conversation until stopped.
        if self._dance_loop_active:
            # Re-assert "dancing": a turn interrupted by the dance may still
            # finish its finally and flip the dashboard back to idle.
            if self._state != "dancing":
                self._set_state("dancing")
            await asyncio.sleep(0.2)
            return
        # Waiting for a wake word can take tens of seconds and must not own
        # the response lock: typed Dashboard messages need to preempt that
        # passive microphone wait immediately.
        if self._turn_lock.locked():
            await asyncio.sleep(0.05)
            return
        await self._run_voice_turn()

    async def _run_voice_turn(self) -> None:
        """One full turn: listen → think → speak."""
        # ── 1. Listen ──
        self._set_state("listening")

        listen_task = asyncio.create_task(self._listen_for_speech())
        self._voice_listen_task = listen_task
        try:
            user_text = await listen_task
        except asyncio.CancelledError:
            # A Dashboard text turn cancels only the passive listener.  A
            # lifecycle cancellation (engine.stop / gesture takeover) must
            # still propagate and terminate the conversation loop.
            if not self._voice_listener_preempted:
                raise
            self._voice_listener_preempted = False
            return
        finally:
            if self._voice_listen_task is listen_task:
                self._voice_listen_task = None

        async with self._turn_lock:
            if not user_text:
                # During the 30s wake-free window, pause briefly before retrying
                # to avoid hammering the cloud ASR with rapid reconnections.
                if time.monotonic() < self._wake_word_active_until:
                    await asyncio.sleep(1.0)
                await self._maybe_recover_audio()
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
                self._wake_word_active_until = time.monotonic() + (
                    0.0
                    if self._off_axis_ambient
                    else (
                        self.config.followup_window_s
                        if self.config.audio_frontend_v2
                        else self.config.wake_word_timeout_s
                    )
                )
                self._off_axis_ambient = False

    async def _cancel_voice_listener(self) -> None:
        """Cancel a passive wake/ASR wait before a Dashboard text turn."""
        task = self._voice_listen_task
        if task is None or task.done() or task is asyncio.current_task():
            return
        self._voice_listener_preempted = True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # ── ASR: mic → text ────────────────────────────────────────────────
    async def _listen_for_speech(self) -> str:
        """Capture audio and stream to Bailian ASR. Returns final transcript.

        Wake word is detected locally (sherpa-onnx KWS) before the cloud ASR
        session is opened; the cloud transcript check in ``_accept_transcript``
        remains as the ``cloud`` engine fallback.  One duplex capture stream
        is shared by the wake/VAD gate and the ASR session so no audio is
        lost between wake-word detection and ASR connection.
        """
        if self._dance_loop_active or self._external_interaction_active:
            return ""
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

        capture = self._audio.start_capture().__aiter__()
        try:
            if self._needs_wake_word():
                self._set_state("wake_listening")
                pre_roll = await self._wait_for_wake_word(capture)
                if not pre_roll:
                    self._last_asr_end_reason = "wake_word_timeout"
                    return ""
                # Wake confirmed — refresh the grace window for follow-ups.
                self._wake_word_active_until = (
                    time.monotonic() + self.config.wake_word_timeout_s
                )
                self._set_state("listening")
                self._emit_asr_status("检测到唤醒词，正在连接识别服务")
                return await self._listen_cloud_asr(
                    capture=capture, initial_audio=pre_roll
                )

            initial_audio = await self._wait_for_voice_activity(capture)
            if not initial_audio:
                return ""

            self._set_state("listening")
            logger.info("🎤 [聆听] 检测到语音，建立 ASR 连接...")
            self._emit_asr_status("检测到语音，正在连接识别服务")
            return await self._listen_cloud_asr(
                capture=capture, initial_audio=initial_audio
            )
        finally:
            close_capture = getattr(capture, "aclose", None)
            if close_capture is not None:
                try:
                    await close_capture()
                except Exception:
                    logger.debug("listen capture close failed", exc_info=True)

    async def _maybe_recover_audio(self) -> None:
        """Restart the audio backend after consecutive silent timeouts.

        The SDK GStreamer recording pipeline can stall (appsink silently
        stops delivering samples) while playback keeps working — the engine
        would then loop through 20 s silent listen timeouts forever.  When
        samples stop flowing entirely (last sample older than 5 s), restart
        the backend (pipeline NULL → PLAYING) and try again.
        """
        age = getattr(self._audio, "last_sample_age_s", None)
        if age is not None and age < 5.0:
            self._silent_timeouts = 0  # samples still flowing — mic is alive
            return
        rms = float(getattr(self._audio, "capture_rms", 0.0) or 0.0)
        if age is None and rms > 0.01:
            self._silent_timeouts = 0  # no age probe (sounddevice), RMS alive
            return
        self._silent_timeouts += 1
        if self._silent_timeouts < 2:
            logger.warning(
                "🎤 麦克风无样本流入 (age=%.1fs, RMS=%.4f)，再观察一轮",
                age if age is not None else float("inf"),
                rms,
            )
            return
        self._silent_timeouts = 0
        logger.warning("🎤 麦克风连续两轮无样本流入，重启音频后端...")
        if self._audio is not None:
            try:
                await self._audio.close()
                await self._audio.open()
                logger.info("✅ 音频后端已重启")
            except Exception:
                logger.exception("音频后端重启失败")

    def _needs_wake_word(self) -> bool:
        """True when the local KWS gate must run before the cloud ASR."""
        return (
            self.config.enable_wake_word
            and self.config.wake_engine == "local"
            and self._wake is not None
            and time.monotonic() >= self._wake_word_active_until
        )

    async def _wait_for_wake_word(self, capture: Any) -> list[bytes]:
        """Run local KWS over the mic stream until the wake word is spoken.

        Returns up to ~1 s of pre-roll audio so the utterance that follows
        the wake word is preserved for the cloud ASR session.  Returns an
        empty list on timeout or stream end.
        """
        assert self._wake is not None
        self._emit_asr_status("等待唤醒词…（本地识别）")
        deadline = time.monotonic() + self.config.wake_listen_timeout_s
        pre_roll: deque[bytes] = deque(maxlen=10)  # ~1 s @ 100 ms chunks
        try:
            while time.monotonic() < deadline:
                remaining = max(0.01, min(0.2, deadline - time.monotonic()))
                try:
                    chunk = await asyncio.wait_for(anext(capture), timeout=remaining)
                except asyncio.TimeoutError:
                    continue
                except StopAsyncIteration:
                    break
                pre_roll.append(chunk)
                if self._wake.hit(chunk):
                    logger.info("🗣 [唤醒] 本地 KWS 命中，进入聆听")
                    get_doa = (
                        getattr(self._audio, "get_doa", None)
                        if self.config.audio_frontend_v2 and self.config.doa_enabled
                        else None
                    )
                    self._locked_doa = get_doa() if callable(get_doa) else None
                    self._audio_frontend_metrics["locked_doa"] = self._locked_doa
                    return list(pre_roll)
        finally:
            self._wake.reset()
        self._emit_asr_status("未检测到唤醒词")
        return []

    async def _wait_for_voice_activity(self, capture: Any) -> list[bytes]:
        """Wait locally for speech before opening a cloud ASR session.

        A short local gate avoids cloud recognition while nobody is speaking
        and retains up to 500 ms of pre-roll so the first syllable is not
        lost.  Consumes the caller's capture iterator.
        """
        assert self._audio is not None
        deadline = time.monotonic() + self.config.asr_initial_silence_timeout_s
        pre_roll: deque[bytes] = deque(maxlen=5)
        frontend = SpeechEndpoint(
            sample_rate=self.config.audio_sample_rate,
            model_path=self.config.vad_model_path,
            on_threshold=self.config.vad_on_threshold,
            off_threshold=self.config.vad_off_threshold,
            min_speech_ms=self.config.min_utterance_ms,
            silence_ms=self.config.endpoint_silence_ms,
            max_utterance_s=self.config.max_utterance_s,
        )
        legacy_voice_frames = 0
        self._emit_asr_status("等待语音活动（未连接云端识别）")
        while time.monotonic() < deadline:
            remaining = max(0.01, min(0.2, deadline - time.monotonic()))
            try:
                chunk = await asyncio.wait_for(anext(capture), timeout=remaining)
            except asyncio.TimeoutError:
                continue
            except StopAsyncIteration:
                break
            pre_roll.append(chunk)
            frame = frontend.update(chunk)
            backend_rms = float(getattr(self._audio, "capture_rms", 0.0) or 0.0)
            if (
                (not self.config.audio_frontend_v2 or not frontend.vad.available)
                and backend_rms >= self.config.voice_activity_threshold
            ):
                legacy_voice_frames += 1
            else:
                legacy_voice_frames = 0
            if frame.speech or legacy_voice_frames >= 2:
                logger.info(
                    "🎤 [本地 VAD] 检测到语音 RMS=%.4f SNR=%.1fdB P=%.2f",
                    frame.rms,
                    frame.snr_db,
                    frame.vad_probability,
                )
                return list(pre_roll)
        self._last_asr_end_reason = "initial_silence_timeout"
        rms = float(getattr(self._audio, "capture_rms", 0.0) or 0.0)
        backend = getattr(self._audio, "backend_name", "?")
        logger.info(
            "🎤 [聆听] %.0fs 未检测到语音 (RMS=%.4f, backend=%s)",
            self.config.asr_initial_silence_timeout_s,
            rms,
            backend,
        )
        self._emit_asr_status("未检测到用户开口（未连接云端识别）")
        return []

    async def _listen_cloud_asr(
        self,
        capture: Any,
        initial_audio: list[bytes] | None = None,
    ) -> str:
        """Stream mic audio to Bailian realtime ASR via WebSocket.

        ``capture`` is the duplex capture iterator opened by the caller
        (shared with the local VAD / wake-word gate); audio arriving while
        the ASR connection is established is buffered in ``audio_queue`` so
        nothing spoken after the wake word is lost.

        Server-side VAD (``turn_detection.server_vad``) detects
        speech-start / speech-stop and the ASR service returns the
        final transcript once the user stops speaking.

        Periodic diagnostic logging helps identify audio-quality issues
        (clipping, low level, dropped frames) without external tools.
        """
        assert self._audio is not None

        audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=256)
        for chunk in initial_audio or []:
            try:
                audio_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                break
        capture_done = asyncio.Event()
        local_endpoint = asyncio.Event()
        feed_done = asyncio.Event()
        dropped_frames = 0
        speech_start_time: float | None = None
        _last_level_log = 0.0
        _peak_since_last_log = 0.0
        _clip_since_last_log = 0
        finish_sent = False
        endpoint = SpeechEndpoint(
            sample_rate=self.config.audio_sample_rate,
            model_path=self.config.vad_model_path,
            on_threshold=self.config.vad_on_threshold,
            off_threshold=self.config.vad_off_threshold,
            min_speech_ms=self.config.min_utterance_ms,
            silence_ms=self.config.endpoint_silence_ms,
            max_utterance_s=self.config.max_utterance_s,
        )
        direction = DirectionGate(
            self.config.doa_tolerance_deg, self.config.doa_mismatch_ms
        )
        direction.lock(self._locked_doa)

        async def _capture() -> None:
            nonlocal \
                dropped_frames, \
                _last_level_log, \
                _peak_since_last_log, \
                _clip_since_last_log, \
                speech_start_time
            async for chunk in capture:
                if capture_done.is_set():
                    break
                if self._listening_is_blocked():
                    continue
                doa_getter = (
                    getattr(self._audio, "get_doa", None)
                    if self.config.audio_frontend_v2 and self.config.doa_enabled
                    else None
                )
                doa = doa_getter() if callable(doa_getter) else None
                accepted_direction = (
                    direction.accepts(doa) if self.config.audio_frontend_v2 else True
                )
                gated_chunk = chunk if accepted_direction else bytes(len(chunk))
                frame = endpoint.update(gated_chunk)
                if frame.speech and speech_start_time is None:
                    speech_start_time = time.monotonic()
                if direction.muted and self.config.audio_frontend_v2:
                    self._off_axis_ambient = True
                    self._wake_word_active_until = 0.0
                self._audio_frontend_metrics = {
                    "vad_probability": round(frame.vad_probability, 3),
                    "rms": round(frame.rms, 5),
                    "dbfs": round(frame.dbfs, 1),
                    "snr_db": round(frame.snr_db, 1),
                    "doa": None if doa is None else round(float(doa), 1),
                    "locked_doa": direction.locked_doa,
                    "direction_muted": direction.muted,
                    "silero_loaded": endpoint.vad.available,
                    "endpoint_reason": frame.endpoint_reason,
                }
                try:
                    audio_queue.put_nowait(gated_chunk)
                except asyncio.QueueFull:
                    dropped_frames += 1
                if frame.endpoint and self.config.audio_frontend_v2:
                    self._last_asr_end_reason = frame.endpoint_reason
                    logger.info("🎤 [本地端点] %s", frame.endpoint_reason)
                    local_endpoint.set()
                    break

                # ── Audio quality diagnostics (every ~5 s) ──────────
                now = time.monotonic()
                if now - _last_level_log >= 5.0:
                    _last_level_log = now
                    rms = self._audio.capture_rms if self._audio else 0.0
                    db = round(20.0 * math.log10(max(rms, 1e-8)), 1)
                    logger.info(
                        "🎤 [诊断] 电平 %5.1f dBFS | 丢帧 %d | 削波 %d",
                        db,
                        dropped_frames,
                        _clip_since_last_log,
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
                    nonlocal finish_sent
                    while not capture_done.is_set():
                        try:
                            chunk = await asyncio.wait_for(
                                audio_queue.get(), timeout=0.1
                            )
                            if self._listening_is_blocked():
                                continue
                            await asr.send_audio(chunk)
                        except asyncio.TimeoutError:
                            if local_endpoint.is_set() and audio_queue.empty():
                                await asr.finish()
                                finish_sent = True
                                feed_done.set()
                                return
                            continue

                feed_task = asyncio.create_task(_feed_asr())

                results = asr.results().__aiter__()
                result_task: asyncio.Task | None = None
                stable_partial = ""
                stable_count = 0
                try:
                    while True:
                        if speech_start_time is None:
                            result_timeout = self.config.asr_initial_silence_timeout_s
                            timeout_reason = "initial_silence_timeout"
                        else:
                            result_timeout = max(
                                0.0,
                                self.config.asr_speech_max_duration_s
                                - (time.monotonic() - speech_start_time),
                            )
                            timeout_reason = "speech_max_duration_timeout"
                        try:
                            if result_task is None:
                                result_task = asyncio.create_task(anext(results))
                            poll_timeout = min(max(0.01, result_timeout), 0.2)
                            result = await asyncio.wait_for(
                                asyncio.shield(result_task), timeout=poll_timeout
                            )
                            result_task = None
                        except asyncio.TimeoutError:
                            if feed_done.is_set() and result_task is not None:
                                try:
                                    result = await asyncio.wait_for(
                                        asyncio.shield(result_task),
                                        timeout=self.config.asr_finalize_timeout_s,
                                    )
                                    result_task = None
                                except asyncio.TimeoutError:
                                    self._last_asr_end_reason = "finalize_timeout"
                                    if stable_count >= 2:
                                        final_text = stable_partial
                                        logger.warning(
                                            "ASR final 超时，提交稳定 partial: %r",
                                            final_text,
                                        )
                                    break
                            elif result_timeout > 0.2:
                                continue
                            else:
                                if result_task is not None:
                                    result_task.cancel()
                            self._last_asr_end_reason = timeout_reason
                            message = (
                                "未检测到用户开口"
                                if timeout_reason == "initial_silence_timeout"
                                else "未检测到完整语句，请缩短后重新说一次"
                            )
                            logger.warning(
                                "ASR %s (initial=%.1fs, speech_max=%.1fs, "
                                "queue=%d, dropped=%d)",
                                timeout_reason,
                                self.config.asr_initial_silence_timeout_s,
                                self.config.asr_speech_max_duration_s,
                                audio_queue.qsize(),
                                dropped_frames,
                            )
                            self._emit_asr_status(message)
                            break
                        except StopAsyncIteration:
                            self._last_asr_end_reason = "result_stream_closed"
                            self._emit_asr_status("语音识别连接已结束，请重试")
                            break

                        if result.error:
                            self._last_asr_end_reason = "asr_error"
                            self._emit_asr_status("语音识别失败，请重试")
                            raise RuntimeError(result.error)
                        if result.speech_started:
                            speech_count += 1
                            speech_start_time = time.monotonic()
                            self._emit_asr_status("正在聆听，请继续说完")
                            if (
                                self.config.barge_in_enabled
                                and self._tts_playing
                                and self._barge_in
                            ):
                                logger.info("⏸ [打断] 用户开始说话，停止播放")
                                self._audio.stop_playback()
                                self._stop_talk_motion_immediately()
                                self._tts_playing = False
                                self._barge_in_occurred = True

                        if not result.is_final and result.text:
                            if result.text == stable_partial:
                                stable_count += 1
                            else:
                                stable_partial = result.text
                                stable_count = 1
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
                            self._last_asr_end_reason = "completed"
                            if self._on_transcript:
                                self._on_transcript(result.text, True)
                            break
                finally:
                    capture_done.set()
                    if result_task is not None and not result_task.done():
                        result_task.cancel()
                    feed_task.cancel()
                    try:
                        await feed_task
                    except asyncio.CancelledError:
                        pass

                if not finish_sent:
                    await asr.finish()

                # ── Empty-result diagnostics ─────────────────────────
                if not final_text.strip():
                    rms = self._audio.capture_rms if self._audio else 0.0
                    db = round(20.0 * math.log10(max(rms, 1e-8)), 1)
                    logger.info(
                        "🎤 [ASR] 无识别结果 | 电平 %5.1f dBFS | "
                        "speech_count=%d | dropped=%d — "
                        "%s",
                        db,
                        speech_count,
                        dropped_frames,
                        "可能是无人说话"
                        if speech_count == 0
                        else "有语音但未识别出文字",
                    )
                else:
                    logger.info(
                        "🎤 [ASR] 完整转写 chars=%d speech=%.1fs "
                        "vad_silence=%dms queue=%d dropped=%d",
                        len(final_text.strip()),
                        (time.monotonic() - speech_start_time)
                        if speech_start_time
                        else 0.0,
                        self.config.asr_vad_silence_ms,
                        audio_queue.qsize(),
                        dropped_frames,
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
        if not self.config.source_router_v2_enabled:
            if decision.intent == TurnIntent.JOURNEY_RECALL:
                decision = IntentDecision(TurnIntent.JOURNAL, "V2 source router disabled")
            elif decision.intent in {TurnIntent.ORG_KNOWLEDGE, TurnIntent.LOCATION_UPDATE}:
                decision = IntentDecision(TurnIntent.GENERAL, "V2 source router disabled")
        elif (
            decision.intent == TurnIntent.LOCATION_UPDATE
            and not self.config.session_state_v2_enabled
        ):
            decision = IntentDecision(TurnIntent.GENERAL, "V2 session state disabled")
        turn_plan = plan_turn(
            text,
            decision,
            self.config.search_policy,
            self.config.vision_policy,
        )
        self._active_search_mode = turn_plan.search_mode
        self._active_response_tools = []
        self._active_response_tool_handler = None
        self._current_visual_observation = None
        self._last_search_status = {
            "mode": turn_plan.search_mode,
            "used": False,
            "error": "",
            "source_count": 0,
        }
        self._last_vision_status = {
            "policy": self.config.vision_policy,
            "decision": turn_plan.vision_mode,
            "called": False,
            "reason": (
                "semantic candidate (shadow only)"
                if turn_plan.vision_shadow
                else turn_plan.reason
            ),
            "scope": turn_plan.camera_scope if turn_plan.vision_mode != "off" else None,
            "observed_at": None,
            "age_s": None,
            "capture_latency_ms": None,
            "vlm_latency_ms": None,
            "total_latency_ms": None,
            "error": "",
        }
        journal_context = ""
        vision_context = ""
        location_context = self._session_location_context()
        organization_context = ""
        reply = ""
        emotion = ""
        error: str | None = None
        journal_denied = False
        location_tool_required = False
        journal_intents = {TurnIntent.JOURNAL, TurnIntent.JOURNEY_RECALL}

        async def _handle_visual_tool(
            name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            nonlocal vision_context
            if name != "observe_scene":
                return {"ok": False, "error": "不允许的工具"}
            scope = str(arguments.get("scope") or "front").strip().lower()
            if scope not in {"front", "rear"}:
                return {"ok": False, "error": "scope 必须是 front 或 rear"}
            # The deterministic planner owns camera direction. A model cannot
            # silently turn a front-view question into an exterior capture.
            if turn_plan.camera_scope == "rear":
                scope = "rear"
            elif decision.intent != TurnIntent.REAR_CAMERA:
                scope = "front"
            focus = str(arguments.get("focus") or text).strip()[:300]
            observation = await self._observe_scene(
                scope,
                focus=focus,
                fallback_image=image_bytes,
                prefer_cache=self._should_reuse_visual(text, scope),
            )
            if not self._last_vision_status.get("called"):
                self._last_vision_status.update(
                    {
                        "called": True,
                        "scope": observation.scope,
                        "observed_at": observation.captured_at,
                        "age_s": 0.0,
                        "error": observation.error,
                    }
                )
            self._current_visual_observation = observation
            vision_context = observation.to_prompt()
            self._register_visual_source(observation)
            return observation.to_dict()

        try:
            if turn_plan.vision_mode == "required":
                self._emit_turn_event("turn_status", status="observing")
                observation = await self._observe_scene(
                    turn_plan.camera_scope,
                    focus=text,
                    fallback_image=image_bytes,
                    prefer_cache=self._should_reuse_visual(text, turn_plan.camera_scope),
                )
                self._current_visual_observation = observation
                vision_context = observation.to_prompt()
                self._register_visual_source(observation)
                if not observation.ok:
                    reply = observation.error or "我现在没看清楚，可以再让我看一次吗？"
            elif (
                turn_plan.vision_mode == "auto"
                and not turn_plan.vision_shadow
                and self.config.vision_policy == "semantic"
            ):
                self._active_response_tools = [_OBSERVE_SCENE_TOOL]
                self._active_response_tool_handler = _handle_visual_tool
            elif turn_plan.vision_shadow:
                logger.info("视觉 shadow 命中但不采集: %s", text)
            elif decision.intent in {
                TurnIntent.FRONT_CAMERA,
                TurnIntent.REAR_CAMERA,
                TurnIntent.AMBIGUOUS_CAMERA,
            }:
                reply = "我现在暂时看不到前面。"

            if decision.intent in {
                TurnIntent.FRONT_CAMERA,
                TurnIntent.REAR_CAMERA,
                TurnIntent.AMBIGUOUS_CAMERA,
            }:
                pass
            elif decision.intent == TurnIntent.LOCATION_UPDATE:
                place = extract_location_update(text)
                if not place:
                    reply = "我没听清具体地点，你可以说“我们现在在某某地方”。"
                else:
                    self._set_session_location(place)
                    location_context = self._session_location_context()
                    reply = (
                        f"记下啦：你说现在在{place}。"
                        "我会把它作为本次会话备注，但不会用它覆盖实时定位。"
                    )
            elif decision.intent == TurnIntent.LOCATION:
                self._emit_turn_event("turn_status", status="retrieving")
                location_tool_required = True
            elif decision.intent in journal_intents:
                self._emit_turn_event("turn_status", status="retrieving")
                journal_context = await self._verified_journal_context(text)
                if journal_context and is_journey_overview_query(text):
                    format_overview = getattr(
                        self._memory, "format_journey_overview", None
                    )
                    if callable(format_overview):
                        reply = str(format_overview() or "")
                elif not journal_context:
                    # Let the model explain the precise evidence gap naturally
                    # instead of bypassing it with the same canned sentence.
                    journal_denied = True
                    if not self.config.bailian_api_key and _requires_journal_evidence(text):
                        # Offline diagnostics/tests have no generator available;
                        # keep a truthful fallback without attempting the network.
                        reply = _JOURNAL_UNKNOWN
            elif decision.intent == TurnIntent.ORG_KNOWLEDGE:
                organization_context = self.config.organization_knowledge()
                if organization_context:
                    self._current_sources.extend(
                        [
                            {
                                "type": "organization",
                                "title": "柴火创客官方介绍",
                                "url": "https://www.chaihuo.org/space/list",
                            },
                            {
                                "type": "organization",
                                "title": "柴火基地车官网",
                                "url": "https://mcv.chaihuo.org/",
                            },
                        ]
                    )
            elif decision.intent == TurnIntent.MOTION:
                reply = await self._execute_deterministic_motion(text)
            elif (
                decision.intent == TurnIntent.GENERAL
                and re.sub(r"[\s，。！？!?、～~]+", "", text).lower() in _SHORT_ACKS
            ):
                reply = "嗯嗯，我在呢。你慢慢说～"

            if not reply:
                system_prompt = _build_system_prompt(
                    self.config,
                    journal_context,
                    inject_location=location_context,
                    include_location=not is_journey_overview_query(text),
                )
                if journal_denied:
                    system_prompt += (
                        "\n\n【提示】本轮未能从基地车日记检索到相关记录。"
                        "不要拿无关地点、其他高校、组织背景或常识补写我们的经历。"
                        "请自然说明日记覆盖缺口；如果问题还包含可由常识回答的部分，"
                        "可以明确区分来源后提供有帮助的信息。"
                    )
                if organization_context:
                    system_prompt += (
                        f"\n\n【柴火创客官方知识】\n{organization_context}\n"
                        "回答组织介绍时可以展开；不得把这些背景冒充成具体旅途经历。"
                    )
                if vision_context:
                    system_prompt += f"\n\n【本轮视觉观察】\n{vision_context}"
                elif self._active_response_tools:
                    system_prompt += (
                        "\n\n【按需视觉工具】本轮问题可能依赖此刻可见事实。"
                        "只有确实需要看当前的人、物、动作、数量、颜色或文字时，"
                        "调用 observe_scene；能用常识或对话上下文回答就不要调用。"
                        "工具成功后直接自然回答，不得暴露拍摄或识别机制。"
                    )
                if turn_plan.search_mode != "off":
                    system_prompt += (
                        "\n\n【联网规则】你可以使用联网工具核对公开信息。"
                        "只有实际调用联网工具时，才在回答开头简短说“已联网核对”；"
                        "不要朗读网址。未联网时不要声称已核对。"
                    )
                # 本回合刚跳完舞: 注入即兴接话提示, 不说风格
                if (
                    self._pending_dance
                    and self._pending_dance.get("turn_id") == self._current_turn_id
                ):
                    system_prompt += _DANCE_REPLY_INJECTION
                    self._pending_dance = None
                messages: list[dict[str, str]] = [
                    {"role": "system", "content": system_prompt},
                ]
                # Prior assistant replies remain conversational context for
                # general questions, but protected journal facts are rebuilt
                # solely from this turn's verified source material.
                if decision.intent not in journal_intents:
                    messages.extend(self._conversation_history)
                messages.append({"role": "user", "content": text})

                if location_tool_required:
                    messages = await self._prepare_location_tool_messages(messages)

                self._set_state("thinking")
                if speak and self._audio is not None:
                    reply, emotion = await self._think_and_speak(messages)
                else:
                    reply, emotion = await self._think_text_only(messages)
                if self._current_visual_observation is not None:
                    vision_context = self._current_visual_observation.to_prompt()
                if turn_plan.vision_mode == "auto" and not (
                    self._current_visual_observation
                    and self._current_visual_observation.ok
                ):
                    reply = (
                        self._current_visual_observation.error
                        if self._current_visual_observation
                        else "我现在还没看清楚，把想让我看的东西放到我面前一点吧。"
                    )
                if (
                    self._current_visual_observation
                    and self._current_visual_observation.ok
                    and _VISUAL_MECHANISM_RE.search(reply)
                ):
                    logger.warning(
                        "Replaced mechanism-heavy visual reply: %s", reply
                    )
                    reply = _natural_visual_reply(self._current_visual_observation)
                if (
                    decision.intent == TurnIntent.GENERAL
                    and not (
                        self._current_visual_observation
                        and self._current_visual_observation.ok
                    )
                    and _UNGROUNDED_VISUAL_CLAIM_RE.search(reply)
                ):
                    logger.warning(
                        "Blocked ungrounded visual claim in general reply: %s",
                        reply,
                    )
                    reply = (
                        "我现在还没有真正看清楚，不能假装知道眼前是什么。"
                        "你可以把想让我看的东西放到我面前。"
                    )
                elif decision.intent in journal_intents and journal_context:
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
                self._record_history(text, reply, vision_context=vision_context)
                self._last_topic = decision.intent.value
                self._last_evidence_ids = [
                    str(item.get("slug") or item.get("title") or "")
                    for item in self._current_sources
                    if item.get("slug") or item.get("title")
                ]
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
            self._active_response_tools = []
            self._active_response_tool_handler = None
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

    def _record_history(
        self, user_text: str, reply: str, vision_context: str = ""
    ) -> None:
        self._conversation_history.append({"role": "user", "content": user_text})
        assistant_content = reply
        observation = self._current_visual_observation
        if vision_context and observation and observation.ok:
            # Keep only compact structured facts in conversational memory. The
            # JPEG remains in the short-lived in-memory cache and is never put
            # into model history or written to disk here.
            assistant_content += f"\n【最近视觉观察】{observation.to_prompt()}"
        self._conversation_history.append(
            {"role": "assistant", "content": assistant_content}
        )
        max_messages = self.config.max_history_turns * 2
        if len(self._conversation_history) > max_messages:
            self._conversation_history = self._conversation_history[-max_messages:]

    async def _think_text_only(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        parts: list[str] = []
        async with BailianLLMClient(self.config) as llm:
            it = llm.response_stream(
                messages,
                search_mode=self._active_search_mode,
                tools=self._active_response_tools or None,
                tool_handler=self._active_response_tool_handler,
                max_tool_rounds=self.config.vision_max_calls_per_turn,
            ).__aiter__()
            try:
                while True:
                    try:
                        token = await asyncio.wait_for(anext(it), timeout=120.0)
                    except StopAsyncIteration:
                        break
                    parts.append(token)
                    if self._on_llm_token:
                        self._on_llm_token(token)
                    self._emit_turn_event("chat_message_delta", delta=token)
            except asyncio.TimeoutError:
                logger.error("LLM 流式生成超时 (text-only)，返回部分结果")
            self._merge_llm_search_status(llm)
        raw = "".join(parts)
        emotion = _extract_emotion(raw)
        return _TAG_RE.sub("", raw).strip(), emotion

    def _merge_llm_search_status(self, llm: BailianLLMClient) -> None:
        """Expose sanitized web evidence for the Dashboard and turn result."""
        sources = [
            {
                "type": "web",
                "title": item["title"],
                "url": item["url"],
                "retrieved_at": item["retrieved_at"],
            }
            for item in llm.last_sources
        ]
        known = {str(item.get("url") or "") for item in self._current_sources}
        self._current_sources.extend(
            item for item in sources if str(item.get("url") or "") not in known
        )
        self._last_search_status = {
            "mode": self._active_search_mode,
            "used": llm.last_search_used,
            "error": llm.last_search_error,
            "source_count": len(sources),
            "retrieved_at": int(time.time()) if llm.last_search_used else None,
        }

    async def _verified_journal_context(self, query: str) -> str:
        """Online-check the corpus, revalidate candidates, then return evidence."""
        from chaihuo_reachy.memory.journal_fetcher import journal_sync_lock

        if self._memory is None:
            return ""
        # Respect the cross-process sync lock (systemd timer / dashboard
        # auto-sync): skip this tick when another sync owns the corpus.
        with journal_sync_lock(self.config.journal_cache_dir) as acquired:
            if acquired:
                try:
                    await self._journal_fetcher.sync(memory_store=self._memory)
                except Exception:
                    health = self._journal_fetcher.health()
                    if not health["complete"]:
                        logger.warning(
                            "No verified journal cache is available", exc_info=True
                        )
                        return ""
                    logger.warning(
                        "Journal directory is partially unavailable; continuing with "
                        "%d individually verified cached entries (%d expected)",
                        health["complete"],
                        health["expected"],
                    )

        target_date = _extract_target_date(query)
        recent_days = _extract_recent_window(query)
        journey_overview = is_journey_overview_query(query)
        search_journey_overview = getattr(
            self._memory, "search_journey_overview", None
        )
        overview_results = (
            search_journey_overview(k=80)
            if journey_overview and callable(search_journey_overview)
            else []
        )
        search_journey_scope = getattr(self._memory, "search_journey_scope", None)
        journey_scope = (
            search_journey_scope(query, k=6)
            if not journey_overview
            and not target_date
            and not recent_days
            and callable(search_journey_scope)
            else []
        )
        results = overview_results or journey_scope or (
            self._memory.search_by_date(target_date, k=3)
            if target_date
            else (
                self._memory.search_recent(recent_days, k=6)
                if recent_days
                else (
                    _keyword_search(self._memory, query, k=6)
                    or self._memory.search(query, k=self.config.memory_top_k)
                )
            )
        )
        if not results:
            return ""

        # search_by_date only returns manifest entries which cover target_date,
        # including multi-day title ranges whose primary ``date`` differs.
        exact_date = bool(target_date and results)
        if (
            not journey_overview
            and not exact_date
            and float(results[0].get("score") or 0)
            < self.config.journal_relevance_threshold
        ):
            return ""

        candidate_slugs = [str(r.get("slug") or r.get("id") or "") for r in results]
        candidate_slugs = [slug for slug in candidate_slugs if slug]
        if not journey_overview:
            with journal_sync_lock(self.config.journal_cache_dir) as acquired:
                if not acquired:
                    candidate_slugs = []  # busy: answer from the verified cache
            try:
                await self._journal_fetcher.sync(
                    memory_store=self._memory,
                    refresh_slugs=candidate_slugs,
                )
                journey_scope = (
                    search_journey_scope(query, k=6)
                    if not target_date
                    and not recent_days
                    and callable(search_journey_scope)
                    else []
                )
                results = journey_scope or (
                    self._memory.search_by_date(target_date, k=3)
                    if target_date
                    else (
                        self._memory.search_recent(recent_days, k=6)
                        if recent_days
                        else (
                            _keyword_search(self._memory, query, k=6)
                            or self._memory.search(
                                query, k=self.config.memory_top_k
                            )
                        )
                    )
                )
            except Exception:
                logger.warning(
                    "Candidate journal revalidation was partial; using its "
                    "individually verified cached copy"
                )

        selected = (
            results
            if journey_overview
            else (results[:1] if exact_date else results[: 6 if journey_scope else 3])
        )
        health = self._journal_fetcher.health()
        cache_note = (
            f"缓存最后完整校验：{health['last_success_at']}。"
            if health.get("failures")
            else "本轮已在线校验官方目录和候选正文。"
        )
        blocks = [cache_note]
        if journey_overview:
            blocks.append(
                f"这是历史旅程足迹汇总，不是当前位置查询。以下 {len(selected)} 篇"
                "已验证日记标题覆盖目前可确认的完整路线。请按时间阶段概括主要省区、"
                "城市和代表性站点；合并重复地点，不要回答‘我们现在在哪里’，也不要"
                "把当前定位混入历史路线。"
            )
        elif target_date:
            relative_word = _relative_date_word(query)
            label = f"（用户原话：{relative_word}）" if relative_word else ""
            blocks.append(
                f"程序已确定本题目标日期：{_format_chinese_date(target_date)}{label}。"
                "这是确定性日历计算；回答时不得改成其他日期。"
            )
        elif recent_days:
            blocks.append(
                f"程序已按用户时间词检索最近 {recent_days} 天内的日记（确定性日期窗口）。"
                "回答必须基于以下这些日记的内容；日期以日记原文为准。"
            )
        for item in selected:
            source = {
                "type": "journey",
                "slug": item.get("slug") or item.get("id"),
                "title": item.get("title", ""),
                "date": item.get("date", ""),
                "url": item.get("source_url", ""),
                "source_updated_at": item.get("source_updated_at", ""),
                "score": round(float(item.get("score") or 0.0), 3),
            }
            self._current_sources.append(source)
            if journey_overview:
                blocks.append(
                    f"{source['date'] or '日期未知'}｜{source['title']}"
                )
                continue
            evidence = str(item.get("snippet") or "").strip()
            if not evidence:
                evidence = str(item.get("content") or "")[:3600 if exact_date else 2200]
            blocks.append(
                f"日期：{source['date'] or '未知'}\n"
                f"标题：{source['title']}\n"
                f"来源：{source['url']}\n"
                f"匹配证据片段：\n{evidence}"
            )
        return "\n\n---\n\n".join(blocks)

    async def _execute_deterministic_motion(self, text: str) -> str:
        # 未指定风格 → 随机挑一个, 用户以为机器人自己即兴编排
        style = "random"
        if "机械" in text or "robot" in text:
            style = "robot"
        elif "摇摆" in text or "swing" in text or "慢" in text or "温柔" in text:
            style = "swing"
        elif "优雅" in text or "轻柔" in text or "舒缓" in text or "华尔兹" in text:
            style = "elegant"
        elif "动感" in text or "蹦迪" in text or "funky" in text:
            style = "funky"
        elif "搞怪" in text or "搞笑" in text or "silly" in text:
            style = "silly"
        elif "欢快" in text or "开心" in text or "高兴" in text:
            style = "happy"
        elif "随便" in text or "随机" in text or "随意" in text or "自由" in text:
            style = "random"
        if "跳" in text or "舞" in text:
            return await self._tool_dance(style)
        if "点头" in text:
            return await self._tool_gesture("nod", 2)
        if "摇头" in text:
            return await self._tool_gesture("shake_head", 2)
        if "挥" in text or "招呼" in text:
            return await self._tool_gesture("wave", 1)
        return await self._tool_pose(
            "sleep" if "睡" in text or "休息" in text else "wake_up"
        )

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
        # Do not interpret microphone noise as an interrupt while the LLM is
        # still thinking. The first actual TTS PCM callback opens this gate.
        await self._tts_audio_started.wait()

        threshold = max(0.01, min(0.95, self.config.barge_in_sensitivity))
        holdoff_s = 0.3  # ignore initial speaker echo

        start = time.monotonic()
        loud_frames = 0
        while not self._barge_in_requested:
            try:
                # Dance owns the audio: barge-in must never kill the music.
                if self._dance_loop_active:
                    return
                # Grace period — don't barge-in on the first ~300ms of TTS output
                if time.monotonic() - start < holdoff_s:
                    await asyncio.sleep(0.1)
                    continue

                rms = self._audio.capture_rms
                if rms > threshold:
                    loud_frames += 1
                    if loud_frames >= 6:  # ~300 ms, filters short echo spikes
                        self._barge_in_requested = True
                        logger.info(
                            "🗣 [打断] 检测到持续语音 RMS=%.4f (阈值=%.3f) — 停止当前回答",
                            rms,
                            threshold,
                        )
                        self._audio.stop_playback()
                        self._stop_talk_motion_immediately()
                        self._barge_in_occurred = True
                        return
                else:
                    loud_frames = 0
            except Exception:
                pass
            await asyncio.sleep(0.05)

    # ── LLM → TTS ──────────────────────────────────────────────────────
    async def _think_and_speak(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        """Run LLM and stream tokens to TTS. Returns (full_text, emotion)."""
        assert self._audio is not None
        if self._dance_loop_active:
            logger.info("🎵 连跳进行中，跳过本轮回答")
            return "", ""

        # Set TTS output rate (Qwen Realtime = 24kHz)
        self._audio.set_output_sample_rate(self.config.tts_sample_rate)

        full_parts: list[str] = []
        emotion = ""
        tts_text_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
        llm_done_flag = asyncio.Event()
        tts_done_flag = asyncio.Event()

        # Reset barge-in for this turn and monitor the AEC input while TTS is
        # active.  The watcher stops PCM immediately; the stream checks the
        # flag before queuing any additional TTS chunks.
        self._barge_in_requested = False
        self._tts_audio_started.clear()
        self._tts_generation += 1
        generation = self._tts_generation
        self._active_tts_generation = generation

        async with self._speaking_scope():
            barge_watcher = (
                asyncio.create_task(self._watch_barge_in())
                if self.config.barge_in_enabled
                else None
            )

            async with BailianLLMClient(self.config) as llm:
                # TTS worker
                async def _tts_worker() -> None:
                    tts = BailianTTSClient(
                        self.config,
                        on_audio=lambda pcm, sample_rate: self._queue_tts_audio(
                            pcm, sample_rate, generation
                        ),
                        on_done=tts_done_flag.set,
                    )
                    # Do not open a realtime TTS session until the LLM has
                    # produced actual text.  Calling finish() on a session
                    # that never received text makes Qwen wait for its full
                    # 60-second timeout and leaves the Dashboard looking
                    # permanently stuck on "thinking".
                    opened = False
                    fed_text = False
                    try:
                        while not llm_done_flag.is_set() or not tts_text_queue.empty():
                            if self._barge_in_requested:
                                break
                            try:
                                text = await asyncio.wait_for(
                                    tts_text_queue.get(), timeout=0.2
                                )
                                if self._barge_in_requested:
                                    break
                                if not opened:
                                    await tts.open()
                                    opened = True
                                await tts.feed(text)
                                fed_text = True
                            except asyncio.TimeoutError:
                                continue
                        if fed_text and not self._barge_in_requested:
                            await tts.flush()
                    except Exception:
                        logger.exception("TTS worker error")
                        tts_done_flag.set()
                    finally:
                        if opened:
                            await tts.close()

                tts_task = asyncio.create_task(_tts_worker())

                async def _consume_llm() -> None:
                    token_buf = ""
                    async for token in llm.response_stream(
                        messages,
                        search_mode=self._active_search_mode,
                        tools=self._active_response_tools or None,
                        tool_handler=self._active_response_tool_handler,
                        max_tool_rounds=self.config.vision_max_calls_per_turn,
                    ):
                        if self._barge_in_requested:
                            return

                        full_parts.append(token)
                        if self._on_llm_token:
                            self._on_llm_token(token)
                        self._emit_turn_event("chat_message_delta", delta=token)

                        token_buf += token
                        if (
                            any(
                                token_buf.rstrip().endswith(p)
                                for p in ("。", "！", "？", ".", "!", "?", "\n")
                            )
                            or len(token_buf) >= 40
                        ):
                            clean = _TAG_RE.sub("", token_buf.strip())
                            if clean:
                                await tts_text_queue.put(clean)
                            token_buf = ""

                    if token_buf.strip() and not self._barge_in_requested:
                        clean = _TAG_RE.sub("", token_buf.strip())
                        if clean:
                            await tts_text_queue.put(clean)

                llm_task = asyncio.create_task(_consume_llm())
                try:
                    # Hard ceiling on LLM streaming so a hung upstream never
                    # leaves the dashboard stuck on "thinking" forever.
                    if barge_watcher is None:
                        await asyncio.wait_for(llm_task, timeout=120.0)
                    else:
                        done, _ = await asyncio.wait_for(
                            asyncio.wait(
                                (llm_task, barge_watcher),
                                return_when=asyncio.FIRST_COMPLETED,
                            ),
                            timeout=120.0,
                        )
                except asyncio.TimeoutError:
                    logger.error("LLM 流式生成超时，终止当前回合")
                    llm_task.cancel()
                    try:
                        await llm_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    done, _ = await asyncio.wait(
                        (
                            llm_task,
                            barge_watcher or asyncio.create_task(asyncio.sleep(0)),
                        ),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if (
                        barge_watcher is not None
                        and barge_watcher in done
                        and self._barge_in_requested
                    ):
                        logger.info("⏸ [打断] 取消 LLM 流 — 用户开始说话")
                        self._audio.stop_playback()
                        llm_task.cancel()
                        try:
                            await llm_task
                        except asyncio.CancelledError:
                            pass
                    elif barge_watcher is not None:
                        await llm_task
                except Exception:
                    logger.exception("LLM → TTS pipeline error")
                    if not llm_task.done():
                        llm_task.cancel()
                    try:
                        await llm_task
                    except asyncio.CancelledError:
                        pass
                finally:
                    llm_done_flag.set()
                    try:
                        await tts_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.debug(
                            "TTS worker ended during turn cleanup", exc_info=True
                        )
                    if barge_watcher is not None:
                        barge_watcher.cancel()
                        try:
                            await barge_watcher
                        except asyncio.CancelledError:
                            pass
                self._merge_llm_search_status(llm)

            if self._active_tts_generation == generation:
                self._active_tts_generation = None

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

    def _queue_tts_audio(
        self, pcm: bytes, sample_rate: int, generation: int | None = None
    ) -> None:
        """Bridge sync TTS callback (from SDK thread) → async audio playback."""
        if self._audio is None or self._loop is None:
            return
        if self._dance_loop_active:
            logger.debug("Dropping TTS PCM while beat dance active")
            return
        if generation is not None and generation != self._active_tts_generation:
            logger.debug("Dropping late TTS PCM for generation %d", generation)
            return
        if generation is not None:
            self._loop.call_soon_threadsafe(self._tts_audio_started.set)
        future = asyncio.run_coroutine_threadsafe(
            self._play_tts_audio(pcm, sample_rate, generation), self._loop
        )
        self._pending_playbacks.add(future)
        future.add_done_callback(self._pending_playbacks.discard)

    async def _play_tts_audio(
        self, pcm: bytes, sample_rate: int, generation: int | None = None
    ) -> None:
        """Queue conversational PCM; the speaker observer drives motion."""
        assert self._audio is not None
        if generation is not None and generation != self._active_tts_generation:
            return
        if self._dance_loop_active:
            return  # music owns the buffer — never interleave TTS
        self._audio.set_output_sample_rate(sample_rate)
        # Start motion before exposing PCM to the speaker buffer.  This closes
        # the first-chunk race where the status could report audible playback
        # for one poll while the talk-motion task had not started yet.
        if (
            not self._speech_pcm_active
            and self._motion is not None
            and self.config.wobbling_enabled
        ):
            start_motion = getattr(self._motion, "start_talk_motion", None)
            if callable(start_motion):
                start_motion()
        self._speech_pcm_active = True
        await self._audio.play(pcm)

    def _on_speaker_pcm(self, pcm: bytes, sample_rate: int) -> None:
        """Forward only currently audible conversational PCM to motion.

        This callback runs on ALSA/PortAudio's real-time output thread.  The
        motion controller's feed method is deliberately non-blocking and
        drop-oldest, so speaker timing can never wait for a motor command.
        """
        if (
            not self._speech_pcm_active
            or self._dance_loop_active
            or self._motion is None
            or not self.config.wobbling_enabled
        ):
            return
        feed = getattr(self._motion, "feed_talk_audio", None)
        if callable(feed):
            feed(pcm, sample_rate)

    def _stop_talk_motion_immediately(self) -> None:
        if self._motion is None:
            return
        stop_motion = getattr(self._motion, "stop_talk_motion", None)
        if callable(stop_motion):
            stop_motion(immediate=True)

    async def _wait_for_playback_drain(self) -> None:
        """Wait until all queued audio has been played."""
        assert self._audio is not None
        while any(not future.done() for future in tuple(self._pending_playbacks)):
            await asyncio.sleep(0.01)
        if self._dance_loop_active:
            return  # buffer is owned by music; drain would wait for the whole dance
        while self._audio.is_playing:
            await asyncio.sleep(0.05)

    def _listening_is_blocked(self) -> bool:
        return (
            self._tts_playing
            or self._external_interaction_active
            or time.monotonic() < self._listen_not_before
            or self._state in ("thinking", "speaking")
        )

    async def _wait_for_listening_gate(self) -> None:
        while True:
            remaining = self._listen_not_before - time.monotonic()
            if (
                not self._tts_playing
                and not self._external_interaction_active
                and remaining <= 0
            ):
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
            self._speech_pcm_active = False
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
                    # The audio backend is authoritative for speech end.  TTS
                    # callbacks may finish seconds before queued PCM is heard,
                    # so stopping anywhere earlier truncates the head motion.
                    if self._motion is not None:
                        stop_motion = getattr(self._motion, "stop_talk_motion", None)
                        if callable(stop_motion):
                            stop_motion(immediate=self._barge_in_occurred)
                    self._speech_pcm_active = False
                    self._tts_playing = False
                    # Skip echo gate when barge-in occurred — user is already talking
                    if not self._barge_in_occurred:
                        self._listen_not_before = max(
                            self._listen_not_before,
                            time.monotonic()
                            + max(0.0, self.config.post_playback_silence_s),
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
        if self.config.wake_engine != "cloud":
            # Local KWS already gated this turn; only a bare wake word gets
            # the canned response, everything else is a real instruction.
            wake_words = [
                phrase.strip().lower()
                for phrase in self.config.wake_words.split(",")
                if phrase.strip()
            ]
            if stripped.lower() in wake_words:
                return _WAKE_ONLY
            return normalized

        now = time.monotonic()

        # Check for wake word (cloud engine only)
        remaining = self._strip_wake_word(normalized)
        if remaining is not None:
            self._wake_word_active_until = now + self.config.wake_word_timeout_s
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
        """Return text after a wake word at the start, or None if absent.

        Only a *leading* wake word is stripped: a wake word in the middle or
        end of a sentence is the user's own words (e.g. "我不吃皮皮虾",
        "跳个正常点的舞蹈。皮皮虾。") and must not truncate the instruction.
        """
        wake_words = [
            phrase.strip().lower()
            for phrase in self.config.wake_words.split(",")
            if phrase.strip()
        ]
        lowered = normalized.lower()
        for ww in wake_words:
            if lowered.startswith(ww):
                return normalized[len(ww) :].strip(" ，,。！？!?")
        return None

    async def _speak_wake_response(self) -> None:
        """Play the canned wake-word acknowledgment."""
        assert self._audio is not None
        if self._dance_loop_active:
            return
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
        if self._dance_loop_active:
            return
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
        dance_status: dict[str, object] = {}
        if self._beat_dance is not None and self._dance_loop_active:
            dance_status = dict(self._beat_dance.status())
            dance_status["dance_loop_active"] = True
        if self._audio:
            ri = self._audio.resolved_info
            audio = ri if isinstance(ri, dict) else ri.to_dict()
        else:
            audio = None
        # Audio level diagnostics
        # Use play_rms() method (available on all AudioBackend implementations)
        capture_rms = (
            float(getattr(self._audio, "capture_rms", 0.0) or 0.0)
            if self._audio
            else 0.0
        )
        capture_db = round(20.0 * math.log10(max(capture_rms, 1e-8)), 1)
        speaker_playing = bool(self._audio and self._audio.is_playing)
        speech_audio_playing = bool(speaker_playing and self._speech_pcm_active)
        talk_motion_active = bool(
            self._motion is not None and getattr(self._motion, "is_talk_shaking", False)
        )
        location = (
            self._location.latest_position.to_dict()
            if self._location and self._location.latest_position
            else None
        )
        try:
            journal_health = self._memory.health() if self._memory is not None else None
        except Exception:
            journal_health = None
        vision_status = dict(self._last_vision_status)
        if self._visual_cache is not None:
            vision_status["age_s"] = round(
                max(0.0, now - self._visual_cache.stored_monotonic), 3
            )
            vision_status["cache_fresh"] = self._visual_cache.is_fresh(
                self.config.visual_context_ttl_s,
                self._visual_cache.observation.scope,
            )
        else:
            vision_status["cache_fresh"] = False
        return {
            **dance_status,
            "state": self._state,
            "external_interaction_active": self._external_interaction_active,
            "language": self._language,
            "audio": audio,
            "audio_level_rms": round(capture_rms, 4),
            "audio_level_db": capture_db,
            "echo_gate_remaining_s": round(max(0.0, self._listen_not_before - now), 3),
            "tts_playing": self._tts_playing,
            "speaker_playing": speaker_playing,
            "speech_audio_playing": speech_audio_playing,
            "talk_motion_active": talk_motion_active,
            "talk_motion_backend": getattr(
                self._motion, "talk_motion_backend", "disabled"
            ),
            "talk_motion_envelope": round(
                float(getattr(self._motion, "talk_motion_envelope", 0.0)), 3
            ),
            "talk_motion_error": getattr(self._motion, "talk_motion_error", ""),
            "talk_motion_dropped_frames": int(
                getattr(self._motion, "talk_motion_dropped_frames", 0)
            ),
            "barge_in_occurred": self._barge_in_occurred,
            "wake_word_active": now < self._wake_word_active_until,
            "model": self.config.bailian_llm_model,
            "search_policy": self.config.search_policy,
            "search": dict(self._last_search_status),
            "vision": vision_status,
            "asr": {
                "vad_silence_ms": self.config.asr_vad_silence_ms,
                "initial_silence_timeout_s": self.config.asr_initial_silence_timeout_s,
                "speech_max_duration_s": self.config.asr_speech_max_duration_s,
                "last_end_reason": self._last_asr_end_reason,
                "frontend_v2": self.config.audio_frontend_v2,
                **self._audio_frontend_metrics,
            },
            "conversation_turns": len(self._conversation_history) // 2,
            "conversation_idle_reset_s": self.config.session_reset_idle_s,
            "last_topic": self._last_topic,
            "last_evidence_ids": list(self._last_evidence_ids),
            "location": location,
            "journal_health": journal_health,
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
        if self._external_interaction_active:
            return {
                "reply": "手势模式占用中，请先退出手势交互。",
                "emotion": "",
                "memory_context": "",
                "vision_context": "",
                "intent": TurnIntent.GENERAL.value,
                "sources": [],
                "error": "gesture_mode_busy",
                "turn_id": "",
            }
        if self._dance_loop_active:
            return {
                "reply": "正在跳舞，等音乐停了再聊吧～",
                "emotion": "",
                "memory_context": "",
                "vision_context": "",
                "intent": TurnIntent.GENERAL.value,
                "sources": [],
                "error": "",
                "turn_id": "",
            }
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
            await self._cancel_voice_listener()
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
        if self._dance_loop_active:
            return
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

    def _set_session_location(self, place: str) -> None:
        now = datetime.now(timezone.utc)
        self._session_location = {
            "ok": False,
            "place": place.strip().strip("'\"“”"),
            "province": "",
            "city": "",
            "adcode": "",
            "coordinates": None,
            "accuracy_m": None,
            "precision": "unverified_note",
            "source": "session_note",
            "observed_at": now.isoformat(),
            "is_stale": True,
        }

    def _session_location_context(self) -> str:
        if not self._session_location:
            return ""
        place = str(self._session_location.get("place") or "")
        return (
            f"用户在本次会话中声明地点为{place}。这只是未经传感器验证的会话备注；"
            "不能覆盖 GPS、浏览器或高德实时定位，也不能据此编造过去的旅途故事。"
        )

    async def _tool_get_current_location(
        self, *, refresh: bool = False
    ) -> dict[str, Any]:
        """Return a normalized, source-labelled location tool payload."""

        if self._location is not None:
            try:
                pos = await self._location.get_position(refresh=refresh)
                if pos.source != "unavailable":
                    if pos.lat is not None and pos.lon is not None:
                        logger.info("📍 位置查询成功 (%s, %s)", pos.source, pos.precision)
                    return {
                        "ok": True,
                        "place": pos.address or pos.city or pos.province,
                        "province": pos.province,
                        "city": pos.city,
                        "adcode": pos.adcode,
                        "coordinates": (
                            {"latitude": pos.lat, "longitude": pos.lon}
                            if pos.lat is not None and pos.lon is not None
                            else None
                        ),
                        "accuracy_m": pos.accuracy_m,
                        "radius_m": pos.radius_m,
                        "coordinate_system": pos.coordinate_system,
                        "precision": pos.precision,
                        "source": pos.source,
                        "observed_at": datetime.fromtimestamp(
                            pos.timestamp, tz=timezone.utc
                        ).isoformat(),
                        "age_s": max(0.0, time.time() - pos.timestamp),
                        "is_stale": time.time() - pos.timestamp > pos.stale_after_s,
                    }
                location_error = pos.error
            except Exception:
                logger.exception("Location tool failed")
                location_error = "位置查询异常"
        else:
            location_error = "定位服务未启动"

        return {
            "ok": False,
            "place": "",
            "province": "",
            "city": "",
            "adcode": "",
            "coordinates": None,
            "accuracy_m": None,
            "precision": "unknown",
            "source": "unavailable",
            "session_note": (
                str(self._session_location.get("place") or "")
                if self._session_location
                else ""
            ),
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "is_stale": False,
            "error": location_error or "当前无法获取定位",
        }

    async def _prepare_location_tool_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Force one registered location tool call, then return its tool result."""

        tool_call_id = f"location_{uuid.uuid4().hex[:12]}"
        assistant_message: dict[str, Any] | None = None
        refresh = False
        if self.config.bailian_api_key:
            try:
                async with BailianLLMClient(self.config) as llm:
                    response = await llm.chat(
                        messages,
                        tools=[_LOCATION_TOOL_SCHEMA],
                        tool_choice={
                            "type": "function",
                            "function": {"name": "get_current_location"},
                        },
                    )
                candidate = (response.get("choices") or [{}])[0].get("message") or {}
                calls = candidate.get("tool_calls") or []
                if calls:
                    call = calls[0]
                    if (call.get("function") or {}).get("name") == "get_current_location":
                        assistant_message = candidate
                        tool_call_id = str(call.get("id") or tool_call_id)
                        try:
                            arguments = json.loads(
                                str((call.get("function") or {}).get("arguments") or "{}")
                            )
                            refresh = bool(arguments.get("refresh", False))
                        except (json.JSONDecodeError, AttributeError):
                            refresh = False
            except Exception:
                # Deterministic source routing remains authoritative if the
                # model's tool-planning request is unavailable.
                logger.warning("Location tool planning failed; executing directly", exc_info=True)

        if assistant_message is None:
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "get_current_location",
                            "arguments": json.dumps(
                                {"refresh": refresh}, ensure_ascii=False
                            ),
                        },
                    }
                ],
            }

        try:
            payload = await asyncio.wait_for(
                self._tool_get_current_location(refresh=refresh), timeout=3.0
            )
        except asyncio.TimeoutError:
            payload = {
                "ok": False,
                "place": "",
                "precision": "unknown",
                "source": "unavailable",
                "error": "定位工具执行超时",
            }
        self._current_sources.append(
            {
                "type": "live_location",
                "title": "实时位置工具",
                "source": payload.get("source", "unavailable"),
                "precision": payload.get("precision", "unknown"),
            }
        )
        return [
            *messages,
            assistant_message,
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": "get_current_location",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

    def _should_reuse_visual(self, text: str, scope: str) -> bool:
        cache = self._visual_cache
        return bool(
            cache
            and cache.is_fresh(self.config.visual_context_ttl_s, scope)
            and _VISUAL_REFERENCE_RE.search(text)
            and not _VISUAL_REFRESH_RE.search(text)
        )

    async def _capture_visual_jpeg(
        self, scope: str, fallback_image: bytes | None = None
    ) -> bytes | None:
        if scope == "rear":
            from chaihuo_reachy.ezviz import capture_rear_view

            return await capture_rear_view(self.config)

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
        return jpeg

    async def _observe_scene(
        self,
        scope: str,
        *,
        focus: str,
        fallback_image: bytes | None = None,
        prefer_cache: bool = False,
    ) -> VisualObservation:
        """Capture or reuse a frame and return internal grounded facts."""
        scope = "rear" if scope == "rear" else "front"
        started = time.monotonic()
        capture_ms = 0
        vlm_ms = 0
        jpeg: bytes | None = None
        reused_frame = False
        cache = self._visual_cache
        if prefer_cache and cache and cache.is_fresh(
            self.config.visual_context_ttl_s, scope
        ):
            jpeg = cache.jpeg
            reused_frame = True
            if cache.focus == focus and cache.observation.ok:
                observation = cache.observation
                self._update_vision_status(
                    observation,
                    capture_ms=0,
                    vlm_ms=0,
                    total_ms=int((time.monotonic() - started) * 1000),
                    reason="reused structured observation",
                )
                return observation

        if jpeg is None:
            capture_started = time.monotonic()
            self._emit_turn_event("turn_status", status="observing")
            try:
                jpeg = await self._capture_visual_jpeg(scope, fallback_image)
            except Exception:
                logger.exception("%s camera capture failed", scope)
                error = (
                    "我暂时看不到车外，请检查后视设备连接。"
                    if scope == "rear"
                    else "我现在暂时看不到前面，可能是眼睛还没准备好。"
                )
                observation = VisualObservation.failure(scope=scope, error=error)
                self._update_vision_status(
                    observation,
                    capture_ms=int((time.monotonic() - capture_started) * 1000),
                    vlm_ms=0,
                    total_ms=int((time.monotonic() - started) * 1000),
                    reason="capture failed",
                )
                return observation
            capture_ms = int((time.monotonic() - capture_started) * 1000)

        if not jpeg:
            error = (
                "我暂时看不到车外，请检查后视设备连接。"
                if scope == "rear"
                else "我现在暂时看不到前面，可能是眼睛还没准备好。"
            )
            observation = VisualObservation.failure(scope=scope, error=error)
            self._update_vision_status(
                observation,
                capture_ms=capture_ms,
                vlm_ms=0,
                total_ms=int((time.monotonic() - started) * 1000),
                reason="no frame",
            )
            return observation

        if self._on_snapshot and not reused_frame:
            try:
                self._on_snapshot(jpeg, scope)
            except Exception:
                logger.debug("%s snapshot callback error", scope, exc_info=True)

        issue = visual_quality_issue(jpeg)
        if issue:
            logger.warning("%s frame rejected: %s", scope, issue)
            observation = VisualObservation.failure(
                scope=scope,
                error=_natural_visual_quality_error(issue),
                quality="rejected",
            )
            self._update_vision_status(
                observation,
                capture_ms=capture_ms,
                vlm_ms=0,
                total_ms=int((time.monotonic() - started) * 1000),
                reason="quality rejected",
            )
            return observation

        vlm_started = time.monotonic()
        try:
            async with BailianVLMClient(self.config) as vlm:
                raw = await vlm.understand(
                    jpeg,
                    question=(
                        "你是 Reachy 的内部视觉感知模块，不负责对用户说话。"
                        f"观察范围：{'基地车后方/车外' if scope == 'rear' else 'Reachy 面前'}。"
                        f"本轮关注点：{focus}。"
                        "只报告清晰可见且与关注点有关的事实；不要猜人物身份、具体地点、"
                        "画外信息或被遮挡内容。请只输出 JSON："
                        '{"facts":["事实1","事实2"],"uncertainties":["不确定项"],'
                        '"quality":"ok"}。facts 不要使用“照片、图片、画面、摄像头、识别到”'
                        "等面向实现的说法。看不清的内容放入 uncertainties。"
                    ),
                )
            vlm_ms = int((time.monotonic() - vlm_started) * 1000)
            raw = _VISUAL_PREFIX_RE.sub("", raw).strip()
            observation = parse_vlm_observation(raw, scope=scope)
        except Exception:
            logger.exception("%s VLM observation failed", scope)
            vlm_ms = int((time.monotonic() - vlm_started) * 1000)
            observation = VisualObservation.failure(
                scope=scope,
                error="我刚才没看清楚，可以把它靠近一点再让我看看吗？",
                quality="analysis_failed",
            )

        if observation.ok:
            self._visual_cache = VisualCache(
                observation=observation,
                jpeg=jpeg,
                stored_monotonic=time.monotonic(),
                focus=focus,
            )
        self._update_vision_status(
            observation,
            capture_ms=capture_ms,
            vlm_ms=vlm_ms,
            total_ms=int((time.monotonic() - started) * 1000),
            reason="cached frame refocus" if reused_frame else "fresh observation",
        )
        return observation

    def _update_vision_status(
        self,
        observation: VisualObservation,
        *,
        capture_ms: int,
        vlm_ms: int,
        total_ms: int,
        reason: str,
    ) -> None:
        self._last_vision_status.update(
            {
                "called": True,
                "reason": reason,
                "scope": observation.scope,
                "observed_at": observation.captured_at,
                "age_s": 0.0,
                "capture_latency_ms": capture_ms,
                "vlm_latency_ms": vlm_ms,
                "total_latency_ms": total_ms,
                "error": observation.error,
            }
        )

    def _register_visual_source(self, observation: VisualObservation) -> None:
        if any(
            item.get("observation_id") == observation.observation_id
            for item in self._current_sources
        ):
            return
        self._current_sources.append(
            {
                "type": "live_vision",
                "title": "Reachy 实时观察",
                "scope": observation.scope,
                "observed_at": observation.captured_at,
                "quality": observation.quality,
                "observation_id": observation.observation_id,
            }
        )

    async def _tool_capture_rear_view(self) -> str:
        """Compatibility wrapper for callers that explicitly request rear view."""
        observation = await self._observe_scene("rear", focus="车外现在有什么")
        return _natural_visual_reply(observation)

    async def _tool_take_photo(self, fallback_image: bytes | None = None) -> str:
        """Compatibility wrapper; successful replies use first-person sight."""
        observation = await self._observe_scene(
            "front", focus="面前现在有什么", fallback_image=fallback_image
        )
        return _natural_visual_reply(observation)

    # ── Motion tool handlers ─────────────────────────────────────────

    async def _tool_dance(
        self, style: str = "random", duration_limit_s: float | None = None
    ) -> str:
        """Execute a dance via the MotionController, with backing music.

        ``style`` may be "random": it is resolved to a concrete style here
        so the backing track and the choreography always match.  The
        track's BPM paces the choreography; the dance runs up to
        ``duration_limit_s`` (default: min(track length, 60 s)) and stops
        early when the music is shorter.  Without a track the dance runs a
        single pass at a default tempo.

        On success returns "" and records the finished dance in
        ``_pending_dance`` so ``_coordinate_turn`` lets the LLM improvise a
        natural post-dance remark (never naming the style).  Failure paths
        return canned short replies.
        """
        from pathlib import Path

        from chaihuo_reachy.motion import resolve_dance_style
        from chaihuo_reachy.music import (
            STYLE_BPM,
            detect_bpm,
            read_track,
            resolve_track,
        )

        if self._motion is None:
            return "运动控制未启用，无法跳舞。"
        # 音乐与动作必须用同一个具体风格 (random.wav 已被具体风格取代)
        style = resolve_dance_style(style)
        beat_s = 0.5
        duration_s: float | None = None
        track = resolve_track(Path(self.config.dance_music_dir), style)
        if track is not None:
            info = read_track(track)
            if info is not None:
                sr, pcm = info
                # 自研合成曲目用已知 BPM 表; 外部曲目回退到能量自相关检测
                bpm = STYLE_BPM.get(style) or detect_bpm(pcm, sr)
                if bpm:
                    beat_s = 60.0 / bpm
                limit = min(
                    len(pcm) / 2 / sr,
                    duration_limit_s or self.config.dance_duration_s,
                )
                duration_s = max(10.0, limit)  # never shorter than 10 s
                logger.info(
                    "🎵 伴奏: %s (%.1fs, BPM=%s, 拍长 %.3fs, 舞时长 %.0fs)",
                    track.name,
                    len(pcm) / 2 / sr,
                    bpm if bpm else "未检出",
                    beat_s,
                    duration_s,
                )
        music_task: asyncio.Task | None = None
        if self._audio is not None:
            music_task = asyncio.create_task(self._play_dance_music(style, track))
        try:
            summary = await self._motion.dance(
                style, duration_s=duration_s, beat_s=beat_s
            )
            skipped = int(summary.get("skipped") or 0)
            if skipped:
                logger.warning(
                    "Dance %s finished with %d skipped step(s)", style, skipped
                )
                return f"跳完啦！有{skipped}个动作没跟上，但整体完成。"
            logger.info(
                "💃 舞毕: %s (%.1fs)",
                style,
                float(summary.get("duration") or 0.0),
            )
            self._pending_dance = {
                "style": style,
                "skipped": skipped,
                "duration": float(summary.get("duration") or 0.0),
                "turn_id": self._current_turn_id,
            }
            return ""  # 交给 LLM 即兴接话, 不透露风格
        except Exception as e:
            logger.exception("Dance failed")
            return f"跳舞失败: {e}"
        finally:
            if music_task is not None:
                music_task.cancel()
                try:
                    await music_task
                except asyncio.CancelledError:
                    pass

    async def _play_dance_music(self, style: str, track: Path | None = None) -> None:
        """Loop a backing track while the dance runs (best-effort).

        ``track`` is the resolved music file (or None to resolve here).
        Missing/unreadable tracks are fine — the dance proceeds silently.
        Chunks are fed to the speaker at real-time pace; the task is
        cancelled when the dance finishes.
        """
        from pathlib import Path

        from chaihuo_reachy.music import read_track, resolve_track

        assert self._audio is not None
        if track is None:
            track = resolve_track(Path(self.config.dance_music_dir), style)
        if track is None:
            logger.info("未找到伴奏音乐，安静跳舞")
            return
        info = read_track(track)
        if info is None:
            return
        sr, frames = info
        logger.info("🎵 播放伴奏: %s (%.1fs)", track.name, len(frames) / 2 / sr)
        self._audio.set_output_sample_rate(sr)
        chunk_size = sr * 2 // 5  # 0.2s 一块，实时节奏喂给播放队列
        try:
            while True:
                for start in range(0, len(frames), chunk_size):
                    await self._audio.play(frames[start : start + chunk_size])
                    # seconds, not bytes (int16 = 2 bytes/frame)
                    await asyncio.sleep(chunk_size / 2 / sr)
        except asyncio.CancelledError:
            logger.debug("伴奏播放结束: %s", track.name)
            raise

    # ── Beat-dance loop (infinite beat-synced dance, suspends voice) ──
    async def _suspend_voice_for_dance(self) -> None:
        """Give the dance exclusive ownership of the audio device.

        Called once per ``start_beat_dance`` BEFORE the music task starts.
        Aborts any in-flight TTS (barge-in flag + generation bump), engages
        the ``_dance_loop_active`` gates, flushes the shared playback buffer
        twice around the in-flight coroutine wait so no TTS chunk survives
        into the music stream.
        """
        if self._tts_playing or self._state in ("speaking", "thinking"):
            logger.info("🎵 连跳接管：取消当前回答")
        # Abort the in-flight reply through the existing barge-in path
        # (the watcher exits without calling stop_playback()).
        self._barge_in_requested = True
        # Invalidate every in-flight TTS generation (mirrors _think_and_speak).
        self._tts_generation += 1
        self._active_tts_generation = self._tts_generation
        # Engage all conversation gates before the music task starts.
        self._dance_loop_active = True
        if self._motion is not None:
            self._stop_talk_motion_immediately()
        if self._audio is not None:
            self._audio.stop_playback()
        # Let in-flight _play_tts_audio coroutines that already passed the
        # generation check finish (each is sub-millisecond), then flush once
        # more to catch the chunk between its check and play().
        while any(not future.done() for future in tuple(self._pending_playbacks)):
            await asyncio.sleep(0.01)
        if self._audio is not None:
            self._audio.stop_playback()

    async def start_beat_dance(self) -> str:
        """Start the infinite beat dance. Returns a user-facing message."""
        if self._beat_dance is None:
            return "节拍连跳未启用（缺少 beat 控制器）"
        if self._dance_loop_active:
            return "已经在跳啦"
        await self._suspend_voice_for_dance()
        info = self._beat_dance.start()
        if info is None:
            return "节拍音乐或节拍数据缺失，无法连跳"
        # One-shot resample to the output rate so per-chunk playback never
        # pays the numpy interpolation cost (asyncio jitter ⇒ stutter).
        sr, pcm = info
        target_sr = int(getattr(self._audio, "output_sr", 16000) or 16000)
        if sr != target_sr:
            from chaihuo_reachy.music import resample_pcm

            logger.info("🎵 伴奏重采样 %dHz → %dHz（一次性）", sr, target_sr)
            pcm = resample_pcm(pcm, sr, target_sr)
            sr = target_sr
        self._dance_loop_active = True
        self._set_state("dancing")
        self._beat_music_task = asyncio.create_task(self._play_beat_music((sr, pcm)))
        logger.info("🎵 无限节拍连跳开始（语音挂起）")
        return "开始跳舞！"

    async def stop_beat_dance(self) -> str:
        """Stop the infinite beat dance and resume conversation."""
        if self._beat_music_task is not None:
            self._beat_music_task.cancel()
            try:
                await self._beat_music_task
            except (asyncio.CancelledError, Exception):
                pass
            self._beat_music_task = None
        if self._audio is not None:
            stop = getattr(self._audio, "stop_playback", None)
            if callable(stop):
                stop()
        if self._beat_dance is not None:
            self._beat_dance.stop()
        self._dance_loop_active = False
        self._set_state("idle")
        logger.info("⏹ 无限节拍连跳停止，恢复语音对话")
        return "停啦"

    async def _play_beat_music(self, info: tuple[int, bytes]) -> None:
        """Feed beat.mp3 PCM to the speaker in a real-time loop (infinite).

        Prefills ~2 s of audio so asyncio scheduling jitter (KWS polling,
        websocket traffic) never empties the playback buffer — an empty
        buffer is heard as stutter.  Then replenishes one 1 s chunk per
        second, keeping the buffer at a stable ~2 s cushion.
        """
        assert self._audio is not None
        sr, frames = info
        logger.info("🎵 无限循环播放: beat.mp3 (%.1fs @%dHz)", len(frames) / 2 / sr, sr)
        self._audio.set_output_sample_rate(sr)
        chunk_size = sr * 2  # 1.0s 一块
        total = len(frames) // chunk_size
        if total == 0:
            total = 1
        idx = 0

        def next_chunk() -> bytes:
            nonlocal idx
            start = (idx % total) * chunk_size
            chunk = frames[start : start + chunk_size]
            idx += 1
            return chunk

        # Prefill ~2s to absorb scheduling jitter before real-time pacing.
        await self._audio.play(next_chunk())
        await self._audio.play(next_chunk())
        try:
            while True:
                await self._audio.play(next_chunk())
                # chunk_size bytes = 1s of int16 mono; sleep must be in
                # SECONDS (÷2 bytes/frame) — a raw chunk_size/sr slept 2s
                # per 1s chunk and starved the buffer into 1s-on/1s-off.
                await asyncio.sleep(chunk_size / 2 / sr)
        except asyncio.CancelledError:
            raise

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


def _keyword_search(memory: Any, query: str, k: int = 6) -> list[dict[str, Any]]:
    """Entity-term full-text retrieval with a defensive fallback.

    Some memory stand-ins (tests, downgraded stores) predate
    ``search_keywords``; treat them as "no keyword hits" and let the vector
    search take over.
    """
    fn = getattr(memory, "search_keywords", None)
    if not callable(fn):
        return []
    try:
        return fn(query, k=k)
    except Exception:
        logger.debug("keyword search failed", exc_info=True)
        return []


def _extract_recent_window(query: str) -> int | None:
    """Detect "最近发生了什么 / 这几天 / 近来" queries → days window.

    Vector search has no notion of recency, so these relative-time queries
    need a date-window retrieval (``MemoryStore.search_recent``) instead of
    similarity search.
    """
    compact = re.sub(r"\s+", "", query)
    if any(word in compact for word in ("这几天", "这两天", "近来", "近期")):
        return 3
    if "最近" in compact:
        match = re.search(r"最近(\d+)天", compact)
        if match:
            return min(30, max(1, int(match.group(1))))
        return 7
    return None


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
        "大前天": 3,
        "前天": 2,
        "昨天": 1,
        "今天": 0,
        "今日": 0,
        "明天": -1,
        "后天": -2,
        "大后天": -3,
    }
    for word, offset in relative_days.items():
        if word in q:
            return (today - timedelta(days=offset)).isoformat()

    # Day of week: 周一/... / 上周一/... / 下周一/...
    weekday_map = {
        "一": 0,
        "二": 1,
        "三": 2,
        "四": 3,
        "五": 4,
        "六": 5,
        "日": 6,
        "天": 6,
    }
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
                    return date(
                        int(m.group(1)), int(m.group(2)), int(m.group(3))
                    ).isoformat()
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
