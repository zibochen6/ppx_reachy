"""Configuration management — layered: env vars > YAML file > defaults.

Supports two deployment targets:
  - ``mac`` — development on macOS (uses local mic/camera)
  - ``jetson`` — production on Jetson Orin (uses Reachy SDK + Docker)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("chaihuo_reachy.config")

# Auto-load .env file from project root on import
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


@dataclass
class Config:
    """Application configuration with sensible defaults."""

    # ── Deployment target ──────────────────────────────────────────────
    target: str = "mac"  # "mac" | "jetson"

    # ── Bailian / DashScope ────────────────────────────────────────────
    bailian_api_key: str = ""
    bailian_workspace_id: str = ""
    bailian_region: str = "cn-beijing"
    bailian_asr_model: str = "qwen3-asr-flash-realtime"
    bailian_llm_model: str = "qwen3.7-plus"
    bailian_tts_model: str = "qwen3-tts-flash-realtime"
    bailian_tts_voice: str = "Cherry"
    bailian_vlm_model: str = "qwen-vl-plus"

    # ── Web search ────────────────────────────────────────────────────
    # ``enable_search`` remains as a compatibility view for older Dashboard
    # clients. New code uses the three-state policy below.
    enable_search: bool = True
    search_policy: str = "auto"  # "auto" | "explicit" | "off"
    search_timeout_s: float = 20.0
    search_circuit_breaker_s: float = 30.0

    # ── ASR settings ───────────────────────────────────────────────────
    asr_language: str = "zh"
    asr_vad_silence_ms: int = 600  # Server-side second-layer endpoint window
    asr_vad_threshold: float = 0.50  # Lower values admit more background speech
    asr_initial_silence_timeout_s: float = 20.0  # Wait for first speech onset
    asr_speech_max_duration_s: float = 15.0  # Compatibility hard ceiling
    asr_finalize_timeout_s: float = 2.0
    asr_min_chars: int = 2

    # ── LLM settings ───────────────────────────────────────────────────
    llm_temperature: float = 0.4
    llm_max_tokens: int = 1600
    llm_system_prompt: str = ""

    # ── TTS settings ───────────────────────────────────────────────────
    tts_speech_rate: float = 1.0
    tts_pitch_rate: float = 1.0
    tts_volume: int = 90  # Higher for Reachy Mini speaker
    tts_sample_rate: int = 24000

    # ── Wake word ──────────────────────────────────────────────────────
    # Wake word is detected locally with sherpa-onnx KWS by default; the
    # cloud ASR transcript matching in ``_accept_transcript()`` remains as
    # the ``cloud`` fallback.  Once detected, a ``wake_word_timeout_s``
    # grace window keeps the conversation open.
    enable_wake_word: bool = True
    wake_engine: str = (
        "local"  # "local" (sherpa-onnx KWS) | "cloud" (ASR text match) | "off"
    )
    wake_words: str = "皮皮虾"
    wake_word_timeout_s: float = 30.0
    wake_response: str = "我在呢！有什么可以帮你的？"
    feedback_sounds_enabled: bool = True  # Play beeps on state transitions

    # ── Local KWS (sherpa-onnx) ────────────────────────────────────────
    # Model dir must contain tokens.txt + encoder/decoder/joiner .onnx and
    # keywords.txt, as produced by scripts/download_kws_model.py.
    kws_model_dir: str = "models/kws"
    # Tuned for a visitor speaking at a normal near-field volume.  The old
    # 0.35/1.0 pair required shouting on the Reachy XMOS microphone.
    kws_threshold: float = 0.20  # Lower = easier to trigger; 0.20 avoids shouting
    kws_score: float = 1.5  # Modest beam boost for the custom Chinese keyword
    wake_listen_timeout_s: float = 60.0  # Max time waiting for the wake word locally
    # The SDK AEC channel has an idle RMS around 0.08-0.14 on the current
    # hardware. Keep standby recognition and interruption above that floor.
    voice_activity_threshold: float = 0.16
    barge_in_enabled: bool = False
    barge_in_sensitivity: float = 0.18  # Mic RMS threshold for interrupting TTS (0-1)

    # ── Audio ──────────────────────────────────────────────────────────
    # Safe default: auto-detect a unique Reachy full-duplex device. Use the
    # explicit value "default" to opt into Mac/system input + output.
    audio_device: str | int | None = "auto"
    audio_sample_rate: int = 16000
    audio_input_channel: int = 1  # Reachy ch1 = AEC-processed (echo-cancelled)
    audio_mic_gain: float = 1.0  # Unity gain; avoid clipping that degrades ASR
    audio_frontend_v2: bool = True
    vad_model_path: str = "models/vad/silero_vad.onnx"
    vad_on_threshold: float = 0.60
    vad_off_threshold: float = 0.35
    endpoint_silence_ms: int = 700
    max_utterance_s: float = 15.0
    min_utterance_ms: int = 200
    doa_tolerance_deg: float = 35.0
    doa_mismatch_ms: int = 800
    followup_window_s: float = 8.0

    # ── SDK Media backend ───────────────────────────────────────────────
    # "auto" → SDK daemon 模式下用 local，standalone 用 no_media
    # "local" → 强制用 SDK GStreamer 后端（需 daemon）
    # "no_media" → 回退到 sounddevice + OpenCV 直接访问硬件
    media_backend: str = "auto"

    # ── Daemon ──────────────────────────────────────────────────────────
    # "auto" → 连接健康 daemon；本机没有 daemon 时才拉起
    # "connect" → 只连接已有 SDK daemon
    # "spawn"   → 仅回收本项目 daemon 后拉起
    daemon_mode: str = "auto"
    daemon_host: str = "reachy-mini.local"
    daemon_port: int = 8000
    daemon_serial_port: str = ""  # Optional explicit motor-controller serial port
    daemon_simulation: bool = False  # Simulation must be explicitly opted into
    daemon_state_file: str = "state/daemon.json"
    auto_wake_up: bool = True  # 启动时自动让机器人站起来
    auto_sleep: bool = True  # 关闭时自动让机器人休眠

    # ── Motion / Dance ──────────────────────────────────────────────────
    dance_enabled: bool = True  # 启用运动控制（Dashboard 按钮 + 语音指令）
    wobbling_enabled: bool = True  # 说话时头部自然晃动
    talk_motion_yaw_max_deg: float = 20.0
    talk_motion_pitch_max_deg: float = 3.2
    talk_motion_roll_max_deg: float = 2.0
    talk_motion_gain: float = 1.0
    doa_enabled: bool = True  # 麦克风阵列声源定位 / audio direction gate
    # 跳舞伴奏音乐目录: 放 <style>.wav (happy/swing/robot) 或任意 *.wav;
    # 缺失时安静跳舞（不报错）。
    dance_music_dir: str = "music"
    dance_duration_s: float = 15.0  # 舞蹈时长上限（音乐更短则随音乐停）
    # 无限节拍连跳（继承 beat.mp3 项目）：音乐 + 舞蹈无限循环，按钮播放/停止
    beat_dance_enabled: bool = True
    beat_music_path: str = "music/beat.mp3"
    beat_timeline_path: str = "data/dance/beat_timeline.json"

    # ── Camera (Reachy Mini USB camera) ────────────────────────────────
    # Mac: auto-detect by name containing "Reachy". Jetson: /dev/video0 or index
    camera_device: int | str = "auto"  # "auto" | int index | "/dev/video0"
    camera_width: int = 640
    camera_height: int = 480

    # ── Single-hand pose interaction ──────────────────────────────────
    gesture_backend: str = "auto"  # auto | coreml | mps | tensorrt
    gesture_coreml_model_path: str = "models/hand_pose/hand_pose_resnet18.mlpackage"
    gesture_torchscript_model_path: str = "models/hand_pose/hand_pose_resnet18.ts"
    gesture_tensorrt_engine_path: str = "models/hand_pose/hand_pose_resnet18_fp16.engine"
    gesture_torch2trt_model_path: str = "models/hand_pose/hand_pose_resnet18_torch2trt.pth"
    gesture_inference_fps: float = 15.0
    gesture_keypoint_confidence: float = 0.15
    gesture_confirmation_ms: int = 300
    gesture_lost_timeout_ms: int = 800
    gesture_tracking_deadzone: float = 0.06
    gesture_tracking_alpha: float = 0.35
    gesture_tracking_max_step_deg: float = 2.0
    gesture_head_yaw_max_deg: float = 20.0
    gesture_head_pitch_max_deg: float = 20.0
    gesture_body_yaw_max_deg: float = 30.0

    # ── Location / GPS ──────────────────────────────────────────────────
    location_gpsd_enabled: bool = True  # Try GPSD daemon for real GPS
    location_gpsd_host: str = "127.0.0.1"
    location_gpsd_port: int = 2947
    location_poll_interval_s: float = 2.0  # GPS poll interval
    location_gps_fresh_s: float = 30.0
    location_browser_fresh_s: float = 60.0
    # Operator-declared current location. This is text so a known venue or
    # city can be supplied even when no live source is available. It is now a
    # final fallback rather than an absolute override.
    manual_location: str = ""  # Deprecated: session note only, never live location
    location_wifi_enabled: bool = True
    location_wifi_scan_interval_s: float = 10.0
    location_wifi_fresh_s: float = 30.0
    amap_web_key: str = ""
    amap_web_private_key: str = ""
    amap_timeout_s: float = 3.0
    amap_cache_ttl_s: float = 600.0

    # ── Conversation ───────────────────────────────────────────────────
    language: str = "zh"  # "zh" | "en"
    max_history_turns: int = 20
    session_reset_idle_s: float = 1800.0
    post_playback_silence_s: float = 0.8  # avoid echo self-trigger (was 0.5)

    # ── Memory / Journal ───────────────────────────────────────────────
    chroma_persist_dir: str = "data/chroma"
    # Read the Yuque knowledge-base TOC directly.  The public aggregation
    # page can lag behind newly published journals by hours, which made an
    # otherwise available "yesterday" entry invisible to the assistant.
    journal_url: str = "https://www.yuque.com/mouseart/mcv/guaaeocvtm3mtl99"
    journal_cache_dir: str = "data/journals"
    memory_top_k: int = 5  # number of journal snippets to retrieve
    journal_relevance_threshold: float = 0.48
    journal_auto_sync_interval_minutes: int = 30  # 0 = sync only on startup
    journal_index_v3_enabled: bool = True
    source_router_v2_enabled: bool = True
    session_state_v2_enabled: bool = True

    # ── EZVIZ rear-view camera ────────────────────────────────────────
    ezviz_app_key: str = ""
    ezviz_app_secret: str = ""
    ezviz_device_serial: str = ""
    ezviz_channel_no: str = "1"

    # ── Dashboard ──────────────────────────────────────────────────────
    dashboard_enabled: bool = True
    dashboard_port: int = 8640
    chat_history_limit: int = 100
    capture_history_limit: int = 20
    camera_frame_max_age_s: float = 3.0

    # ── Semantic vision ────────────────────────────────────────────────
    # semantic: implicit + explicit vision; semantic_shadow: log implicit
    # decisions without capturing; explicit: keyword routes only; off: disabled.
    vision_policy: str = "semantic"
    visual_context_ttl_s: float = 15.0
    vision_max_calls_per_turn: int = 1

    # ── Persona ────────────────────────────────────────────────────────
    persona: str = "xiao_chai"

    @property
    def profile_dir(self) -> Path:
        project_profile = (
            Path(__file__).parent.parent.parent / "profiles" / self.persona
        )
        if project_profile.exists():
            return project_profile
        return Path(__file__).with_name("profiles") / self.persona

    def system_prompt(self) -> str:
        """Load the persona's system prompt from profile."""
        p = self.profile_dir / "instructions.txt"
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""

    def organization_knowledge(self) -> str:
        """Load curated, source-labelled Chaihuo organization facts."""
        p = self.profile_dir / "organization_knowledge.txt"
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""

    def system_prompt_with_context(
        self, journal_context: str = "", vision_context: str = ""
    ) -> str:
        """Build the full system prompt with dynamic context."""
        prompt = self.system_prompt()
        if journal_context:
            prompt += f"\n\n【基地车近期日记参考】\n{journal_context}"
        if vision_context:
            prompt += f"\n\n【当前视觉信息】\n{vision_context}"
        return prompt


# ── Env-var mapping ─────────────────────────────────────────────────────────
_ENV: dict[str, str] = {
    "bailian_api_key": "BAILIAN_API_KEY",
    "bailian_workspace_id": "BAILIAN_WORKSPACE_ID",
    "bailian_region": "BAILIAN_REGION",
    "bailian_asr_model": "BAILIAN_ASR_MODEL",
    "bailian_llm_model": "BAILIAN_LLM_MODEL",
    "bailian_tts_model": "BAILIAN_TTS_MODEL",
    "bailian_tts_voice": "BAILIAN_TTS_VOICE",
    "bailian_vlm_model": "BAILIAN_VLM_MODEL",
    "enable_search": "BAILIAN_ENABLE_SEARCH",
    "search_policy": "REACHY_SEARCH_POLICY",
    "search_timeout_s": "REACHY_SEARCH_TIMEOUT_S",
    "search_circuit_breaker_s": "REACHY_SEARCH_CIRCUIT_BREAKER_S",
    "target": "REACHY_TARGET",
    "asr_language": "BAILIAN_ASR_LANGUAGE",
    "asr_vad_silence_ms": "BAILIAN_ASR_VAD_SILENCE_MS",
    "asr_vad_threshold": "BAILIAN_ASR_VAD_THRESHOLD",
    "asr_initial_silence_timeout_s": "BAILIAN_ASR_INITIAL_SILENCE_TIMEOUT_S",
    "asr_speech_max_duration_s": "BAILIAN_ASR_SPEECH_MAX_DURATION_S",
    "asr_min_chars": "BAILIAN_ASR_MIN_CHARS",
    "asr_finalize_timeout_s": "BAILIAN_ASR_FINALIZE_TIMEOUT_S",
    "llm_temperature": "BAILIAN_LLM_TEMPERATURE",
    "llm_max_tokens": "BAILIAN_LLM_MAX_TOKENS",
    "llm_system_prompt": "BAILIAN_LLM_SYSTEM_PROMPT",
    "tts_speech_rate": "BAILIAN_TTS_SPEECH_RATE",
    "tts_volume": "BAILIAN_TTS_VOLUME",
    "tts_sample_rate": "BAILIAN_TTS_SAMPLE_RATE",
    "enable_wake_word": "BAILIAN_ENABLE_WAKE_WORD",
    "wake_engine": "REACHY_WAKE_ENGINE",
    "wake_words": "BAILIAN_WAKE_WORDS",
    "wake_word_timeout_s": "BAILIAN_WAKE_WORD_TIMEOUT_S",
    "wake_response": "BAILIAN_WAKE_RESPONSE",
    "kws_model_dir": "REACHY_KWS_MODEL_DIR",
    "kws_threshold": "REACHY_KWS_THRESHOLD",
    "kws_score": "REACHY_KWS_SCORE",
    "wake_listen_timeout_s": "REACHY_WAKE_LISTEN_TIMEOUT_S",
    "voice_activity_threshold": "REACHY_VOICE_ACTIVITY_THRESHOLD",
    "barge_in_enabled": "REACHY_BARGE_IN_ENABLED",
    "barge_in_sensitivity": "REACHY_BARGE_IN_SENSITIVITY",
    "audio_device": "REACHY_AUDIO_DEVICE",
    "audio_input_channel": "REACHY_AUDIO_INPUT_CHANNEL",
    "audio_mic_gain": "REACHY_MIC_GAIN",
    "audio_frontend_v2": "REACHY_AUDIO_FRONTEND_V2",
    "vad_model_path": "REACHY_VAD_MODEL_PATH",
    "vad_on_threshold": "REACHY_VAD_ON_THRESHOLD",
    "vad_off_threshold": "REACHY_VAD_OFF_THRESHOLD",
    "endpoint_silence_ms": "REACHY_ENDPOINT_SILENCE_MS",
    "max_utterance_s": "REACHY_MAX_UTTERANCE_S",
    "min_utterance_ms": "REACHY_MIN_UTTERANCE_MS",
    "doa_tolerance_deg": "REACHY_DOA_TOLERANCE_DEG",
    "doa_mismatch_ms": "REACHY_DOA_MISMATCH_MS",
    "followup_window_s": "REACHY_FOLLOWUP_WINDOW_S",
    "camera_device": "REACHY_CAMERA_DEVICE",
    "dashboard_port": "REACHY_DASHBOARD_PORT",
    "language": "REACHY_LANGUAGE",
    "max_history_turns": "REACHY_MAX_HISTORY_TURNS",
    "manual_location": "REACHY_MANUAL_LOCATION",
    "location_gps_fresh_s": "REACHY_LOCATION_GPS_FRESH_S",
    "location_browser_fresh_s": "REACHY_LOCATION_BROWSER_FRESH_S",
    "location_wifi_enabled": "REACHY_LOCATION_WIFI_ENABLED",
    "location_wifi_scan_interval_s": "REACHY_LOCATION_WIFI_SCAN_INTERVAL_S",
    "location_wifi_fresh_s": "REACHY_LOCATION_WIFI_FRESH_S",
    "amap_web_key": "AMAP_WEB_KEY",
    "amap_web_private_key": "AMAP_WEB_PRIVATE_KEY",
    "amap_timeout_s": "AMAP_TIMEOUT_S",
    "amap_cache_ttl_s": "AMAP_CACHE_TTL_S",
    "session_reset_idle_s": "REACHY_SESSION_RESET_IDLE_S",
    "media_backend": "REACHY_MEDIA_BACKEND",
    "daemon_mode": "REACHY_DAEMON_MODE",
    "daemon_host": "REACHY_DAEMON_HOST",
    "daemon_port": "REACHY_DAEMON_PORT",
    "daemon_serial_port": "REACHY_DAEMON_SERIAL_PORT",
    "daemon_simulation": "REACHY_DAEMON_SIMULATION",
    "daemon_state_file": "REACHY_DAEMON_STATE_FILE",
    "auto_wake_up": "REACHY_AUTO_WAKE_UP",
    "auto_sleep": "REACHY_AUTO_SLEEP",
    "dance_enabled": "REACHY_DANCE_ENABLED",
    "wobbling_enabled": "REACHY_WOBBLING_ENABLED",
    "talk_motion_yaw_max_deg": "REACHY_TALK_YAW_MAX_DEG",
    "talk_motion_pitch_max_deg": "REACHY_TALK_PITCH_MAX_DEG",
    "talk_motion_roll_max_deg": "REACHY_TALK_ROLL_MAX_DEG",
    "talk_motion_gain": "REACHY_TALK_MOTION_GAIN",
    "doa_enabled": "REACHY_DOA_ENABLED",
    "dance_music_dir": "REACHY_DANCE_MUSIC_DIR",
    "dance_duration_s": "REACHY_DANCE_DURATION_S",
    "beat_dance_enabled": "REACHY_BEAT_DANCE_ENABLED",
    "beat_music_path": "REACHY_BEAT_MUSIC_PATH",
    "beat_timeline_path": "REACHY_BEAT_TIMELINE_PATH",
    "gesture_backend": "REACHY_GESTURE_BACKEND",
    "gesture_coreml_model_path": "REACHY_GESTURE_COREML_MODEL",
    "gesture_torchscript_model_path": "REACHY_GESTURE_TORCHSCRIPT_MODEL",
    "gesture_tensorrt_engine_path": "REACHY_GESTURE_TENSORRT_ENGINE",
    "gesture_torch2trt_model_path": "REACHY_GESTURE_TORCH2TRT_MODEL",
    "gesture_inference_fps": "REACHY_GESTURE_INFERENCE_FPS",
    "gesture_keypoint_confidence": "REACHY_GESTURE_KEYPOINT_CONFIDENCE",
    "gesture_confirmation_ms": "REACHY_GESTURE_CONFIRMATION_MS",
    "gesture_lost_timeout_ms": "REACHY_GESTURE_LOST_TIMEOUT_MS",
    "gesture_tracking_deadzone": "REACHY_GESTURE_TRACKING_DEADZONE",
    "gesture_tracking_alpha": "REACHY_GESTURE_TRACKING_ALPHA",
    "gesture_tracking_max_step_deg": "REACHY_GESTURE_TRACKING_MAX_STEP_DEG",
    "gesture_head_yaw_max_deg": "REACHY_GESTURE_HEAD_YAW_MAX_DEG",
    "gesture_head_pitch_max_deg": "REACHY_GESTURE_HEAD_PITCH_MAX_DEG",
    "gesture_body_yaw_max_deg": "REACHY_GESTURE_BODY_YAW_MAX_DEG",
    "ezviz_app_key": "EZVIZ_APP_KEY",
    "ezviz_app_secret": "EZVIZ_APP_SECRET",
    "ezviz_device_serial": "EZVIZ_DEVICE_SERIAL",
    "ezviz_channel_no": "EZVIZ_CHANNEL_NO",
    "journal_auto_sync_interval_minutes": "JOURNAL_AUTO_SYNC_INTERVAL_MINUTES",
    "journal_index_v3_enabled": "JOURNAL_INDEX_V3",
    "source_router_v2_enabled": "SOURCE_ROUTER_V2",
    "session_state_v2_enabled": "SESSION_STATE_V2",
    "vision_policy": "REACHY_VISION_POLICY",
    "visual_context_ttl_s": "REACHY_VISUAL_CONTEXT_TTL_S",
    "vision_max_calls_per_turn": "REACHY_VISION_MAX_CALLS_PER_TURN",
}


def _coerce(value: str, typ: Any) -> Any:
    if typ in (bool, "bool"):
        return value.lower() in ("1", "true", "yes", "on")
    if typ in (int, "int"):
        return int(value)
    if typ in (float, "float"):
        return float(value)
    return value


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Build Config from defaults, optional YAML, then environment overrides."""
    data: dict[str, Any] = {}

    # 1. YAML config file (lowest priority above defaults)
    if path and Path(path).exists():
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            known = {f.name for f in fields(Config)}
            data = {k: v for k, v in loaded.items() if k in known}

    cfg = Config(**data)

    # 2. Environment overrides (highest priority)
    type_by_name = {f.name: f.type for f in fields(Config)}
    for field_name, env_name in _ENV.items():
        raw = os.environ.get(env_name)
        if raw is not None:
            setattr(cfg, field_name, _coerce(raw, type_by_name.get(field_name, str)))

    # Backward-compatible camera selector used by the existing deployment.
    # REACHY_CAMERA_DEVICE is the canonical name and wins when both are set.
    camera_raw = os.environ.get("REACHY_CAMERA_DEVICE")
    if camera_raw is None:
        camera_raw = os.environ.get("CHAIHUO_CAMERA_DEVICE")
    if camera_raw is not None:
        camera_raw = camera_raw.strip()
        cfg.camera_device = int(camera_raw) if camera_raw.isdigit() else camera_raw

    # A previous deployment shipped this truncated spelling. Normalize it at
    # the boundary so an old .env cannot route TTS into the wrong HTTP client.
    if cfg.bailian_tts_model == "qwen3-tts-flash-realtim":
        logger.warning(
            "修正旧 TTS 模型拼写 qwen3-tts-flash-realtim -> qwen3-tts-flash-realtime"
        )
        cfg.bailian_tts_model = "qwen3-tts-flash-realtime"

    cfg.search_policy = str(cfg.search_policy or "auto").strip().lower()
    if cfg.search_policy not in {"auto", "explicit", "off"}:
        logger.warning("未知搜索策略 %r，回退为 auto", cfg.search_policy)
        cfg.search_policy = "auto"
    # The legacy flag may only turn search off; it can no longer silently
    # downgrade auto planning into an unobservable in-memory toggle.
    legacy_search = os.environ.get("BAILIAN_ENABLE_SEARCH")
    if legacy_search is not None and not _coerce(legacy_search, bool):
        cfg.search_policy = "off"
    cfg.enable_search = cfg.search_policy != "off"

    cfg.vision_policy = str(cfg.vision_policy or "semantic").strip().lower()
    if cfg.vision_policy not in {"semantic", "semantic_shadow", "explicit", "off"}:
        logger.warning("未知视觉策略 %r，回退为 semantic", cfg.vision_policy)
        cfg.vision_policy = "semantic"
    cfg.visual_context_ttl_s = max(0.0, float(cfg.visual_context_ttl_s))
    cfg.vision_max_calls_per_turn = max(1, int(cfg.vision_max_calls_per_turn))

    cfg.gesture_backend = str(cfg.gesture_backend or "auto").strip().lower()
    if cfg.gesture_backend not in {"auto", "coreml", "mps", "tensorrt"}:
        logger.warning("未知手势推理后端 %r，回退为 auto", cfg.gesture_backend)
        cfg.gesture_backend = "auto"
    cfg.gesture_inference_fps = max(1.0, float(cfg.gesture_inference_fps))
    cfg.gesture_confirmation_ms = max(50, int(cfg.gesture_confirmation_ms))
    cfg.gesture_lost_timeout_ms = max(100, int(cfg.gesture_lost_timeout_ms))
    cfg.gesture_tracking_deadzone = max(0.0, min(0.25, float(cfg.gesture_tracking_deadzone)))
    cfg.gesture_tracking_alpha = max(0.01, min(1.0, float(cfg.gesture_tracking_alpha)))
    cfg.gesture_tracking_max_step_deg = max(
        0.1, min(10.0, float(cfg.gesture_tracking_max_step_deg))
    )
    cfg.gesture_head_yaw_max_deg = max(
        1.0, min(45.0, float(cfg.gesture_head_yaw_max_deg))
    )
    cfg.gesture_head_pitch_max_deg = max(
        1.0, min(35.0, float(cfg.gesture_head_pitch_max_deg))
    )
    cfg.gesture_body_yaw_max_deg = max(
        1.0, min(60.0, float(cfg.gesture_body_yaw_max_deg))
    )

    return cfg
