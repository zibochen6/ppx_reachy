"""Configuration management — layered: env vars > YAML file > defaults.

Supports two deployment targets:
  - ``mac`` — development on macOS (uses local mic/camera)
  - ``jetson`` — production on Jetson Orin (uses Reachy SDK + Docker)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

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
    bailian_llm_model: str = "qwen-plus"  # better function calling than qwen-turbo
    bailian_tts_model: str = "qwen3-tts-flash-realtime"
    bailian_tts_voice: str = "Cherry"
    bailian_vlm_model: str = "qwen-vl-plus"

    # ── Web search ────────────────────────────────────────────────────
    enable_search: bool = False  # Enable Bailian built-in web search

    # ── ASR settings ───────────────────────────────────────────────────
    asr_language: str = "zh"
    asr_vad_silence_ms: int = 500   # server-side VAD silence threshold (xiaozhi-style)
    asr_vad_threshold: float = 0.35  # VAD sensitivity — lower = more sensitive
    asr_turn_timeout_s: float = 15.0
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
    # Wake word is detected server-side via cloud ASR transcript matching.
    # No local wake-word model is used — the mic is always streamed to the
    # cloud, and ``_accept_transcript()`` checks for the wake word in the
    # recognised text.  Once detected, a 30 s grace window keeps the
    # conversation open without requiring the wake word on every turn.
    enable_wake_word: bool = True
    wake_words: str = "皮皮虾"
    wake_word_timeout_s: float = 30.0
    wake_response: str = "我在呢！有什么可以帮你的？"
    feedback_sounds_enabled: bool = True   # Play beeps on state transitions
    barge_in_sensitivity: float = 0.06     # Mic RMS threshold for interrupting TTS (0-1)

    # ── Audio ──────────────────────────────────────────────────────────
    # Safe default: auto-detect a unique Reachy full-duplex device. Use the
    # explicit value "default" to opt into Mac/system input + output.
    audio_device: str | int | None = "auto"
    audio_sample_rate: int = 16000
    audio_input_channel: int = 1    # Reachy ch1 = AEC-processed (echo-cancelled)
    audio_mic_gain: float = 1.0     # Unity gain; avoid clipping that degrades ASR

    # ── SDK Media backend ───────────────────────────────────────────────
    # "auto" → SDK daemon 模式下用 local，standalone 用 no_media
    # "local" → 强制用 SDK GStreamer 后端（需 daemon）
    # "no_media" → 回退到 sounddevice + OpenCV 直接访问硬件
    media_backend: str = "auto"

    # ── Daemon ──────────────────────────────────────────────────────────
    # "connect" → 连接已有 SDK daemon
    # "spawn"   → 自动拉起 SDK daemon 子进程
    daemon_mode: str = "connect"
    daemon_host: str = "reachy-mini.local"
    daemon_port: int = 8000
    auto_wake_up: bool = True      # 启动时自动让机器人站起来
    auto_sleep: bool = True        # 关闭时自动让机器人休眠

    # ── Motion / Dance ──────────────────────────────────────────────────
    dance_enabled: bool = True     # 启用运动控制（Dashboard 按钮 + 语音指令）
    wobbling_enabled: bool = True  # 说话时头部自然晃动
    doa_enabled: bool = False      # 麦克风阵列声源定位

    # ── Camera (Reachy Mini USB camera) ────────────────────────────────
    # Mac: auto-detect by name containing "Reachy". Jetson: /dev/video0 or index
    camera_device: int | str = "auto"  # "auto" | int index | "/dev/video0"
    camera_width: int = 640
    camera_height: int = 480

    # ── Location / GPS ──────────────────────────────────────────────────
    location_gpsd_enabled: bool = True     # Try GPSD daemon for real GPS
    location_gpsd_host: str = "127.0.0.1"
    location_gpsd_port: int = 2947
    location_poll_interval_s: float = 2.0   # GPS poll interval

    # ── Conversation ───────────────────────────────────────────────────
    language: str = "zh"  # "zh" | "en"
    max_history_turns: int = 10
    session_reset_idle_s: float = 120.0
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

    # ── Persona ────────────────────────────────────────────────────────
    persona: str = "xiao_chai"

    @property
    def profile_dir(self) -> Path:
        project_profile = Path(__file__).parent.parent.parent / "profiles" / self.persona
        if project_profile.exists():
            return project_profile
        return Path(__file__).with_name("profiles") / self.persona

    def system_prompt(self) -> str:
        """Load the persona's system prompt from profile."""
        p = self.profile_dir / "instructions.txt"
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
    "target": "REACHY_TARGET",
    "asr_language": "BAILIAN_ASR_LANGUAGE",
    "asr_vad_silence_ms": "BAILIAN_ASR_VAD_SILENCE_MS",
    "asr_vad_threshold": "BAILIAN_ASR_VAD_THRESHOLD",
    "asr_min_chars": "BAILIAN_ASR_MIN_CHARS",
    "llm_temperature": "BAILIAN_LLM_TEMPERATURE",
    "llm_max_tokens": "BAILIAN_LLM_MAX_TOKENS",
    "llm_system_prompt": "BAILIAN_LLM_SYSTEM_PROMPT",
    "tts_speech_rate": "BAILIAN_TTS_SPEECH_RATE",
    "tts_volume": "BAILIAN_TTS_VOLUME",
    "tts_sample_rate": "BAILIAN_TTS_SAMPLE_RATE",
    "enable_wake_word": "BAILIAN_ENABLE_WAKE_WORD",
    "wake_words": "BAILIAN_WAKE_WORDS",
    "wake_word_timeout_s": "BAILIAN_WAKE_WORD_TIMEOUT_S",
    "wake_response": "BAILIAN_WAKE_RESPONSE",
    "audio_device": "REACHY_AUDIO_DEVICE",
    "audio_input_channel": "REACHY_AUDIO_INPUT_CHANNEL",
    "audio_mic_gain": "REACHY_MIC_GAIN",
    "camera_device": "REACHY_CAMERA_DEVICE",
    "dashboard_port": "REACHY_DASHBOARD_PORT",
    "language": "REACHY_LANGUAGE",
    "session_reset_idle_s": "REACHY_SESSION_RESET_IDLE_S",
    "media_backend": "REACHY_MEDIA_BACKEND",
    "daemon_mode": "REACHY_DAEMON_MODE",
    "daemon_host": "REACHY_DAEMON_HOST",
    "daemon_port": "REACHY_DAEMON_PORT",
    "auto_wake_up": "REACHY_AUTO_WAKE_UP",
    "auto_sleep": "REACHY_AUTO_SLEEP",
    "dance_enabled": "REACHY_DANCE_ENABLED",
    "wobbling_enabled": "REACHY_WOBBLING_ENABLED",
    "doa_enabled": "REACHY_DOA_ENABLED",
    "ezviz_app_key": "EZVIZ_APP_KEY",
    "ezviz_app_secret": "EZVIZ_APP_SECRET",
    "ezviz_device_serial": "EZVIZ_DEVICE_SERIAL",
    "ezviz_channel_no": "EZVIZ_CHANNEL_NO",
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

    return cfg
