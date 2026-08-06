from __future__ import annotations

from chaihuo_reachy.config import load_config
from chaihuo_reachy.bailian.tts_client import _is_qwen_realtime_model, _is_streaming_model


def test_truncated_legacy_tts_model_is_normalized(monkeypatch) -> None:
    monkeypatch.setenv("BAILIAN_TTS_MODEL", "qwen3-tts-flash-realtim")
    cfg = load_config()
    assert cfg.bailian_tts_model == "qwen3-tts-flash-realtime"
    assert _is_qwen_realtime_model(cfg.bailian_tts_model)
    assert _is_streaming_model(cfg.bailian_tts_model)
