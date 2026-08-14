from __future__ import annotations

import asyncio
import json
from typing import Any
from types import SimpleNamespace

import pytest

from chaihuo_reachy.bailian.asr_client import ASRResult, BailianASRClient
from chaihuo_reachy.config import Config
from chaihuo_reachy.engine import ConversationEngine


class _FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, message: str) -> None:
        self.messages.append(json.loads(message))


@pytest.mark.asyncio
async def test_asr_configures_vad_once_before_audio_ingestion() -> None:
    client = BailianASRClient(Config())
    websocket = _FakeWebSocket()
    client._ws = websocket  # type: ignore[assignment]
    await client.configure()
    assert len(websocket.messages) == 1
    assert websocket.messages[0]["type"] == "session.update"
    turn_detection = websocket.messages[0]["session"]["turn_detection"]
    assert turn_detection["silence_duration_ms"] == 600
    assert turn_detection["threshold"] == 0.5


def test_runtime_status_exposes_asr_endpoint_policy() -> None:
    engine = ConversationEngine(Config())
    assert engine.runtime_status()["asr"] == {
        "vad_silence_ms": 600,
        "initial_silence_timeout_s": 20.0,
        "speech_max_duration_s": 15.0,
        "last_end_reason": "",
        "frontend_v2": True,
    }


class _FakeAudio:
    capture_rms = 0.0

    async def start_capture(self):
        while True:
            yield b"\0" * 320
            await asyncio.sleep(0.001)


class _FakeASR:
    def __init__(self, events: list[tuple[float, ASRResult]]) -> None:
        self.events = events
        self.finished = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def configure(self) -> None:
        return None

    async def send_audio(self, _chunk: bytes) -> None:
        return None

    async def finish(self) -> None:
        self.finished = True

    async def results(self):
        for delay, result in self.events:
            if delay:
                await asyncio.sleep(delay)
            yield result
        await asyncio.Event().wait()


def _engine_with_fake_asr(monkeypatch, events, *, initial: float = 0.03, maximum: float = 0.06):
    import chaihuo_reachy.engine as engine_module

    asr = _FakeASR(events)
    monkeypatch.setattr(engine_module, "BailianASRClient", lambda _cfg: asr)
    engine = ConversationEngine(
        Config(
            asr_initial_silence_timeout_s=initial,
            asr_speech_max_duration_s=maximum,
        ),
        audio_backend=_FakeAudio(),  # type: ignore[arg-type]
    )
    return engine, asr


def _fake_capture() -> Any:
    """An endless capture iterator, as opened by _listen_for_speech."""
    return _FakeAudio().start_capture().__aiter__()


@pytest.mark.asyncio
async def test_partial_and_final_transcript_are_committed_without_session_update(monkeypatch) -> None:
    engine, asr = _engine_with_fake_asr(monkeypatch, [
        (0.0, ASRResult(text="", speech_started=True)),
        (0.0, ASRResult(text="请介绍一下基地车")),
        (0.0, ASRResult(text="请介绍一下基地车。", is_final=True)),
    ])
    transcripts: list[tuple[str, bool]] = []
    engine.on_transcript(lambda text, final: transcripts.append((text, final)))

    assert await engine._listen_cloud_asr(capture=_fake_capture()) == "请介绍一下基地车。"
    assert transcripts[-1] == ("请介绍一下基地车。", True)
    assert engine._last_asr_end_reason == "completed"


@pytest.mark.asyncio
async def test_initial_silence_timeout_is_distinct_from_speech_timeout(monkeypatch) -> None:
    engine, asr = _engine_with_fake_asr(monkeypatch, [], initial=0.01, maximum=0.06)
    statuses: list[str] = []
    engine.on_asr_status(statuses.append)

    assert await engine._listen_cloud_asr(capture=_fake_capture()) == ""
    assert asr.finished
    assert engine._last_asr_end_reason == "initial_silence_timeout"
    assert statuses[-1] == "未检测到用户开口"


@pytest.mark.asyncio
async def test_speech_can_exceed_initial_window_but_stops_at_speech_maximum(monkeypatch) -> None:
    engine, _asr = _engine_with_fake_asr(monkeypatch, [
        (0.0, ASRResult(text="", speech_started=True)),
        (0.02, ASRResult(text="这是一个较长的问题", is_final=True)),
    ], initial=0.01, maximum=0.05)
    assert await engine._listen_cloud_asr(capture=_fake_capture()) == "这是一个较长的问题"

    timed_out, _asr = _engine_with_fake_asr(monkeypatch, [
        (0.0, ASRResult(text="", speech_started=True)),
    ], initial=0.01, maximum=0.01)
    assert await timed_out._listen_cloud_asr(capture=_fake_capture()) == ""
    assert timed_out._last_asr_end_reason == "speech_max_duration_timeout"


@pytest.mark.asyncio
async def test_local_endpoint_commits_repeated_stable_partial(monkeypatch) -> None:
    import chaihuo_reachy.engine as engine_module

    class Endpoint:
        vad = SimpleNamespace(available=True)

        def __init__(self, **_kwargs) -> None:
            pass

        def update(self, _chunk):
            return SimpleNamespace(
                rms=0.1,
                dbfs=-20.0,
                snr_db=15.0,
                vad_probability=0.9,
                speech=True,
                endpoint=True,
                endpoint_reason="local_silence",
            )

    monkeypatch.setattr(engine_module, "SpeechEndpoint", Endpoint)
    engine, asr = _engine_with_fake_asr(
        monkeypatch,
        [
            (0.0, ASRResult(text="", speech_started=True)),
            (0.0, ASRResult(text="请介绍基地车")),
            (0.0, ASRResult(text="请介绍基地车")),
        ],
        maximum=1.0,
    )
    engine.config.asr_finalize_timeout_s = 0.01

    assert await engine._listen_cloud_asr(capture=_fake_capture()) == "请介绍基地车"
    assert asr.finished
    assert engine._last_asr_end_reason == "finalize_timeout"
