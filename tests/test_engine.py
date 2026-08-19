from __future__ import annotations

import asyncio
import struct
import time
import wave
from pathlib import Path

import pytest
import time

from chaihuo_reachy.config import Config
from chaihuo_reachy.engine import (
    ConversationEngine,
    _DANCE_REPLY_INJECTION,
    _JOURNAL_UNKNOWN,
    _WAKE_ONLY,
    _build_system_prompt,
    _enforce_journal_target_date,
    _extract_target_date,
)
from chaihuo_reachy.vision import VisualObservation


def test_manual_location_is_not_injected_as_live_context() -> None:
    prompt = _build_system_prompt(Config(manual_location="上海市徐汇区西岸艺术中心"))
    assert "上海市徐汇区西岸艺术中心" not in prompt


@pytest.mark.asyncio
async def test_manual_location_is_not_a_realtime_fallback() -> None:
    engine = ConversationEngine(Config(manual_location="北京市朝阳区三里屯"))
    result = await engine._tool_get_current_location()
    assert result["place"] == ""
    assert result["source"] == "unavailable"
    assert result["ok"] is False


def test_cloud_engine_bare_wake_word_returns_sentinel_and_opens_followup_window() -> (
    None
):
    engine = ConversationEngine(Config(wake_engine="cloud"))
    assert engine._accept_transcript("皮皮虾") == _WAKE_ONLY
    assert engine._accept_transcript("今天天气怎么样") == "今天天气怎么样"


def test_wake_word_in_sentence_middle_or_end_is_not_stripped() -> None:
    # A wake word inside the user's own sentence must not truncate the
    # instruction: "跳个正常点的舞蹈。皮皮虾。" previously degraded to a
    # bare-wake-word canned response and the request was lost.
    engine = ConversationEngine(Config(wake_engine="cloud"))
    assert engine._accept_transcript("皮皮虾") == _WAKE_ONLY  # open the window
    assert engine._accept_transcript("跳个正常点的舞蹈。皮皮虾。") == (
        "跳个正常点的舞蹈。皮皮虾。"
    )
    assert engine._accept_transcript("我不吃皮皮虾") == "我不吃皮皮虾"
    # Leading wake word is still stripped correctly.
    assert engine._accept_transcript("皮皮虾今天天气怎么样") == "今天天气怎么样"


def test_local_engine_only_treats_bare_wake_word_as_wake() -> None:
    # With the local KWS engine, a transcript that is exactly the wake word
    # still gets the canned response; anything else is a real instruction
    # (no substring stripping — KWS already gated the turn).
    engine = ConversationEngine(Config(wake_engine="local"))
    assert engine._accept_transcript("皮皮虾") == _WAKE_ONLY
    assert engine._accept_transcript("今天天气怎么样") == "今天天气怎么样"
    assert engine._accept_transcript("我不吃皮皮虾") == "我不吃皮皮虾"


@pytest.mark.asyncio
async def test_bare_wake_response_is_awaited_inside_turn() -> None:
    engine = ConversationEngine(Config())
    order: list[str] = []

    async def listen(**kwargs) -> str:
        order.append("listen")
        return "皮皮虾"

    async def speak() -> None:
        order.append("speak-start")
        await asyncio.sleep(0.01)
        order.append("speak-end")

    engine._listen_for_speech = listen  # type: ignore[method-assign]
    engine._speak_wake_response = speak  # type: ignore[method-assign]
    await engine._run_turn()
    order.append("turn-returned")
    assert order == ["listen", "speak-start", "speak-end", "turn-returned"]


@pytest.mark.asyncio
async def test_listening_gate_waits_until_echo_window_expires() -> None:
    engine = ConversationEngine(Config())
    engine._listen_not_before = time.monotonic() + 0.04
    started = time.monotonic()
    await engine._wait_for_listening_gate()
    assert time.monotonic() - started >= 0.03


@pytest.mark.asyncio
async def test_listening_gate_waits_while_tts_is_active() -> None:
    engine = ConversationEngine(Config())
    engine._tts_playing = True

    async def release() -> None:
        await asyncio.sleep(0.03)
        engine._tts_playing = False

    release_task = asyncio.create_task(release())
    await engine._wait_for_listening_gate()
    await release_task


def test_runtime_status_distinguishes_speech_from_other_speaker_audio() -> None:
    class Audio:
        resolved_info: dict[str, object] = {}
        is_playing = True

        @staticmethod
        def play_rms() -> float:
            return 0.0

    class Motion:
        is_talk_shaking = True

    engine = ConversationEngine(
        Config(),
        audio_backend=Audio(),
        motion=Motion(),  # type: ignore[arg-type]
    )
    status = engine.runtime_status()
    assert status["speaker_playing"] is True
    assert status["speech_audio_playing"] is False

    engine._speech_pcm_active = True
    status = engine.runtime_status()
    assert status["speech_audio_playing"] is True
    assert status["talk_motion_active"] is True


class _VoiceGateAudio:
    def __init__(self, levels: list[float]) -> None:
        self.levels = iter(levels)
        self.capture_rms = 0.0
        self.chunks = [b"pre-roll-1", b"pre-roll-2", b"speech"]

    async def start_capture(self):
        for chunk in self.chunks:
            self.capture_rms = next(self.levels, 0.0)
            yield chunk
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_standby_waits_for_local_voice_before_opening_cloud_asr() -> None:
    engine = ConversationEngine(
        Config(
            asr_initial_silence_timeout_s=0.2,
            voice_activity_threshold=0.06,
            vad_model_path="models/vad/not-installed.onnx",
        ),
        audio_backend=_VoiceGateAudio([0.0, 0.08, 0.08]),  # type: ignore[arg-type]
    )
    received: list[list[bytes]] = []

    async def fake_asr(capture, initial_audio=None):
        received.append(initial_audio or [])
        return "皮皮虾你好"

    engine._listen_cloud_asr = fake_asr  # type: ignore[method-assign]
    assert await engine._listen_for_speech() == "皮皮虾你好"
    assert received == [[b"pre-roll-1", b"pre-roll-2", b"speech"]]


@pytest.mark.asyncio
async def test_barge_in_requires_sustained_voice_and_stops_playback() -> None:
    class Audio:
        def __init__(self) -> None:
            self.values = iter([0.0] * 6 + [0.1] * 6)
            self.stopped = False

        @property
        def capture_rms(self):
            return next(self.values, 0.1)

        def stop_playback(self):
            self.stopped = True

    audio = Audio()
    engine = ConversationEngine(
        Config(barge_in_enabled=True, barge_in_sensitivity=0.06),
        audio_backend=audio,
    )  # type: ignore[arg-type]
    engine._tts_audio_started.set()
    await engine._watch_barge_in()
    assert engine._barge_in_requested
    assert audio.stopped


class EmptyJournalMemory:
    def search(self, _query: str, k: int = 3) -> list[dict]:
        return []

    def search_by_date(self, _date: str, k: int = 3) -> list[dict]:
        return []


class CompleteFetcher:
    async def sync(self, memory_store=None, **_kwargs) -> list[dict]:
        return []

    def health(self) -> dict:
        return {
            "expected": 52,
            "complete": 52,
            "last_checked_at": "2026-07-30T00:00:00Z",
            "last_success_at": "2026-07-30T00:00:00Z",
            "failures": [],
        }


class PartialFetcher:
    async def sync(self, memory_store=None, **_kwargs) -> list[dict]:
        raise RuntimeError("one private TOC entry returned 401")

    def health(self) -> dict:
        return {
            "expected": 54,
            "complete": 53,
            "last_checked_at": "2026-07-31T01:45:00+08:00",
            "last_success_at": "2026-07-31T01:39:00+08:00",
            "failures": ["private-entry: 401"],
        }


class ExactDateMemory:
    def __init__(self, target_date: str) -> None:
        self.target_date = target_date

    def search(self, _query: str, k: int = 3) -> list[dict]:
        return []

    def search_by_date(self, date_str: str, k: int = 3) -> list[dict]:
        assert date_str == self.target_date
        return [
            {
                "slug": "exact-date",
                "title": f"基地车日记 {date_str}",
                "date": date_str,
                "source_url": "https://example.test/exact-date",
                "source_updated_at": f"{date_str}T16:54:40Z",
                "content": "这是已经完整下载并验证过的日记正文。" * 20,
                "score": 1.0,
            }
        ]


class JourneyScopeMemory:
    def __init__(self) -> None:
        self.entries = [
            {
                "slug": slug,
                "title": title,
                "date": date,
                "source_url": f"https://example.test/{slug}",
                "source_updated_at": f"{date}T00:00:00Z",
                "content": body * 30,
                "score": 1.0,
            }
            for slug, title, date, body in (
                ("start", "山西启程", "2026-07-29", "驶入山西临汾。"),
                ("middle", "临汾交流", "2026-07-31", "在临汾三中交流。"),
                ("end", "太原活动", "2026-08-03", "太原站点活动。"),
            )
        ]

    def search_journey_scope(self, _query: str, k: int = 6) -> list[dict]:
        return self.entries[:k]

    def search(self, _query: str, k: int = 3) -> list[dict]:
        return self.entries[:k]

    def search_by_date(self, _date: str, k: int = 3) -> list[dict]:
        return self.entries[:k]


class JourneyOverviewMemory(JourneyScopeMemory):
    def search_journey_overview(self, k: int = 80) -> list[dict]:
        return self.entries[:k]

    def format_journey_overview(self) -> str:
        return "按已验证日记，我们走过山西临汾和太原等站点。"


@pytest.mark.asyncio
async def test_vehicle_fact_without_evidence_returns_fixed_unknown() -> None:
    engine = ConversationEngine(Config())
    engine._memory = EmptyJournalMemory()  # type: ignore[assignment]
    engine._journal_fetcher = CompleteFetcher()  # type: ignore[assignment]
    result = await engine.process_text("基地车有多少名队员？")
    assert result["reply"] == _JOURNAL_UNKNOWN
    assert result["intent"] == "journal"


@pytest.mark.asyncio
async def test_partial_directory_keeps_individually_verified_exact_date() -> None:
    target_date = _extract_target_date("昨天发生了什么？")
    assert target_date is not None
    engine = ConversationEngine(Config())
    engine._memory = ExactDateMemory(target_date)  # type: ignore[assignment]
    engine._journal_fetcher = PartialFetcher()  # type: ignore[assignment]

    context = await engine._verified_journal_context("昨天发生了什么？")

    assert target_date in context
    assert "这是已经完整下载并验证过的日记正文" in context
    assert engine._current_sources[0]["slug"] == "exact-date"


@pytest.mark.asyncio
async def test_journey_scope_context_includes_all_verified_route_days() -> None:
    engine = ConversationEngine(Config())
    engine._memory = JourneyScopeMemory()  # type: ignore[assignment]
    engine._journal_fetcher = CompleteFetcher()  # type: ignore[assignment]

    context = await engine._verified_journal_context(
        "我们在山西都去了哪些站点，帮我回忆一下"
    )

    assert all(title in context for title in ("山西启程", "临汾交流", "太原活动"))
    assert [source["slug"] for source in engine._current_sources] == [
        "start",
        "middle",
        "end",
    ]


@pytest.mark.asyncio
async def test_full_journey_overview_is_historical_and_includes_all_titles() -> None:
    engine = ConversationEngine(Config())
    engine._memory = JourneyOverviewMemory()  # type: ignore[assignment]
    engine._journal_fetcher = CompleteFetcher()  # type: ignore[assignment]

    context = await engine._verified_journal_context("我们都去过什么地方？")

    assert "不是当前位置查询" in context
    assert "不要回答‘我们现在在哪里’" in context
    assert all(title in context for title in ("山西启程", "临汾交流", "太原活动"))


@pytest.mark.asyncio
async def test_full_journey_question_never_calls_live_location_tool() -> None:
    engine = ConversationEngine(Config(manual_location="北京市"))
    engine._memory = JourneyOverviewMemory()  # type: ignore[assignment]
    engine._journal_fetcher = CompleteFetcher()  # type: ignore[assignment]
    engine._set_session_location("北京市清华大学")
    prompts: list[list[dict]] = []

    async def summarize(messages: list[dict]) -> tuple[str, str]:
        prompts.append(messages)
        return "我们走过山西临汾和太原等站点。", ""

    async def forbidden_location_tool(_messages: list[dict]) -> list[dict]:
        raise AssertionError("historical route question must not call live location")

    engine._think_text_only = summarize  # type: ignore[method-assign]
    engine._prepare_location_tool_messages = forbidden_location_tool  # type: ignore[method-assign]

    result = await engine.process_text("我们都去过什么地方？")

    assert result["intent"] == "journey_recall"
    assert result["reply"] == "按已验证日记，我们走过山西临汾和太原等站点。"
    assert prompts == []


def test_relative_journal_date_guard_corrects_neighboring_date() -> None:
    reply = "大前天是2026年07月29日，但日记记录的是另一天。"
    guarded = _enforce_journal_target_date(
        "大前天发生了什么？",
        reply,
        "2026-07-28",
    )
    assert guarded.startswith("大前天是2026年7月28日")
    assert "大前天是2026年07月29日" not in guarded


@pytest.mark.asyncio
async def test_ambiguous_visual_request_defaults_to_front() -> None:
    engine = ConversationEngine(Config())
    called: list[str] = []

    async def observe(scope: str, **_kwargs) -> VisualObservation:
        called.append(scope)
        return VisualObservation.failure(scope=scope, error="我现在暂时看不到前面。")

    engine._observe_scene = observe  # type: ignore[method-assign]
    result = await engine.process_text("帮我看看")
    assert "前" in result["reply"]
    assert called == ["front"]
    assert result["intent"] == "ambiguous_camera"


@pytest.mark.asyncio
async def test_front_and_rear_routes_use_different_handlers() -> None:
    engine = ConversationEngine(Config())
    called: list[str] = []

    async def observe(scope: str, **_kwargs) -> VisualObservation:
        called.append(scope)
        return VisualObservation.success(scope=scope, facts=f"{scope}有一个杯子")

    async def synthesize(_messages) -> tuple[str, str]:
        assert engine._current_visual_observation is not None
        scope = engine._current_visual_observation.scope
        return ("我看到前面有一个杯子。" if scope == "front" else "我看到车外有一辆车。", "")

    engine._observe_scene = observe  # type: ignore[method-assign]
    engine._think_text_only = synthesize  # type: ignore[method-assign]
    front_result = await engine.process_text("你能看到什么？")
    rear_result = await engine.process_text("外面有什么？")
    assert called == ["front", "rear"]
    assert front_result["intent"] == "front_camera"
    assert rear_result["intent"] == "rear_camera"


@pytest.mark.asyncio
async def test_general_reply_cannot_claim_camera_view_without_capture() -> None:
    engine = ConversationEngine(Config())

    async def hallucinated_reply(_messages) -> tuple[str, str]:
        return (
            "我正用摄像头看着车里呢——阳光照进来，仪表盘上有一杯冰美式。",
            "",
        )

    engine._think_text_only = hallucinated_reply  # type: ignore[method-assign]
    result = await engine.process_text("随便聊两句")

    assert "没有真正看清楚" in result["reply"]
    assert "冰美式" not in result["reply"]


@pytest.mark.asyncio
async def test_semantic_visual_turn_exposes_tool_and_uses_grounded_result() -> None:
    engine = ConversationEngine(Config(vision_policy="semantic"))
    observed: list[tuple[str, str]] = []

    async def observe(scope: str, *, focus: str, **_kwargs) -> VisualObservation:
        observed.append((scope, focus))
        return VisualObservation.success(scope=scope, facts="手里拿着一个黄色万用表")

    async def tool_aware_reply(_messages) -> tuple[str, str]:
        assert engine._active_response_tools[0]["name"] == "observe_scene"
        assert engine._active_response_tool_handler is not None
        result = await engine._active_response_tool_handler(
            "observe_scene", {"scope": "front", "focus": "确认手里的物体"}
        )
        assert result["ok"] is True
        return "这是一个黄色万用表。", ""

    engine._observe_scene = observe  # type: ignore[method-assign]
    engine._think_text_only = tool_aware_reply  # type: ignore[method-assign]
    result = await engine.process_text("我手里拿的是什么？")

    assert observed == [("front", "确认手里的物体")]
    assert result["reply"] == "这是一个黄色万用表。"
    assert result["sources"][0]["type"] == "live_vision"
    assert engine.runtime_status()["vision"]["called"] is True


@pytest.mark.asyncio
async def test_nonvisual_turn_does_not_expose_camera_tool() -> None:
    engine = ConversationEngine(Config(vision_policy="semantic"))

    async def reply(_messages) -> tuple[str, str]:
        assert engine._active_response_tools == []
        return "我觉得关键在于先明确目标。", ""

    engine._think_text_only = reply  # type: ignore[method-assign]
    result = await engine.process_text("你怎么看这个问题？")
    assert result["reply"].startswith("我觉得")
    assert engine.runtime_status()["vision"]["called"] is False


@pytest.mark.asyncio
async def test_mechanism_heavy_visual_reply_is_replaced() -> None:
    engine = ConversationEngine(Config())

    async def observe(scope: str, **_kwargs) -> VisualObservation:
        return VisualObservation.success(scope=scope, facts="桌上放着一个红色杯子")

    async def bad_reply(_messages) -> tuple[str, str]:
        return "根据照片分析，画面中显示一个红色杯子。", ""

    engine._observe_scene = observe  # type: ignore[method-assign]
    engine._think_text_only = bad_reply  # type: ignore[method-assign]
    result = await engine.process_text("你看到了什么？")
    assert result["reply"] == "我看到桌上放着一个红色杯子。"
    assert all(
        word not in result["reply"]
        for word in ("拍照", "照片", "图片", "画面", "摄像头", "根据图片")
    )


@pytest.mark.asyncio
async def test_referential_followup_reuses_short_lived_visual_observation() -> None:
    engine = ConversationEngine(Config(visual_context_ttl_s=15))
    observation = VisualObservation.success(scope="front", facts="一个蓝色杯子")
    from chaihuo_reachy.vision import VisualCache

    engine._visual_cache = VisualCache(
        observation=observation,
        jpeg=b"private-jpeg",
        stored_monotonic=time.monotonic(),
        focus="它是什么颜色",
    )

    async def forbidden_capture(*_args, **_kwargs):
        raise AssertionError("fresh visual follow-up must not recapture")

    engine._capture_visual_jpeg = forbidden_capture  # type: ignore[method-assign]
    assert engine._should_reuse_visual("它是什么颜色？", "front")
    reused = await engine._observe_scene(
        "front", focus="它是什么颜色", prefer_cache=True
    )
    assert reused.observation_id == observation.observation_id
    assert not engine._should_reuse_visual("它现在还在吗？", "front")


# ── Dance backing music ───────────────────────────────────────────────────


class _MusicFakeAudio:
    backend_name = "fake"

    def __init__(self) -> None:
        self.plays: list[bytes] = []
        self.output_sr: int | None = None
        self.capture_rms = 0.0
        self.stopped = False

    def set_output_sample_rate(self, sr: int) -> None:
        self.output_sr = sr

    async def play(self, pcm: bytes) -> None:
        self.plays.append(pcm)

    def stop_playback(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_talk_motion_stops_only_after_audible_playback_drains() -> None:
    class Audio:
        def __init__(self) -> None:
            self._playing = False
            self._drain_at = 0.0

        async def play(self, _pcm: bytes) -> None:
            self._playing = True

        def set_output_sample_rate(self, _sr: int) -> None:
            pass

        def mark_playback_done(self) -> None:
            self._drain_at = time.monotonic() + 0.08

        @property
        def is_playing(self) -> bool:
            if self._playing and time.monotonic() >= self._drain_at > 0:
                self._playing = False
            return self._playing

        async def start_capture(self):
            while True:
                await asyncio.sleep(1)
                yield b""

    class Motion:
        def __init__(self) -> None:
            self.active = False
            self.started_at = 0.0
            self.stopped_at = 0.0

        def start_talk_motion(self) -> None:
            self.active = True
            self.started_at = time.monotonic()

        def stop_talk_motion(self, *, immediate: bool = False) -> None:
            self.active = False
            self.stopped_at = time.monotonic()

    audio = Audio()
    motion = Motion()
    engine = ConversationEngine(
        Config(wobbling_enabled=True),
        audio_backend=audio,  # type: ignore[arg-type]
        motion=motion,  # type: ignore[arg-type]
    )

    async with engine._speaking_scope():
        pcm = b"\x00\x10" * 1600
        await engine._play_tts_audio(pcm, 16000)
        assert motion.active

    assert not motion.active
    assert motion.stopped_at >= audio._drain_at


def test_speaker_observer_only_forwards_conversational_audio() -> None:
    class Motion:
        def __init__(self) -> None:
            self.blocks: list[tuple[bytes, int]] = []

        def feed_talk_audio(self, pcm: bytes, sample_rate: int) -> None:
            self.blocks.append((pcm, sample_rate))

    motion = Motion()
    engine = ConversationEngine(Config(wobbling_enabled=True), motion=motion)  # type: ignore[arg-type]
    engine._on_speaker_pcm(b"prompt", 16_000)
    assert motion.blocks == []

    engine._speech_pcm_active = True
    engine._on_speaker_pcm(b"speech", 16_000)
    assert motion.blocks == [(b"speech", 16_000)]

    engine._dance_loop_active = True
    engine._on_speaker_pcm(b"music", 48_000)
    assert motion.blocks == [(b"speech", 16_000)]


def _make_wav(path: Path, sr: int = 16000, seconds: float = 0.4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = b"".join(struct.pack("<h", 0) for _ in range(int(sr * seconds)))
        wf.writeframes(frames)


@pytest.mark.asyncio
async def test_play_dance_music_loops_until_cancelled(tmp_path) -> None:
    _make_wav(tmp_path / "happy.wav", sr=16000, seconds=0.4)
    audio = _MusicFakeAudio()
    engine = ConversationEngine(
        Config(dance_music_dir=str(tmp_path)),
        audio_backend=audio,  # type: ignore[arg-type]
    )
    task = asyncio.create_task(engine._play_dance_music("happy"))
    await asyncio.sleep(0.3)  # let a couple of chunks feed
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert len(audio.plays) > 0
    assert audio.output_sr == 16000  # track sample rate forwarded to audio


@pytest.mark.asyncio
async def test_play_dance_music_silent_without_track(tmp_path) -> None:
    audio = _MusicFakeAudio()
    engine = ConversationEngine(
        Config(dance_music_dir=str(tmp_path)),
        audio_backend=audio,  # type: ignore[arg-type]
    )
    await engine._play_dance_music("happy")  # no music dir → returns quietly
    assert audio.plays == []


@pytest.mark.asyncio
async def test_tool_dance_plays_music_while_dancing(tmp_path) -> None:
    _make_wav(tmp_path / "happy.wav", sr=16000, seconds=0.4)

    class _FakeMotion:
        async def dance(self, style: str, **kwargs) -> dict:
            await asyncio.sleep(0.2)  # pretend to dance
            return {"style": style, "skipped": 0, "duration": 0.2}

    audio = _MusicFakeAudio()
    engine = ConversationEngine(
        Config(dance_music_dir=str(tmp_path)),
        audio_backend=audio,  # type: ignore[arg-type]
        motion=_FakeMotion(),  # type: ignore[arg-type]
    )
    reply = await engine._tool_dance("happy")
    # Success hands the turn to the LLM for an improvised remark, so the
    # tool itself returns "" and records the finished dance.
    assert reply == ""
    assert engine._pending_dance is not None
    assert engine._pending_dance["style"] == "happy"
    assert engine._pending_dance["turn_id"] == engine._current_turn_id
    assert audio.plays  # music was fed while dancing


@pytest.mark.asyncio
async def test_tool_dance_random_resolves_concrete_style(tmp_path) -> None:
    _make_wav(tmp_path / "swing.wav", sr=16000, seconds=0.4)

    seen: dict[str, str] = {}

    class _FakeMotion:
        async def dance(self, style: str, **kwargs) -> dict:
            seen["style"] = style
            return {"style": style, "skipped": 0, "duration": 0.1}

    audio = _MusicFakeAudio()
    engine = ConversationEngine(
        Config(dance_music_dir=str(tmp_path)),
        audio_backend=audio,  # type: ignore[arg-type]
        motion=_FakeMotion(),  # type: ignore[arg-type]
    )
    await engine._tool_dance("random")
    # random resolves to one concrete choreography, never the literal
    # "random" (which has no matching track anymore).
    assert seen["style"] in {"happy", "swing", "robot", "elegant", "funky", "silly"}
    assert engine._pending_dance is not None
    assert engine._pending_dance["style"] == seen["style"]


@pytest.mark.asyncio
async def test_tool_dance_failure_paths(tmp_path) -> None:
    engine = ConversationEngine(Config(dance_music_dir=str(tmp_path)))
    assert engine._motion is None
    reply = await engine._tool_dance("random")
    assert reply == "运动控制未启用，无法跳舞。"
    assert engine._pending_dance is None


@pytest.mark.asyncio
async def test_motion_style_keyword_mapping() -> None:
    engine = ConversationEngine(Config())
    seen: list[str] = []

    async def fake_dance(style: str, **kwargs) -> str:
        seen.append(style)
        return ""

    engine._tool_dance = fake_dance  # type: ignore[method-assign]
    cases = {
        "跳个舞": "random",
        "随便跳个舞": "random",
        "跳个优雅的舞": "elegant",
        "来段机械舞": "robot",
        "跳个动感的舞": "funky",
        "跳个搞笑的舞": "silly",
        "跳个欢快的舞": "happy",
        "跳个慢一点的舞": "swing",
    }
    for text, expected in cases.items():
        seen.clear()
        await engine._execute_deterministic_motion(text)
        assert seen == [expected], f"{text!r} → {seen}, want {expected!r}"


@pytest.mark.asyncio
async def test_dance_turn_injects_llm_context(tmp_path) -> None:
    class _FakeMotion:
        async def dance(self, style: str, **kwargs) -> dict:
            return {"style": style, "skipped": 0, "duration": 0.1}

    captured: list[list[dict[str, str]]] = []

    async def fake_think(messages: list[dict[str, str]]) -> tuple[str, str]:
        captured.append(messages)
        return "刚才随音乐随便扭了一段，真开心！", ""

    engine = ConversationEngine(
        Config(dance_music_dir=str(tmp_path)),
        motion=_FakeMotion(),  # type: ignore[arg-type]
    )
    engine._think_text_only = fake_think  # type: ignore[method-assign]
    result = await engine.process_text("跳个舞")

    assert result["intent"] == "motion"
    assert result["reply"] == "刚才随音乐随便扭了一段，真开心！"
    assert captured, "LLM branch must run after a successful dance"
    system = captured[0][0]["content"]
    assert "舞蹈提示" in system
    # The style is never revealed to the model — the robot improvises.
    for style in ("swing", "robot", "elegant", "funky", "silly"):
        assert style not in system
    # "happy" collides with the persona's [happy] emotion tag, so verify the
    # injection text itself never names any style instead.
    assert "happy" not in _DANCE_REPLY_INJECTION
    assert engine._pending_dance is None  # consumed by this turn
    assert len(engine._conversation_history) == 2  # user + improvised reply


# ── Beat-dance loop (infinite dance, voice suspended) ─────────────────


class _FakeBeatDance:
    """Records start/stop; returns tiny PCM for the music feeder."""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self._pcm = b"\x00\x00" * 1600  # 0.1s of silence @16k

    def start(self):
        self.started += 1
        return (16000, self._pcm)

    def stop(self) -> None:
        self.stopped += 1

    def status(self) -> dict:
        return {"active": True, "elapsed": 1.0, "mode_label": "MID", "loop_count": 1}


@pytest.mark.asyncio
async def test_run_turn_suspended_while_dance_loop_active() -> None:
    class _CountingAudio(_MusicFakeAudio):
        def __init__(self) -> None:
            super().__init__()
            self.listen_calls = 0

    audio = _CountingAudio()
    engine = ConversationEngine(
        Config(),
        audio_backend=audio,  # type: ignore[arg-type]
        beat_dance=_FakeBeatDance(),  # type: ignore[arg-type]
    )
    engine._dance_loop_active = True
    # _run_turn must return quickly without listening while dancing,
    # and re-assert the dancing state (an interrupted turn's finally
    # would otherwise flip the dashboard back to idle).
    await asyncio.wait_for(engine._run_turn(), timeout=1.0)
    assert engine._state == "dancing"


@pytest.mark.asyncio
async def test_process_text_blocked_while_dance_loop_active() -> None:
    engine = ConversationEngine(Config(), beat_dance=_FakeBeatDance())  # type: ignore[arg-type]
    engine._dance_loop_active = True
    result = await engine.process_text("你好")
    assert "跳舞" in result["reply"]


@pytest.mark.asyncio
async def test_start_stop_beat_dance() -> None:
    audio = _MusicFakeAudio()
    beat = _FakeBeatDance()
    engine = ConversationEngine(
        Config(),
        audio_backend=audio,  # type: ignore[arg-type]
        beat_dance=beat,  # type: ignore[arg-type]
    )
    reply = await engine.start_beat_dance()
    assert "开始跳舞" in reply
    assert engine._dance_loop_active
    assert beat.started == 1
    await asyncio.sleep(0.05)  # let the music feeder task run a beat
    assert audio.plays  # music chunks fed to the speaker
    # idempotent second start
    reply2 = await engine.start_beat_dance()
    assert "已经在跳" in reply2

    reply3 = await engine.stop_beat_dance()
    assert "停啦" in reply3
    assert not engine._dance_loop_active
    assert beat.stopped == 1
    assert audio.stopped  # playback buffer cleared


@pytest.mark.asyncio
async def test_start_beat_dance_without_controller() -> None:
    engine = ConversationEngine(Config())
    reply = await engine.start_beat_dance()
    assert "未启用" in reply
    assert not engine._dance_loop_active


@pytest.mark.asyncio
async def test_start_beat_dance_invalidates_inflight_tts() -> None:
    audio = _MusicFakeAudio()
    engine = ConversationEngine(
        Config(),
        audio_backend=audio,  # type: ignore[arg-type]
        beat_dance=_FakeBeatDance(),  # type: ignore[arg-type]
    )
    engine._tts_generation = 5
    engine._active_tts_generation = 5
    engine._tts_playing = True  # a reply is mid-flight when dance is clicked
    reply = await engine.start_beat_dance()
    assert "开始跳舞" in reply
    assert engine._active_tts_generation == 6  # in-flight TTS invalidated
    assert audio.stopped  # playback buffer flushed before music starts
    # Late chunks — both the old generation and the current one — must not
    # interleave with the music while the dance owns the buffer.
    await engine._play_tts_audio(b"old-gen", 16000, generation=5)
    await engine._play_tts_audio(b"fresh", 16000, generation=6)
    assert not any(b"old-gen" in p for p in audio.plays)
    assert not any(b"fresh" in p for p in audio.plays)
    # After the dance stops, speech playback resumes for the current generation.
    await engine.stop_beat_dance()
    await engine._play_tts_audio(b"after-stop", 16000, generation=6)
    assert any(b"after-stop" in p for p in audio.plays)


@pytest.mark.asyncio
async def test_queue_tts_audio_dropped_while_dancing() -> None:
    audio = _MusicFakeAudio()
    engine = ConversationEngine(
        Config(),
        audio_backend=audio,  # type: ignore[arg-type]
    )
    engine._loop = asyncio.get_running_loop()
    engine._dance_loop_active = True
    # generation=None is the _speak_reply/_speak_wake_response path — the
    # generation bump cannot reach it, so the funnel guard must.
    engine._queue_tts_audio(b"no-gen", 16000, generation=None)
    await asyncio.sleep(0.02)
    assert audio.plays == []


@pytest.mark.asyncio
async def test_barge_in_does_not_kill_music_while_dancing() -> None:
    class Audio:
        def __init__(self) -> None:
            self.values = iter([0.1] * 12)
            self.stopped = False

        @property
        def capture_rms(self):
            return next(self.values, 0.1)

        def stop_playback(self):
            self.stopped = True

    audio = Audio()
    engine = ConversationEngine(
        Config(barge_in_enabled=True, barge_in_sensitivity=0.06),
        audio_backend=audio,  # type: ignore[arg-type]
    )
    engine._tts_audio_started.set()
    engine._dance_loop_active = True
    # Loud speech while dancing: the watcher exits without stopping the music.
    await engine._watch_barge_in()
    assert not engine._barge_in_requested
    assert not audio.stopped


@pytest.mark.asyncio
async def test_speaking_paths_skipped_while_dancing() -> None:
    audio = _MusicFakeAudio()
    engine = ConversationEngine(
        Config(),
        audio_backend=audio,  # type: ignore[arg-type]
    )
    engine._dance_loop_active = True
    # No LLM/TTS client is instantiated: the gate fires before any I/O.
    text, emotion = await engine._think_and_speak([])
    assert text == "" and emotion == ""
    await engine._speak_reply("你好")
    await engine._speak_wake_response()
    await engine._speak_greeting()
    assert audio.plays == []


@pytest.mark.asyncio
async def test_empty_llm_stream_does_not_open_or_flush_tts(monkeypatch) -> None:
    import chaihuo_reachy.engine as engine_module

    tts_events: list[str] = []

    class EmptyLLM:
        last_sources: list[dict] = []
        last_search_used = False
        last_search_error = ""

        def __init__(self, _config) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def response_stream(self, *_args, **_kwargs):
            if False:
                yield ""

    class TrackingTTS:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def open(self) -> None:
            tts_events.append("open")

        async def feed(self, _text: str) -> None:
            tts_events.append("feed")

        async def flush(self) -> None:
            tts_events.append("flush")

        async def close(self) -> None:
            tts_events.append("close")

    class Audio:
        capture_rms = 0.0
        is_playing = False

        def set_output_sample_rate(self, _sample_rate: int) -> None:
            pass

        async def start_capture(self):
            if False:
                yield b""

        async def play(self, _pcm: bytes) -> None:
            pass

        def mark_playback_done(self) -> None:
            pass

    monkeypatch.setattr(engine_module, "BailianLLMClient", EmptyLLM)
    monkeypatch.setattr(engine_module, "BailianTTSClient", TrackingTTS)
    engine = ConversationEngine(
        Config(barge_in_enabled=False, post_playback_silence_s=0),
        audio_backend=Audio(),  # type: ignore[arg-type]
    )

    text, emotion = await asyncio.wait_for(
        engine._think_and_speak([{"role": "user", "content": "你好"}]),
        timeout=0.5,
    )

    assert text == ""
    assert emotion == ""
    assert tts_events == []


@pytest.mark.asyncio
async def test_dashboard_text_preempts_passive_voice_listener() -> None:
    engine = ConversationEngine(Config())
    listening = asyncio.Event()
    cancelled = asyncio.Event()

    async def passive_listen() -> str:
        listening.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return ""

    async def coordinate(text: str, **kwargs):
        return {
            "reply": "对话正常。",
            "emotion": "",
            "memory_context": "",
            "vision_context": "",
            "intent": "general",
            "sources": [],
            "error": None,
            "turn_id": "test-turn",
            "text": text,
            "source": kwargs.get("source"),
        }

    engine._listen_for_speech = passive_listen  # type: ignore[method-assign]
    engine._coordinate_turn = coordinate  # type: ignore[method-assign]
    voice_turn = asyncio.create_task(engine._run_voice_turn())
    await asyncio.wait_for(listening.wait(), timeout=0.2)

    result = await asyncio.wait_for(
        engine.process_text("稳定性测试", client_message_id="web-test"),
        timeout=0.2,
    )
    await asyncio.wait_for(voice_turn, timeout=0.2)

    assert cancelled.is_set()
    assert result["reply"] == "对话正常。"
    assert result["source"] == "dashboard"


@pytest.mark.asyncio
async def test_playback_drain_skips_is_playing_while_dancing() -> None:
    class _FullAudio(_MusicFakeAudio):
        @property
        def is_playing(self) -> bool:
            return True  # music keeps the buffer topped for the whole dance

    audio = _FullAudio()
    engine = ConversationEngine(
        Config(),
        audio_backend=audio,  # type: ignore[arg-type]
    )
    engine._dance_loop_active = True
    # The interrupted turn's _speaking_scope drain must not hang on the music.
    await asyncio.wait_for(engine._wait_for_playback_drain(), timeout=0.5)


# ── Journal-evidence flexibility (科普/常识不拒绝) ─────────────────────


def test_journal_evidence_terms() -> None:
    from chaihuo_reachy.engine import _requires_journal_evidence

    assert _requires_journal_evidence("基地车上一共有多少名队员？")
    assert _requires_journal_evidence("你记不记得我们昨天去了哪里？")
    assert not _requires_journal_evidence("科普一下什么是人工智能？")
    assert not _requires_journal_evidence("城市为什么会有堵车现象？")


@pytest.mark.asyncio
async def test_journal_without_evidence_still_refuses_base_car_facts() -> None:
    engine = ConversationEngine(Config())
    engine._memory = EmptyJournalMemory()  # type: ignore[assignment]
    engine._journal_fetcher = CompleteFetcher()  # type: ignore[assignment]
    result = await engine.process_text("基地车上一共有多少名队员？")
    assert result["reply"] == _JOURNAL_UNKNOWN


@pytest.mark.asyncio
async def test_journal_without_evidence_falls_back_to_general_answer() -> None:
    class _EchoLLMEngine(ConversationEngine):
        async def _think_text_only(self, messages, **kwargs):
            # 如果 system prompt 包含降级提示且不是拒绝，说明走了常识回答
            sys_prompt = messages[0]["content"]
            assert "未能从基地车日记检索到相关记录" in sys_prompt
            return "科普回答：人工智能是……", ""

    engine = _EchoLLMEngine(Config())
    engine._memory = EmptyJournalMemory()  # type: ignore[assignment]
    engine._journal_fetcher = CompleteFetcher()  # type: ignore[assignment]
    # "昨天" 触发日记意图但问题不是基地车事实 → 降级常识回答
    result = await engine.process_text("昨天我在路上看到一只猫，它为什么一直跟着我？")
    assert "科普回答" in result["reply"]


@pytest.mark.asyncio
async def test_missing_journey_evidence_gets_natural_gap_not_hallucination() -> None:
    class _GapEngine(ConversationEngine):
        async def _think_text_only(self, messages, **kwargs):
            prompt = messages[0]["content"]
            assert "未能从基地车日记检索到相关记录" in prompt
            assert "其他高校" in prompt
            return (
                "日记里暂时没有我们在清华发生过什么的记录。"
                "你刚告诉我的当前位置可以单独记住，但我不能把它编成过去的经历。",
                "",
            )

    engine = _GapEngine(Config(bailian_api_key="test"))
    engine._memory = EmptyJournalMemory()  # type: ignore[assignment]
    engine._journal_fetcher = CompleteFetcher()  # type: ignore[assignment]
    result = await engine.process_text("我们在清华大学有什么故事")

    assert result["intent"] == "journey_recall"
    assert "暂时没有" in result["reply"]
    assert "过去的经历" in result["reply"]


@pytest.mark.asyncio
async def test_location_declaration_updates_only_current_session() -> None:
    engine = ConversationEngine(Config())
    result = await engine.process_text("我们现在在北京清华大学")
    assert result["intent"] == "location_update"
    assert "北京清华大学" in result["reply"]
    assert engine._session_location is not None

    engine.clear_conversation()
    assert engine._session_location is None


@pytest.mark.asyncio
async def test_chaihuo_introduction_uses_dedicated_official_knowledge() -> None:
    class _OrgEngine(ConversationEngine):
        async def _think_text_only(self, messages, **kwargs):
            prompt = messages[0]["content"]
            assert "【柴火创客官方知识】" in prompt
            assert "成立于 2011 年" in prompt
            assert "基地车是柴火创客发起" in prompt
            return "柴火创客成立于2011年，基地车是我们发起的移动AI实验室。", ""

    engine = _OrgEngine(Config())
    result = await engine.process_text("介绍一下柴火创客")

    assert result["intent"] == "org_knowledge"
    assert "2011" in result["reply"]
    assert {source["type"] for source in result["sources"]} == {"organization"}


def test_conversation_defaults_are_twenty_turns_and_thirty_minutes() -> None:
    cfg = Config()
    assert cfg.max_history_turns == 20
    assert cfg.session_reset_idle_s == 1800.0
