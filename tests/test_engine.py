from __future__ import annotations

import asyncio
import time

import pytest

from chaihuo_reachy.config import Config
from chaihuo_reachy.engine import (
    ConversationEngine,
    _JOURNAL_UNKNOWN,
    _WAKE_ONLY,
    _build_system_prompt,
    _enforce_journal_target_date,
    _extract_target_date,
)


def test_manual_location_is_injected_as_natural_location_context() -> None:
    prompt = _build_system_prompt(Config(manual_location="上海市徐汇区西岸艺术中心"))
    assert "【当前位置】" in prompt
    assert "上海市徐汇区西岸艺术中心" in prompt
    assert "【人工设置当前位置】" not in prompt


@pytest.mark.asyncio
async def test_manual_location_takes_precedence_for_location_queries() -> None:
    engine = ConversationEngine(Config(manual_location="北京市朝阳区三里屯"))
    assert await engine._tool_get_current_location() == (
        "我们现在在北京市朝阳区三里屯。"
    )


def test_bare_wake_word_returns_sentinel_and_opens_followup_window() -> None:
    engine = ConversationEngine(Config())
    assert engine._accept_transcript("皮皮虾") == _WAKE_ONLY
    assert engine._accept_transcript("今天天气怎么样") == "今天天气怎么样"


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
        Config(asr_initial_silence_timeout_s=0.2, voice_activity_threshold=0.06),
        audio_backend=_VoiceGateAudio([0.0, 0.08, 0.08]),  # type: ignore[arg-type]
    )
    received: list[list[bytes]] = []

    async def fake_asr(initial_audio=None):
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
        return [{
            "slug": "exact-date",
            "title": f"基地车日记 {date_str}",
            "date": date_str,
            "source_url": "https://example.test/exact-date",
            "source_updated_at": f"{date_str}T16:54:40Z",
            "content": "这是已经完整下载并验证过的日记正文。" * 20,
            "score": 1.0,
        }]


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
        "start", "middle", "end",
    ]


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
async def test_ambiguous_visual_request_asks_direction_without_camera() -> None:
    engine = ConversationEngine(Config())
    result = await engine.process_text("帮我看看")
    assert "前" in result["reply"]
    assert "车外" in result["reply"]
    assert result["intent"] == "ambiguous_camera"


@pytest.mark.asyncio
async def test_front_and_rear_routes_use_different_handlers() -> None:
    engine = ConversationEngine(Config())
    called: list[str] = []

    async def front(_image=None) -> str:
        called.append("front")
        return "前置画面"

    async def rear() -> str:
        called.append("rear")
        return "后视画面"

    engine._tool_take_photo = front  # type: ignore[method-assign]
    engine._tool_capture_rear_view = rear  # type: ignore[method-assign]
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

    assert "这轮没有拍照" in result["reply"]
    assert "冰美式" not in result["reply"]
