from __future__ import annotations

import asyncio

import pytest

from chaihuo_reachy.dashboard import (
    ChatMessageStore,
    DashboardHub,
    run_websocket_session,
)
from chaihuo_reachy.main import DASHBOARD_HTML


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.controls: asyncio.Queue[dict | None] = asyncio.Queue()

    async def send_json(self, message: dict) -> None:
        self.sent.append(message)

    async def receive_json(self) -> dict:
        message = await self.controls.get()
        if message is None:
            raise RuntimeError("disconnected")
        return message


@pytest.mark.asyncio
async def test_websocket_pushes_live_partial_without_client_polling() -> None:
    hub = DashboardHub()
    websocket = FakeWebSocket()

    async def handle_control(_ws: FakeWebSocket, _message: dict) -> None:
        return None

    task = asyncio.create_task(
        run_websocket_session(
            websocket,
            hub,
            lambda: [{"type": "state", "state": "listening"}],
            handle_control,
        )
    )
    await asyncio.sleep(0)
    event = hub.publish({"type": "transcript", "text": "正在实时识别", "final": False})
    await asyncio.sleep(0.01)
    assert event in websocket.sent
    await websocket.controls.put(None)
    with pytest.raises(RuntimeError, match="disconnected"):
        await task
    assert hub.subscriber_count == 0


@pytest.mark.asyncio
async def test_websocket_receives_controls_while_sender_is_waiting() -> None:
    hub = DashboardHub()
    websocket = FakeWebSocket()
    handled = asyncio.Event()

    async def handle_control(_ws: FakeWebSocket, message: dict) -> None:
        if message["type"] == "get_state":
            handled.set()

    task = asyncio.create_task(
        run_websocket_session(websocket, hub, lambda: [], handle_control)
    )
    await websocket.controls.put({"type": "get_state"})
    await asyncio.wait_for(handled.wait(), timeout=0.2)
    await websocket.controls.put(None)
    with pytest.raises(RuntimeError):
        await task


def test_history_keeps_final_not_stale_partial_trail() -> None:
    hub = DashboardHub()
    hub.publish({"type": "transcript", "text": "你", "final": False})
    hub.publish({"type": "transcript", "text": "你好", "final": False})
    final = hub.publish({"type": "transcript", "text": "你好小柴", "final": True})
    replayed = hub.replay()
    # Final transcript is included
    assert final in replayed
    # ASR history replay contains the final text
    history_events = [e for e in replayed if e.get("type") == "asr_history_replay"]
    assert len(history_events) == 1
    items = history_events[0].get("items", [])
    assert any(item["text"] == "你好小柴" for item in items)
    # No stale partials
    partial_in_replay = [
        e for e in replayed
        if e.get("type") == "transcript" and not e.get("final")
    ]
    assert not partial_in_replay


def test_browser_uses_page_host_and_normalized_chat_protocol() -> None:
    assert 'window.location.host+"/ws"' in DASHBOARD_HTML
    assert 'window.location.protocol==="https:"?"wss:":"ws:"' in DASHBOARD_HTML
    assert 'type:"chat_send"' in DASHBOARD_HTML
    assert 'm.type==="chat_history"' in DASHBOARD_HTML
    assert 'm.type==="chat_message_upsert"' in DASHBOARD_HTML
    assert 'm.type==="chat_message_delta"' in DASHBOARD_HTML
    # The WeChat-style Dashboard must retain the original device controls.
    assert 'id="wakeBtn"' in DASHBOARD_HTML
    assert 'id="volumeRange"' in DASHBOARD_HTML
    assert 'type:"get_wake_word"' in DASHBOARD_HTML
    assert 'type:"set_wake_word"' in DASHBOARD_HTML
    assert 'type:"get_volume"' in DASHBOARD_HTML
    assert 'type:"set_volume"' in DASHBOARD_HTML
    # Opening the HTML file directly must not issue file:// API requests.
    assert 'window.location.protocol==="file:"' in DASHBOARD_HTML
    assert 'http://localhost:8640/' in DASHBOARD_HTML
    assert "dbCam" not in DASHBOARD_HTML
    assert 'm.type==="asr_status"' in DASHBOARD_HTML
    assert 'id="modelInfo"' in DASHBOARD_HTML
    assert 'id="searchInfo"' in DASHBOARD_HTML
    assert 'id="audioFrontend"' in DASHBOARD_HTML
    assert 'id="locationInfo"' in DASHBOARD_HTML
    assert "已联网核验" in DASHBOARD_HTML
    assert "navigator.geolocation.watchPosition" in DASHBOARD_HTML
    assert 'type:"browser_location"' in DASHBOARD_HTML


def test_chat_store_replays_both_sides_sources_and_capture() -> None:
    store = ChatMessageStore(message_limit=10, capture_limit=2)
    user, assistant = store.begin_turn(
        "turn-1", "外面有什么？", source="dashboard", client_message_id="client-1"
    )
    assert user["role"] == "user"
    assert assistant["role"] == "assistant"
    capture_id, updated = store.attach_capture(
        "turn-1", b"\xff\xd8photo", label="车外后视"
    )
    assert updated is not None
    final = store.finalize(
        "turn-1",
        "画面中清晰可见一辆车。",
        sources=[{"title": "日记", "url": "https://example.test"}],
    )
    assert final and final["attachments"][0]["capture_id"] == capture_id
    history = store.history_event()
    assert [message["role"] for message in history["messages"]] == ["user", "assistant"]
    assert history["messages"][1]["sources"]
    assert store.get_capture(capture_id) == b"\xff\xd8photo"


def test_chat_store_limits_captures_and_clear() -> None:
    store = ChatMessageStore(message_limit=4, capture_limit=1)
    store.begin_turn("turn-1", "一", source="voice")
    first, _ = store.attach_capture("turn-1", b"first", label="Reachy 前置")
    second, _ = store.attach_capture("turn-1", b"second", label="Reachy 前置")
    assert store.get_capture(first) is None
    assert store.get_capture(second) == b"second"
    store.clear()
    assert store.history_event()["messages"] == []
