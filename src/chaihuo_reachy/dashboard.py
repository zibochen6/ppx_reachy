"""Realtime Dashboard event transport shared by standalone and SDK modes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime, timezone
import uuid
from typing import Any


class DashboardHub:
    """Sequence live events and fan them out to per-client queues.

    Final transcripts are retained for finite replay. Partials are transient:
    only the latest one is included when a browser first connects.
    """

    def __init__(self, history_limit: int = 40) -> None:
        self._history_limit = history_limit
        self._seq = 0
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._history: list[dict[str, Any]] = []
        self._latest_transcript: dict[str, Any] | None = None
        self._asr_history: list[dict[str, Any]] = []  # recent final transcripts
        self._last_audio_level: dict[str, Any] | None = None  # latest level event

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def publish(self, message: dict[str, Any]) -> dict[str, Any]:
        self._seq += 1
        event = {**message, "seq": self._seq}
        msg_type = event.get("type", "")
        if msg_type == "transcript":
            self._latest_transcript = event
            if bool(event.get("final")):
                self._history.append(event)
                self._history = self._history[-self._history_limit :]
                # Track ASR history (max 20 recent finals)
                self._asr_history.append({
                    "text": event.get("text", ""),
                    "seq": self._seq,
                })
                if len(self._asr_history) > 20:
                    self._asr_history = self._asr_history[-20:]
        elif msg_type == "audio_level":
            self._last_audio_level = event
        for queue in tuple(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A stalled browser must not block ASR. Drop its oldest event
                # and retain the newest state/transcript update.
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
        return event

    def replay(self) -> list[dict[str, Any]]:
        events = list(self._history)
        latest = self._latest_transcript
        if latest is not None and not bool(latest.get("final")):
            events.append(latest)
        # Append recent ASR history as a replay batch
        if self._asr_history:
            events.append({"type": "asr_history_replay", "items": list(self._asr_history)})
        if self._last_audio_level:
            events.append(self._last_audio_level)
        return events


class ChatMessageStore:
    """In-memory source of truth for one process' chat and camera captures."""

    def __init__(self, message_limit: int = 100, capture_limit: int = 20) -> None:
        self._message_limit = max(2, message_limit)
        self._capture_limit = max(1, capture_limit)
        self._messages: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._captures: dict[str, bytes] = {}
        self._capture_order: list[str] = []

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _put(self, message: dict[str, Any]) -> dict[str, Any]:
        self._messages.append(message)
        self._by_id[message["id"]] = message
        while len(self._messages) > self._message_limit:
            removed = self._messages.pop(0)
            self._by_id.pop(removed["id"], None)
        return dict(message)

    def begin_turn(
        self,
        turn_id: str,
        text: str,
        *,
        source: str,
        client_message_id: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        now = self._now()
        user = {
            "id": client_message_id or f"user-{uuid.uuid4().hex}",
            "turn_id": turn_id,
            "role": "user",
            "text": text,
            "status": "done",
            "created_at": now,
            "sources": [],
            "attachments": [],
            "error": None,
            "input_source": source,
        }
        assistant = {
            "id": f"assistant-{uuid.uuid4().hex}",
            "turn_id": turn_id,
            "role": "assistant",
            "text": "",
            "status": "thinking",
            "created_at": now,
            "sources": [],
            "attachments": [],
            "error": None,
        }
        return self._put(user), self._put(assistant)

    def assistant_for_turn(self, turn_id: str) -> dict[str, Any] | None:
        for message in reversed(self._messages):
            if message["turn_id"] == turn_id and message["role"] == "assistant":
                return message
        return None

    def append_delta(self, turn_id: str, delta: str) -> dict[str, Any] | None:
        message = self.assistant_for_turn(turn_id)
        if message is None:
            return None
        message["text"] += delta
        message["status"] = "streaming"
        return dict(message)

    def set_status(self, turn_id: str, status: str) -> dict[str, Any] | None:
        message = self.assistant_for_turn(turn_id)
        if message is None:
            return None
        if message["status"] not in {"done", "error"}:
            message["status"] = status
        return dict(message)

    def finalize(
        self,
        turn_id: str,
        text: str,
        *,
        sources: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        message = self.assistant_for_turn(turn_id)
        if message is None:
            return None
        message["text"] = text
        message["sources"] = list(sources or [])
        message["error"] = error
        message["status"] = "error" if error else "done"
        return dict(message)

    def attach_capture(
        self,
        turn_id: str,
        jpeg: bytes,
        *,
        label: str,
        captured_at: str | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        capture_id = uuid.uuid4().hex
        self._captures[capture_id] = jpeg
        self._capture_order.append(capture_id)
        while len(self._capture_order) > self._capture_limit:
            old = self._capture_order.pop(0)
            self._captures.pop(old, None)
        attachment = {
            "type": "image",
            "capture_id": capture_id,
            "url": f"/captures/{capture_id}",
            "label": label,
            "captured_at": captured_at or self._now(),
        }
        message = self.assistant_for_turn(turn_id)
        if message is not None:
            message["attachments"].append(attachment)
            return capture_id, dict(message)
        return capture_id, None

    def get_capture(self, capture_id: str) -> bytes | None:
        return self._captures.get(capture_id)

    def history_event(self) -> dict[str, Any]:
        return {"type": "chat_history", "messages": [dict(m) for m in self._messages]}

    def clear(self) -> None:
        self._messages.clear()
        self._by_id.clear()
        self._captures.clear()
        self._capture_order.clear()

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def capture_count(self) -> int:
        return len(self._captures)


async def run_websocket_session(
    websocket: Any,
    hub: DashboardHub,
    snapshot: Callable[[], Iterable[dict[str, Any]]],
    handle_control: Callable[[Any, dict[str, Any]], Awaitable[None]],
) -> None:
    """Run independent server→browser and browser→server loops."""
    queue = hub.subscribe()

    async def _send() -> None:
        for message in snapshot():
            await websocket.send_json(message)
        for message in hub.replay():
            await websocket.send_json(message)
        while True:
            await websocket.send_json(await queue.get())

    async def _receive() -> None:
        while True:
            data = await websocket.receive_json()
            if isinstance(data, dict):
                await handle_control(websocket, data)

    send_task = asyncio.create_task(_send())
    receive_task = asyncio.create_task(_receive())
    try:
        done, pending = await asyncio.wait(
            {send_task, receive_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except (asyncio.CancelledError, RuntimeError):
                pass
        for task in done:
            exception = task.exception()
            if exception is not None:
                raise exception
    finally:
        hub.unsubscribe(queue)
