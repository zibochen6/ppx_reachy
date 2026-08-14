"""LLM HTTP client for Bailian OpenAI-compatible Chat and Responses APIs.

Uses ``httpx`` for async streaming — no ``openai`` package needed.
Qwen3.7 Responses supplies public-web tools; protected turns use plain chat.
"""

from __future__ import annotations

import json
import logging
import time
import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

from chaihuo_reachy.config import Config

logger = logging.getLogger("chaihuo_reachy.llm")
_SEARCH_CIRCUIT_OPEN_UNTIL = 0.0


class BailianLLMClient:
    """Async HTTP client for Bailian OpenAI-compatible chat completions.

    Usage::

        llm = BailianLLMClient(cfg)
        async with llm:
            async for token in llm.chat_stream(messages):
                print(token, end="", flush=True)
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._client: httpx.AsyncClient | None = None
        self.last_sources: list[dict[str, str]] = []
        self.last_search_used = False
        self.last_search_error = ""
        self.last_function_calls: list[dict[str, Any]] = []

    @property
    def _base_url(self) -> str:
        host = f"{self._cfg.bailian_workspace_id}.{self._cfg.bailian_region}.maas.aliyuncs.com"
        return f"https://{host}/compatible-mode/v1"

    async def __aenter__(self) -> "BailianLLMClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._cfg.bailian_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream LLM tokens. Yields one token string at a time."""
        body: dict = {
            "model": self._cfg.bailian_llm_model,
            "messages": messages,
            "max_completion_tokens": self._cfg.llm_max_tokens,
            "temperature": self._cfg.llm_temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "enable_thinking": False,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        assert self._client is not None
        async with self._client.stream(
            "POST", "/chat/completions", json=body
        ) as response:
            if response.status_code != 200:
                text = await response.aread()
                logger.error(
                    "LLM HTTP %d: %s", response.status_code, text.decode(errors="replace")
                )
                raise RuntimeError(f"LLM request failed: HTTP {response.status_code}")

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict:
        """Non-streaming chat completion. Returns full response dict."""
        body: dict = {
            "model": self._cfg.bailian_llm_model,
            "messages": messages,
            "max_completion_tokens": self._cfg.llm_max_tokens,
            "temperature": self._cfg.llm_temperature,
            "stream": False,
            "enable_thinking": False,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"

        assert self._client is not None
        response = await self._client.post("/chat/completions", json=body)
        if response.status_code != 200:
            text = await response.aread()
            logger.error(
                "LLM HTTP %d: %s", response.status_code, text.decode(errors="replace")
            )
            raise RuntimeError(f"LLM request failed: HTTP {response.status_code}")
        return response.json()

    async def response_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        search_mode: str = "auto",
        tools: list[dict[str, Any]] | None = None,
        tool_handler: (
            Callable[[str, dict[str, Any]], Awaitable[dict[str, Any] | str]] | None
        ) = None,
        max_tool_rounds: int = 1,
    ) -> AsyncIterator[str]:
        """Stream a Responses API turn with native web tools and citations.

        Retries are allowed only before the first visible token.  If the web
        path remains unavailable, the turn falls back to model knowledge with
        an explicit verification disclaimer.
        """
        self.last_sources = []
        self.last_search_used = False
        self.last_search_error = ""
        self.last_function_calls = []
        if tools:
            try:
                async for token in self._response_stream_with_functions(
                    messages,
                    search_mode=search_mode,
                    tools=tools,
                    tool_handler=tool_handler,
                    max_tool_rounds=max_tool_rounds,
                ):
                    yield token
            except Exception as exc:
                logger.warning("Responses 视觉工具链失败，降级为普通回答: %s", exc)
                self.last_search_error = str(exc)
                async for token in self._tool_fallback_stream(messages, search_mode):
                    yield token
            return
        if search_mode == "off":
            async for token in self.chat_stream(messages):
                yield token
            return
        global _SEARCH_CIRCUIT_OPEN_UNTIL
        if time.monotonic() < _SEARCH_CIRCUIT_OPEN_UNTIL:
            self.last_search_error = "search circuit breaker open"
            async for token in self._fallback_stream(messages):
                yield token
            return

        body: dict[str, Any] = {
            "model": self._cfg.bailian_llm_model,
            "input": messages,
            "max_output_tokens": self._cfg.llm_max_tokens,
            "temperature": self._cfg.llm_temperature,
            "stream": True,
            "store": False,
            "enable_thinking": False,
            "tools": [{"type": "web_search"}, {"type": "web_extractor"}],
            "tool_choice": "auto",
        }
        if search_mode == "required":
            # Bailian only permits tool_choice=required with a single tool;
            # forced_search is the supported way to require web search while
            # keeping web_extractor available for follow-up page reading.
            body["search_options"] = {"forced_search": True}
        assert self._client is not None
        last_error: Exception | None = None
        for attempt in range(3):
            emitted = False
            try:
                async with self._client.stream(
                    "POST", "/responses", json=body,
                    timeout=httpx.Timeout(self._cfg.search_timeout_s, connect=5.0),
                ) as response:
                    if response.status_code != 200:
                        detail = (await response.aread()).decode(errors="replace")[:300]
                        raise RuntimeError(
                            f"Responses API HTTP {response.status_code}: {detail}"
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw == "[DONE]":
                            return
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        event_type = str(event.get("type") or "")
                        if (
                            "web_search" in event_type
                            or "web_extractor" in event_type
                            or _contains_web_tool(event)
                        ):
                            self.last_search_used = True
                        self._collect_sources(event)
                        if event_type == "response.output_text.delta":
                            delta = str(event.get("delta") or "")
                            if delta:
                                emitted = True
                                yield delta
                        elif event_type in {"response.failed", "error"}:
                            raise RuntimeError(_response_error(event))
                return
            except Exception as exc:
                last_error = exc
                if emitted:
                    raise
                logger.warning(
                    "Responses 搜索请求失败，第 %d/3 次: %s", attempt + 1, exc
                )
                if attempt < 2:
                    await asyncio.sleep(0.15 * (2**attempt))

        self.last_search_error = str(last_error or "web search unavailable")
        _SEARCH_CIRCUIT_OPEN_UNTIL = (
            time.monotonic() + self._cfg.search_circuit_breaker_s
        )
        async for token in self._fallback_stream(messages):
            yield token

    async def _response_stream_with_functions(
        self,
        messages: list[dict[str, Any]],
        *,
        search_mode: str,
        tools: list[dict[str, Any]],
        tool_handler: (
            Callable[[str, dict[str, Any]], Awaitable[dict[str, Any] | str]] | None
        ),
        max_tool_rounds: int,
    ) -> AsyncIterator[str]:
        """Run a stateless Responses function-call loop.

        Text from a round that requests a function is intentionally buffered,
        so it can never reach TTS before the camera observation is available.
        """
        current_input: list[dict[str, Any]] = list(messages)
        custom_tools = list(tools)
        rounds = 0
        while True:
            active_tools: list[dict[str, Any]] = []
            if search_mode != "off":
                active_tools.extend([{"type": "web_search"}, {"type": "web_extractor"}])
            if rounds < max_tool_rounds:
                active_tools.extend(custom_tools)
            body: dict[str, Any] = {
                "model": self._cfg.bailian_llm_model,
                "input": current_input,
                "max_output_tokens": self._cfg.llm_max_tokens,
                "temperature": self._cfg.llm_temperature,
                "stream": True,
                "store": False,
                "enable_thinking": False,
            }
            if active_tools:
                body["tools"] = active_tools
                body["tool_choice"] = "auto"
            if search_mode == "required":
                body["search_options"] = {"forced_search": True}

            text_parts, output_items, calls = await self._read_response_round(body)
            if not calls:
                for token in text_parts:
                    yield token
                return

            rounds += 1
            current_input.extend(output_items)
            for call in calls:
                name = str(call.get("name") or "")
                raw_arguments = call.get("arguments") or "{}"
                try:
                    arguments = (
                        json.loads(raw_arguments)
                        if isinstance(raw_arguments, str)
                        else dict(raw_arguments)
                    )
                    if not isinstance(arguments, dict):
                        raise TypeError("function arguments must be an object")
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    arguments = {}
                    result: dict[str, Any] | str = {
                        "ok": False,
                        "error": f"工具参数无效: {exc}",
                    }
                else:
                    if tool_handler is None:
                        result = {"ok": False, "error": "工具执行器不可用"}
                    else:
                        try:
                            result = await asyncio.wait_for(
                                tool_handler(name, arguments), timeout=20.0
                            )
                        except Exception as exc:
                            logger.exception("Responses function %s failed", name)
                            result = {"ok": False, "error": f"工具执行失败: {exc}"}
                self.last_function_calls.append(
                    {"name": name, "arguments": arguments, "ok": _tool_result_ok(result)}
                )
                current_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(call.get("call_id") or call.get("id") or ""),
                        "output": (
                            result
                            if isinstance(result, str)
                            else json.dumps(result, ensure_ascii=False)
                        ),
                    }
                )

            # The next round may still use built-in web tools, but custom
            # functions are removed once their per-turn limit is reached.
            if rounds >= max_tool_rounds:
                custom_tools = []

    async def _read_response_round(
        self, body: dict[str, Any]
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        assert self._client is not None
        text_parts: list[str] = []
        output_items: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        async with self._client.stream(
            "POST",
            "/responses",
            json=body,
            timeout=httpx.Timeout(self._cfg.search_timeout_s, connect=5.0),
        ) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode(errors="replace")[:300]
                raise RuntimeError(f"Responses API HTTP {response.status_code}: {detail}")
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event_type = str(event.get("type") or "")
                if (
                    "web_search" in event_type
                    or "web_extractor" in event_type
                    or _contains_web_tool(event)
                ):
                    self.last_search_used = True
                self._collect_sources(event)
                if event_type == "response.output_text.delta":
                    delta = str(event.get("delta") or "")
                    if delta:
                        text_parts.append(delta)
                elif event_type == "response.output_item.done":
                    item = event.get("item")
                    if isinstance(item, dict):
                        output_items.append(item)
                        if item.get("type") == "function_call":
                            calls.append(item)
                elif event_type in {"response.failed", "error"}:
                    raise RuntimeError(_response_error(event))
        return text_parts, output_items, calls

    async def _tool_fallback_stream(
        self, messages: list[dict[str, Any]], search_mode: str
    ) -> AsyncIterator[str]:
        fallback_messages = [
            {
                "role": "system",
                "content": (
                    "本轮实时视觉工具暂不可用。不要声称看到了当前人物、物体或环境；"
                    "请自然说明现在没看清楚。"
                    + (
                        "联网工具也不可用，请明确说明当前无法联网核实。"
                        if search_mode != "off"
                        else ""
                    )
                ),
            },
            *messages,
        ]
        async for token in self.chat_stream(fallback_messages):
            yield token

    async def _fallback_stream(
        self, messages: list[dict[str, Any]]
    ) -> AsyncIterator[str]:
        fallback_messages = [
            {
                "role": "system",
                "content": (
                    "当前联网搜索不可用。只用已有常识回答，并明确说明"
                    "“当前无法联网核实”；不得编造最新数据或来源。"
                ),
            },
            *messages,
        ]
        async for token in self.chat_stream(fallback_messages):
            yield token

    def _collect_sources(self, value: object) -> None:
        if isinstance(value, dict):
            url = value.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                item = {
                    "title": str(value.get("title") or value.get("name") or url),
                    "url": url,
                    "retrieved_at": str(int(time.time())),
                }
                if not any(old["url"] == url for old in self.last_sources):
                    self.last_sources.append(item)
            for child in value.values():
                self._collect_sources(child)
        elif isinstance(value, list):
            for child in value:
                self._collect_sources(child)


def _response_error(event: dict[str, Any]) -> str:
    error = event.get("error") or event.get("response", {}).get("error") or {}
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "Responses API failed")
    return str(error or "Responses API failed")


def _contains_web_tool(value: object) -> bool:
    if isinstance(value, dict):
        if value.get("type") in {"web_search_call", "web_extractor_call"}:
            return True
        return any(_contains_web_tool(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_web_tool(child) for child in value)
    return False


def _tool_result_ok(result: dict[str, Any] | str) -> bool:
    if isinstance(result, dict):
        return bool(result.get("ok", True))
    return bool(result)
