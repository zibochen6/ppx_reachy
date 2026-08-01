"""LLM HTTP client for Bailian OpenAI-compatible chat completions.

Uses ``httpx`` for async streaming — no ``openai`` package needed.
Default model: ``qwen-turbo`` (~337ms TTFT) for low-latency voice conversation.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from chaihuo_reachy.config import Config

logger = logging.getLogger("chaihuo_reachy.llm")


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
        messages: list[dict[str, str]],
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
        if self._cfg.enable_search:
            body["enable_search"] = True
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
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
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
        if self._cfg.enable_search:
            body["enable_search"] = True
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        assert self._client is not None
        response = await self._client.post("/chat/completions", json=body)
        if response.status_code != 200:
            text = await response.aread()
            logger.error(
                "LLM HTTP %d: %s", response.status_code, text.decode(errors="replace")
            )
            raise RuntimeError(f"LLM request failed: HTTP {response.status_code}")
        return response.json()
