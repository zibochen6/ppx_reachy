"""VLM (Vision Language Model) client for scene understanding.

Uses Bailian's OpenAI-compatible endpoint with ``qwen-vl-plus`` to analyze
images captured from the Reachy Mini camera.

Usage::

    vlm = BailianVLMClient(cfg)
    async with vlm:
        description = await vlm.understand(frame_jpeg_bytes, "描述这个场景")
"""

from __future__ import annotations

import base64
import logging
from typing import AsyncIterator

import httpx

from chaihuo_reachy.config import Config

logger = logging.getLogger("chaihuo_reachy.vlm")


class BailianVLMClient:
    """Async client for Bailian VLM (qwen-vl-plus / qwen2.5-vl-72b).

    Uses the same OpenAI-compatible endpoint as LLM but sends image + text.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._client: httpx.AsyncClient | None = None

    @property
    def _base_url(self) -> str:
        host = f"{self._cfg.bailian_workspace_id}.{self._cfg.bailian_region}.maas.aliyuncs.com"
        return f"https://{host}/compatible-mode/v1"

    async def __aenter__(self) -> "BailianVLMClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._cfg.bailian_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def understand(
        self,
        image_bytes: bytes,
        question: str = "请用中文简短描述你看到的场景，包括人物、物体、环境。",
        image_format: str = "jpeg",
    ) -> str:
        """Send an image to VLM and get a text description.

        Args:
            image_bytes: Raw JPEG/PNG image bytes.
            question: What to ask about the image.
            image_format: "jpeg" or "png".

        Returns:
            Text description from the VLM.
        """
        data_url = f"data:image/{image_format};base64,{base64.b64encode(image_bytes).decode()}"

        body = {
            "model": self._cfg.bailian_vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                }
            ],
            "max_tokens": 200,
            "temperature": 0.0,
        }

        assert self._client is not None
        response = await self._client.post("/chat/completions", json=body)
        if response.status_code != 200:
            text = await response.aread()
            logger.error(
                "VLM HTTP %d: %s", response.status_code, text.decode(errors="replace")
            )
            raise RuntimeError(f"VLM request failed: HTTP {response.status_code}")

        data = response.json()
        content = data["choices"][0]["message"].get("content", "")
        logger.info("VLM response: %s", content[:100])
        return content

    async def understand_stream(
        self,
        image_bytes: bytes,
        question: str = "描述这个场景",
        image_format: str = "jpeg",
    ) -> AsyncIterator[str]:
        """Stream VLM response tokens (for real-time display)."""
        import json

        data_url = f"data:image/{image_format};base64,{base64.b64encode(image_bytes).decode()}"

        body = {
            "model": self._cfg.bailian_vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": 200,
            "temperature": 0.3,
            "stream": True,
        }

        assert self._client is not None
        async with self._client.stream("POST", "/chat/completions", json=body) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise RuntimeError(f"VLM request failed: HTTP {response.status_code}")

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
