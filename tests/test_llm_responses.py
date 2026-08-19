from __future__ import annotations

import json

import httpx
import pytest

import chaihuo_reachy.bailian.llm_client as llm_module
from chaihuo_reachy.bailian.llm_client import BailianLLMClient
from chaihuo_reachy.config import Config


@pytest.mark.asyncio
async def test_responses_stream_collects_web_source_and_text() -> None:
    events = [
        {"type": "response.web_search_call.in_progress"},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "content": [
                    {
                        "annotations": [
                            {"title": "官方天气", "url": "https://example.com/weather"}
                        ]
                    }
                ],
            },
        },
        {"type": "response.output_text.delta", "delta": "深圳今天晴。"},
    ]
    # Bailian emits SSE without a space after ``data:``.
    body = "".join(f"data:{json.dumps(item)}\n\n" for item in events) + "data:[DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/responses")
        payload = json.loads(request.content)
        assert payload["tool_choice"] == "auto"
        assert payload["search_options"]["forced_search"] is True
        assert payload["enable_thinking"] is False
        assert payload["tools"] == [{"type": "web_search"}]
        return httpx.Response(200, text=body)

    llm = BailianLLMClient(Config(bailian_api_key="test"))
    llm._client = httpx.AsyncClient(
        base_url="https://example.invalid", transport=httpx.MockTransport(handler)
    )
    try:
        tokens = [
            token
            async for token in llm.response_stream(
                [{"role": "user", "content": "今天深圳天气"}],
                search_mode="required",
            )
        ]
    finally:
        await llm._client.aclose()

    assert tokens == ["深圳今天晴。"]
    assert llm.last_search_used
    assert llm.last_sources[0]["url"] == "https://example.com/weather"


@pytest.mark.asyncio
async def test_search_failure_retries_before_fallback(monkeypatch) -> None:
    monkeypatch.setattr(llm_module, "_SEARCH_CIRCUIT_OPEN_UNTIL", 0.0)
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="unavailable")

    llm = BailianLLMClient(
        Config(bailian_api_key="test", search_circuit_breaker_s=10)
    )
    llm._client = httpx.AsyncClient(
        base_url="https://example.invalid", transport=httpx.MockTransport(handler)
    )

    async def fallback(_messages):
        yield "当前无法联网核实。"

    monkeypatch.setattr(llm, "_fallback_stream", fallback)
    try:
        tokens = [
            token
            async for token in llm.response_stream(
                [{"role": "user", "content": "最新新闻"}],
                search_mode="required",
            )
        ]
    finally:
        await llm._client.aclose()

    assert calls == 3
    assert tokens == ["当前无法联网核实。"]
    assert llm.last_search_error


@pytest.mark.asyncio
async def test_responses_failure_event_without_sse_space_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(llm_module, "_SEARCH_CIRCUIT_OPEN_UNTIL", 0.0)
    calls = 0
    error = {
        "type": "response.failed",
        "response": {
            "error": {
                "code": "InvalidParameter",
                "message": "normal mode does not support this tool",
            }
        },
    }

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text=f"data:{json.dumps(error)}\n\n")

    llm = BailianLLMClient(Config(bailian_api_key="test"))
    llm._client = httpx.AsyncClient(
        base_url="https://example.invalid", transport=httpx.MockTransport(handler)
    )

    async def fallback(_messages):
        yield "已回退到普通回答。"

    monkeypatch.setattr(llm, "_fallback_stream", fallback)
    try:
        tokens = [
            token
            async for token in llm.response_stream(
                [{"role": "user", "content": "为什么"}], search_mode="auto"
            )
        ]
    finally:
        await llm._client.aclose()

    assert calls == 3
    assert tokens == ["已回退到普通回答。"]
    assert "normal mode" in llm.last_search_error


@pytest.mark.asyncio
async def test_empty_responses_stream_retries_then_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(llm_module, "_SEARCH_CIRCUIT_OPEN_UNTIL", 0.0)
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            text='data:{"type":"response.completed"}\n\n',
        )

    llm = BailianLLMClient(Config(bailian_api_key="test"))
    llm._client = httpx.AsyncClient(
        base_url="https://example.invalid", transport=httpx.MockTransport(handler)
    )

    async def fallback(_messages):
        yield "模型暂时无文本，已自动回退。"

    monkeypatch.setattr(llm, "_fallback_stream", fallback)
    try:
        tokens = [
            token
            async for token in llm.response_stream(
                [{"role": "user", "content": "你好"}], search_mode="auto"
            )
        ]
    finally:
        await llm._client.aclose()

    assert calls == 3
    assert tokens == ["模型暂时无文本，已自动回退。"]
    assert "without output text" in llm.last_search_error


@pytest.mark.asyncio
async def test_responses_function_call_waits_for_output_before_final_text() -> None:
    requests: list[dict] = []
    rounds = [
        [
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "observe_scene",
                    "arguments": json.dumps(
                        {"scope": "front", "focus": "识别用户手中的物体"},
                        ensure_ascii=False,
                    ),
                },
            }
        ],
        [{"type": "response.output_text.delta", "delta": "这是一个万用表。"}],
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        events = rounds[len(requests) - 1]
        body = "".join(
            f"data:{json.dumps(item, ensure_ascii=False)}\n\n" for item in events
        )
        return httpx.Response(200, text=body + "data:[DONE]\n\n")

    observed: list[dict] = []

    async def run_tool(name: str, arguments: dict) -> dict:
        observed.append({"name": name, **arguments})
        return {"ok": True, "facts": "手里拿着一个万用表"}

    llm = BailianLLMClient(Config(bailian_api_key="test"))
    llm._client = httpx.AsyncClient(
        base_url="https://example.invalid", transport=httpx.MockTransport(handler)
    )
    tool = {
        "type": "function",
        "name": "observe_scene",
        "description": "观察当前场景",
        "parameters": {"type": "object", "properties": {}},
    }
    try:
        tokens = [
            token
            async for token in llm.response_stream(
                [{"role": "user", "content": "这是什么"}],
                search_mode="off",
                tools=[tool],
                tool_handler=run_tool,
                max_tool_rounds=1,
            )
        ]
    finally:
        await llm._client.aclose()

    assert tokens == ["这是一个万用表。"]
    assert observed == [
        {"name": "observe_scene", "scope": "front", "focus": "识别用户手中的物体"}
    ]
    assert any(item.get("type") == "function_call_output" for item in requests[1]["input"])
    assert "tools" not in requests[1]
    assert llm.last_function_calls[0]["ok"] is True
