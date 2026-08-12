"""Compare Qwen and Doubao on exactly the same grounded evidence prompts.

This is deliberately separate from production routing.  It prints JSON and
never changes the configured production provider.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "scripts" / "eval_cases" / "dialog_grounding.json"


def _providers(selected: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if selected in {"all", "qwen"}:
        workspace = os.environ.get("BAILIAN_WORKSPACE_ID", "").strip()
        region = os.environ.get("BAILIAN_REGION", "cn-beijing").strip()
        host = "dashscope.aliyuncs.com"
        if workspace:
            host = f"{workspace}.{region}.dashscope.aliyuncs.com"
        items.append(
            {
                "name": "qwen",
                "base_url": f"https://{host}/compatible-mode/v1",
                "api_key": os.environ.get("BAILIAN_API_KEY", ""),
                "model": os.environ.get("BAILIAN_LLM_MODEL", "qwen3.7-plus"),
            }
        )
    if selected in {"all", "doubao"}:
        items.append(
            {
                "name": "doubao",
                "base_url": os.environ.get(
                    "DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
                ),
                "api_key": os.environ.get("DOUBAO_API_KEY", ""),
                "model": os.environ.get("DOUBAO_MODEL", ""),
            }
        )
    return items


async def _run_case(
    client: httpx.AsyncClient,
    provider: dict[str, str],
    case: dict[str, Any],
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "你是皮皮虾。只根据给定证据回答；证据没有支持的经历不能编造。"
                "用自然、温暖的简体中文回答。"
            ),
        },
        {
            "role": "user",
            "content": f"【证据】\n{case['evidence']}\n\n【问题】\n{case['question']}",
        },
    ]
    started = time.perf_counter()
    response = await client.post(
        "/chat/completions",
        json={
            "model": provider["model"],
            "messages": messages,
            "temperature": 0.2,
            "max_completion_tokens": 600,
            "stream": False,
        },
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    response.raise_for_status()
    data = response.json()
    text = str(data["choices"][0]["message"].get("content") or "")
    required = [term for term in case.get("required", []) if term not in text]
    forbidden = [term for term in case.get("forbidden", []) if term in text]
    return {
        "id": case["id"],
        "text": text,
        "pass": not required and not forbidden,
        "missing_required": required,
        "hit_forbidden": forbidden,
        "latency_ms": latency_ms,
        "usage": data.get("usage") or {},
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("all", "qwen", "doubao"), default="all")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    report: dict[str, Any] = {"cases": str(args.cases), "providers": {}}
    for provider in _providers(args.provider):
        if not provider["api_key"] or not provider["model"]:
            report["providers"][provider["name"]] = {
                "skipped": True,
                "reason": "missing API key or model",
            }
            continue
        async with httpx.AsyncClient(
            base_url=provider["base_url"],
            headers={"Authorization": f"Bearer {provider['api_key']}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        ) as client:
            results = [await _run_case(client, provider, case) for case in cases]
        latencies = [float(item["latency_ms"]) for item in results]
        report["providers"][provider["name"]] = {
            "model": provider["model"],
            "pass_rate": sum(bool(item["pass"]) for item in results) / len(results),
            "latency_median_ms": statistics.median(latencies),
            "latency_max_ms": max(latencies),
            "results": results,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
