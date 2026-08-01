#!/usr/bin/env python3
"""Synchronize the complete public MCV Yuque journal, including images.

Designed for both humans and agents:

    uv run python scripts/sync_journals.py
    uv run python scripts/sync_journals.py --refresh-all --json

Exit code is non-zero unless every visible journal and every inline image is
available locally and represented in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from chaihuo_reachy.memory import JournalFetcher, MemoryStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = "https://www.yuque.com/mouseart/mcv/guaaeocvtm3mtl99"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "journals"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "chroma"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="完整同步柴火基地车语雀日记正文、原始 HTML 和所有图片"
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SEED,
        help="语雀知识库内任一公开日记 URL（默认使用用户指定的入口）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="日记输出目录",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX,
        help="Chroma 向量索引目录",
    )
    parser.add_argument(
        "--refresh-all",
        action="store_true",
        help="重新校验并下载全部正文和图片；默认仅新增/缺失项",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出适合 Agent 读取的单行 JSON 结果",
    )
    return parser.parse_args()


async def sync(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.expanduser().resolve()
    index_dir = args.index_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(persist_dir=str(index_dir), journal_dir=str(output))
    fetcher = JournalFetcher(listing_url=args.source, cache_dir=output)
    sync_error = ""
    try:
        results = await fetcher.sync(
            memory_store=store,
            refresh_all=bool(args.refresh_all),
        )
    except Exception as exc:
        # The fetcher writes an honest partial manifest before raising.  Return
        # that state to the calling Agent instead of reducing it to plain text.
        results = []
        sync_error = str(exc)
    health = fetcher.health()
    markdown_count = len(list(output.glob("*.md")))
    raw_count = len(list((output / "raw").glob("*.html")))
    ok = (
        health["expected"] > 0
        and health["expected"] == health["complete"] == markdown_count == raw_count
        and health["images"] == health["images_complete"]
    )
    report: dict[str, object] = {
        "ok": ok,
        "source": args.source,
        "output": str(output),
        "official_journals": health["expected"],
        "complete_journals": health["complete"],
        "markdown_files": markdown_count,
        "raw_html_files": raw_count,
        "images": health["images"],
        "complete_images": health["images_complete"],
        "new": sum(1 for item in results if item.get("new")),
        "changed": sum(1 for item in results if item.get("changed")),
        "stale": sum(1 for item in results if item.get("stale")),
        "last_checked_at": health["last_checked_at"],
        "last_success_at": health["last_success_at"],
        "manifest": str(fetcher.manifest_path),
        "index_chunks": store.chunk_count(),
        "failures": health["failures"],
        "error": sync_error or None,
    }
    return report


def main() -> None:
    args = parse_args()
    try:
        report = asyncio.run(sync(args))
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"❌ 日记同步失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    elif report["ok"]:
        print(
            "✅ 同步完成："
            f"{report['complete_journals']}/{report['official_journals']} 篇日记，"
            f"{report['complete_images']}/{report['images']} 张图片；"
            f"保存到 {report['output']}"
        )
    else:
        print(
            "❌ 日记同步未通过完整性检查："
            f"{report['complete_journals']}/{report['official_journals']} 篇，"
            f"{report['complete_images']}/{report['images']} 张图片；"
            f"{report['error']}",
            file=sys.stderr,
        )
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
