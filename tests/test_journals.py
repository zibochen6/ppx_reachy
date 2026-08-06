from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote
import asyncio
import json

import httpx
import pytest

from chaihuo_reachy.memory.journal_fetcher import JournalFetcher
from chaihuo_reachy.memory.store import MemoryStore, _title_range_contains


def yuque_page(slug: str, title: str, content: str) -> str:
    app_data = {
        "doc": {
            "slug": slug,
            "title": title,
            "created_at": "2026-07-01T08:00:00Z",
            "updated_at": "2026-07-02T08:00:00Z",
            "_cachedContent": {
                "_cache_decrypted_body": f"<div><p>{content}</p></div>"
            },
        }
    }
    encoded = quote(json.dumps(app_data, ensure_ascii=False), safe="")
    return f'<script>window.appData = JSON.parse(decodeURIComponent("{encoded}"));</script>'


def test_listing_accepts_every_title_style_and_deduplicates() -> None:
    html = """
    <a href="https://www.yuque.com/mouseart/mcv/slugone">普通标题</a>
    https://www.yuque.com/mouseart/mcv/slugtwo
    <a data-x="1" href="https://www.yuque.com/mouseart/mcv/slugthree">2026.7.1</a>
    https://www.yuque.com/mouseart/mcv/slugone
    """
    entries = JournalFetcher.parse_listing(html)
    assert [item["slug"] for item in entries] == ["slugone", "slugtwo", "slugthree"]


def test_listing_can_start_from_embedded_yuque_knowledge_base_toc() -> None:
    toc = """
- type: META
  count: 3
- type: DOC
  title: 基地车日记｜2026.07.01｜第一篇
  url: firstslug
  visible: 1
- type: DOC
  title: 使用说明
  url: instructions
  visible: 1
- type: DOC
  title: 基地车日记｜2026.06.30｜第二篇
  url: secondslug
  visible: 1
"""
    app_data = {"book": {"toc_yml": toc}}
    encoded = quote(json.dumps(app_data, ensure_ascii=False), safe="")
    html = (
        '<script>window.appData = '
        f'JSON.parse(decodeURIComponent("{encoded}"));</script>'
    )
    entries = JournalFetcher.parse_listing(html)
    assert [item["slug"] for item in entries] == ["firstslug", "secondslug"]


def test_extracts_complete_yuque_appdata_body() -> None:
    body = "完整正文。" * 80
    result = JournalFetcher.extract_yuque_document(
        yuque_page("abc123", "基地车日记｜2026.07.01 测试", body),
        "https://www.yuque.com/mouseart/mcv/abc123",
    )
    assert result["slug"] == "abc123"
    assert result["date"] == "2026-07-01"
    assert len(result["content"]) >= 200
    assert "完整正文" in result["content"]


def test_extracts_all_dates_from_compact_title_range() -> None:
    result = JournalFetcher.extract_yuque_document(
        yuque_page("range123", "基地车日记｜2026.05.18-20｜沿途记录", "完整正文。" * 80),
        "https://www.yuque.com/mouseart/mcv/range123",
    )
    assert result["dates"] == ["2026-05-18", "2026-05-19", "2026-05-20"]


@pytest.mark.parametrize("content", ["内容待抓取", "太短"])
def test_rejects_placeholders_and_short_content(content: str) -> None:
    with pytest.raises(ValueError, match="incomplete|placeholder"):
        JournalFetcher.extract_yuque_document(
            yuque_page("abc123", "标题", content),
            "https://www.yuque.com/mouseart/mcv/abc123",
        )


@pytest.mark.asyncio
async def test_inline_images_are_downloaded_and_rewritten_locally(tmp_path: Path) -> None:
    source = "https://cdn.example.test/image.jpeg?crop=1"
    body = f'<div><p>{"正文。" * 80}</p><img src="{source}" alt="现场照片"></div>'
    app_data = {
        "doc": {
            "slug": "image123",
            "title": "基地车日记 2026.07.01",
            "updated_at": "2026-07-01T00:00:00Z",
            "_cachedContent": {"_cache_decrypted_body": body},
        }
    }
    encoded = quote(json.dumps(app_data, ensure_ascii=False), safe="")
    document = JournalFetcher.extract_yuque_document(
        f'<script>window.appData = JSON.parse(decodeURIComponent("{encoded}"));</script>',
        "https://www.yuque.com/mouseart/mcv/image123",
    )
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            content=b"\xff\xd8\xff" + b"image-payload" * 10,
            headers={"content-type": "image/jpeg"},
        )
    )
    fetcher = JournalFetcher(cache_dir=tmp_path)
    async with httpx.AsyncClient(transport=transport) as client:
        await fetcher._localize_images(  # noqa: SLF001 - integrity unit test
            document, client, asyncio.Semaphore(1)
        )
    image = document["images"][0]
    assert Path(image["file"]).is_file()
    assert source not in document["content"]
    assert image["relative_path"] in document["content"]


class FakeMemory:
    def __init__(self) -> None:
        self.upserts: list[str] = []
        self.expected: set[str] = set()

    def upsert_journal(self, entry: dict, _content: str) -> None:
        self.upserts.append(entry["slug"])

    def remove_missing_journals(self, expected: set[str]) -> None:
        self.expected = set(expected)


@pytest.mark.asyncio
async def test_sync_requires_and_saves_every_official_document(tmp_path: Path) -> None:
    fetcher = JournalFetcher(cache_dir=str(tmp_path))
    listing = [
        {"slug": f"slug{i}", "url": f"https://example.test/slug{i}"}
        for i in range(3)
    ]

    async def listing_stub(_client=None):
        return listing

    async def download_stub(url: str, _client=None):
        slug = url.rsplit("/", 1)[-1]
        return {
            "slug": slug,
            "title": f"基地车日记 2026.07.0{int(slug[-1]) + 1}",
            "date": f"2026-07-0{int(slug[-1]) + 1}",
            "content": f"{slug} 的完整正文。" * 50,
            "source_url": url,
            "source_updated_at": "2026-07-02T08:00:00Z",
        }

    fetcher.fetch_listing = listing_stub  # type: ignore[method-assign]
    fetcher.download_journal = download_stub  # type: ignore[method-assign]
    memory = FakeMemory()
    results = await fetcher.sync(memory_store=memory)
    manifest = fetcher.load_manifest()

    assert len(results) == 3
    assert manifest["expected_count"] == manifest["complete_count"] == 3
    assert len(list(tmp_path.glob("*.md"))) == 3
    assert set(memory.upserts) == {"slug0", "slug1", "slug2"}
    assert memory.expected == {"slug0", "slug1", "slug2"}
    assert all("内容待抓取" not in path.read_text() for path in tmp_path.glob("*.md"))


@pytest.mark.asyncio
async def test_failed_refresh_uses_last_complete_cache_and_keeps_cutoff(
    tmp_path: Path,
) -> None:
    fetcher = JournalFetcher(cache_dir=str(tmp_path))
    listing = [{"slug": "slug0", "url": "https://example.test/slug0"}]

    async def listing_stub(_client=None):
        return listing

    async def first_download(url: str, _client=None):
        return {
            "slug": "slug0",
            "title": "基地车日记 2026.07.01",
            "date": "2026-07-01",
            "content": "可靠缓存正文。" * 50,
            "source_url": url,
            "source_updated_at": "2026-07-02T08:00:00Z",
        }

    fetcher.fetch_listing = listing_stub  # type: ignore[method-assign]
    fetcher.download_journal = first_download  # type: ignore[method-assign]
    await fetcher.sync()
    cutoff = fetcher.health()["last_success_at"]

    async def broken_download(_url: str, _client=None):
        raise RuntimeError("offline")

    fetcher.download_journal = broken_download  # type: ignore[method-assign]
    results = await fetcher.sync(refresh_all=True)
    assert results[0]["stale"] is True
    assert fetcher.health()["last_success_at"] == cutoff
    assert fetcher.health()["failures"]


def test_exact_chinese_title_match_beats_wrong_vector_neighbor(tmp_path: Path) -> None:
    journal_dir = tmp_path / "journals"
    journal_dir.mkdir()
    exact_file = journal_dir / "exact.md"
    wrong_file = journal_dir / "wrong.md"
    exact_file.write_text("西安理工的完整记录。" * 30)
    wrong_file.write_text("相邻一天的其他记录。" * 30)
    manifest = {
        "entries": {
            "exact": {
                "slug": "exact",
                "status": "complete",
                "title": "基地车日记｜西安理工",
                "date": "2026-07-27",
                "file": str(exact_file),
                "source_url": "https://example.test/exact",
            },
            "wrong": {
                "slug": "wrong",
                "status": "complete",
                "title": "基地车日记｜西安收官",
                "date": "2026-07-28",
                "file": str(wrong_file),
                "source_url": "https://example.test/wrong",
            },
        }
    }
    (journal_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    class FakeCollection:
        def count(self) -> int:
            return 1

        def query(self, **_kwargs):
            return {
                "ids": [["wrong:0"]],
                "metadatas": [[{
                    "slug": "wrong",
                    "title": "基地车日记｜西安收官",
                    "date": "2026-07-28",
                    "file": str(wrong_file),
                    "source_url": "https://example.test/wrong",
                }]],
                "documents": [["相邻内容"]],
                "distances": [[0.1]],
            }

    store = MemoryStore.__new__(MemoryStore)
    store._journal_dir = journal_dir
    store._manifest_path = journal_dir / "manifest.json"
    store._collection = FakeCollection()
    result = store.search("基地车去过西安理工发生了什么", k=1)
    assert result[0]["slug"] == "exact"
    assert result[0]["score"] == 0.95


def test_journey_scope_returns_every_matching_region_day_in_date_order(tmp_path: Path) -> None:
    journal_dir = tmp_path / "journals"
    journal_dir.mkdir()
    entries = {}
    for slug, date, content in (
        ("shanxi-start", "2026-07-29", "基地车驶入山西临汾隰县。"),
        ("shanxi-middle", "2026-07-31", "今天继续在山西临汾拜访学校。"),
        ("shanxi-end", "2026-08-03", "山西太原的活动圆满结束。"),
        ("xian", "2026-07-28", "西安收官日。"),
    ):
        path = journal_dir / f"{slug}.md"
        path.write_text(content * 40, encoding="utf-8")
        entries[slug] = {
            "slug": slug,
            "status": "complete",
            "title": f"基地车日记 {slug}",
            "date": date,
            "file": str(path),
            "source_url": f"https://example.test/{slug}",
        }
    (journal_dir / "manifest.json").write_text(
        json.dumps({"entries": entries}, ensure_ascii=False), encoding="utf-8"
    )

    store = MemoryStore.__new__(MemoryStore)
    store._journal_dir = journal_dir
    store._manifest_path = journal_dir / "manifest.json"
    result = store.search_journey_scope("我们在山西都去了哪些站点，帮我回忆一下", k=6)

    assert [item["slug"] for item in result] == [
        "shanxi-start", "shanxi-middle", "shanxi-end",
    ]


def test_compact_multi_day_title_covers_every_day() -> None:
    title = "基地车日记｜2026.05.18-20｜成都-江油-唐家河"
    assert _title_range_contains(title, "2026-05-18")
    assert _title_range_contains(title, "2026-05-19")
    assert _title_range_contains(title, "2026-05-20")
    assert not _title_range_contains(title, "2026-05-21")
