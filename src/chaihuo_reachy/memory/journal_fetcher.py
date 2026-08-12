"""Fetch every public MCV journal and persist verified full text.

The public journal index links to Yuque documents.  Yuque's ordinary read page
does not render article text into the HTML response, while its ``/markdown``
view embeds the complete Lake document in ``window.appData``.  We decode that
payload, normalize the article to readable text, validate it, and maintain a
manifest keyed by the stable Yuque slug.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
import contextlib
import fcntl
import hashlib
import html as html_module
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import httpx
import yaml

logger = logging.getLogger("chaihuo_reachy.journal_fetcher")


@contextlib.contextmanager
def journal_sync_lock(cache_dir: str | Path) -> Iterator[bool]:
    """Non-blocking cross-process mutual exclusion for journal sync.

    Locks ``<cache_dir>/.sync.lock`` with ``flock(LOCK_EX | LOCK_NB)`` and
    yields ``True`` when the caller owns the lock and should run the sync,
    ``False`` when another sync holds it (any process — flock treats
    separately-opened fds as distinct even within one process, so the
    dashboard's auto-sync and the engine's per-answer sync exclude each
    other too).  Released on context exit or process death, so a crashed
    sync can never wedge the periodic timer.
    """
    lock_path = Path(cache_dir) / ".sync.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+")  # never truncate; flock keys on the fd
    try:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    finally:
        lock_file.close()


_YUQUE_URL_RE = re.compile(
    r"https://www\.yuque\.com/mouseart/mcv/(?P<slug>[a-z0-9]+)"
)
_APP_DATA_RE = re.compile(
    r'window\.appData\s*=\s*JSON\.parse\(decodeURIComponent\("(.*?)"\)\)',
    re.DOTALL,
)
_DATE_PATTERNS = (
    re.compile(r"(20\d{2})[.\-/年](\d{1,2})[.\-/月](\d{1,2})"),
    re.compile(r"(20\d{2})\.(\d{2})(\d{2})"),
)
_INVALID_MARKERS = (
    "内容待抓取",
    "请访问原文链接查看完整日记",
)


class _LakeTextParser(HTMLParser):
    """Small dependency-free HTML-to-text converter for Yuque Lake output."""

    _BLOCK_TAGS = {
        "p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5",
        "h6", "blockquote", "pre", "table", "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.images: list[dict[str, str]] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag in self._BLOCK_TAGS or tag == "br":
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "img":
            values = dict(attrs)
            src = values.get("src") or values.get("data-src") or ""
            if src.startswith(("https://", "http://")):
                alt = values.get("alt") or values.get("title") or "日记图片"
                self.images.append({"source_url": src, "alt": alt})
                self.parts.append(f"\n![{alt}]({src})\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if not self._ignored_depth and (tag in self._BLOCK_TAGS or tag == "li"):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts).replace("\xa0", " ")
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r" *\n *", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()


def _extract_date(*values: str) -> str:
    for value in values:
        for pattern in _DATE_PATTERNS:
            match = pattern.search(value or "")
            if match:
                try:
                    return (
                        f"{int(match.group(1)):04d}-"
                        f"{int(match.group(2)):02d}-"
                        f"{int(match.group(3)):02d}"
                    )
                except (ValueError, IndexError):
                    continue
    return ""


def _extract_dates(*values: str) -> list[str]:
    """Return every covered day, including compact ranges such as 05.18-20."""
    from datetime import date, timedelta

    dates: set[str] = set()
    primary = _extract_date(*values)
    if primary:
        dates.add(primary)
    range_pattern = re.compile(
        r"(20\d{2})[.\-/年](\d{1,2})[.\-/月](\d{1,2})"
        r"\s*(?:-|—|–|至|~)\s*(?:(\d{1,2})[.\-/月])?(\d{1,2})"
    )
    for value in values:
        match = range_pattern.search(value or "")
        if not match:
            continue
        try:
            start = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            end = date(
                start.year,
                int(match.group(4) or start.month),
                int(match.group(5)),
            )
        except ValueError:
            continue
        if end < start or (end - start).days > 31:
            continue
        current = start
        while current <= end:
            dates.add(current.isoformat())
            current += timedelta(days=1)
    return sorted(dates)


def _is_complete_content(content: str) -> bool:
    text_only = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", content)
    compact = re.sub(r"\s+", "", text_only)
    return len(compact) >= 200 and not any(marker in content for marker in _INVALID_MARKERS)


class JournalFetcher:
    """Synchronize the official journal index into a complete local corpus."""

    def __init__(
        self,
        listing_url: str = "https://mcv.chaihuo.org/journals",
        cache_dir: str | Path = "data/journals",
    ) -> None:
        self._listing_url = listing_url
        self._cache_dir = Path(cache_dir)
        self._manifest_path = self._cache_dir / "manifest.json"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def load_manifest(self) -> dict[str, Any]:
        if not self._manifest_path.exists():
            return {"version": 3, "last_checked_at": "", "entries": {}}
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("entries"), dict):
                return data
        except Exception:
            logger.warning("Journal manifest is unreadable", exc_info=True)
        return {"version": 3, "last_checked_at": "", "entries": {}}

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        tmp = self._manifest_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(self._manifest_path)

    async def fetch_listing(self, client: httpx.AsyncClient | None = None) -> list[dict[str, str]]:
        owns_client = client is None
        if client is None:
            client = self._client()
        try:
            request_url = self._listing_url
            if _YUQUE_URL_RE.fullmatch(request_url) and not request_url.endswith("/markdown"):
                request_url = f"{request_url}/markdown"
            response = await client.get(request_url)
            response.raise_for_status()
            return self.parse_listing(response.text)
        except Exception:
            logger.exception("Failed to fetch journal listing")
            return []
        finally:
            if owns_client:
                await client.aclose()

    @staticmethod
    def parse_listing(text: str) -> list[dict[str, str]]:
        """Return every unique public Yuque journal link, regardless of title style."""
        seen: set[str] = set()
        entries: list[dict[str, str]] = []

        # A Yuque knowledge-base document embeds the complete published TOC.
        # This lets the standalone sync script start from the seed URL supplied
        # by the user rather than depending on a separate website index.
        app_match = _APP_DATA_RE.search(text)
        if app_match:
            try:
                app_data = json.loads(unquote(app_match.group(1)))
                toc = yaml.safe_load((app_data.get("book") or {}).get("toc_yml") or "")
                for item in toc if isinstance(toc, list) else []:
                    if (
                        not isinstance(item, dict)
                        or item.get("type") != "DOC"
                        or not item.get("visible", 1)
                        or "基地车日记" not in str(item.get("title") or "")
                    ):
                        continue
                    slug = str(item.get("url") or "")
                    if not re.fullmatch(r"[a-z0-9]+", slug) or slug in seen:
                        continue
                    seen.add(slug)
                    entries.append(
                        {
                            "slug": slug,
                            "doc_id": slug,
                            "url": f"https://www.yuque.com/mouseart/mcv/{slug}",
                            "title": str(item.get("title") or ""),
                            "date": _extract_date(str(item.get("title") or "")),
                        }
                    )
            except Exception:
                logger.debug("Could not parse embedded Yuque TOC", exc_info=True)

        for match in _YUQUE_URL_RE.finditer(text):
            slug = match.group("slug")
            if slug in seen:
                continue
            seen.add(slug)
            entries.append(
                {
                    "slug": slug,
                    "doc_id": slug,  # backwards-compatible name
                    "url": f"https://www.yuque.com/mouseart/mcv/{slug}",
                    "title": "",
                    "date": "",
                }
            )
        return entries

    @staticmethod
    def extract_yuque_document(html: str, source_url: str) -> dict[str, Any]:
        """Decode a Yuque ``/markdown`` response into verified document fields."""
        match = _APP_DATA_RE.search(html)
        if not match:
            raise ValueError("Yuque page did not contain window.appData")
        try:
            app_data = json.loads(unquote(match.group(1)))
        except Exception as exc:
            raise ValueError("Yuque appData could not be decoded") from exc

        doc = app_data.get("doc") or {}
        cached = doc.get("_cachedContent") or {}
        body = (
            cached.get("_cache_decrypted_body")
            or cached.get("body")
            or doc.get("body_html")
            or ""
        )
        if not body:
            raise ValueError("Yuque appData contained no readable document body")

        parser = _LakeTextParser()
        parser.feed(body)
        content = parser.text()
        if not _is_complete_content(content):
            raise ValueError("Yuque document body is empty, incomplete, or a placeholder")

        title = str(doc.get("title") or "").strip()
        updated_at = str(
            doc.get("content_updated_at")
            or doc.get("updated_at")
            or app_data.get("timestamp")
            or ""
        )
        created_at = str(doc.get("created_at") or "")
        date = _extract_date(title, created_at, updated_at)
        slug = str(doc.get("slug") or source_url.rstrip("/").rsplit("/", 1)[-1])
        return {
            "slug": slug,
            "title": title,
            "date": date,
            "dates": _extract_dates(title, created_at, updated_at),
            "content": content,
            "source_url": source_url,
            "source_updated_at": updated_at,
            "raw_body": body,
            "images": parser.images,
        }

    async def download_journal(
        self,
        url: str,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        owns_client = client is None
        if client is None:
            client = self._client()
        markdown_url = f"{url.rstrip('/')}/markdown"
        last_error: Exception | None = None
        try:
            for attempt in range(3):
                try:
                    response = await client.get(markdown_url)
                    response.raise_for_status()
                    return self.extract_yuque_document(response.text, url)
                except Exception as exc:
                    last_error = exc
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (attempt + 1))
            raise RuntimeError(
                f"failed to download complete journal: {url}: {last_error}"
            ) from last_error
        finally:
            if owns_client:
                await client.aclose()

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(35.0, connect=10.0),
            follow_redirects=True,
            headers={
                # Yuque redirects malformed/unknown browser signatures to its
                # browser-upgrade page, which contains no document appData.
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/xhtml+xml",
            },
        )

    async def _localize_images(
        self,
        document: dict[str, Any],
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        *,
        force: bool = False,
    ) -> None:
        """Download every inline image and rewrite body references to local files."""
        slug = str(document["slug"])
        asset_dir = self._cache_dir / "assets" / slug
        asset_dir.mkdir(parents=True, exist_ok=True)
        unique: dict[str, dict[str, str]] = {}
        for image in document.get("images") or []:
            source_url = str(image.get("source_url") or "")
            if source_url:
                unique.setdefault(source_url, image)

        async def download_one(
            index: int, source_url: str, image: dict[str, str]
        ) -> dict[str, Any]:
            suffix = Path(urlparse(source_url).path).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}:
                suffix = ".img"
            filename = f"{index:03d}-{hashlib.sha1(source_url.encode()).hexdigest()[:12]}{suffix}"
            path = asset_dir / filename
            if not path.is_file() or force:
                async with semaphore:
                    response = await client.get(source_url)
                    response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                payload = response.content
                if (
                    len(payload) < 32
                    or not (
                        content_type.startswith("image/")
                        or payload.startswith((b"\xff\xd8", b"\x89PNG", b"GIF8", b"RIFF", b"<svg"))
                    )
                ):
                    raise ValueError(f"image response is invalid: {source_url}")
                temp = path.with_suffix(path.suffix + ".tmp")
                temp.write_bytes(payload)
                temp.replace(path)
            payload = path.read_bytes()
            if len(payload) < 32:
                raise ValueError(f"cached image is invalid: {path}")
            relative = path.relative_to(self._cache_dir).as_posix()
            return {
                "source_url": source_url,
                "alt": image.get("alt") or "日记图片",
                "file": str(path),
                "relative_path": relative,
                "content_hash": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "status": "complete",
            }

        records = await asyncio.gather(
            *(
                download_one(index, source_url, image)
                for index, (source_url, image) in enumerate(unique.items(), 1)
            )
        )
        replacements = {item["source_url"]: item["relative_path"] for item in records}
        content = str(document["content"])
        raw_body = str(document.get("raw_body") or "")
        for source_url, relative in replacements.items():
            content = content.replace(f"]({source_url})", f"]({relative})")
            raw_body = raw_body.replace(source_url, f"../{relative}")
            raw_body = raw_body.replace(
                html_module.escape(source_url, quote=True),
                html_module.escape(f"../{relative}", quote=True),
            )
        document["content"] = content
        document["raw_body"] = raw_body
        document["images"] = records

    def _persist_document(self, document: dict[str, Any]) -> dict[str, Any]:
        slug = document["slug"]
        path = self._cache_dir / f"{slug}.md"
        raw_dir = self._cache_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{slug}.html"
        content = document["content"].strip()
        rendered = (
            f"# {document['title']}\n\n"
            f"- 日期：{document['date'] or '未知'}\n"
            f"- 原文：{document['source_url']}\n"
            f"- 原文更新时间：{document['source_updated_at'] or '未知'}\n\n"
            f"{content}\n"
        )
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        temp_path = path.with_suffix(".md.tmp")
        temp_path.write_text(rendered, encoding="utf-8")
        temp_path.replace(path)
        raw_temp = raw_path.with_suffix(".html.tmp")
        raw_temp.write_text(str(document.get("raw_body") or ""), encoding="utf-8")
        raw_temp.replace(raw_path)
        now = datetime.now(timezone.utc).isoformat()
        return {
            "slug": slug,
            "title": document["title"],
            "date": document["date"],
            "dates": list(document.get("dates") or ([document["date"]] if document["date"] else [])),
            "source_url": document["source_url"],
            "source_updated_at": document["source_updated_at"],
            "fetched_at": now,
            "content_hash": content_hash,
            "file": str(path),
            "raw_file": str(raw_path),
            "status": "complete",
            "content_chars": len(content),
            "image_count": len(document.get("images") or []),
            "images": list(document.get("images") or []),
            "images_complete": all(
                image.get("status") == "complete"
                for image in document.get("images") or []
            ),
        }

    async def sync(
        self,
        memory_store: Any | None = None,
        *,
        refresh_all: bool = False,
        refresh_slugs: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Synchronize all official entries and return one result per listing entry.

        New or incomplete documents are always downloaded.  ``refresh_all`` is
        used by the full corpus command; ``refresh_slugs`` revalidates documents
        selected for an answer before that answer is generated.
        """

        refresh_set = set(refresh_slugs or ())
        manifest = self.load_manifest()
        old_entries: dict[str, Any] = dict(manifest.get("entries") or {})

        async with self._client() as client:
            listing = await self.fetch_listing(client)
            if not listing:
                raise RuntimeError("官方日记目录不可用，无法校验完整性")

            semaphore = asyncio.Semaphore(4)
            image_semaphore = asyncio.Semaphore(8)

            async def sync_one(entry: dict[str, str]) -> dict[str, Any]:
                slug = entry["slug"]
                existing = old_entries.get(slug) or {}
                existing_file = Path(str(existing.get("file") or ""))
                cached_images = existing.get("images") or []
                cached_assets_complete = (
                    existing.get("images_complete") is True
                    and all(
                        Path(str(image.get("file") or "")).is_file()
                        for image in cached_images
                    )
                )
                must_fetch = (
                    refresh_all
                    or slug in refresh_set
                    or existing.get("status") != "complete"
                    or not existing_file.is_file()
                    or not Path(str(existing.get("raw_file") or "")).is_file()
                    or not cached_assets_complete
                )
                if not must_fetch:
                    return {**existing, "new": False, "changed": False}

                async with semaphore:
                    document = await self.download_journal(entry["url"], client)
                    await self._localize_images(
                        document,
                        client,
                        image_semaphore,
                        force=refresh_all,
                    )
                persisted = self._persist_document(document)
                changed = persisted["content_hash"] != existing.get("content_hash")
                if memory_store is not None:
                    memory_store.upsert_journal(persisted, document["content"])
                return {
                    **persisted,
                    "new": not bool(existing),
                    "changed": changed,
                }

            raw_results = await asyncio.gather(
                *(sync_one(entry) for entry in listing),
                return_exceptions=True,
            )

        results: list[dict[str, Any]] = []
        failures: list[str] = []
        for entry, result in zip(listing, raw_results):
            if isinstance(result, BaseException):
                slug = entry["slug"]
                cached = old_entries.get(slug)
                if cached and cached.get("status") == "complete" and Path(
                    str(cached.get("file") or "")
                ).is_file():
                    results.append({**cached, "new": False, "changed": False, "stale": True})
                    failures.append(f"{slug}: {result}")
                else:
                    failures.append(f"{slug}: {result}")
                continue
            results.append(result)

        complete_by_slug = {
            item["slug"]: {k: v for k, v in item.items() if k not in {"new", "changed", "stale"}}
            for item in results
            if item.get("status") == "complete"
        }
        expected_slugs = {entry["slug"] for entry in listing}
        missing = sorted(expected_slugs - set(complete_by_slug))
        now = datetime.now(timezone.utc).isoformat()
        new_manifest = {
            "version": 3,
            "listing_url": self._listing_url,
            "last_checked_at": now,
            "last_success_at": (
                now
                if not missing and not failures
                else manifest.get("last_success_at", "")
            ),
            "expected_count": len(expected_slugs),
            "complete_count": len(complete_by_slug),
            "failures": failures,
            "entries": complete_by_slug,
        }
        self._write_manifest(new_manifest)

        if missing:
            logger.warning(
                "日记同步不完整：官方 %d 篇，完整保存 %d 篇，缺少 %d 篇；%s",
                len(expected_slugs),
                len(complete_by_slug),
                len(missing),
                "; ".join(failures[:3]),
            )

        if memory_store is not None:
            memory_store.remove_missing_journals(expected_slugs)
        logger.info(
            "Journal sync complete: official=%d complete=%d refreshed=%d",
            len(expected_slugs),
            len(complete_by_slug),
            sum(1 for item in results if item.get("changed")),
        )
        return results

    def health(self) -> dict[str, Any]:
        manifest = self.load_manifest()
        entries = manifest.get("entries") or {}
        return {
            "expected": int(manifest.get("expected_count") or len(entries)),
            "complete": sum(1 for item in entries.values() if item.get("status") == "complete"),
            "images": sum(int(item.get("image_count") or 0) for item in entries.values()),
            "images_complete": sum(
                1
                for item in entries.values()
                for image in item.get("images") or []
                if image.get("status") == "complete"
                and Path(str(image.get("file") or "")).is_file()
            ),
            "last_checked_at": manifest.get("last_checked_at") or "",
            "last_success_at": manifest.get("last_success_at") or "",
            "failures": list(manifest.get("failures") or []),
        }
