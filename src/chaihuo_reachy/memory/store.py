"""Versioned, updateable journal retrieval store."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable

logger = logging.getLogger("chaihuo_reachy.memory")

_COLLECTION_NAME = "mcv_journals_v2"
_QUERY_STOP_PHRASES = (
    "请问", "帮我", "讲讲", "说说", "介绍一下", "完整总结", "详细总结",
    "基地车", "旅途日记", "日记", "发生了什么", "有什么", "怎么样",
    "是什么", "为什么", "怎么", "哪里", "哪儿", "去过", "到过",
    "去了", "当天", "那天", "这件事", "一下", "什么",
)
_JOURNEY_QUERY_MARKERS = (
    "哪些站点", "都去了", "都去过", "去过哪些", "行程", "旅程", "路线",
    "途经", "途径", "一路", "回忆", "回顾",
)
_JOURNEY_QUERY_NOISE_TERMS = {
    "我们", "你们", "哪些", "哪里", "站点", "精华", "回忆", "回顾",
    "帮我", "一下", "请问", "基地", "地车", "日记", "行程", "旅程",
    "路线", "发生", "什么", "都有", "都去", "去了", "去过", "哪些",
    "有哪", "哪些", "给我", "一下", "一下", "回忆", "回顾", "精华",
}
_JOURNEY_TERM_EDGE_NOISE = set("我你他她它们在到从向往去都的了啊呀呢吗和与及把被给帮请问有哪些什么")


def _body_from_cached_file(text: str) -> str:
    """Remove the generated metadata header from a canonical journal file."""
    marker = "\n\n"
    if text.startswith("# "):
        # The canonical writer emits title + four metadata lines + blank line.
        parts = text.split(marker, 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text.strip()


def _title_range_contains(title: str, target_date: str) -> bool:
    """Match compact title ranges such as ``2026.05.18-20``."""
    from datetime import date, timedelta

    try:
        target = date.fromisoformat(target_date)
    except ValueError:
        return False
    pattern = re.compile(
        r"(20\d{2})[.\-/年](\d{1,2})[.\-/月](\d{1,2})"
        r"\s*(?:-|—|–|至|~)\s*(?:(\d{1,2})[.\-/月])?(\d{1,2})"
    )
    match = pattern.search(title)
    if not match:
        return False
    try:
        start = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        end = date(start.year, int(match.group(4) or start.month), int(match.group(5)))
    except ValueError:
        return False
    return start <= target <= end and end - start <= timedelta(days=31)


def _chunks(content: str, *, max_chars: int = 1200, overlap: int = 160) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + max_chars)
            chunks.append(paragraph[start:end])
            if end == len(paragraph):
                current = ""
                break
            start = max(start + 1, end - overlap)
    if current:
        chunks.append(current)
    return chunks


class MemoryStore:
    """Chroma-backed journal index with full-text source retrieval.

    Chroma stores only searchable chunks.  Returned answers load the complete
    canonical file, so a date-specific summary never relies on a truncated
    vector-store metadata field.
    """

    def __init__(
        self,
        persist_dir: str = "data/chroma",
        journal_dir: str = "data/journals",
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        self._journal_dir = Path(journal_dir)
        self._journal_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._journal_dir / "manifest.json"

        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "schema_version": 2},
        )
        self._conversation_collection = self._client.get_or_create_collection(
            name="conversation_memory",
            metadata={"hnsw:space": "cosine"},
        )
        self._index_cached_journals()

    def _manifest_entries(self) -> dict[str, dict[str, Any]]:
        if not self._manifest_path.exists():
            return {}
        try:
            value = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            entries = value.get("entries") if isinstance(value, dict) else {}
            return entries if isinstance(entries, dict) else {}
        except Exception:
            logger.warning("Cannot read journal manifest", exc_info=True)
            return {}

    def _index_cached_journals(self) -> None:
        for entry in self._manifest_entries().values():
            if entry.get("status") != "complete":
                continue
            path = Path(str(entry.get("file") or ""))
            if not path.is_file():
                continue
            try:
                content = _body_from_cached_file(path.read_text(encoding="utf-8"))
                self.upsert_journal(entry, content, skip_if_current=True)
            except Exception:
                logger.warning("Failed to index %s", path, exc_info=True)

    def upsert_journal(
        self,
        entry: dict[str, Any],
        content: str,
        *,
        skip_if_current: bool = False,
    ) -> None:
        slug = str(entry["slug"])
        content_hash = str(
            entry.get("content_hash")
            or hashlib.sha256(content.encode("utf-8")).hexdigest()
        )
        existing = self._collection.get(
            where={"slug": slug},
            include=["metadatas"],
        )
        if existing.get("ids"):
            metadatas = existing.get("metadatas") or []
            if metadatas and all(meta.get("content_hash") == content_hash for meta in metadatas):
                return

        chunks = _chunks(content)
        if not chunks:
            return
        ids = [
            f"{slug}:{index}:{content_hash[:10]}"
            for index in range(len(chunks))
        ]
        metadatas = [
            {
                "slug": slug,
                "title": str(entry.get("title") or ""),
                "date": str(entry.get("date") or ""),
                "dates": "|".join(entry.get("dates") or ([entry.get("date")] if entry.get("date") else [])),
                "source_url": str(entry.get("source_url") or ""),
                "source_updated_at": str(entry.get("source_updated_at") or ""),
                "content_hash": content_hash,
                "file": str(entry.get("file") or ""),
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]
        # New content hashes produce new IDs.  Add the complete replacement
        # first, then retire the old generation so a failed embedding/upsert
        # never destroys the last valid searchable version.
        self._collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        if existing.get("ids"):
            self._collection.delete(ids=list(existing["ids"]))

    def remove_missing_journals(self, expected_slugs: Iterable[str]) -> None:
        expected = set(expected_slugs)
        data = self._collection.get(include=["metadatas"])
        stale_ids = [
            doc_id
            for doc_id, metadata in zip(data.get("ids") or [], data.get("metadatas") or [])
            if metadata.get("slug") not in expected
        ]
        if stale_ids:
            self._collection.delete(ids=stale_ids)

    @staticmethod
    def _item_from_metadata(
        metadata: dict[str, Any],
        *,
        score: float,
        snippet: str = "",
    ) -> dict[str, Any]:
        path = Path(str(metadata.get("file") or ""))
        content = ""
        if path.is_file():
            content = _body_from_cached_file(path.read_text(encoding="utf-8"))
        return {
            "id": metadata.get("slug", ""),
            "slug": metadata.get("slug", ""),
            "content": content,
            "snippet": snippet,
            "title": metadata.get("title", ""),
            "date": metadata.get("date", ""),
            "source_url": metadata.get("source_url", ""),
            "source_updated_at": metadata.get("source_updated_at", ""),
            "score": score,
        }

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        if self._collection.count() == 0:
            return []
        compact_query = re.sub(r"\s+", "", query)
        cleaned_query = compact_query
        for phrase in _QUERY_STOP_PHRASES:
            cleaned_query = cleaned_query.replace(phrase, " ")
        exact_terms = [
            token
            for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", cleaned_query)
            if len(token) >= 2
        ]

        # Scan the small canonical manifest before vector search.  Chinese
        # embeddings can rank adjacent dates above an exact school/place name;
        # exact source text must win for factual answers.
        best: dict[str, dict[str, Any]] = {}
        for entry in self._manifest_entries().values():
            path = Path(str(entry.get("file") or ""))
            if not path.is_file():
                continue
            content = _body_from_cached_file(path.read_text(encoding="utf-8"))
            title = str(entry.get("title") or "")
            compact_title = re.sub(r"\s+", "", title)
            compact_content = re.sub(r"\s+", "", content)
            title_hit = any(term.lower() in compact_title.lower() for term in exact_terms)
            content_hit = any(term.lower() in compact_content.lower() for term in exact_terms)
            if title_hit or content_hit:
                metadata = {
                    "slug": entry.get("slug", ""),
                    "title": title,
                    "date": entry.get("date", ""),
                    "source_url": entry.get("source_url", ""),
                    "source_updated_at": entry.get("source_updated_at", ""),
                    "file": str(path),
                }
                best[str(entry.get("slug") or "")] = self._item_from_metadata(
                    metadata,
                    score=0.95 if title_hit else 0.82,
                    snippet=content[:1200],
                )

        result = self._collection.query(
            query_texts=[query],
            n_results=min(max(k * 3, 5), self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for _, metadata, document, distance in zip(ids, metadatas, documents, distances):
            slug = str(metadata.get("slug") or "")
            score = 1.0 - float(distance)
            current = best.get(slug)
            if current is None or score > current["score"]:
                best[slug] = self._item_from_metadata(
                    metadata, score=score, snippet=document
                )

        # Exact title/date/place substring matches are trustworthy even when
        # the generic embedding model scores Chinese text conservatively.
        for item in best.values():
            haystack = re.sub(
                r"\s+", "", f"{item['date']} {item['title']} {item['snippet']}"
            )
            if any(token.lower() in haystack.lower() for token in exact_terms):
                item["score"] = max(item["score"], 0.72)

        ranked = sorted(best.values(), key=lambda item: item["score"], reverse=True)
        return ranked[:k]

    def search_journey_scope(self, query: str, k: int = 6) -> list[dict[str, Any]]:
        """Return a chronological group of entries for a region-wide journey.

        Vector retrieval is useful for a specific event, but it can collapse a
        question such as "山西都去了哪些站点" into one semantically similar day.
        For explicit itinerary/recap questions, identify short place anchors
        from the query that recur in the verified corpus and return all of the
        matching diary entries in date order.
        """
        if not any(marker in query for marker in _JOURNEY_QUERY_MARKERS):
            return []

        compact_query = re.sub(r"\s+", "", query)
        terms = {
            compact_query[start:end]
            for size in range(2, 5)
            for start in range(max(0, len(compact_query) - size + 1))
            for end in (start + size,)
            if compact_query[start:end] not in _JOURNEY_QUERY_NOISE_TERMS
            and compact_query[start] not in _JOURNEY_TERM_EDGE_NOISE
            and compact_query[end - 1] not in _JOURNEY_TERM_EDGE_NOISE
        }
        entries: list[tuple[dict[str, Any], str, str]] = []
        hits_by_term: dict[str, list[int]] = {term: [] for term in terms}
        for entry in self._manifest_entries().values():
            if entry.get("status") != "complete":
                continue
            path = Path(str(entry.get("file") or ""))
            if not path.is_file():
                continue
            title = str(entry.get("title") or "")
            content = _body_from_cached_file(path.read_text(encoding="utf-8"))
            haystack = re.sub(r"\s+", "", f"{title}\n{content}")
            index = len(entries)
            entries.append((entry, title, content))
            for term in terms:
                if term in haystack:
                    hits_by_term[term].append(index)

        # A geographical anchor should connect several days, but terms that
        # match most of the corpus are generic prose rather than a location.
        anchor_terms = {
            term
            for term, indexes in hits_by_term.items()
            if 2 <= len(indexes) <= max(12, len(entries) // 3)
        }
        if not anchor_terms:
            return []

        selected_indexes = {
            index
            for term in anchor_terms
            for index in hits_by_term[term]
        }
        selected: list[dict[str, Any]] = []
        for index in selected_indexes:
            entry, title, content = entries[index]
            metadata = {
                "slug": entry.get("slug", ""),
                "title": title,
                "date": entry.get("date", ""),
                "source_url": entry.get("source_url", ""),
                "source_updated_at": entry.get("source_updated_at", ""),
                "file": entry.get("file", ""),
            }
            selected.append(
                self._item_from_metadata(metadata, score=1.0, snippet=content[:1200])
            )
        selected.sort(key=lambda item: (str(item.get("date") or ""), str(item.get("slug") or "")))
        return selected[:k]

    def search_by_date(self, date_str: str, k: int = 3) -> list[dict[str, Any]]:
        if self._collection.count() == 0:
            return []
        manifest_items: list[dict[str, Any]] = []
        for entry in self._manifest_entries().values():
            dates = set(entry.get("dates") or [])
            if entry.get("date"):
                dates.add(str(entry["date"]))
            # Older manifests predate the explicit date-range field.
            title = str(entry.get("title") or "")
            if date_str not in dates and not _title_range_contains(title, date_str):
                continue
            metadata = {
                "slug": entry.get("slug", ""),
                "title": title,
                "date": entry.get("date", ""),
                "source_url": entry.get("source_url", ""),
                "source_updated_at": entry.get("source_updated_at", ""),
                "file": entry.get("file", ""),
            }
            manifest_items.append(self._item_from_metadata(metadata, score=1.0))
        if manifest_items:
            return manifest_items[:k]
        data = self._collection.get(
            where={"date": date_str},
            include=["documents", "metadatas"],
        )
        seen: set[str] = set()
        items: list[dict[str, Any]] = []
        for metadata, document in zip(
            data.get("metadatas") or [], data.get("documents") or []
        ):
            slug = str(metadata.get("slug") or "")
            if slug in seen:
                continue
            seen.add(slug)
            items.append(self._item_from_metadata(metadata, score=1.0, snippet=document))
        return items[:k]

    def get_by_slugs(self, slugs: Iterable[str]) -> list[dict[str, Any]]:
        wanted = list(dict.fromkeys(slugs))
        items: list[dict[str, Any]] = []
        for slug in wanted:
            data = self._collection.get(
                where={"slug": slug},
                include=["documents", "metadatas"],
            )
            metadatas = data.get("metadatas") or []
            documents = data.get("documents") or []
            if metadatas:
                items.append(
                    self._item_from_metadata(
                        metadatas[0],
                        score=1.0,
                        snippet=documents[0] if documents else "",
                    )
                )
        return items

    def add_journal(self, title: str, content: str, date: str = "") -> str:
        """Compatibility helper used by older callers and tests."""
        slug = hashlib.md5(f"{title}{date}".encode()).hexdigest()[:12]
        path = self._journal_dir / f"{slug}.md"
        path.write_text(content, encoding="utf-8")
        entry = {
            "slug": slug,
            "title": title,
            "date": date,
            "source_url": "",
            "source_updated_at": "",
            "file": str(path),
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        }
        self.upsert_journal(entry, content)
        return slug

    def add_conversation_turn(self, role: str, content: str) -> None:
        index = len(self._conversation_collection.get()["ids"])
        doc_id = hashlib.md5(f"{role}{content}{index}".encode()).hexdigest()[:12]
        self._conversation_collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[{"role": role, "timestamp": str(__import__("time").time())}],
        )

    def count(self) -> int:
        return len(self._manifest_entries())

    def chunk_count(self) -> int:
        return self._collection.count()

    def health(self) -> dict[str, Any]:
        entries = self._manifest_entries()
        complete = sum(1 for item in entries.values() if item.get("status") == "complete")
        return {
            "documents": len(entries),
            "complete": complete,
            "chunks": self._collection.count(),
            "ready": bool(entries) and complete == len(entries),
        }

    def reset(self) -> None:
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "schema_version": 2},
        )
