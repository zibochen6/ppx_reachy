"""Versioned, updateable journal retrieval store."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Iterable

logger = logging.getLogger("chaihuo_reachy.memory")

_COLLECTION_NAME = "mcv_journals_v2"
_COLLECTION_NAME_V3 = "mcv_journals_v3"
_QUERY_STOP_PHRASES = (
    "请问", "帮我", "讲讲", "说说", "介绍一下", "完整总结", "详细总结",
    "基地车", "旅途日记", "日记", "发生了什么", "有什么", "怎么样",
    "是什么", "为什么", "怎么", "哪里", "哪儿", "去过", "到过",
    "去了", "当天", "那天", "这件事", "一下", "什么",
)
_JOURNEY_QUERY_MARKERS = (
    "哪些站点", "都去了", "都去过", "去过哪些", "行程", "旅程", "路线",
    "途经", "途径", "一路", "回忆", "回顾", "都做了什么", "干了什么",
)
_JOURNEY_QUERY_NOISE_TERMS = {
    "我们", "你们", "哪些", "哪里", "站点", "精华", "回忆", "回顾",
    "帮我", "一下", "请问", "基地", "地车", "日记", "行程", "旅程",
    "路线", "发生", "什么", "都有", "都去", "去了", "去过", "哪些",
    "有哪", "哪些", "给我", "一下", "一下", "回忆", "回顾", "精华",
}
_JOURNEY_TERM_EDGE_NOISE = set("我你他她它们在到从向往去都的了啊呀呢吗和与及把被给帮请问有哪些什么")
_ENTITY_NOISE = {
    "我们", "咱们", "基地车", "皮皮虾", "日记", "故事", "经历", "活动",
    "记得", "还记得", "做了什么", "发生了什么", "有什么", "介绍", "详细",
    "大学", "学院", "城市", "地方", "现在", "目前", "当天", "那天",
}
_REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "内蒙古": ("内蒙古", "内蒙古自治区", "晋蒙交界"),
    "北京": ("北京", "北京市"),
    "山西": ("山西", "晋南", "晋蒙交界"),
    "陕西": ("陕西", "陕北", "陕蒙交界"),
    "宁夏": ("宁夏", "宁夏回族自治区"),
    "甘肃": ("甘肃",),
    "青海": ("青海",),
    "西藏": ("西藏", "西藏自治区"),
    "四川": ("四川",),
    "贵州": ("贵州",),
    "广西": ("广西", "广西壮族自治区"),
    "广东": ("广东",),
}

# Canonical province-level ownership for principal stops that appear in diary
# titles. This avoids asking an LLM to remember administrative geography while
# summarizing a protected journey fact (for example, 哈密 is新疆, not甘肃).
_ROUTE_PLACE_CATALOG: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("广东", (("广东科学中心", "广东科学中心"), ("阳江", "阳江"))),
    (
        "广西",
        (
            ("玉林", "玉林"),
            ("广西科技馆", "广西科技馆"),
            ("柳州", "柳州"),
            ("柳江", "柳江"),
            ("三都镇", "三都镇"),
            ("七百弄", "七百弄"),
            ("浩坤湖", "浩坤湖"),
        ),
    ),
    (
        "贵州",
        (
            ("格凸河", "格凸河"),
            ("贵阳", "贵阳"),
            ("贵州师院", "贵州师院"),
            ("贵州大学", "贵州大学"),
            ("赫章", "赫章"),
        ),
    ),
    (
        "四川",
        (
            ("宜宾", "宜宾"),
            ("成都", "成都"),
            ("绵阳", "绵阳"),
            ("江油", "江油"),
            ("老河沟驿站", "老河沟驿站"),
            ("清溪镇", "清溪镇"),
            ("唐家河", "唐家河"),
            ("雅安", "雅安"),
            ("塔公", "塔公"),
            ("雅江", "雅江"),
            ("理塘", "理塘"),
            ("巴塘", "巴塘"),
        ),
    ),
    (
        "西藏",
        (
            ("入藏", "入藏"),
            ("芒康", "芒康"),
            ("如美", "如美"),
            ("东达山垭口", "东达山垭口"),
            ("左贡", "左贡"),
        ),
    ),
    ("新疆", (("伊吾", "伊吾"), ("哈密", "哈密"))),
    (
        "甘肃",
        (
            ("敦煌", "敦煌"),
            ("玉门", "玉门"),
            ("酒泉", "酒泉"),
            ("肃南", "肃南"),
            ("兰州", "兰州"),
        ),
    ),
    ("宁夏", (("吴忠", "吴忠"), ("银川", "银川"))),
    ("陕西", (("定边", "定边"), ("榆林", "榆林"), ("西安", "西安"))),
    (
        "山西",
        (
            ("隰县", "隰县"),
            ("临汾", "临汾"),
            ("太原", "太原"),
            ("晋蒙交界", "晋蒙交界"),
        ),
    ),
    ("内蒙古", (("呼和浩特", "呼和浩特"),)),
    ("北京", (("北京", "北京"), ("清华园", "清华园"), ("iCenter", "iCenter"))),
)


class _ChineseHashEmbedding:
    """Dependency-free, stable Chinese character n-gram vectorizer for V3."""

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in input:
            compact = re.sub(r"\s+", "", text.lower())
            vector = [0.0] * self.dimensions
            grams = [
                compact[start : start + size]
                for size in (2, 3, 4)
                for start in range(max(0, len(compact) - size + 1))
            ]
            for gram in grams:
                digest = hashlib.sha256(gram.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                vector[index] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    @staticmethod
    def name() -> str:
        return "chaihuo-chinese-hash-v1"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "_ChineseHashEmbedding":
        return _ChineseHashEmbedding(int(config.get("dimensions", 512)))

    def get_config(self) -> dict[str, Any]:
        return {"dimensions": self.dimensions}

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]


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


def _extract_query_entities(query: str) -> list[tuple[str, tuple[str, ...]]]:
    """Extract place/school entities whose identity must match a diary."""

    compact = re.sub(r"\s+", "", query)
    entities: list[tuple[str, tuple[str, ...]]] = []
    for canonical, aliases in _REGION_ALIASES.items():
        if any(alias in compact for alias in aliases):
            entities.append((canonical, aliases))

    patterns = (
        r"(?:在|到|去过|来到|记得)([\u4e00-\u9fff]{2,14}?)(?=都|做|干|发生|有|去|走|经历|吗|呢|[，。！？!?]|$)",
        r"([\u4e00-\u9fff]{2,12}(?:大学|学院|市|县|镇|村|景区))",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, compact):
            entity = match.group(1).strip("我们咱们现在目前还")
            if entity in _ENTITY_NOISE or len(entity) < 2:
                continue
            if any(entity == canonical for canonical, _ in entities):
                continue
            entities.append((entity, (entity,)))
    return entities


def _meaningful_query_terms(query: str) -> list[str]:
    compact = re.sub(r"\s+", "", query)
    for phrase in (*_QUERY_STOP_PHRASES, *_ENTITY_NOISE):
        compact = compact.replace(phrase, " ")
    terms = [
        token
        for token in re.findall(r"[\u4e00-\u9fff]{2,12}|[A-Za-z0-9]{2,}", compact)
        if token not in _ENTITY_NOISE and len(token) >= 2
    ]
    return sorted(set(terms), key=len, reverse=True)


def _matching_snippet(content: str, terms: Iterable[str], max_chars: int = 2200) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    wanted = [term for term in terms if term]
    matched = [p for p in paragraphs if any(term in p for term in wanted)]
    if not matched:
        matched = paragraphs[:3]
    result = ""
    for paragraph in matched:
        candidate = f"{result}\n\n{paragraph}".strip()
        if len(candidate) > max_chars:
            if not result:
                return paragraph[:max_chars]
            break
        result = candidate
    return result


def _entry_entity_score(
    title: str,
    content: str,
    aliases: Iterable[str],
) -> tuple[float, list[str]]:
    matched = [alias for alias in aliases if alias in title or alias in content]
    if not matched:
        return 0.0, []
    title_hits = sum(1 for alias in matched if alias in title)
    body_hits = sum(min(content.count(alias), 3) for alias in matched)
    score = min(0.99, 0.58 + title_hits * 0.18 + body_hits * 0.04)
    return score, matched


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
        *,
        use_v3: bool = True,
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
        self._collection_name = _COLLECTION_NAME_V3 if use_v3 else _COLLECTION_NAME
        self._use_v3 = use_v3
        if use_v3:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine", "schema_version": 3},
                embedding_function=_ChineseHashEmbedding(),
            )
        else:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
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
        if getattr(self, "_use_v3", False):
            return self._search_v3(query, k=k)
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

    def _search_v3(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """Entity-gated hybrid retrieval with literal and Chinese-vector scores."""

        entities = _extract_query_entities(query)
        terms = _meaningful_query_terms(query)
        candidates: dict[str, dict[str, Any]] = {}
        for entry in self._manifest_entries().values():
            if entry.get("status") != "complete":
                continue
            path = Path(str(entry.get("file") or ""))
            if not path.is_file():
                continue
            title = str(entry.get("title") or "")
            content = _body_from_cached_file(path.read_text(encoding="utf-8"))
            matched_terms: list[str] = []
            scores: list[float] = []
            if entities:
                for _, aliases in entities:
                    score, matched = _entry_entity_score(title, content, aliases)
                    if score <= 0:
                        scores = []
                        break
                    scores.append(score)
                    matched_terms.extend(matched)
                if not scores:
                    continue
                score = sum(scores) / len(scores)
            else:
                title_hits = [term for term in terms if term in title]
                body_hits = [term for term in terms if term in content]
                matched_terms = list(dict.fromkeys([*title_hits, *body_hits]))
                if not matched_terms:
                    continue
                coverage = len(matched_terms) / max(1, len(terms))
                score = min(0.93, 0.46 + coverage * 0.28 + len(title_hits) * 0.12)

            metadata = {
                "slug": entry.get("slug", ""),
                "title": title,
                "date": entry.get("date", ""),
                "source_url": entry.get("source_url", ""),
                "source_updated_at": entry.get("source_updated_at", ""),
                "file": str(path),
            }
            candidates[str(entry.get("slug") or "")] = self._item_from_metadata(
                metadata,
                score=score,
                snippet=_matching_snippet(content, matched_terms or terms),
            )

        # An explicit place/school entity is a hard constraint. If the corpus
        # does not contain it, returning a semantically similar place is worse
        # than returning no evidence.
        if entities:
            ranked = sorted(
                candidates.values(),
                key=lambda item: (float(item["score"]), str(item.get("date") or "")),
                reverse=True,
            )
            return ranked[:k]

        if self._collection.count() and query.strip():
            result = self._collection.query(
                query_texts=[query],
                n_results=min(max(k * 2, 4), self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            metadatas = (result.get("metadatas") or [[]])[0]
            documents = (result.get("documents") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            for metadata, document, distance in zip(metadatas, documents, distances):
                score = 1.0 - float(distance)
                if score < 0.52:
                    continue
                slug = str(metadata.get("slug") or "")
                current = candidates.get(slug)
                if current is None or score > float(current["score"]):
                    candidates[slug] = self._item_from_metadata(
                        metadata,
                        score=score,
                        snippet=str(document or "")[:2200],
                    )

        return sorted(
            candidates.values(),
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:k]

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

        if getattr(self, "_use_v3", False):
            entities = _extract_query_entities(query)
            if not entities:
                return []
            # Reuse entity-gated retrieval with a deliberately large internal
            # limit, then present a regional recap chronologically.
            selected = self._search_v3(query, k=max(k, len(self._manifest_entries())))
            entity_aliases = {
                alias for _, aliases in entities for alias in aliases
            }
            selected = [
                item
                for item in selected
                if any(
                    alias
                    in f"{item.get('title', '')}\n{str(item.get('content') or '')[:1200]}"
                    for alias in entity_aliases
                )
            ]
            selected.sort(
                key=lambda item: (str(item.get("date") or ""), str(item.get("slug") or ""))
            )
            return selected[:k]

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

    def search_keywords(self, query: str, k: int = 6) -> list[dict[str, Any]]:
        """Entity-term full-text retrieval over titles and bodies.

        Vector similarity fails narrative queries ("我们在兰州发生了什么"
        ranks unrelated days above the Lanzhou journal), so extract 2-4 char
        n-grams from the query and match them literally against every
        journal.  Journals hit by the most distinct terms win; scores are
        pinned to 1.0 so the relevance threshold does not discard them.
        """
        if getattr(self, "_use_v3", False):
            return self._search_v3(query, k=k)

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
        terms = {term for term in terms if len(term) >= 2 and not term.isdigit()}
        if not terms:
            return []
        scored: list[tuple[int, dict[str, Any], str]] = []
        for entry in self._manifest_entries().values():
            if entry.get("status") != "complete":
                continue
            path = Path(str(entry.get("file") or ""))
            if not path.is_file():
                continue
            title = str(entry.get("title") or "")
            content = _body_from_cached_file(path.read_text(encoding="utf-8"))
            title_flat = re.sub(r"\s+", "", title)
            body_flat = re.sub(r"\s+", "", content)
            # Title hits weigh double: "酒泉职大" in a title beats a passing
            # mention in another day's body.
            hits = sum(2 if term in title_flat else 1 for term in terms if term in body_flat)
            if hits:
                metadata = {
                    "slug": entry.get("slug", ""),
                    "title": title,
                    "date": str(entry.get("date") or ""),
                    "source_url": str(entry.get("source_url") or ""),
                    "source_updated_at": str(entry.get("source_updated_at") or ""),
                    "file": str(entry.get("file") or ""),
                }
                scored.append((hits, metadata, content))
        if not scored:
            return []
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0][0]
        items: list[dict[str, Any]] = []
        for hits, metadata, content in scored[:k]:
            items.append(
                self._item_from_metadata(
                    metadata,
                    score=1.0,  # literal hit — never filtered by relevance
                    snippet=content[:1600],
                )
            )
        return items

    def search_recent(self, days: int = 7, k: int = 6) -> list[dict[str, Any]]:
        """Return the newest journals within the last ``days`` days, newest first.

        Handles "最近发生了什么 / 这几天 / 近来" — queries where vector
        similarity has no time meaning.  Entries whose date or title range
        falls inside the window are returned with full body snippets so the
        LLM can summarize them.
        """
        from datetime import date, timedelta

        today = date.today()
        cutoff = (today - timedelta(days=max(1, days))).isoformat()
        items: list[tuple[str, dict[str, Any], str]] = []
        for entry in self._manifest_entries().values():
            if entry.get("status") != "complete":
                continue
            path = Path(str(entry.get("file") or ""))
            if not path.is_file():
                continue
            entry_date = str(entry.get("date") or "")
            if entry_date and entry_date >= cutoff and entry_date <= today.isoformat():
                content = _body_from_cached_file(path.read_text(encoding="utf-8"))
                metadata = {
                    "slug": entry.get("slug", ""),
                    "title": str(entry.get("title") or ""),
                    "date": entry_date,
                    "source_url": str(entry.get("source_url") or ""),
                    "source_updated_at": str(entry.get("source_updated_at") or ""),
                    "file": str(entry.get("file") or ""),
                }
                items.append((entry_date, metadata, content))
        items.sort(key=lambda item: item[0], reverse=True)
        return [
            self._item_from_metadata(metadata, score=1.0, snippet=content[:1600])
            for _, metadata, content in items[:k]
        ]

    def search_journey_overview(self, k: int = 80) -> list[dict[str, Any]]:
        """Return every verified journey title in chronological order.

        A route overview needs broad corpus coverage, but injecting every full
        article would add noise and latency. Diary titles already contain the
        date and principal stops, so this path provides those compact verified
        facts while retaining source metadata for traceability.
        """

        items: list[dict[str, Any]] = []
        for entry in self._manifest_entries().values():
            if entry.get("status") != "complete":
                continue
            path = Path(str(entry.get("file") or ""))
            title = str(entry.get("title") or "").strip()
            if not path.is_file() or not title:
                continue
            metadata = {
                "slug": entry.get("slug", ""),
                "title": title,
                "date": str(entry.get("date") or ""),
                "source_url": str(entry.get("source_url") or ""),
                "source_updated_at": str(entry.get("source_updated_at") or ""),
                "file": str(path),
            }
            items.append(self._item_from_metadata(metadata, score=1.0, snippet=title))
        items.sort(
            key=lambda item: (
                str(item.get("date") or ""),
                str(item.get("slug") or ""),
            )
        )
        return items[: max(1, k)]

    def format_journey_overview(self) -> str:
        """Compose a grounded route answer without model geography guesses."""

        items = self.search_journey_overview(k=200)
        if not items:
            return ""
        titles = "\n".join(str(item.get("title") or "") for item in items)
        stages: list[str] = []
        for region, catalog in _ROUTE_PLACE_CATALOG:
            places: list[str] = []
            for marker, display in catalog:
                if marker in titles and display not in places:
                    places.append(display)
            if places:
                stages.append(f"{region}（{'、'.join(places)}）")
        if not stages:
            return ""
        first_date = str(items[0].get("date") or "日期未知")
        last_date = str(items[-1].get("date") or "日期未知")
        return (
            f"按目前已完整验证的 {len(items)} 篇基地车日记，从 {first_date} 到 "
            f"{last_date}，我们走过的主要路线是：{' → '.join(stages)}。"
            "这是按日记标题确认的历史足迹；重复停留已经合并，日记没有明确记录的"
            "地点、总里程和行政区数量我不额外猜。"
        )

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
        indexed_data = self._collection.get(include=["metadatas"])
        indexed_slugs = {
            str(metadata.get("slug") or "")
            for metadata in indexed_data.get("metadatas") or []
        }
        coverage: dict[str, dict[str, Any]] = {}
        for entry in entries.values():
            dates = entry.get("dates") or [entry.get("date")]
            for date in (str(value or "") for value in dates):
                if not date:
                    continue
                coverage[date] = {
                    "discovered": True,
                    "fetched": bool(entry.get("fetched_at")),
                    "validated": entry.get("status") == "complete",
                    "indexed": str(entry.get("slug") or "") in indexed_slugs,
                    "slug": str(entry.get("slug") or ""),
                }
        return {
            "documents": len(entries),
            "complete": complete,
            "chunks": self._collection.count(),
            "schema_version": 3 if self._use_v3 else 2,
            "collection": self._collection_name,
            "coverage": coverage,
            "ready": (
                bool(entries)
                and complete == len(entries)
                and all(item["indexed"] for item in coverage.values())
            ),
        }

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        kwargs: dict[str, Any] = {
            "name": self._collection_name,
            "metadata": {
                "hnsw:space": "cosine",
                "schema_version": 3 if self._use_v3 else 2,
            },
        }
        if self._use_v3:
            kwargs["embedding_function"] = _ChineseHashEmbedding()
        self._collection = self._client.get_or_create_collection(**kwargs)
