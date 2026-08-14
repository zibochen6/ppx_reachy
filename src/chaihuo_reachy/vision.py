"""Structured, ephemeral observations from Reachy's live cameras."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
import time
import uuid
from typing import Any


@dataclass(frozen=True)
class VisualObservation:
    """Grounded facts produced by the VLM, never user-facing prose."""

    observation_id: str
    scope: str
    captured_at: str
    facts: str
    uncertainties: str = ""
    quality: str = "ok"
    source: str = "live_camera"
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.facts.strip()) and not self.error

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}

    def to_prompt(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def success(
        cls,
        *,
        scope: str,
        facts: str,
        uncertainties: str = "",
        quality: str = "ok",
    ) -> "VisualObservation":
        return cls(
            observation_id=uuid.uuid4().hex,
            scope=scope,
            captured_at=datetime.now(timezone.utc).isoformat(),
            facts=facts.strip(),
            uncertainties=uncertainties.strip(),
            quality=quality,
        )

    @classmethod
    def failure(cls, *, scope: str, error: str, quality: str = "unavailable") -> "VisualObservation":
        return cls(
            observation_id=uuid.uuid4().hex,
            scope=scope,
            captured_at=datetime.now(timezone.utc).isoformat(),
            facts="",
            quality=quality,
            error=error.strip(),
        )


@dataclass
class VisualCache:
    observation: VisualObservation
    jpeg: bytes
    stored_monotonic: float
    focus: str

    def is_fresh(self, ttl_s: float, scope: str) -> bool:
        return (
            self.observation.scope == scope
            and ttl_s > 0
            and time.monotonic() - self.stored_monotonic <= ttl_s
        )


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_MECHANISM_RE = re.compile(
    r"(?:根据|从)?(?:这张|当前|刚才)?(?:照片|图片|画面|镜头|摄像头)"
    r"(?:里|中|内|来看)?(?:可以|能够|能)?(?:看到|看见|显示|识别到)?[，,:： ]*"
)


def _clean_facts(value: object) -> str:
    text = str(value or "").strip()
    text = _MECHANISM_RE.sub("", text)
    return re.sub(r"^(?:可以看到|可见|显示|识别到)[，,:： ]*", "", text).strip()


def parse_vlm_observation(raw: str, *, scope: str) -> VisualObservation:
    """Accept strict JSON when available and safely degrade to factual text."""
    text = _CODE_FENCE_RE.sub("", str(raw or "").strip()).strip()
    if not text:
        return VisualObservation.failure(scope=scope, error="我现在没看清楚，可以再让我看一次吗？")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        facts = payload.get("facts", "")
        if isinstance(facts, list):
            facts = "；".join(str(item).strip() for item in facts if str(item).strip())
        uncertainty = payload.get("uncertainties", "")
        if isinstance(uncertainty, list):
            uncertainty = "；".join(
                str(item).strip() for item in uncertainty if str(item).strip()
            )
        facts_text = _clean_facts(facts)
        if facts_text:
            return VisualObservation.success(
                scope=scope,
                facts=facts_text,
                uncertainties=str(uncertainty or "").strip(),
                quality=str(payload.get("quality") or "ok"),
            )
    cleaned = _clean_facts(text)
    if not cleaned:
        return VisualObservation.failure(scope=scope, error="我现在没看清楚，可以再让我看一次吗？")
    return VisualObservation.success(scope=scope, facts=cleaned)
