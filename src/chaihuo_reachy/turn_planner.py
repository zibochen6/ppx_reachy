"""Explicit per-turn capability planning.

Private/local facts are isolated from web content before an LLM request is
created.  Only general conversation can use the public web.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from chaihuo_reachy.intent import IntentDecision, TurnIntent


_EXPLICIT_WEB = re.compile(
    r"查一下|搜一下|搜索|联网|网上|核实|查查|看新闻|web\s*search",
    re.IGNORECASE,
)
_FRESH_INFORMATION = re.compile(
    r"最新|今天|今日|现在|目前|刚刚|实时|天气|气温|新闻|价格|多少钱|"
    r"汇率|股价|比赛|比分|赛程|政策|规定|日程|几点|开不开门|营业时间",
    re.IGNORECASE,
)
_PROTECTED = {
    TurnIntent.JOURNAL,
    TurnIntent.JOURNEY_RECALL,
    TurnIntent.ORG_KNOWLEDGE,
    TurnIntent.FRONT_CAMERA,
    TurnIntent.REAR_CAMERA,
    TurnIntent.AMBIGUOUS_CAMERA,
    TurnIntent.LOCATION,
    TurnIntent.LOCATION_UPDATE,
    TurnIntent.MOTION,
}

_VISUAL_DEPENDENCY = re.compile(
    r"(?:"
    r"这(?:个|些|件|张|块|台|是什么|上面)|那(?:个|些|件|张|块|台|上面)|"
    r"我(?:手里|手上|拿着|穿着|穿的|戴着|戴的|举着|指着|在做)|"
    r"(?:桌|台|地|墙|门|牌子|屏幕|衣服|杯子|盒子)上|"
    r"眼前|面前|旁边|附近|周围|"
    r"读(?:一下|出来)|念(?:一下|出来)|"
    r"什么颜色|几(?:个|只|件|辆|人)|多少(?:个|只|件|辆|人)|"
    r"是什么东西|什么东西|什么牌子|什么型号|我的造型|我的穿搭"
    r")",
    re.IGNORECASE,
)
_VISUAL_FALSE_FRIENDS = re.compile(
    r"(?:你怎么看|你的看法|你看呢|看起来|没看懂|看不懂|"
    r"这个(?:问题|方案|想法|观点|词|代码|新闻|故事|日记)|"
    r"昨天我看到|以前我看到|曾经看到|历史上)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TurnPlan:
    intent: TurnIntent
    search_mode: str  # off | auto | required
    vision_mode: str  # off | auto | required
    camera_scope: str  # front | rear
    allowed_sources: tuple[str, ...]
    protected: bool
    reason: str
    vision_shadow: bool = False


def is_semantic_visual_candidate(text: str) -> bool:
    """Broad local gate; the LLM makes the final tool decision inside it."""
    normalized = re.sub(r"\s+", "", text).strip()
    return bool(
        normalized
        and _VISUAL_DEPENDENCY.search(normalized)
        and not _VISUAL_FALSE_FRIENDS.search(normalized)
    )


def plan_turn(
    text: str,
    decision: IntentDecision,
    policy: str = "auto",
    vision_policy: str = "semantic",
) -> TurnPlan:
    policy = (policy or "auto").lower()
    vision_policy = (vision_policy or "semantic").lower()
    explicit_vision = decision.intent in {
        TurnIntent.FRONT_CAMERA,
        TurnIntent.REAR_CAMERA,
        TurnIntent.AMBIGUOUS_CAMERA,
    }
    camera_scope = "rear" if decision.intent == TurnIntent.REAR_CAMERA else "front"
    visual_candidate = decision.intent == TurnIntent.GENERAL and is_semantic_visual_candidate(text)
    if vision_policy == "off":
        vision_mode = "off"
        vision_shadow = False
    elif explicit_vision:
        vision_mode = "required"
        vision_shadow = False
    elif vision_policy in {"semantic", "semantic_shadow"} and visual_candidate:
        vision_mode = "auto"
        vision_shadow = vision_policy == "semantic_shadow"
    else:
        vision_mode = "off"
        vision_shadow = False

    if decision.intent in _PROTECTED:
        sources = ("live_vision",) if explicit_vision and vision_mode != "off" else ("local_verified",)
        return TurnPlan(
            intent=decision.intent,
            search_mode="off",
            vision_mode=vision_mode,
            camera_scope=camera_scope,
            allowed_sources=sources,
            protected=True,
            reason="protected deterministic source route",
            vision_shadow=vision_shadow,
        )
    if policy == "off":
        mode = "off"
    elif _EXPLICIT_WEB.search(text) or _FRESH_INFORMATION.search(text):
        mode = "required"
    elif policy == "auto":
        mode = "auto"
    else:  # explicit
        mode = "off"
    return TurnPlan(
        intent=decision.intent,
        search_mode=mode,
        vision_mode=vision_mode,
        camera_scope=camera_scope,
        allowed_sources=tuple(
            source
            for source in (
                "model_knowledge",
                "public_web" if mode != "off" else "",
                "live_vision" if vision_mode != "off" else "",
            )
            if source
        ),
        protected=False,
        reason=(
            "fresh/explicit web request" if mode == "required" else f"policy={policy}"
        ),
        vision_shadow=vision_shadow,
    )
