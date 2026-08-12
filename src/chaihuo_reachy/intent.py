"""Deterministic intent routing for fact and camera-sensitive turns.

Critical tool selection must not depend on the LLM guessing correctly.  This
module therefore handles the small set of intents where choosing the wrong
source would produce an ungrounded answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class TurnIntent(StrEnum):
    GENERAL = "general"
    JOURNAL = "journal"
    JOURNEY_RECALL = "journey_recall"
    ORG_KNOWLEDGE = "org_knowledge"
    FRONT_CAMERA = "front_camera"
    REAR_CAMERA = "rear_camera"
    AMBIGUOUS_CAMERA = "ambiguous_camera"
    LOCATION = "location"
    LOCATION_UPDATE = "location_update"
    MOTION = "motion"


@dataclass(frozen=True)
class IntentDecision:
    intent: TurnIntent
    reason: str


_REAR_TERMS = (
    "车外",
    "外面",
    "车后",
    "后方",
    "后面",
    "后视",
    "车尾",
)
_FRONT_PHRASES = (
    "你看到什么",
    "你能看到什么",
    "你可以看到什么",
    "你看到了什么",
    "看到了什么",
    "你看见什么",
    "你能看见什么",
    "你可以看见什么",
    "你现在看到什么",
    "看看现在",
    "看一下现在",
    "看下现在",
    "拍一下现在",
    "现在看到什么",
    "现在是什么",
    "你面前",
    "前面是什么",
    "前面有什么",
    "拍张照",
    "拍个照",
    "拍照看看",
    "看看前面",
    "看下前面",
    "周围有什么",
    "拍照",
    "拍一张",
    "拍个照片",
    "拍一下",
    "拍照片",
    "帮我拍",
)
_FRONT_VISION_QUERY_RE = re.compile(
    r"(?:你)?(?:现在)?(?:能不能|能|可以)?(?:看到|看见)(?:些什么|什么|啥)"
)
_VISION_NEGATIVES = (
    "怎么看",
    "你的看法",
    "你看呢",
    "看起来",
    "查看",
    "没看懂",
    "看不懂",
    "看法",
    "日记",
)
_AMBIGUOUS_CAMERA = {
    "帮我看看",
    "你看看",
    "看一下",
    "看看",
    "帮忙看看",
}
_LIVE_LOCATION_RE = re.compile(
    r"(?:当前位置|实时位置|定位(?:一下)?|"
    r"(?:我们|咱们|基地车)?(?:现在|目前|当前|这会儿|此刻)"
    r"(?:在|到|走到|来到)?(?:哪(?:里|儿)?|什么地方)|"
    r"(?:我们|咱们|基地车)(?:在|到|走到)?哪(?:里|儿)?|"
    r"(?:我们|咱们|基地车)?(?:到|走到)哪(?:里|儿)?(?:了|啦))"
)
_ORG_TERMS = (
    "柴火创客",
    "柴火空间",
    "柴火创客空间",
    "介绍柴火",
    "柴火是谁",
    "柴火和基地车",
    "基地车和柴火",
    "皮皮虾代表谁",
    "我们代表谁",
    "介绍基地车",
    "基地车是什么",
    "什么是基地车",
    "讲讲基地车",
)
_MOTION_TERMS = (
    "跳个舞",
    "跳舞",
    "点点头",
    "摇摇头",
    "挥挥天线",
    "打个招呼",
    "去睡觉",
    "休息吧",
    "站起来",
    "醒醒",
    "摇摆",
    "蹦迪",
    "机械舞",
    "来一段舞",
    "来段舞",
    "来支舞",
)
# "跳一段舞" / "跳支舞" / "跳一下舞" — 跳与舞之间隔了修饰字,
# 连续子串匹配不到,用宽松模式兜底。
_MOTION_DANCE_RE = re.compile(r"跳.{0,4}舞")
_JOURNAL_TERMS = (
    # 明确基地车/旅程语境的词；通用词（城市/高校/乡村等）不在此列，
    # 否则科普常识问题（如"城市为什么堵车"）会被误判为日记检索。
    "日记",
    "旅途",
    "行程",
    "去过",
    "到过",
    "走过",
    "发生了什么",
    "基地车",
    "哪一站",
    "队友",
    "团队成员",
    "领队",
    "昨天",
    "前天",
    "大前天",
    "上周",
    "本周",
    "最近几天",
)
_JOURNEY_RECALL_RE = re.compile(
    r"(?:还记得|记不记得|回忆|我们|咱们).{0,18}"
    r"(?:做了什么|干了什么|发生(?:过|了)?什么|有什么故事|经历了什么|"
    r"都去了|都去过|走过|到过|哪一站)"
)
_JOURNEY_OVERVIEW_HISTORY_TERMS = (
    "去过",
    "到过",
    "走过",
    "经过",
    "途经",
    "途径",
    "足迹",
    "路线",
    "行程",
    "旅程",
)
_JOURNEY_OVERVIEW_SCOPE_TERMS = (
    "都",
    "哪些",
    "什么地方",
    "哪里",
    "哪儿",
    "一路",
    "完整",
    "所有",
)
_LOCATION_UPDATE_PATTERNS = (
    re.compile(
        r"(?:^|告诉你[，,:：]?|跟你说[，,:：]?)"
        r"(?:我们|咱们)(?:现在|目前|这会儿)?(?:已经)?"
        r"(?:在|到(?:了)?|来到(?:了)?)"
        r"(?P<place>[^，。！？!?]{2,32})(?:了)?[。！!]?$"
    ),
    re.compile(
        r"^(?:当前位置是|把当前位置(?:设为|改成)|位置更新为)"
        r"(?P<place>[^，。！？!?]{2,32})[。！!]?$"
    ),
)
_LOCATION_UPDATE_BLOCKERS = (
    "哪里",
    "哪儿",
    "什么地方",
    "做了什么",
    "发生",
    "故事",
    "经历",
    "为什么",
)


def extract_location_update(text: str) -> str:
    """Extract an explicit present-location declaration, not a place mention."""

    normalized = re.sub(r"\s+", "", text).strip()
    for pattern in _LOCATION_UPDATE_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        place = match.group("place").strip("，。！？!?、 了")
        if len(place) >= 2 and not any(
            blocker in place for blocker in _LOCATION_UPDATE_BLOCKERS
        ):
            return place
    return ""


def is_journey_overview_query(text: str) -> bool:
    """Whether the user asks for historical route coverage, not live location."""

    normalized = re.sub(r"\s+", "", text).lower().strip("，。！？!?、 ")
    has_history = any(term in normalized for term in _JOURNEY_OVERVIEW_HISTORY_TERMS)
    has_scope = any(term in normalized for term in _JOURNEY_OVERVIEW_SCOPE_TERMS)
    journey_subject = any(term in normalized for term in ("我们", "咱们", "基地车"))
    return (
        has_history
        and has_scope
        and (journey_subject or normalized.startswith(_JOURNEY_OVERVIEW_HISTORY_TERMS))
    )


def classify_intent(text: str) -> IntentDecision:
    """Classify a user turn with deterministic precedence.

    Rear/exterior language wins over generic visual words.  Ambiguous visual
    requests deliberately ask for clarification instead of taking a photo.
    """

    normalized = re.sub(r"\s+", "", text).lower().strip("，。！？!?、 ")
    semantic_visual = any(term in normalized for term in _VISION_NEGATIVES)
    if (
        not semantic_visual
        and any(term in normalized for term in _REAR_TERMS)
        and any(term in normalized for term in ("看", "什么", "场景", "情况", "拍"))
    ):
        return IntentDecision(
            TurnIntent.REAR_CAMERA, "explicit exterior/rear-view phrase"
        )

    if normalized in _AMBIGUOUS_CAMERA:
        return IntentDecision(TurnIntent.AMBIGUOUS_CAMERA, "visual target is ambiguous")

    if not semantic_visual:
        if any(
            phrase in normalized for phrase in _FRONT_PHRASES
        ) or _FRONT_VISION_QUERY_RE.search(normalized):
            return IntentDecision(
                TurnIntent.FRONT_CAMERA, "explicit robot/front-view phrase"
            )

    if extract_location_update(normalized):
        return IntentDecision(
            TurnIntent.LOCATION_UPDATE, "explicit current-location declaration"
        )

    # Historical coverage must win over the generic phrase "什么地方". Without
    # this guard, "我们都去过什么地方" incorrectly calls live geolocation.
    if is_journey_overview_query(normalized):
        return IntentDecision(
            TurnIntent.JOURNEY_RECALL,
            "full journey overview requires all verified diary titles",
        )

    if _LIVE_LOCATION_RE.search(normalized):
        return IntentDecision(TurnIntent.LOCATION, "live location requested")

    if any(term in normalized for term in _MOTION_TERMS) or _MOTION_DANCE_RE.search(
        normalized
    ):
        return IntentDecision(TurnIntent.MOTION, "explicit robot motion requested")

    if _JOURNEY_RECALL_RE.search(normalized):
        return IntentDecision(
            TurnIntent.JOURNEY_RECALL,
            "team journey recall requires verified diary evidence",
        )

    if any(term in normalized for term in _ORG_TERMS):
        return IntentDecision(
            TurnIntent.ORG_KNOWLEDGE, "Chaihuo organization knowledge requested"
        )

    if any(term in normalized for term in _JOURNAL_TERMS) or re.search(
        r"(?:20\d{2}[-.年])?\d{1,2}[.月]\d{1,2}(?:日)?", normalized
    ):
        return IntentDecision(
            TurnIntent.JOURNAL, "vehicle/journey fact requires journal evidence"
        )

    return IntentDecision(TurnIntent.GENERAL, "no protected factual source required")
