"""Deterministic intent routing for fact and camera-sensitive turns.

Critical tool selection must not depend on the LLM guessing correctly.  This
module therefore handles the small set of intents where choosing the wrong
source would produce an ungrounded answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re


class TurnIntent(StrEnum):
    GENERAL = "general"
    JOURNAL = "journal"
    FRONT_CAMERA = "front_camera"
    REAR_CAMERA = "rear_camera"
    AMBIGUOUS_CAMERA = "ambiguous_camera"
    LOCATION = "location"
    MOTION = "motion"


@dataclass(frozen=True)
class IntentDecision:
    intent: TurnIntent
    reason: str


_REAR_TERMS = (
    "车外", "外面", "车后", "后方", "后面", "后视", "车尾",
)
_FRONT_PHRASES = (
    "你看到什么", "你能看到什么", "你可以看到什么",
    "你看到了什么", "看到了什么", "你看见什么",
    "你能看见什么", "你可以看见什么",
    "你现在看到什么", "看看现在", "看一下现在", "看下现在",
    "拍一下现在", "现在看到什么", "现在是什么",
    "你面前", "前面是什么", "前面有什么", "拍张照", "拍个照",
    "拍照看看", "看看前面", "看下前面", "周围有什么",
)
_FRONT_VISION_QUERY_RE = re.compile(
    r"(?:你)?(?:现在)?(?:能不能|能|可以)?(?:看到|看见)(?:些什么|什么|啥)"
)
_VISION_NEGATIVES = (
    "怎么看", "你的看法", "你看呢", "看起来", "查看", "没看懂",
    "看不懂", "看法", "日记",
)
_AMBIGUOUS_CAMERA = {
    "帮我看看", "你看看", "看一下", "看看", "帮忙看看",
}
_LOCATION_TERMS = (
    "我们在哪", "现在在哪", "当前位置", "到哪了", "什么地方",
    "基地车在哪", "走到哪里", "定位",
)
_MOTION_TERMS = (
    "跳个舞", "跳舞", "点点头", "摇摇头", "挥挥天线", "打个招呼",
    "去睡觉", "休息吧", "站起来", "醒醒",
)
_JOURNAL_TERMS = (
    "日记", "旅途", "行程", "去过", "到过", "走过", "发生了什么",
    "基地车", "哪一站", "城市", "高校", "乡村", "队友", "团队成员",
    "领队", "昨天", "前天", "大前天", "上周", "本周", "最近几天",
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
        return IntentDecision(TurnIntent.REAR_CAMERA, "explicit exterior/rear-view phrase")

    if normalized in _AMBIGUOUS_CAMERA:
        return IntentDecision(TurnIntent.AMBIGUOUS_CAMERA, "visual target is ambiguous")

    if not semantic_visual:
        if (
            any(phrase in normalized for phrase in _FRONT_PHRASES)
            or _FRONT_VISION_QUERY_RE.search(normalized)
        ):
            return IntentDecision(TurnIntent.FRONT_CAMERA, "explicit robot/front-view phrase")

    if any(term in normalized for term in _LOCATION_TERMS):
        return IntentDecision(TurnIntent.LOCATION, "live location requested")

    if any(term in normalized for term in _MOTION_TERMS):
        return IntentDecision(TurnIntent.MOTION, "explicit robot motion requested")

    if any(term in normalized for term in _JOURNAL_TERMS) or re.search(
        r"(?:20\d{2}[-.年])?\d{1,2}[.月]\d{1,2}(?:日)?", normalized
    ):
        return IntentDecision(TurnIntent.JOURNAL, "vehicle/journey fact requires journal evidence")

    return IntentDecision(TurnIntent.GENERAL, "no protected factual source required")
