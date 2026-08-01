from __future__ import annotations

import pytest

from chaihuo_reachy.intent import TurnIntent, classify_intent


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("你看到什么", TurnIntent.FRONT_CAMERA),
        ("你能看到什么？", TurnIntent.FRONT_CAMERA),
        ("你现在可以看见啥", TurnIntent.FRONT_CAMERA),
        ("看看你面前", TurnIntent.FRONT_CAMERA),
        ("拍张照", TurnIntent.FRONT_CAMERA),
        ("看看车外面的场景", TurnIntent.REAR_CAMERA),
        ("看看车外面", TurnIntent.REAR_CAMERA),
        ("外面有什么？", TurnIntent.REAR_CAMERA),
        ("看看车后方", TurnIntent.REAR_CAMERA),
        ("你怎么看这个问题", TurnIntent.GENERAL),
        ("你怎么看外面的世界", TurnIntent.GENERAL),
        ("查看昨天的日记", TurnIntent.JOURNAL),
        ("我没看懂", TurnIntent.GENERAL),
        ("帮我看看这个方案", TurnIntent.GENERAL),
        ("帮我看看", TurnIntent.AMBIGUOUS_CAMERA),
        ("基地车去过哪里", TurnIntent.JOURNAL),
        ("我们现在在哪", TurnIntent.LOCATION),
    ],
)
def test_critical_intent_routes(text: str, expected: TurnIntent) -> None:
    assert classify_intent(text).intent == expected
