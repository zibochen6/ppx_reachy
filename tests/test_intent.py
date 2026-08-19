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
        ("基地车去过哪里", TurnIntent.JOURNEY_RECALL),
        ("我们都到过什么地方", TurnIntent.JOURNEY_RECALL),
        ("我们都去过什么地方", TurnIntent.JOURNEY_RECALL),
        ("我们去过哪些地方", TurnIntent.JOURNEY_RECALL),
        ("基地车都走过哪里", TurnIntent.JOURNEY_RECALL),
        ("我们现在在哪", TurnIntent.LOCATION),
        ("我们现在在什么地方", TurnIntent.LOCATION),
        ("我们到哪了", TurnIntent.LOCATION),
        ("我们现在在北京清华大学", TurnIntent.LOCATION_UPDATE),
        ("告诉你，我们现在在北京清华大学。", TurnIntent.LOCATION_UPDATE),
        ("我们在内蒙古都做了什么", TurnIntent.JOURNEY_RECALL),
        ("你还记得我们在内蒙古做了什么吗", TurnIntent.JOURNEY_RECALL),
        ("我们在清华大学有什么故事", TurnIntent.JOURNEY_RECALL),
        ("介绍一下柴火创客", TurnIntent.ORG_KNOWLEDGE),
        ("柴火和基地车是什么关系", TurnIntent.ORG_KNOWLEDGE),
        ("跳个舞", TurnIntent.MOTION),
        ("跳一段舞", TurnIntent.MOTION),
        ("按照你的性格随便跳一段舞", TurnIntent.MOTION),
        ("跳一个不一样的舞蹈", TurnIntent.MOTION),
        ("跳支舞吧", TurnIntent.MOTION),
        ("来一段摇摆", TurnIntent.MOTION),
        ("来支舞", TurnIntent.MOTION),
        ("机械舞", TurnIntent.MOTION),
        ("点点头", TurnIntent.MOTION),
        ("挥挥天线", TurnIntent.MOTION),
        ("去睡觉", TurnIntent.MOTION),
        ("你刚才跳的是骑马舞吗", TurnIntent.GENERAL),
        ("他，你是跳的骑马舞吗？", TurnIntent.GENERAL),
    ],
)
def test_critical_intent_routes(text: str, expected: TurnIntent) -> None:
    assert classify_intent(text).intent == expected
