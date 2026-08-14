from chaihuo_reachy.intent import IntentDecision, TurnIntent
from chaihuo_reachy.turn_planner import plan_turn


def test_fresh_information_forces_search() -> None:
    plan = plan_turn(
        "今天深圳天气怎么样", IntentDecision(TurnIntent.GENERAL, "test"), "auto"
    )
    assert plan.search_mode == "required"


def test_general_knowledge_uses_auto_search() -> None:
    plan = plan_turn(
        "解释一下量子纠缠", IntentDecision(TurnIntent.GENERAL, "test"), "auto"
    )
    assert plan.search_mode == "auto"


def test_private_journal_route_never_uses_web() -> None:
    plan = plan_turn(
        "基地车昨天去了哪里", IntentDecision(TurnIntent.JOURNAL, "test"), "auto"
    )
    assert plan.search_mode == "off"
    assert plan.protected


def test_explicit_and_ambiguous_vision_require_front_observation() -> None:
    for intent in (TurnIntent.FRONT_CAMERA, TurnIntent.AMBIGUOUS_CAMERA):
        plan = plan_turn(
            "帮我看看", IntentDecision(intent, "test"), "auto", "semantic"
        )
        assert plan.vision_mode == "required"
        assert plan.camera_scope == "front"


def test_semantic_visual_candidate_uses_auto_tool() -> None:
    for text in (
        "这是什么？",
        "我手里拿的是什么？",
        "我穿的什么颜色？",
        "桌上有几个杯子？",
        "读一下这个牌子",
    ):
        plan = plan_turn(
            text, IntentDecision(TurnIntent.GENERAL, "test"), "auto", "semantic"
        )
        assert plan.vision_mode == "auto", text
        assert "live_vision" in plan.allowed_sources


def test_nonvisual_false_friends_never_open_camera() -> None:
    for text in (
        "你怎么看 AI？",
        "帮我看看这个方案",
        "我昨天看到一只猫",
        "红色为什么显眼",
        "查看系统状态",
    ):
        plan = plan_turn(
            text, IntentDecision(TurnIntent.GENERAL, "test"), "auto", "semantic"
        )
        assert plan.vision_mode == "off", text


def test_shadow_records_candidate_without_enabling_capture() -> None:
    plan = plan_turn(
        "这个按钮怎么用？",
        IntentDecision(TurnIntent.GENERAL, "test"),
        "auto",
        "semantic_shadow",
    )
    assert plan.vision_mode == "auto"
    assert plan.vision_shadow
