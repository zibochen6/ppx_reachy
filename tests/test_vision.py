from __future__ import annotations

import time

from chaihuo_reachy.vision import VisualCache, VisualObservation, parse_vlm_observation


def test_vlm_json_becomes_structured_grounded_observation() -> None:
    observation = parse_vlm_observation(
        '```json\n{"facts":["照片中可以看到一个红色杯子","杯子在桌上"],'
        '"uncertainties":["品牌看不清"],"quality":"ok"}\n```',
        scope="front",
    )
    assert observation.ok
    assert observation.facts == "一个红色杯子；杯子在桌上"
    assert observation.uncertainties == "品牌看不清"
    assert "照片" not in observation.facts


def test_visual_cache_respects_scope_and_ttl() -> None:
    observation = VisualObservation.success(scope="front", facts="一个杯子")
    cache = VisualCache(
        observation=observation,
        jpeg=b"jpeg",
        stored_monotonic=time.monotonic(),
        focus="这是什么",
    )
    assert cache.is_fresh(15, "front")
    assert not cache.is_fresh(15, "rear")
    cache.stored_monotonic -= 16
    assert not cache.is_fresh(15, "front")
