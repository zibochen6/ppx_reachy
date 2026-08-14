from __future__ import annotations

import json
import time

import pytest

from chaihuo_reachy.config import Config
from chaihuo_reachy.engine import ConversationEngine
from chaihuo_reachy.location import LocationService, Position
from chaihuo_reachy.wifi_scan import WifiAccessPoint


class _FakeAmap:
    available = True

    def __init__(self) -> None:
        self.ip_calls = 0
        self.closed = False

    async def locate_by_ip(self, *, refresh: bool = False):
        self.ip_calls += 1
        return {
            "place": "北京市",
            "province": "北京市",
            "city": "北京市",
            "adcode": "110000",
            "precision": "city",
            "source": "amap_ip",
        }

    async def reverse_geocode(self, lat: float, lon: float):
        return {"place": "北京市海淀区", "city": "北京市", "district": "海淀区"}

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_fresh_browser_position_wins_without_ip_lookup() -> None:
    amap = _FakeAmap()
    service = LocationService(gpsd_host="", amap_client=amap)  # type: ignore[arg-type]
    service.set_browser_position(40.0, 116.3, accuracy_m=25)

    result = await service.get_position()

    assert result.source == "browser"
    assert result.lat == 40.0
    assert result.accuracy_m == 25
    assert amap.ip_calls == 0


@pytest.mark.asyncio
async def test_amap_ip_fallback_has_no_fake_coordinates() -> None:
    amap = _FakeAmap()
    service = LocationService(gpsd_host="", amap_client=amap)  # type: ignore[arg-type]

    result = await service.get_position()

    assert result.source == "amap_ip"
    assert result.address == "北京市"
    assert result.precision == "city"
    assert result.lat is None and result.lon is None


def test_position_reports_coordinate_system_radius_age_and_staleness() -> None:
    position = Position(
        40.0,
        116.3,
        source="amap_wifi",
        timestamp=time.time() - 31,
        radius_m=80,
        coordinate_system="GCJ-02",
        stale_after_s=30,
    ).to_dict()
    assert position["coordinate_system"] == "GCJ-02"
    assert position["radius_m"] == 80
    assert position["age_s"] >= 31
    assert position["is_stale"] is True


@pytest.mark.asyncio
async def test_wifi_point_wins_before_ip_city_fallback() -> None:
    class Scanner:
        async def scan(self):
            return [
                WifiAccessPoint("00:11:22:33:44:55", -50, "hotspot", True),
                WifiAccessPoint("10:20:30:40:50:60", -60, "shop"),
                WifiAccessPoint("20:21:22:23:24:25", -70, "office"),
            ]

    class Amap(_FakeAmap):
        async def locate_by_wifi(self, _points):
            return {
                "latitude": 40.001,
                "longitude": 116.326,
                "radius_m": 85,
                "place": "北京市海淀区",
                "province": "北京市",
                "city": "北京市",
                "district": "海淀区",
                "adcode": "110108",
            }

    amap = Amap()
    service = LocationService(
        gpsd_host="", amap_client=amap, wifi_scanner=Scanner()  # type: ignore[arg-type]
    )
    result = await service.get_position()
    assert result.source == "amap_wifi"
    assert result.coordinate_system == "GCJ-02"
    assert result.radius_m == 85
    assert amap.ip_calls == 0


@pytest.mark.asyncio
async def test_live_location_wins_over_session_note_and_note_clears() -> None:
    class LiveLocation:
        async def get_position(self, *, refresh: bool = False):
            return Position(
                39.9, 116.4, source="gpsd", address="北京市", precision="point"
            )

    engine = ConversationEngine(Config(manual_location="内蒙古呼和浩特"))
    engine._location = LiveLocation()  # type: ignore[assignment]
    engine._set_session_location("北京清华大学")

    session = await engine._tool_get_current_location()
    assert session["place"] == "北京市"
    assert session["source"] == "gpsd"

    engine.clear_conversation()
    live = await engine._tool_get_current_location()
    assert live["place"] == "北京市"
    assert live["source"] == "gpsd"


@pytest.mark.asyncio
async def test_registered_location_tool_appends_structured_tool_result() -> None:
    engine = ConversationEngine(Config())
    engine._set_session_location("北京清华大学")
    messages = [
        {"role": "system", "content": "只根据工具回答"},
        {"role": "user", "content": "我们现在在哪？"},
    ]

    result = await engine._prepare_location_tool_messages(messages)

    assert result[-2]["tool_calls"][0]["function"]["name"] == "get_current_location"
    assert result[-1]["role"] == "tool"
    payload = json.loads(result[-1]["content"])
    assert payload["place"] == ""
    assert payload["source"] == "unavailable"
    assert payload["session_note"] == "北京清华大学"
