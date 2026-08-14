from __future__ import annotations

import hashlib
from urllib.parse import parse_qs

import httpx
import pytest

from chaihuo_reachy.amap import AmapAPIError, AmapWebClient, build_signature


def test_amap_signature_sorts_all_params_and_appends_private_key() -> None:
    params = {"output": "JSON", "key": "public-key", "ip": "1.2.3.4"}
    canonical = "ip=1.2.3.4&key=public-key&output=JSONprivate-secret"
    assert (
        build_signature(params, "private-secret")
        == hashlib.md5(canonical.encode("utf-8")).hexdigest()
    )


@pytest.mark.asyncio
async def test_ip_location_is_city_precision_and_cached() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "status": "1",
                "info": "OK",
                "infocode": "10000",
                "province": "北京市",
                "city": "北京市",
                "adcode": "110000",
                "rectangle": "116.0,39.0;117.0,40.0",
            },
        )

    http = httpx.AsyncClient(
        base_url="https://restapi.amap.com",
        transport=httpx.MockTransport(handler),
    )
    client = AmapWebClient("public-key", "private-secret", client=http, cache_ttl_s=600)
    first = await client.locate_by_ip()
    second = await client.locate_by_ip()

    assert first == second
    assert first["place"] == "北京市"
    assert first["precision"] == "city"
    assert first["source"] == "amap_ip"
    assert len(requests) == 1
    assert requests[0].url.params.get("sig")
    assert requests[0].url.params.get("ip") is None


@pytest.mark.asyncio
async def test_reverse_geocode_converts_wgs84_before_address_lookup() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/coordinate/convert"):
            return httpx.Response(
                200,
                json={
                    "status": "1",
                    "info": "OK",
                    "infocode": "10000",
                    "locations": "116.326000,40.001000",
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "1",
                "info": "OK",
                "infocode": "10000",
                "regeocode": {
                    "formatted_address": "北京市海淀区清华大学附近",
                    "addressComponent": {
                        "province": "北京市",
                        "city": "北京市",
                        "district": "海淀区",
                        "adcode": "110108",
                    },
                    "pois": [{"name": "清华大学"}],
                },
            },
        )

    http = httpx.AsyncClient(
        base_url="https://restapi.amap.com",
        transport=httpx.MockTransport(handler),
    )
    client = AmapWebClient("public-key", "private-secret", client=http)
    result = await client.reverse_geocode(40.0, 116.32)

    assert paths == [
        "/v3/assistant/coordinate/convert",
        "/v3/geocode/regeo",
    ]
    assert result["place"] == "北京市海淀区清华大学附近"
    assert result["nearest_poi"] == "清华大学"
    assert result["gcj02"] == {"latitude": 40.001, "longitude": 116.326}


@pytest.mark.asyncio
async def test_amap_error_is_safe_and_classified() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "0",
                "info": "INVALID_USER_SIGNATURE",
                "infocode": "10007",
            },
        )

    http = httpx.AsyncClient(
        base_url="https://restapi.amap.com",
        transport=httpx.MockTransport(handler),
    )
    client = AmapWebClient("public-key", "private-secret", client=http)
    with pytest.raises(AmapAPIError) as caught:
        await client.locate_by_ip()
    assert caught.value.infocode == "10007"
    assert "数字签名" in str(caught.value)
    assert "public-key" not in str(caught.value)
    assert "private-secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_iot_wifi_location_posts_fingerprint_and_returns_gcj02() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "status": "1",
                "info": "OK",
                "infocode": "10000",
                "position": {
                    "location": "116.326,40.001",
                    "radius": "85",
                    "formatted_address": "北京市海淀区",
                    "addressComponent": {
                        "province": "北京市",
                        "city": "北京市",
                        "district": "海淀区",
                        "adcode": "110108",
                    },
                },
            },
        )

    http = httpx.AsyncClient(
        base_url="https://restapi.amap.com",
        transport=httpx.MockTransport(handler),
    )
    client = AmapWebClient("public-key", client=http)
    result = await client.locate_by_wifi(
        [
            {"bssid": "00:11:22:33:44:55", "signal_dbm": -50, "ssid": "hotspot", "connected": True},
            {"bssid": "10:20:30:40:50:60", "signal_dbm": -60, "ssid": "ap1"},
            {"bssid": "20:21:22:23:24:25", "signal_dbm": -70, "ssid": "ap2"},
        ]
    )

    assert requests[0].method == "POST"
    assert requests[0].url.path == "/v5/position/IoT"
    form = parse_qs(requests[0].content.decode())
    assert "00:11:22:33:44:55" in form["mmac"][0]
    assert "00:11:22:33:44:55" not in form["macs"][0]
    assert form["macs"][0].count("|") == 1
    assert result["coordinate_system"] == "GCJ-02"
    assert result["radius_m"] == 85.0
    assert "00:11:22:33:44:55" not in str(result)
