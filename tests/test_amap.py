from __future__ import annotations

import hashlib

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
