"""Small, server-side AMap Web Service client.

The Web Service key and its signing private key must never be exposed to the
Dashboard or to the language model.  This module therefore returns normalized
data and keeps request parameters out of log messages.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Mapping
from typing import Any

import httpx

_BASE_URL = "https://restapi.amap.com"
_ERROR_LABELS = {
    "10001": "高德 Key 错误或已过期",
    "10002": "高德 Key 没有当前服务权限",
    "10003": "高德接口已超过当天配额",
    "10004": "高德接口调用过于频繁",
    "10005": "高德接口 IP 白名单不匹配",
    "10007": "高德数字签名校验失败",
    "10010": "高德 IP 调用已超过限制",
}


class AmapAPIError(RuntimeError):
    """A safe-to-log AMap failure without credentials or signed URLs."""

    def __init__(self, infocode: str, message: str = "") -> None:
        self.infocode = str(infocode or "unknown")
        safe_message = _ERROR_LABELS.get(self.infocode) or message or "高德定位请求失败"
        super().__init__(f"{safe_message}（{self.infocode}）")


def build_signature(params: Mapping[str, object], private_key: str) -> str:
    """Return AMap's MD5 signature for sorted UTF-8 request parameters."""

    canonical = "&".join(
        f"{name}={params[name]}" for name in sorted(params) if name != "sig"
    )
    return hashlib.md5(f"{canonical}{private_key}".encode()).hexdigest()


class AmapWebClient:
    """AMap IP location, coordinate conversion and reverse geocoding."""

    def __init__(
        self,
        key: str,
        private_key: str = "",
        *,
        timeout_s: float = 3.0,
        cache_ttl_s: float = 600.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._key = key.strip()
        self._private_key = private_key.strip()
        self._timeout_s = max(0.2, float(timeout_s))
        self._cache_ttl_s = max(0.0, float(cache_ttl_s))
        self._client = client
        self._owns_client = client is None
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def available(self) -> bool:
        return bool(self._key)

    @property
    def signing_enabled(self) -> bool:
        return bool(self._private_key)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    async def locate_by_ip(self, *, refresh: bool = False) -> dict[str, Any]:
        """Locate the current server egress IPv4 address at city precision."""

        data = await self._cached_request(
            "ip:self",
            "/v3/ip",
            {"output": "JSON"},
            refresh=refresh,
        )
        province = _text(data.get("province"))
        city = _text(data.get("city"))
        if not province and not city:
            raise AmapAPIError("empty_location", "高德没有返回可用的省市信息")
        place = _join_place(province, city)
        return {
            "place": place,
            "province": province,
            "city": city,
            "adcode": _text(data.get("adcode")),
            "rectangle": _text(data.get("rectangle")),
            "precision": "city",
            "source": "amap_ip",
        }

    async def locate_by_wifi(
        self,
        access_points: list[dict[str, object]],
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Locate an IoT device from at least two fixed surrounding APs.

        Raw BSSIDs stay in the request body only and are never put into cache
        keys, return values or exception messages.
        """
        usable = [
            item for item in access_points if item.get("bssid") and item.get("ssid")
        ]
        if len(usable) < 3:
            raise AmapAPIError("insufficient_wifi", "周边固定 Wi-Fi 少于两个")
        connected = next((item for item in usable if item.get("connected")), None)
        if connected is None:
            raise AmapAPIError("missing_mmac", "未获取到当前连接的 Wi-Fi 信息")
        surrounding = [item for item in usable if not item.get("connected")][:30]
        if len(surrounding) < 2:
            raise AmapAPIError("insufficient_wifi", "周边固定 Wi-Fi 少于两个")
        params: dict[str, object] = {
            "accesstype": "2",
            "macs": "|".join(_format_ap(item) for item in surrounding),
            "output": "JSON",
            "show_fields": "formatted_address,addressComponent",
        }
        params["mmac"] = _format_ap(connected)
        # Wi-Fi fingerprints should not be retained in the generic cache.
        data = await self._request("/v5/position/IoT", params, method="POST")
        position = data.get("position") or data.get("result") or {}
        if not isinstance(position, dict):
            raise AmapAPIError("invalid_position", "高德 Wi-Fi 定位结果无效")
        location = _text(position.get("location"))
        try:
            lon, lat = (float(value) for value in location.split(",", 1))
        except (TypeError, ValueError):
            raise AmapAPIError("invalid_coordinates", "高德 Wi-Fi 坐标无效") from None
        component = position.get("addressComponent") or {}
        if not isinstance(component, dict):
            component = {}
        return {
            "latitude": lat,
            "longitude": lon,
            "radius_m": _float_or_none(position.get("radius")),
            "place": _text(position.get("formatted_address")),
            "province": _text(component.get("province")),
            "city": _text(component.get("city")),
            "district": _text(component.get("district")),
            "adcode": _text(component.get("adcode")),
            "coordinate_system": "GCJ-02",
            "precision": "point",
            "source": "amap_wifi",
        }

    async def convert_coordinates(
        self,
        lat: float,
        lon: float,
        *,
        refresh: bool = False,
    ) -> tuple[float, float]:
        """Convert WGS84 GPS coordinates to AMap GCJ-02 coordinates."""

        cache_key = f"convert:{lat:.5f},{lon:.5f}"
        data = await self._cached_request(
            cache_key,
            "/v3/assistant/coordinate/convert",
            {
                "locations": f"{lon:.6f},{lat:.6f}",
                "coordsys": "gps",
                "output": "JSON",
            },
            refresh=refresh,
        )
        location = _text(data.get("locations"))
        try:
            gcj_lon, gcj_lat = (float(value) for value in location.split(",", 1))
        except (TypeError, ValueError):
            raise AmapAPIError("invalid_coordinates", "高德坐标转换结果无效") from None
        return gcj_lat, gcj_lon

    async def reverse_geocode(
        self,
        lat: float,
        lon: float,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Resolve WGS84 coordinates to an AMap address and nearby POI."""

        gcj_lat, gcj_lon = await self.convert_coordinates(lat, lon, refresh=refresh)
        cache_key = f"regeo:{gcj_lat:.4f},{gcj_lon:.4f}"
        data = await self._cached_request(
            cache_key,
            "/v3/geocode/regeo",
            {
                "location": f"{gcj_lon:.6f},{gcj_lat:.6f}",
                "extensions": "all",
                "radius": "1000",
                "output": "JSON",
            },
            refresh=refresh,
        )
        regeocode = data.get("regeocode") or {}
        component = regeocode.get("addressComponent") or {}
        pois = regeocode.get("pois") or []
        nearest_poi = ""
        if isinstance(pois, list) and pois and isinstance(pois[0], dict):
            nearest_poi = _text(pois[0].get("name"))
        province = _text(component.get("province"))
        city = _text(component.get("city"))
        district = _text(component.get("district"))
        formatted = _text(regeocode.get("formatted_address"))
        place = formatted or _join_place(province, city, district, nearest_poi)
        if not place:
            raise AmapAPIError("empty_address", "高德没有返回可用地址")
        return {
            "place": place,
            "province": province,
            "city": city,
            "district": district,
            "adcode": _text(component.get("adcode")),
            "nearest_poi": nearest_poi,
            "gcj02": {"latitude": gcj_lat, "longitude": gcj_lon},
            "precision": "point",
            "source": "amap_regeo",
        }

    async def _cached_request(
        self,
        cache_key: str,
        path: str,
        params: dict[str, object],
        *,
        refresh: bool,
    ) -> dict[str, Any]:
        if not self._key:
            raise AmapAPIError("not_configured", "高德 Web 服务 Key 尚未配置")
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if not refresh and cached and now - cached[0] < self._cache_ttl_s:
            return dict(cached[1])
        lock = self._locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            cached = self._cache.get(cache_key)
            if not refresh and cached and now - cached[0] < self._cache_ttl_s:
                return dict(cached[1])
            data = await self._request(path, params)
            self._cache[cache_key] = (time.monotonic(), data)
            return dict(data)

    async def _request(
        self,
        path: str,
        params: Mapping[str, object],
        *,
        method: str = "GET",
    ) -> dict[str, Any]:
        request_params: dict[str, object] = {**params, "key": self._key}
        if self._private_key:
            request_params["sig"] = build_signature(request_params, self._private_key)

        client = self._client
        if client is None:
            client = httpx.AsyncClient(
                base_url=_BASE_URL,
                timeout=httpx.Timeout(self._timeout_s),
            )
            self._client = client

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                if method == "POST":
                    response = await client.post(path, data=request_params)
                else:
                    response = await client.get(path, params=request_params)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise AmapAPIError("invalid_response", "高德返回格式无效")
                infocode = _text(data.get("infocode"))
                if str(data.get("status")) != "1" or infocode != "10000":
                    raise AmapAPIError(infocode, _text(data.get("info")))
                return data
            except AmapAPIError:
                raise
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
            ) as exc:
                last_error = exc
                if attempt == 0:
                    await asyncio.sleep(0.05)
                    continue
        raise AmapAPIError("network_error", "高德定位网络请求失败") from last_error


def _text(value: object) -> str:
    if value is None or value == []:
        return ""
    return str(value).strip()


def _join_place(*parts: str) -> str:
    result = ""
    for part in parts:
        if part and part not in result:
            result += part
    return result


def _format_ap(item: Mapping[str, object]) -> str:
    bssid = _text(item.get("bssid")).lower()
    strength = int(float(item.get("signal_dbm") or -80))
    ssid = _text(item.get("ssid")).replace("|", "").replace(",", "")
    return f"{bssid},{strength},{ssid},0"


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
