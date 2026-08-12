"""GPS / location service for the Chaihuo Mobile Creative Vehicle.

Positioning backends, in priority order:
  1. GPSD       — real GPS hardware via gpsd daemon (Jetson / Linux)
  2. Browser    — Dashboard navigator.geolocation (macOS CoreLocation Wi-Fi)
  3. Manual     — user-configured fixed coordinates

Usage::

    loc = LocationService()
    await loc.start()
    pos = await loc.get_position()
    # pos = {"lat": 22.5431, "lon": 113.9544, "source": "gpsd", ...}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from chaihuo_reachy.amap import AmapAPIError, AmapWebClient

logger = logging.getLogger("chaihuo_reachy.location")

# ── Data ────────────────────────────────────────────────────────────────────


@dataclass
class Position:
    lat: float | None
    lon: float | None
    source: str = "unknown"  # "gpsd" | "browser" | "manual" | "unavailable"
    timestamp: float = field(default_factory=time.time)
    altitude_m: float | None = None
    speed_kmh: float | None = None
    heading_deg: float | None = None
    accuracy_m: float | None = None  # browser provides this
    address: str | None = None
    province: str = ""
    city: str = ""
    district: str = ""
    adcode: str = ""
    precision: str = "point"  # "point" | "district" | "city" | "unknown"
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "lat": round(self.lat, 6) if self.lat is not None else None,
            "lon": round(self.lon, 6) if self.lon is not None else None,
            "source": self.source,
            "timestamp": round(self.timestamp, 1),
            "altitude_m": round(self.altitude_m, 1)
            if self.altitude_m is not None
            else None,
            "speed_kmh": round(self.speed_kmh, 1)
            if self.speed_kmh is not None
            else None,
            "heading_deg": round(self.heading_deg, 1)
            if self.heading_deg is not None
            else None,
            "accuracy_m": round(self.accuracy_m, 1)
            if self.accuracy_m is not None
            else None,
            "address": self.address,
            "province": self.province,
            "city": self.city,
            "district": self.district,
            "adcode": self.adcode,
            "precision": self.precision,
            "error": self.error,
        }

    def to_human(self) -> str:
        """Human-readable one-line description."""
        parts = []
        if self.lat is not None and self.lon is not None:
            parts.append(f"纬度 {self.lat:.6f}°, 经度 {self.lon:.6f}°")
        if self.address:
            parts.append(self.address)
        if self.altitude_m is not None:
            parts.append(f"海拔 {self.altitude_m:.0f}m")
        if self.speed_kmh is not None and self.speed_kmh > 0.5:
            parts.append(f"速度 {self.speed_kmh:.1f}km/h")
        if self.accuracy_m is not None:
            parts.append(f"精度 ±{self.accuracy_m:.0f}m")
        source_labels = {
            "gpsd": "🛰 GPS卫星",
            "browser": "📱 设备定位",
            "amap_ip": "🌐 高德IP城市定位",
            "manual": "📍 手动设置",
            "unavailable": "❌ 无信号",
        }
        parts.append(source_labels.get(self.source, self.source))
        return " ".join(parts)

    def to_llm_context(self) -> str:
        """Formatted context string for LLM consumption."""
        lines = [
            f"数据来源：{self.source}",
            f"定位精度级别：{self.precision}",
            f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
        ]
        if self.lat is not None and self.lon is not None:
            lines.insert(0, f"当前位置：纬度 {self.lat:.6f}，经度 {self.lon:.6f}")
        elif self.address:
            lines.insert(0, f"当前位置：{self.address}")
        if self.altitude_m is not None:
            lines.append(f"海拔高度：{self.altitude_m:.1f} 米")
        if self.speed_kmh is not None:
            lines.append(f"移动速度：{self.speed_kmh:.1f} km/h")
        if self.heading_deg is not None:
            lines.append(f"行进方向：{self.heading_deg:.0f}°")
        if self.accuracy_m is not None:
            lines.append(f"定位精度：±{self.accuracy_m:.0f} 米")
        if self.address:
            lines.append(f"参考地址：{self.address}")
        if self.precision == "city":
            lines.append(
                "这是城市级IP定位，只能回答省市，不得推断具体街道、校园或建筑。"
            )
        else:
            lines.append("请基于以上实时坐标或地址回答，不要扩写未经定位支持的地点。")
        return "\n".join(lines)


# ── Location Service ─────────────────────────────────────────────────────────


class LocationService:
    """Multi-source location provider.

    Priority: GPSD (satellite) > Browser (Wi-Fi/CoreLocation) > Manual.
    """

    def __init__(
        self,
        *,
        gpsd_host: str = "127.0.0.1",
        gpsd_port: int = 2947,
        poll_interval_s: float = 2.0,
        stale_timeout_s: float = 30.0,
        browser_stale_timeout_s: float = 60.0,
        amap_client: AmapWebClient | None = None,
    ) -> None:
        self._gpsd_host = gpsd_host
        self._gpsd_port = gpsd_port
        self._poll_interval_s = poll_interval_s
        self._stale_timeout_s = stale_timeout_s
        self._browser_stale_timeout_s = browser_stale_timeout_s
        self._amap = amap_client

        self._latest: Position | None = None
        self._browser_pos: Position | None = None  # latest from Dashboard JS
        self._manual_lat: float | None = None
        self._manual_lon: float | None = None
        self._running = False
        self._task: asyncio.Task | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start periodic GPSD polling (background task)."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "📍 定位服务已启动 (GPSD=%s:%d, Browser=ready)",
            self._gpsd_host,
            self._gpsd_port,
        )

    async def stop(self) -> None:
        """Stop polling."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._amap is not None:
            await self._amap.aclose()

    # ── Public API ──────────────────────────────────────────────────────

    def set_browser_position(
        self,
        lat: float,
        lon: float,
        accuracy_m: float | None = None,
        altitude_m: float | None = None,
        heading_deg: float | None = None,
        speed_kmh: float | None = None,
    ) -> Position:
        """Called by Dashboard JS via WebSocket when browser reports location."""
        pos = Position(
            lat=lat,
            lon=lon,
            source="browser",
            accuracy_m=accuracy_m,
            altitude_m=altitude_m,
            heading_deg=heading_deg,
            speed_kmh=speed_kmh,
        )
        self._browser_pos = pos
        # Browser position is authoritative until GPSD gives a fix
        if (
            self._latest is None
            or self._latest.source != "gpsd"
            or time.time() - self._latest.timestamp >= self._stale_timeout_s
        ):
            self._latest = pos
        logger.info("📍 浏览器定位: %.6f, %.6f (±%.0fm)", lat, lon, accuracy_m or 0)
        # Reverse geocode in background
        asyncio.ensure_future(self._enrich_with_address(pos))
        return pos

    async def get_position(self, *, refresh: bool = False) -> Position:
        """Get the best current position without allowing stale data to win."""
        now = time.time()
        if (
            not refresh
            and self._latest is not None
            and self._latest.source == "gpsd"
            and now - self._latest.timestamp < self._stale_timeout_s
        ):
            return self._latest
        if (
            not refresh
            and self._browser_pos is not None
            and now - self._browser_pos.timestamp < self._browser_stale_timeout_s
        ):
            return self._browser_pos
        # Try GPSD once. A real satellite fix remains the highest live source.
        pos = await self._poll_gpsd()
        if pos is not None:
            self._latest = pos
            asyncio.ensure_future(self._enrich_with_address(pos))
            return pos
        # Browser coordinates are useful only while they are fresh.
        if (
            self._browser_pos is not None
            and now - self._browser_pos.timestamp < self._browser_stale_timeout_s
        ):
            self._latest = self._browser_pos
            return self._browser_pos
        # IP location is city-level, but better than a stale or fabricated fix.
        if self._amap is not None and self._amap.available:
            try:
                item = await self._amap.locate_by_ip(refresh=refresh)
                pos = Position(
                    lat=None,
                    lon=None,
                    source="amap_ip",
                    address=str(item.get("place") or ""),
                    province=str(item.get("province") or ""),
                    city=str(item.get("city") or ""),
                    adcode=str(item.get("adcode") or ""),
                    precision="city",
                )
                self._latest = pos
                return pos
            except AmapAPIError as exc:
                logger.warning("高德IP定位不可用: %s", exc)
                error = str(exc)
            except Exception:
                logger.warning("高德IP定位异常", exc_info=True)
                error = "高德定位异常"
        else:
            error = ""
        return Position(
            lat=None,
            lon=None,
            source="unavailable",
            address="等待 GPS、浏览器或网络定位...",
            precision="unknown",
            error=error,
        )

    def set_manual(self, lat: float, lon: float) -> Position:
        """Manually override position."""
        self._manual_lat = lat
        self._manual_lon = lon
        self._latest = Position(
            lat=lat,
            lon=lon,
            source="manual",
            address=f"手动设置 ({lat:.4f}, {lon:.4f})",
            precision="point",
        )
        logger.info("📍 手动位置: %.6f, %.6f", lat, lon)
        return self._latest

    @property
    def latest_position(self) -> Position | None:
        return self._latest

    @property
    def is_running(self) -> bool:
        return self._running

    # ── GPSD backend ────────────────────────────────────────────────────

    async def _poll_gpsd(self) -> Position | None:
        """Query GPSD for TPV (Time-Position-Velocity) data."""
        if not self._gpsd_host:
            return None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self._gpsd_host, self._gpsd_port),
                timeout=2.0,
            )
            writer.write(b'?WATCH={"enable":true,"json":true};\n')
            await writer.drain()
            await asyncio.sleep(0.3)
            writer.write(b"?POLL;\n")
            await writer.drain()

            data = b""
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                try:
                    chunk = await asyncio.wait_for(reader.read(4096), timeout=0.5)
                except TimeoutError:
                    break
                if not chunk:
                    break
                data += chunk
            writer.close()

            for line in data.decode(errors="replace").split("\n"):
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("class") == "TPV" and obj.get("mode", 0) >= 2:
                    lat = obj.get("lat")
                    lon = obj.get("lon")
                    if lat is None or lon is None:
                        continue
                    alt = obj.get("alt")
                    speed_ms = obj.get("speed")
                    track = obj.get("track")
                    return Position(
                        lat=float(lat),
                        lon=float(lon),
                        source="gpsd",
                        altitude_m=float(alt) if alt is not None else None,
                        speed_kmh=float(speed_ms) * 3.6
                        if speed_ms is not None
                        else None,
                        heading_deg=float(track) if track is not None else None,
                    )
            return None
        except (TimeoutError, OSError, ConnectionRefusedError):
            return None
        except Exception:
            logger.debug("GPSD poll error", exc_info=True)
            return None

    # ── Reverse geocoding ───────────────────────────────────────────────

    async def _enrich_with_address(self, pos: Position) -> None:
        """Reverse-geocode a position to get a human-readable address.

        Called in background after each position update. Uses Nominatim
        (OpenStreetMap, free, no API key). Falls back gracefully.
        """
        if pos.address:
            return  # Already has an address
        if pos.lat is None or pos.lon is None:
            return
        try:
            if self._amap is not None and self._amap.available:
                item = await self._amap.reverse_geocode(pos.lat, pos.lon)
                addr = str(item.get("place") or "")
                pos.province = str(item.get("province") or "")
                pos.city = str(item.get("city") or "")
                pos.district = str(item.get("district") or "")
                pos.adcode = str(item.get("adcode") or "")
            else:
                addr = ""
            if not addr:
                addr = await _reverse_geocode(pos.lat, pos.lon)
            if addr:
                pos.address = addr
                # Also update _latest if it's the same position
                if self._latest is pos or (
                    self._latest
                    and self._latest.lat is not None
                    and self._latest.lon is not None
                    and abs(self._latest.lat - pos.lat) < 0.0001
                    and abs(self._latest.lon - pos.lon) < 0.0001
                ):
                    self._latest.address = addr
                logger.info("📍 逆地理编码: %s", addr)
        except Exception:
            # Keep the precise coordinates and fall back to the existing free
            # reverse geocoder instead of allowing AMap to erase a good fix.
            try:
                addr = await _reverse_geocode(pos.lat, pos.lon)
                if addr:
                    pos.address = addr
            except Exception:
                logger.debug("Reverse geocoding failed", exc_info=True)

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                pos = await self._poll_gpsd()
                if pos is not None:
                    self._latest = pos
                    # Enrich GPS fixes with address
                    if not pos.address:
                        asyncio.ensure_future(self._enrich_with_address(pos))
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("GPSD poll error", exc_info=True)
            await asyncio.sleep(self._poll_interval_s)


# ── Reverse geocoding ────────────────────────────────────────────────────────

# Nominatim (OpenStreetMap) — free, no API key, 1 req/s rate limit
_NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/reverse"
    "?format=json&lat={lat}&lon={lon}&zoom=18&accept-language=zh"
)

_reverse_geocode_cache: dict[str, str] = {}  # simple in-memory cache


async def _reverse_geocode(lat: float, lon: float) -> str | None:
    """Convert lat/lon to a human-readable address via Nominatim.

    Returns something like "深圳市南山区科技园" or "Kwai Chung, Hong Kong".
    Cached per (lat, lon) rounded to 4 decimal places (~11m precision).
    """
    key = f"{round(lat, 4)},{round(lon, 4)}"
    if key in _reverse_geocode_cache:
        return _reverse_geocode_cache[key]

    import httpx

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            headers = {"User-Agent": "ChaihuoReachy/1.0 (mobile-vehicle-location)"}
            url = _NOMINATIM_URL.format(lat=lat, lon=lon)
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                logger.debug("Nominatim HTTP %d", resp.status_code)
                return None
            data = resp.json()
            display_name = data.get("display_name", "")
            if not display_name:
                return None
            # Truncate very long addresses
            if len(display_name) > 120:
                # Take the important parts: city/town/village + road
                address = data.get("address", {})
                parts = []
                for k in ("city", "town", "village", "county", "state", "country"):
                    v = address.get(k)
                    if v:
                        parts.append(v)
                        if len(parts) >= 2:
                            break
                if parts:
                    display_name = ", ".join(parts)
                else:
                    display_name = display_name[:120]
            _reverse_geocode_cache[key] = display_name
            return display_name
    except Exception:
        logger.debug("Nominatim lookup error", exc_info=True)
        return None


# ── Factory ──────────────────────────────────────────────────────────────────


def create_location_service(
    *,
    gpsd_enabled: bool = True,
    gpsd_host: str = "127.0.0.1",
    gpsd_port: int = 2947,
    poll_interval_s: float = 2.0,
    gps_fresh_s: float = 30.0,
    browser_fresh_s: float = 60.0,
    amap_web_key: str = "",
    amap_web_private_key: str = "",
    amap_timeout_s: float = 3.0,
    amap_cache_ttl_s: float = 600.0,
) -> LocationService:
    """Create a LocationService from explicit parameters."""
    amap_client = (
        AmapWebClient(
            amap_web_key,
            amap_web_private_key,
            timeout_s=amap_timeout_s,
            cache_ttl_s=amap_cache_ttl_s,
        )
        if amap_web_key.strip()
        else None
    )
    return LocationService(
        gpsd_host=gpsd_host if gpsd_enabled else "",
        gpsd_port=gpsd_port,
        poll_interval_s=poll_interval_s,
        stale_timeout_s=gps_fresh_s,
        browser_stale_timeout_s=browser_fresh_s,
        amap_client=amap_client,
    )
