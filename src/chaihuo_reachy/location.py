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
from typing import Any

logger = logging.getLogger("chaihuo_reachy.location")

# ── Data ────────────────────────────────────────────────────────────────────


@dataclass
class Position:
    lat: float
    lon: float
    source: str = "unknown"  # "gpsd" | "browser" | "manual" | "unavailable"
    timestamp: float = field(default_factory=time.time)
    altitude_m: float | None = None
    speed_kmh: float | None = None
    heading_deg: float | None = None
    accuracy_m: float | None = None  # browser provides this
    address: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "source": self.source,
            "timestamp": round(self.timestamp, 1),
            "altitude_m": round(self.altitude_m, 1) if self.altitude_m is not None else None,
            "speed_kmh": round(self.speed_kmh, 1) if self.speed_kmh is not None else None,
            "heading_deg": round(self.heading_deg, 1) if self.heading_deg is not None else None,
            "accuracy_m": round(self.accuracy_m, 1) if self.accuracy_m is not None else None,
            "address": self.address,
        }

    def to_human(self) -> str:
        """Human-readable one-line description."""
        parts = [f"纬度 {self.lat:.6f}°, 经度 {self.lon:.6f}°"]
        if self.address:
            parts.append(f"（{self.address}）")
        if self.altitude_m is not None:
            parts.append(f"海拔 {self.altitude_m:.0f}m")
        if self.speed_kmh is not None and self.speed_kmh > 0.5:
            parts.append(f"速度 {self.speed_kmh:.1f}km/h")
        if self.accuracy_m is not None:
            parts.append(f"精度 ±{self.accuracy_m:.0f}m")
        source_labels = {
            "gpsd": "🛰 GPS卫星",
            "browser": "📱 设备定位",
            "manual": "📍 手动设置",
            "unavailable": "❌ 无信号",
        }
        parts.append(source_labels.get(self.source, self.source))
        return " ".join(parts)

    def to_llm_context(self) -> str:
        """Formatted context string for LLM consumption."""
        lines = [
            f"当前位置：纬度 {self.lat:.6f}，经度 {self.lon:.6f}",
            f"数据来源：{self.source}",
            f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
        ]
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
        lines.append(
            "如果用户询问位置相关问题，请基于以上坐标信息回答。"
            "你可以提及城市名、大致区域，但不要编造具体街道名称。"
        )
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
    ) -> None:
        self._gpsd_host = gpsd_host
        self._gpsd_port = gpsd_port
        self._poll_interval_s = poll_interval_s
        self._stale_timeout_s = stale_timeout_s

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
        logger.info("📍 定位服务已启动 (GPSD=%s:%d, Browser=ready)",
                     self._gpsd_host, self._gpsd_port)

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

    # ── Public API ──────────────────────────────────────────────────────

    def set_browser_position(
        self, lat: float, lon: float,
        accuracy_m: float | None = None,
        altitude_m: float | None = None,
        heading_deg: float | None = None,
        speed_kmh: float | None = None,
    ) -> Position:
        """Called by Dashboard JS via WebSocket when browser reports location."""
        pos = Position(
            lat=lat, lon=lon, source="browser",
            accuracy_m=accuracy_m,
            altitude_m=altitude_m,
            heading_deg=heading_deg,
            speed_kmh=speed_kmh,
        )
        self._browser_pos = pos
        # Browser position is authoritative until GPSD gives a fix
        if self._latest is None or self._latest.source in ("browser", "unavailable"):
            self._latest = pos
        logger.info("📍 浏览器定位: %.6f, %.6f (±%.0fm)", lat, lon, accuracy_m or 0)
        # Reverse geocode in background
        asyncio.ensure_future(self._enrich_with_address(pos))
        return pos

    async def get_position(self) -> Position:
        """Get the latest known position."""
        if self._latest is not None:
            age = time.time() - self._latest.timestamp
            if age < self._stale_timeout_s:
                return self._latest
        # Try GPSD once
        pos = await self._poll_gpsd()
        if pos is not None:
            self._latest = pos
            return pos
        # Fall back to browser or last known
        if self._browser_pos is not None:
            return self._browser_pos
        if self._latest is not None:
            return self._latest
        return Position(lat=0.0, lon=0.0, source="unavailable",
                        address="等待 GPS 或浏览器上报位置...")

    def set_manual(self, lat: float, lon: float) -> Position:
        """Manually override position."""
        self._manual_lat = lat
        self._manual_lon = lon
        self._latest = Position(
            lat=lat, lon=lon, source="manual",
            address=f"手动设置 ({lat:.4f}, {lon:.4f})",
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
                except asyncio.TimeoutError:
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
                        speed_kmh=float(speed_ms) * 3.6 if speed_ms is not None else None,
                        heading_deg=float(track) if track is not None else None,
                    )
            return None
        except (asyncio.TimeoutError, OSError, ConnectionRefusedError):
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
        try:
            addr = await _reverse_geocode(pos.lat, pos.lon)
            if addr:
                pos.address = addr
                # Also update _latest if it's the same position
                if self._latest is pos or (
                    self._latest
                    and abs(self._latest.lat - pos.lat) < 0.0001
                    and abs(self._latest.lon - pos.lon) < 0.0001
                ):
                    self._latest.address = addr
                logger.info("📍 逆地理编码: %s", addr)
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
    import math as _math
    key = f"{round(lat, 4)},{round(lon, 4)}"
    if key in _reverse_geocode_cache:
        return _reverse_geocode_cache[key]

    import httpx
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            headers = {
                "User-Agent": "ChaihuoReachy/1.0 (mobile-vehicle-location)"
            }
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
) -> LocationService:
    """Create a LocationService from explicit parameters."""
    return LocationService(
        gpsd_host=gpsd_host if gpsd_enabled else "",
        gpsd_port=gpsd_port,
        poll_interval_s=poll_interval_s,
    )
