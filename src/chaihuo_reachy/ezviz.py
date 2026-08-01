"""Non-blocking EZVIZ rear-view camera capture."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from chaihuo_reachy.config import Config

logger = logging.getLogger("chaihuo_reachy.ezviz")

EZVIZ_BASE = "https://open.ys7.com/api/lapp"
_token_cache: tuple[str, float] | None = None
_token_lock: asyncio.Lock | None = None


def _lock() -> asyncio.Lock:
    global _token_lock
    if _token_lock is None:
        _token_lock = asyncio.Lock()
    return _token_lock


async def _get_token(cfg: Config, client: httpx.AsyncClient) -> str:
    global _token_cache
    now = time.monotonic()
    if _token_cache is not None and now < _token_cache[1]:
        return _token_cache[0]

    async with _lock():
        now = time.monotonic()
        if _token_cache is not None and now < _token_cache[1]:
            return _token_cache[0]
        if not cfg.ezviz_app_key or not cfg.ezviz_app_secret:
            raise RuntimeError("EZVIZ_APP_KEY / EZVIZ_APP_SECRET 未配置")
        response = await client.post(
            f"{EZVIZ_BASE}/token/get",
            data={"appKey": cfg.ezviz_app_key, "appSecret": cfg.ezviz_app_secret},
        )
        response.raise_for_status()
        data = response.json()
        if str(data.get("code")) != "200":
            raise RuntimeError(f"EZVIZ token failed: {data.get('msg', '')}")
        token = str(data["data"]["accessToken"])
        # The service token is long lived; a short local cache reduces rate
        # limiting while still recovering quickly from revocation.
        _token_cache = (token, now + 4 * 60)
        return token


async def capture_rear_view(cfg: Config, *, total_timeout_s: float = 20.0) -> bytes:
    """Capture a fresh rear-view JPEG without blocking the asyncio loop."""
    if not cfg.ezviz_device_serial:
        raise RuntimeError("EZVIZ_DEVICE_SERIAL 未配置")

    async def _capture() -> bytes:
        timeout = httpx.Timeout(8.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            token = await _get_token(cfg, client)
            response = await client.post(
                f"{EZVIZ_BASE}/device/capture",
                data={
                    "accessToken": token,
                    "deviceSerial": cfg.ezviz_device_serial,
                    "channelNo": cfg.ezviz_channel_no or "1",
                },
            )
            response.raise_for_status()
            data = response.json()
            if str(data.get("code")) != "200":
                raise RuntimeError(f"EZVIZ capture failed: {data.get('msg', '')}")
            picture_url = str((data.get("data") or {}).get("picUrl") or "")
            if not picture_url:
                raise RuntimeError("EZVIZ 抓拍结果没有图片地址")

            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    picture = await client.get(picture_url)
                    picture.raise_for_status()
                    jpeg = picture.content
                    if not jpeg.startswith(b"\xff\xd8"):
                        raise ValueError("后视摄像头返回的内容不是 JPEG")
                    end = jpeg.rfind(b"\xff\xd9")
                    return jpeg[: end + 2] if end >= 2 else jpeg
                except Exception as exc:
                    last_error = exc
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (attempt + 1))
            raise RuntimeError("后视照片下载失败") from last_error

    return await asyncio.wait_for(_capture(), timeout=total_timeout_s)

