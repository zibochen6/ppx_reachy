"""NetworkManager Wi-Fi scan with privacy-safe normalized output."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass


_BSSID = re.compile(r"^(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


@dataclass(frozen=True)
class WifiAccessPoint:
    bssid: str
    signal_dbm: int
    ssid: str = ""
    connected: bool = False

    def to_amap(self) -> dict[str, object]:
        return {
            "bssid": self.bssid,
            "signal_dbm": self.signal_dbm,
            "ssid": self.ssid,
            "connected": self.connected,
        }


def parse_nmcli_wifi(output: str) -> list[WifiAccessPoint]:
    """Parse tab-separated nmcli output; reject malformed/multicast BSSIDs."""
    result: list[WifiAccessPoint] = []
    seen: set[str] = set()
    for line in output.splitlines():
        parts = line.rstrip().split("\t", 3)
        if len(parts) < 3:
            continue
        in_use, bssid, signal = parts[:3]
        ssid = parts[3] if len(parts) > 3 else ""
        bssid = bssid.strip().upper()
        if not _BSSID.match(bssid) or bssid in seen:
            continue
        first_octet = int(bssid[:2], 16)
        if first_octet & 1:  # multicast/broadcast, never a physical AP
            continue
        try:
            quality = max(0, min(100, int(signal)))
        except ValueError:
            continue
        # NetworkManager exposes quality 0..100. This widely used conversion
        # is sufficient for the AMap fingerprint (-100..-50 dBm).
        dbm = max(-100, min(-30, quality // 2 - 100))
        result.append(WifiAccessPoint(bssid, dbm, ssid.strip(), in_use.strip() == "*"))
        seen.add(bssid)
    return sorted(result, key=lambda item: item.signal_dbm, reverse=True)


class NetworkManagerWifiScanner:
    async def scan(self) -> list[WifiAccessPoint]:
        try:
            process = await asyncio.create_subprocess_exec(
                "nmcli",
                "-t",
                "--escape",
                "no",
                "-f",
                "IN-USE,BSSID,SIGNAL,SSID",
                "device",
                "wifi",
                "list",
                "--rescan",
                "yes",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5.0)
        except (FileNotFoundError, OSError, asyncio.TimeoutError):
            return []
        if process.returncode != 0:
            return []
        # Convert nmcli's colon mode to a stable tab representation. BSSID
        # colons are escaped as \: unless --escape=no; request multiline mode
        # is not portable, so recover fields from the right-side regex.
        text = stdout.decode(errors="replace")
        rows: list[str] = []
        pattern = re.compile(
            r"^(\*?):((?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}):(\d+):(.*)$"
        )
        for line in text.splitlines():
            match = pattern.match(line)
            if match:
                rows.append("\t".join(match.groups()))
        return parse_nmcli_wifi("\n".join(rows))
