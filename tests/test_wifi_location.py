from __future__ import annotations

from chaihuo_reachy.wifi_scan import parse_nmcli_wifi


def test_parse_nmcli_wifi_normalizes_signal_and_connection() -> None:
    points = parse_nmcli_wifi(
        "*\t00:11:22:33:44:55\t80\tphone-hotspot\n"
        "\t10:20:30:40:50:60\t60\tshop\n"
        "\t20:21:22:23:24:25\t40\toffice\n"
    )
    assert len(points) == 3
    assert points[0].connected
    assert points[0].signal_dbm == -60
    assert sum(not item.connected for item in points) == 2


def test_parse_nmcli_wifi_discards_invalid_and_duplicate_bssid() -> None:
    points = parse_nmcli_wifi(
        "\t00:11:22:33:44:55\t50\ta\n"
        "\t00:11:22:33:44:55\t40\tb\n"
        "\tFF:FF:FF:FF:FF:FF\t90\tbad\n"
    )
    assert [item.ssid for item in points] == ["a"]
