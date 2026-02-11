from __future__ import annotations

from app.desktop.webview_shell import (
    _embed_boot_timeout_seconds,
    _compute_packaged_env,
    _health_url,
    _host_and_port,
    _is_localhost_url,
)


def test_webview_shell_health_url() -> None:
    assert _health_url("http://127.0.0.1:8787") == "http://127.0.0.1:8787/api/v1/health"
    assert _health_url("http://127.0.0.1:8787/") == "http://127.0.0.1:8787/api/v1/health"


def test_webview_shell_localhost_detection() -> None:
    assert _is_localhost_url("http://127.0.0.1:8787")
    assert _is_localhost_url("http://localhost:8787")
    assert not _is_localhost_url("http://192.168.1.88:8787")
    assert not _is_localhost_url("https://example.com")


def test_webview_shell_host_port_resolution() -> None:
    assert _host_and_port("http://127.0.0.1:8787") == ("127.0.0.1", 8787)
    assert _host_and_port("http://localhost") == ("localhost", 80)
    assert _host_and_port("https://localhost") == ("localhost", 443)


def test_compute_packaged_env_non_frozen() -> None:
    updates = _compute_packaged_env(
        frozen=False,
        meipass=None,
        local_app_data=None,
        env={},
    )
    assert updates == {}


def test_compute_packaged_env_defaults_when_frozen() -> None:
    updates = _compute_packaged_env(
        frozen=True,
        meipass=None,
        local_app_data="C:\\Users\\tester\\AppData\\Local",
        env={},
    )
    assert updates["RVC_DATA_DIR"].endswith("RealtimeVoiceChanger\\data")


def test_compute_packaged_env_respects_existing_env() -> None:
    updates = _compute_packaged_env(
        frozen=True,
        meipass=None,
        local_app_data="C:\\Users\\tester\\AppData\\Local",
        env={"RVC_DATA_DIR": "D:\\custom\\data"},
    )
    assert "RVC_DATA_DIR" not in updates


def test_embed_boot_timeout_seconds_default(monkeypatch) -> None:
    monkeypatch.delenv("RVC_EMBED_BOOT_TIMEOUT_S", raising=False)
    assert _embed_boot_timeout_seconds() >= 10.0


def test_embed_boot_timeout_seconds_clamps(monkeypatch) -> None:
    monkeypatch.setenv("RVC_EMBED_BOOT_TIMEOUT_S", "1")
    assert _embed_boot_timeout_seconds() == 10.0
    monkeypatch.setenv("RVC_EMBED_BOOT_TIMEOUT_S", "9999")
    assert _embed_boot_timeout_seconds() == 600.0
    monkeypatch.setenv("RVC_EMBED_BOOT_TIMEOUT_S", "75")
    assert _embed_boot_timeout_seconds() == 75.0
