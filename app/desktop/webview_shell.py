from __future__ import annotations

import os
import sys
import time
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import uvicorn

DEFAULT_URL = "http://127.0.0.1:8787"
DEFAULT_EMBED_BOOT_TIMEOUT_S = 120.0


@dataclass
class BackendRuntime:
    server: uvicorn.Server
    thread: threading.Thread


def _compute_packaged_env(
    *,
    frozen: bool,
    meipass: str | None,
    local_app_data: str | None,
    env: dict[str, str],
) -> dict[str, str]:
    if not frozen:
        return {}

    updates: dict[str, str] = {}
    if "RVC_FRONTEND_DIR" not in env and meipass:
        frontend_dir = Path(meipass) / "app" / "frontend"
        if frontend_dir.exists():
            updates["RVC_FRONTEND_DIR"] = str(frontend_dir)

    if "RVC_DATA_DIR" not in env:
        if local_app_data:
            base = Path(local_app_data)
        else:
            base = Path.home() / "AppData" / "Local"
        updates["RVC_DATA_DIR"] = str(base / "RealtimeVoiceChanger" / "data")
    return updates


def _prepare_runtime_env() -> None:
    frozen = bool(getattr(sys, "frozen", False))
    meipass = getattr(sys, "_MEIPASS", None)
    updates = _compute_packaged_env(
        frozen=frozen,
        meipass=meipass,
        local_app_data=os.getenv("LOCALAPPDATA"),
        env=dict(os.environ),
    )
    for key, value in updates.items():
        os.environ[key] = value


def _health_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/v1/health"


def _is_backend_ready(base_url: str, timeout_seconds: float = 1.2) -> bool:
    req = urllib.request.Request(_health_url(base_url), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            return int(response.status) == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


def _wait_backend_ready(base_url: str, timeout_seconds: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _is_backend_ready(base_url):
            return True
        time.sleep(0.25)
    return False


def _embed_boot_timeout_seconds() -> float:
    raw = os.getenv("RVC_EMBED_BOOT_TIMEOUT_S", str(DEFAULT_EMBED_BOOT_TIMEOUT_S)).strip()
    try:
        value = float(raw)
    except Exception:
        value = DEFAULT_EMBED_BOOT_TIMEOUT_S
    return max(10.0, min(600.0, value))


def _is_localhost_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost"}


def _host_and_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is not None:
        return host, int(parsed.port)
    if parsed.scheme == "https":
        return host, 443
    return host, 80


def _resolve_webview_gui() -> str | None:
    raw = os.getenv("RVC_WEBVIEW_GUI", "auto").strip().lower()
    if raw in {"", "auto", "default"}:
        return None
    allowed = {"edgechromium", "cef", "qt", "gtk", "mshtml"}
    return raw if raw in allowed else None


def _start_backend(base_url: str) -> BackendRuntime:
    host, port = _host_and_port(base_url)
    config = uvicorn.Config(
        "app.backend.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="warning",
    )
    server = uvicorn.Server(config=config)
    thread = threading.Thread(target=server.run, daemon=True, name="rvc-embedded-backend")
    thread.start()
    return BackendRuntime(server=server, thread=thread)


def launch_webview_shell(
    base_url: str = DEFAULT_URL,
    *,
    width: int = 1360,
    height: int = 860,
) -> None:
    _prepare_runtime_env()
    runtime: BackendRuntime | None = None
    owns_backend = False

    if not _is_backend_ready(base_url) and _is_localhost_url(base_url):
        runtime = _start_backend(base_url)
        owns_backend = True
        timeout_s = _embed_boot_timeout_seconds()
        if not _wait_backend_ready(base_url, timeout_seconds=timeout_s):
            runtime.server.should_exit = True
            runtime.thread.join(timeout=5.0)
            raise RuntimeError(
                f"Embedded backend failed to start within {timeout_s:.0f}s: {base_url}"
            )

    try:
        import webview
    except Exception as exc:
        if runtime is not None:
            runtime.server.should_exit = True
            runtime.thread.join(timeout=5.0)
        raise RuntimeError(
            "pywebview is required for web-shell desktop mode. Install with: pip install pywebview"
        ) from exc

    window = webview.create_window(
        "实时变声器",
        url=base_url,
        width=width,
        height=height,
        min_size=(1080, 720),
        text_select=True,
        resizable=True,
    )

    if owns_backend and runtime is not None:
        window.events.closed += lambda: setattr(runtime.server, "should_exit", True)

    try:
        gui = _resolve_webview_gui()
        if gui:
            webview.start(gui=gui, debug=False)
        else:
            webview.start(debug=False)
    finally:
        if owns_backend and runtime is not None:
            runtime.server.should_exit = True
            runtime.thread.join(timeout=6.0)
