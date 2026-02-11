from __future__ import annotations

import importlib

import pytest

from app.backend.db import AppDatabase


def test_legacy_realtime_settings_are_migrated(monkeypatch, tmp_path) -> None:
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient

    data_dir = tmp_path / "data"
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(data_dir))
    data_dir.mkdir(parents=True, exist_ok=True)

    db = AppDatabase(data_dir / "app.db")
    db.init_schema()
    db.upsert_setting(
        "realtime_settings",
        {
            "input_device_id": None,
            "monitor_device_id": None,
            "virtual_mic_device_id": None,
            "monitor_enabled": False,
            "vad_silence_ms": 240,
            "max_segment_ms": 12000,
        },
    )
    db.close()

    module = importlib.import_module("app.backend.main")
    module = importlib.reload(module)
    with TestClient(module.app) as client:
        resp = client.get("/api/v1/realtime/state")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["settings"]["vad_silence_ms"] == 140
        assert payload["settings"]["max_segment_ms"] == 2800


def test_virtual_mic_setting_is_migrated_to_preferred(monkeypatch, tmp_path) -> None:
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient

    data_dir = tmp_path / "data"
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(data_dir))
    data_dir.mkdir(parents=True, exist_ok=True)

    db = AppDatabase(data_dir / "app.db")
    db.init_schema()
    db.upsert_setting(
        "realtime_settings",
        {
            "input_device_id": 1,
            "monitor_device_id": 2,
            "virtual_mic_device_id": 3,
            "monitor_enabled": False,
            "vad_silence_ms": 180,
            "max_segment_ms": 4000,
        },
    )
    db.close()

    module = importlib.import_module("app.backend.main")
    module = importlib.reload(module)
    monkeypatch.setattr(
        module,
        "list_audio_devices",
        lambda _hint: type("Resp", (), {"virtual_mic_output_id": 9})(),
    )
    with TestClient(module.app) as client:
        resp = client.get("/api/v1/realtime/state")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["settings"]["virtual_mic_device_id"] == 9
