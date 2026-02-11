from __future__ import annotations

import importlib

import pytest


def test_health_endpoint(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    module = importlib.import_module("app.backend.main")
    module = importlib.reload(module)
    with TestClient(module.app) as client:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["fake_mode"] is True
        assert payload["gpu_required"] is False
        assert "gpu_available" in payload
        assert payload["asr_backend"] == "qwen3"
        assert payload["model_device"] == "cuda:0"
        assert payload["asr_device"] == "cuda"


def test_frontend_routes(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    module = importlib.import_module("app.backend.main")
    module = importlib.reload(module)
    with TestClient(module.app) as client:
        root_resp = client.get("/")
        assert root_resp.status_code == 200
        assert "实时变声器" in root_resp.text
        assert "/static/main.js" in root_resp.text

        voices_resp = client.get("/voices")
        assert voices_resp.status_code == 200
        assert "音色工作台" in voices_resp.text
        assert "/static/voices.js" in voices_resp.text

        settings_resp = client.get("/settings")
        assert settings_resp.status_code == 200
        assert "/static/settings.js" in settings_resp.text
