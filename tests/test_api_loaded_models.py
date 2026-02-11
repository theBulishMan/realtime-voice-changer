from __future__ import annotations

import importlib

import pytest


def test_loaded_models_endpoint_exists(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    module = importlib.import_module("app.backend.main")
    module = importlib.reload(module)

    with TestClient(module.app) as client:
        resp = client.get("/api/v1/models/loaded")
        assert resp.status_code == 200
        payload = resp.json()
        assert "gpu_available" in payload
        assert "cuda_memory_allocated_mb" in payload
        assert isinstance(payload.get("models"), list)


def test_unload_unknown_model_key_returns_404(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    module = importlib.import_module("app.backend.main")
    module = importlib.reload(module)

    with TestClient(module.app) as client:
        resp = client.delete("/api/v1/models/loaded/not_exists")
        assert resp.status_code == 404
        assert "Unsupported model key" in str(resp.json().get("detail", ""))


def test_unload_rejected_while_realtime_running(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    module = importlib.import_module("app.backend.main")
    module = importlib.reload(module)

    class _RunningState:
        running = True

    class _RunningEngine:
        def state(self):
            return _RunningState()

    monkeypatch.setattr(module, "_get_engine", lambda: _RunningEngine())

    with TestClient(module.app) as client:
        resp = client.delete("/api/v1/models/loaded/tts_base")
        assert resp.status_code == 409
        assert "请先停止实时链路" in str(resp.json().get("detail", ""))
