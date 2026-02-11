from __future__ import annotations

import importlib

import pytest


def test_metrics_reset_endpoint(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    module = importlib.import_module("app.backend.main")
    module = importlib.reload(module)

    with TestClient(module.app) as client:
        before = client.get("/api/v1/metrics/current")
        assert before.status_code == 200

        reset = client.post("/api/v1/metrics/reset?clear_log=true&clear_db=true")
        assert reset.status_code == 200
        payload = reset.json()
        assert payload["ok"] is True
        assert payload["metrics"]["sample_count"] == 0
