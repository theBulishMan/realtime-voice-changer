from __future__ import annotations

import importlib

import pytest


def _load_main(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    module = importlib.import_module("app.backend.main")
    return importlib.reload(module)


def _create_design_voice(client):
    resp = client.post(
        "/api/v1/voices/design",
        json={
            "name": "sim-voice",
            "voice_prompt": "clean and stable",
            "preview_text": "hello smoke",
            "language": "Auto",
            "save": True,
        },
    )
    assert resp.status_code == 200
    return resp.json()["voice"]["id"]


def test_realtime_simulate_requires_running(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)

    with TestClient(module.app) as client:
        voice_id = _create_design_voice(client)
        assert voice_id
        resp = client.post("/api/v1/realtime/simulate", json={"count": 1})
        assert resp.status_code == 409


def test_realtime_simulate_generates_metrics(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)

    with TestClient(module.app) as client:
        voice_id = _create_design_voice(client)
        start = client.post("/api/v1/realtime/start", json={"voice_id": voice_id, "language": "Auto"})
        assert start.status_code == 200

        resp = client.post(
            "/api/v1/realtime/simulate",
            json={"count": 3, "duration_ms": 500, "amplitude": 0.08, "frequency_hz": 220.0},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["ok"] is True
        assert payload["injected_segments"] == 3
        assert payload["metrics"]["sample_count"] >= 3

        stop = client.post("/api/v1/realtime/stop")
        assert stop.status_code == 200


def test_realtime_start_with_fake_capture_input_enabled(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    monkeypatch.setenv("RVC_FAKE_CAPTURE_INPUT", "1")
    module = _load_main(monkeypatch, tmp_path)

    with TestClient(module.app) as client:
        voice_id = _create_design_voice(client)
        start = client.post("/api/v1/realtime/start", json={"voice_id": voice_id, "language": "Auto"})
        assert start.status_code == 200
        stop = client.post("/api/v1/realtime/stop")
        assert stop.status_code == 200
