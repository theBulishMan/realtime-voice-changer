from __future__ import annotations

import importlib

import pytest


def _load_main(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    module = importlib.import_module("app.backend.main")
    return importlib.reload(module)


def _create_voice(client):
    resp = client.post(
        "/api/v1/voices/design",
        json={
            "name": "ptt-voice",
            "voice_prompt": "clear and stable",
            "preview_text": "ptt hello",
            "language": "Auto",
            "save": True,
        },
    )
    assert resp.status_code == 200
    return resp.json()["voice"]["id"]


def test_realtime_ptt_requires_running(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)

    with TestClient(module.app) as client:
        resp = client.post("/api/v1/realtime/ptt", json={"active": True})
        assert resp.status_code == 409


def test_realtime_ptt_updates_state(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)

    with TestClient(module.app) as client:
        voice_id = _create_voice(client)
        start = client.post("/api/v1/realtime/start", json={"voice_id": voice_id, "language": "Auto"})
        assert start.status_code == 200

        on = client.post("/api/v1/realtime/ptt", json={"active": True})
        assert on.status_code == 200
        assert on.json()["ptt_active"] is True

        off = client.post("/api/v1/realtime/ptt", json={"active": False})
        assert off.status_code == 200
        assert off.json()["ptt_active"] is False

        stop = client.post("/api/v1/realtime/stop")
        assert stop.status_code == 200
