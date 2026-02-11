from __future__ import annotations

import importlib

import pytest


def _load_main(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    module = importlib.import_module("app.backend.main")
    return importlib.reload(module)


def _create_voice(client) -> str:
    resp = client.post(
        "/api/v1/voices/design",
        json={
            "name": "delete-me",
            "voice_prompt": "clear and neutral",
            "preview_text": "hello delete",
            "language": "Auto",
            "save": True,
        },
    )
    assert resp.status_code == 200
    return resp.json()["voice"]["id"]


def test_delete_voice_success(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)

    with TestClient(module.app) as client:
        voice_id = _create_voice(client)
        delete_resp = client.delete(f"/api/v1/voices/{voice_id}")
        assert delete_resp.status_code == 200
        payload = delete_resp.json()
        assert payload["ok"] is True
        assert payload["voice_id"] == voice_id

        voices = client.get("/api/v1/voices").json()
        assert all(v["id"] != voice_id for v in voices)


def test_delete_voice_missing(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)

    with TestClient(module.app) as client:
        delete_resp = client.delete("/api/v1/voices/not-exist")
        assert delete_resp.status_code == 404
