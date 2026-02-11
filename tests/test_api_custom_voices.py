from __future__ import annotations

import importlib

import pytest


def _load_main(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    module = importlib.import_module("app.backend.main")
    return importlib.reload(module)


def test_custom_catalog_endpoint(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)
    with TestClient(module.app) as client:
        resp = client.get("/api/v1/voices/custom/catalog")
        assert resp.status_code == 200
        payload = resp.json()
        speakers = payload.get("speakers") or []
        speaker_ids = {item.get("id") for item in speakers}
        assert "Vivian" in speaker_ids
        assert "Ryan" in speaker_ids
        assert "Auto" in (payload.get("languages") or [])


def test_create_custom_voice_endpoint(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)
    with TestClient(module.app) as client:
        resp = client.post(
            "/api/v1/voices/custom",
            json={
                "name": "api-custom-vivian",
                "speaker": "Vivian",
                "preview_text": "这是官方音色固化流程测试。",
                "language": "Chinese",
                "instruct": "平静自然",
                "save": True,
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["voice"]["mode"] == "custom"
        assert payload["preview_audio_b64"]
