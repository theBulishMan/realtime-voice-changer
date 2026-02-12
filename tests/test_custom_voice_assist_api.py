from __future__ import annotations

import importlib

import pytest


def _load_main(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("RVC_SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    module = importlib.import_module("app.backend.main")
    return importlib.reload(module)


def test_custom_voice_assist_fallback(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)

    with TestClient(module.app) as client:
        resp = client.post(
            "/api/v1/voices/custom/assist",
            json={
                "brief": "年轻女声，温柔但有力量，情绪克制，适合直播聊天",
                "language": "Chinese",
                "speaker": "Vivian",
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["source"] == "fallback"
        assert len(str(payload["instruct"])) > 8
        assert len(str(payload["preview_text"])) > 10

