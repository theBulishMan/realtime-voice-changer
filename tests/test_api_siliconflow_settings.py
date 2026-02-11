from __future__ import annotations

import importlib

import pytest


def _load_main(monkeypatch: pytest.MonkeyPatch, data_dir):
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(data_dir))
    module = importlib.import_module("app.backend.main")
    return importlib.reload(module)


def test_get_siliconflow_settings_masks_key(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    data_dir = tmp_path / "data"
    monkeypatch.setenv("RVC_TEXT_CORRECTION_PROVIDER", "siliconflow")
    monkeypatch.setenv("RVC_SILICONFLOW_API_KEY", "sk-1234567890abcdef")
    module = _load_main(monkeypatch, data_dir)

    with TestClient(module.app) as client:
        resp = client.get("/api/v1/settings/siliconflow")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["text_correction_provider"] == "siliconflow"
        assert payload["api_key_present"] is True
        assert payload["api_key_masked"]
        assert "1234567890abcdef" not in payload["api_key_masked"]
        assert payload["compat_mode"] in {"strict_openai", "siliconflow"}
        assert payload["endpoint_model_name"]


def test_patch_siliconflow_settings_persists_after_restart(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    data_dir = tmp_path / "data"
    module = _load_main(monkeypatch, data_dir)

    with TestClient(module.app) as client:
        resp = client.patch(
            "/api/v1/settings/siliconflow",
            json={
                "text_correction_provider": "siliconflow",
                "siliconflow_api_base": "https://api.siliconflow.cn/v1",
                "endpoint_model_name": "zai-org/GLM-4.5-Air",
                "credential_name": "API KEY 1",
                "model_display_name": "GLM 快速纠错",
                "context_length": 131072,
                "max_tokens": 8192,
                "compat_mode": "strict_openai",
                "thinking_enabled": False,
                "siliconflow_timeout_s": 15,
                "siliconflow_api_key": "sk-persisted-key-1234",
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["text_correction_provider"] == "siliconflow"
        assert payload["siliconflow_model"] == "zai-org/GLM-4.5-Air"
        assert payload["endpoint_model_name"] == "zai-org/GLM-4.5-Air"
        assert payload["credential_name"] == "API KEY 1"
        assert payload["model_display_name"] == "GLM 快速纠错"
        assert payload["context_length"] == 131072
        assert payload["max_tokens"] == 8192
        assert payload["compat_mode"] == "strict_openai"
        assert payload["thinking_enabled"] is False
        assert payload["siliconflow_timeout_s"] == 15
        assert payload["api_key_present"] is True

    monkeypatch.setenv("RVC_TEXT_CORRECTION_PROVIDER", "local")
    monkeypatch.setenv("RVC_SILICONFLOW_MODEL", "dummy-model")
    monkeypatch.delenv("RVC_SILICONFLOW_API_KEY", raising=False)
    module = _load_main(monkeypatch, data_dir)

    with TestClient(module.app) as client:
        resp = client.get("/api/v1/settings/siliconflow")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["text_correction_provider"] == "siliconflow"
        assert payload["siliconflow_model"] == "zai-org/GLM-4.5-Air"
        assert payload["endpoint_model_name"] == "zai-org/GLM-4.5-Air"
        assert payload["model_display_name"] == "GLM 快速纠错"
        assert payload["max_tokens"] == 8192
        assert payload["siliconflow_timeout_s"] == 15
        assert payload["api_key_present"] is True


def test_patch_siliconflow_settings_can_clear_key(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    data_dir = tmp_path / "data"
    module = _load_main(monkeypatch, data_dir)

    with TestClient(module.app) as client:
        save = client.patch(
            "/api/v1/settings/siliconflow",
            json={"text_correction_provider": "siliconflow", "siliconflow_api_key": "sk-clear-me"},
        )
        assert save.status_code == 200
        assert save.json()["api_key_present"] is True

        clear = client.patch(
            "/api/v1/settings/siliconflow",
            json={"clear_api_key": True},
        )
        assert clear.status_code == 200
        cleared = clear.json()
        assert cleared["api_key_present"] is False
        assert cleared["api_key_masked"] == ""
