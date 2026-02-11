from __future__ import annotations

import importlib

import pytest

from app.backend.errors import ClientInputError


def _load_main(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    module = importlib.import_module("app.backend.main")
    return importlib.reload(module)


def test_design_voice_client_error_returns_400(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)

    def _raise(*args, **kwargs):
        raise ClientInputError("bad design request")

    monkeypatch.setattr(module.voice_service, "create_design_voice", _raise)
    with TestClient(module.app) as client:
        resp = client.post(
            "/api/v1/voices/design",
            json={
                "name": "demo",
                "voice_prompt": "warm and clear",
                "preview_text": "hello world",
                "language": "Auto",
                "save": True,
            },
        )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "bad design request"


def test_clone_voice_client_error_returns_400(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)

    def _raise(*args, **kwargs):
        raise ClientInputError("bad clone request")

    monkeypatch.setattr(module.voice_service, "create_clone_voice", _raise)
    with TestClient(module.app) as client:
        resp = client.post(
            "/api/v1/voices/clone",
            data={"name": "demo", "language": "Auto", "ref_text": "hello"},
            files={"audio_file": ("ref.wav", b"dummy", "audio/wav")},
        )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "bad clone request"


def test_custom_voice_client_error_returns_400(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)

    def _raise(*args, **kwargs):
        raise ClientInputError("bad custom request")

    monkeypatch.setattr(module.voice_service, "create_custom_voice", _raise)
    with TestClient(module.app) as client:
        resp = client.post(
            "/api/v1/voices/custom",
            json={
                "name": "demo",
                "speaker": "Vivian",
                "preview_text": "hello",
                "language": "Auto",
                "instruct": "",
                "save": True,
            },
        )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "bad custom request"
