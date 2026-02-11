from __future__ import annotations

from app.backend.config import AppConfig
from app.backend.services.text_corrector import TextCorrector


def test_text_corrector_uses_siliconflow_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("RVC_FAKE_MODE", "0")
    monkeypatch.setenv("RVC_TEXT_CORRECTION_PROVIDER", "siliconflow")
    monkeypatch.setenv("RVC_SILICONFLOW_API_KEY", "dummy-key")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))
    config = AppConfig.from_env()
    config.ensure_paths()
    corrector = TextCorrector(config)

    called = {"ok": False}

    def _fake_chat(*, messages, max_tokens, temperature, top_p):
        called["ok"] = True
        assert isinstance(messages, list) and len(messages) >= 2
        assert max_tokens >= 32
        return "聪明啊！"

    monkeypatch.setattr(corrector._siliconflow, "chat", _fake_chat)
    result = corrector.correct_text("聪明 阿", language_hint="Chinese")
    assert called["ok"] is True
    assert result.used_model is True
    assert result.text == "聪明啊！"
