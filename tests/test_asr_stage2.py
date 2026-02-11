from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.backend.asr.asr_stage2 import Stage2ASR
from app.backend.config import AppConfig


class _Seg:
    def __init__(self, text: str) -> None:
        self.text = text


def test_stage2_suppresses_low_confidence_auto_thank_you(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RVC_FAKE_MODE", "0")
    monkeypatch.setenv("RVC_ASR_BACKEND", "faster_whisper")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))
    cfg = AppConfig.from_env()
    cfg.ensure_paths()
    stage2 = Stage2ASR(cfg)

    calls = {"n": 0}

    class DummyModel:
        @staticmethod
        def transcribe(audio, language, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return iter([_Seg("thank you")]), SimpleNamespace(language_probability=0.41)
            return iter([_Seg("")]), SimpleNamespace(language_probability=0.0)

    stage2._model = DummyModel()
    wav = np.full(16000, 0.02, dtype=np.float32)
    text = stage2.transcribe_final(wav, "Auto")
    assert text == ""
    assert calls["n"] == 2


def test_stage2_keeps_high_confidence_auto_thank_you(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RVC_FAKE_MODE", "0")
    monkeypatch.setenv("RVC_ASR_BACKEND", "faster_whisper")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))
    cfg = AppConfig.from_env()
    cfg.ensure_paths()
    stage2 = Stage2ASR(cfg)

    class DummyModel:
        @staticmethod
        def transcribe(audio, language, **kwargs):
            return iter([_Seg("thank you")]), SimpleNamespace(language_probability=0.92)

    stage2._model = DummyModel()
    wav = np.full(16000, 0.03, dtype=np.float32)
    text = stage2.transcribe_final(wav, "Auto")
    assert text == "thank you"
