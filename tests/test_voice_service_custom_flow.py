from __future__ import annotations

from pathlib import Path

import pytest

from app.backend.config import AppConfig
from app.backend.db import AppDatabase
from app.backend.errors import ClientInputError
from app.backend.services.tts_manager import TTSManager
from app.backend.services.voice_service import VoiceService
from app.backend.types import CustomVoiceRequest


def _build_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VoiceService:
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    cfg = AppConfig.from_env()
    cfg.ensure_paths()
    db = AppDatabase(cfg.db_path)
    db.init_schema()
    return VoiceService(cfg, db, TTSManager(cfg))


def test_custom_voice_saved_persists_voice(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    req = CustomVoiceRequest(
        name="vivian-custom",
        speaker="Vivian",
        preview_text="大家好，这是一段测试文本。",
        language="Chinese",
        instruct="语气自然、清晰。",
        save=True,
    )
    result = svc.create_custom_voice(req)

    assert result.voice.mode == "custom"
    assert result.preview_audio_b64
    voices = svc.list_voices()
    assert len(voices) == 1
    assert voices[0].id == result.voice.id


def test_custom_voice_preview_not_saved(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    req = CustomVoiceRequest(
        name="custom-preview-only",
        speaker="Ryan",
        preview_text="This is a preview-only custom voice flow.",
        language="English",
        instruct="Very energetic.",
        save=False,
    )
    result = svc.create_custom_voice(req)
    assert result.voice.mode == "custom"
    assert svc.list_voices() == []
    assert not (svc.config.voices_dir / result.voice.id).exists()


def test_custom_voice_rejects_invalid_speaker(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    req = CustomVoiceRequest(
        name="bad-speaker",
        speaker="NotExists",
        preview_text="hello world",
        language="Auto",
        instruct="",
        save=True,
    )
    with pytest.raises(ClientInputError, match="Unsupported custom speaker"):
        svc.create_custom_voice(req)


def test_custom_voice_save_reuses_preview_cache(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    preview_req = CustomVoiceRequest(
        name="custom-cache-preview",
        speaker="Vivian",
        preview_text="This is a preview-only custom voice flow.",
        language="English",
        instruct="Very energetic.",
        save=False,
    )
    preview_result = svc.create_custom_voice(preview_req)
    assert preview_result.preview_cache_key

    def _should_not_run(*_args, **_kwargs):
        raise RuntimeError("custom preview should be reused from cache")

    svc.tts.generate_custom_voice_preview = _should_not_run  # type: ignore[assignment]

    save_req = CustomVoiceRequest(
        name="custom-cache-saved",
        speaker="Vivian",
        preview_text="This is a preview-only custom voice flow.",
        language="English",
        instruct="Very energetic.",
        save=True,
        preview_cache_key=preview_result.preview_cache_key,
    )
    save_result = svc.create_custom_voice(save_req)
    assert save_result.preview_reused is True
    assert save_result.preview_cache_key is None
    voices = svc.list_voices()
    assert len(voices) == 1
